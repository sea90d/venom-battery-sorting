"""Local image console for T0 and three-battery sequence field tests.

The console deliberately runs on localhost and delegates robot work to the
selected closed-loop runner. It never duplicates motion or gripper logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "web_console" / "index.html"
OUTPUT_ROOT = ROOT / "output" / "competition_console"
RUNNER = ROOT / "tools" / "run_smooth_t0.py"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
# The console selects the timing-trial dynamics. A changed Home/trajectory
# requires its own physical validation; this selection does not certify it.
FORMAL_PROFILE = "ten-second"
SEQUENCE_CONFIG = ROOT / 'config' / 'closed_loop_sequence.json'
TASK_ASSIGNMENTS = {'basic': {'T0': 'red'}, 'sequence': {'P1': 'yellow', 'P2': 'green', 'P3': 'blue'}}


EVENT_PHASES = {
    "startup_empty_reset_requested": "正在初始化夹爪",
    "startup_empty_reset_complete": "夹爪初始化完成",
    "startup_return_plan": "正在返回观察位",
    "startup_home_complete": "已到达折叠待机位",
    "initial_frame_rejected": "正在重新识别现场",
    "initial_window_rejected": "正在复核识别结果",
    "target_frame_rejected": "T0 画面波动，正在自动重试",
    "initial_consensus": "现场识别一致",
    "scene_frame_rejected": "当前深度帧波动，正在同一位置重拍",
    "vision_pass": "坐标计算完成",
    "pick_full_open": "夹爪已完全张开",
    "prepared_waiting_for_start": "设备准备完成，正在自动触发任务",
    "start_trigger": "已触发任务，正在重新采图",
    "cycle_started": "自动抓取进行中",
    "grasped": "红色电池已夹稳",
    "lifted_hold_complete": "举升稳定判定通过",
    "released": "电池已释放，正在稳定判定",
    "home_arrived": "机械臂已折回 Home，计时完成",
    "field_test_started": "第二题现场测试准备中",
    "targets": "三个圆已识别，准备按黄、绿、蓝顺序抓取",
    "battery_complete": "当前电池放置动作完成",
    "fixed_plan_loaded": "已载入固定示教位置",
    "fixed_paths_preparing": "正在准备固定轨迹，尚未开始计时",
    "fixed_paths_prepared": "固定轨迹检查完成，准备连续抓放",
    "fixed_trajectory_checks_skipped": "已跳过轨迹检查，直接执行固定点",
    "fixed_points_ready": "固定点已载入，准备连续抓放",
    "placement_verification_skipped": "放置动作完成，继续执行（未复检）",
}

MOVE_PHASES = {
    "unfold_to_fixed_entry": "正在整体展开至固定抓取入口",
    "unfold_to_observation": "正在整体展开至朝下观察位",
    "return_folded_standby": "正在折回朝前待机位",
    "startup_vertical_clearance": "准备阶段竖直抬高",
    "startup_move_to_corridor": "准备阶段沿区外走廊平移",
    "startup_turn_down": "准备阶段单独转为朝下",
    "startup_return_home": "准备阶段平移至新 Home",
    "startup_sideways_home": "准备阶段单独转为侧向待机",
    "start_observation_pose": "开始后在区外转向桌面拍照",
    "home_sideways_pose": "回到区外并恢复侧向姿态",
    "verification_observation_pose": "正在转向桌面复检",
    "verification_return_sideways_home": "复检后恢复侧向 Home",
    "normalize_high_z": "正在校正安全高度",
    "home_to_red_hover": "正在前往红色电池",
    "red_pick": "正在下降夹取",
    "red_lift": "正在提升电池",
    "red_to_t0": "正在搬运至 T0",
    "red_place": "正在下降放置",
    "direct_return_home": "正在沿直线退回观察位",
}

COLOR_NAMES = {"red": "红色", "yellow": "黄色", "blue": "蓝色", "green": "绿色"}


def humanize_error(message: str) -> str:
    """Translate the small set of expected operator-facing vision failures."""
    if '未检测到指定 RealSense' in message or "Couldn't resolve requests" in message:
        return 'RealSense 未连接或流配置失败，请检查 USB 3 连接和序列号 234222302865'
    if '连接' in message and ('29999' in message or '30004' in message):
        return '机械臂网络未连接，请检查 192.168.1.10 的控制器网线'
    if 'COM11' in message or '串口' in message:
        return '夹爪串口未连接，请检查 USB-RS485 端口'
    if 'outside configured workspace' in message or 'outside measured workspace' in message:
        return '现场坐标超出当前标定工作区，请重新采集手眼/TCP 数据'
    if 'Controller safety limits reject' in message or 'UR 安全限制拒绝' in message:
        return 'UR 拒绝了规划位姿，请检查现场坐标和安全平面配置'
    if message.startswith("Insufficient depth: "):
        color = message.partition(": ")[2]
        return f"{COLOR_NAMES.get(color, color)}区域深度不足"
    if message.startswith("T0 invalid: "):
        return "T0 目标框识别失败：" + message.partition(": ")[2]
    if (message.startswith("Initial vision failed after ")
            and message.endswith("Invalid body/handle geometry: red")):
        return "红色电池主体/把手深度几何持续不稳定，请检查红色电池是否平放及深度画面"
    return message


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _event_epoch(audit: dict[str, Any], event: dict[str, Any]) -> float | None:
    """Convert an audit monotonic offset into a stable wall-clock timestamp."""
    try:
        origin = datetime.fromisoformat(str(audit["timestamp_utc"])).timestamp()
        return origin + float(event["program_elapsed_s"])
    except (KeyError, TypeError, ValueError):
        try:
            return datetime.fromisoformat(str(event["utc"])).timestamp()
        except (KeyError, TypeError, ValueError):
            return None


def phase_for_event(event: dict[str, Any]) -> str:
    name = str(event.get("event", ""))
    if name=='fixed_path_progress':
        return f"固定轨迹检查：{event.get('index',0)}/{event.get('total',0)} 段（尚未计时）"
    if name in {"move", "smooth_path", "joint_move", "joint_path_checked"}:
        label = str(event.get("label", ""))
        parts = label.split(':')
        if len(parts)>1 and parts[0] in COLOR_NAMES:
            action = {'hover':'前往抓取上方','hover_lower':'下降到低位搬运面',
                      'pick':'下降夹取','lift':'提升','place_high':'搬运至目标圆',
                      'place':'下降放置','retreat':'释放后撤离',
                      'verify_corridor':'沿区外走廊复检','verify_raise':'抬升至观察高度',
                      'back_pick':'返回抓取观察位','next_pick_corridor':'返回抓取区',
                      'next_pick_raise':'连续抬升至抓取观察高度','next_pick_view':'准备下一次抓取'}.get(parts[1], '运动')
            return f"{COLOR_NAMES[parts[0]]}电池：{action}"
        return MOVE_PHASES.get(label, {'target_view':'正在观察三个目标圆','verify_view':'正在复检放置结果',
                                     'pick_view':'正在重新识别剩余电池'}.get(parts[0], "机械臂正在运动"))
    if name in {'released', 'battery_complete'} and event.get('color'):
        skipped=event.get('verification_skipped') or event.get('placement_verified') is False
        detail=('已释放，继续执行（未复检）' if name=='released' else '放置动作完成（未复检）') if skipped else ('已释放，正在复检' if name=='released' else '放置及稳定检查完成')
        return f"{COLOR_NAMES.get(event['color'],event['color'])}电池 → {event.get('target','')}：" + detail
    return EVENT_PHASES.get(name, "程序正在运行")


def read_run_audit(output_dir: Path, task: str) -> dict[str, Any] | None:
    if task=='basic':
        return _read_json(output_dir/'audit.json')
    try:
        events=json.loads((output_dir/'events.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(events,list):
        return None
    events=[event for event in events if isinstance(event,dict)]
    summary=_read_json(output_dir/'summary.json') or {}
    failure=next((event for event in reversed(events) if event.get('event')=='failed'),{})
    complete=next((event for event in reversed(events) if event.get('event')=='complete'),{})
    return {'events':events,'completed':bool(summary.get('completed') or complete),
            'timing':summary.get('timing',complete.get('timing',{})), 'error':failure.get('error')}


def run_seconds(timing: dict[str, Any], task: str) -> float:
    return float(timing['task_run_time_s' if task=='sequence' else 'motion_to_home_s'])


@dataclass
class ConsoleState:
    """Thread-safe state shared by the HTTP request and worker threads."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    process: subprocess.Popen[str] | None = None
    generation: int = 0
    state: str = "idle"
    phase: str = "等待开始"
    run_id: str | None = None
    output_dir: Path | None = None
    launched_at: float | None = None
    cycle_started_at: float | None = None
    ended_at: float | None = None
    pick_run_time_s: float | None = None
    error: str | None = None
    recent_output: list[str] = field(default_factory=list)
    start_signal_sent: bool = False
    task: str = 'basic'
    battery_run_times: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            if self.pick_run_time_s is not None:
                elapsed = self.pick_run_time_s
            elif self.cycle_started_at is None:
                elapsed = 0.0
            else:
                elapsed = max(0.0, (self.ended_at or now) - self.cycle_started_at)
            return {
                "state": self.state,
                "phase": self.phase,
                "run_id": self.run_id,
                "output_dir": str(self.output_dir) if self.output_dir else None,
                "launched_at": self.launched_at,
                "cycle_started_at": self.cycle_started_at,
                "ended_at": self.ended_at,
                "elapsed_s": elapsed,
                "pick_run_time_s": self.pick_run_time_s,
                "error": self.error,
                "recent_output": self.recent_output[-8:],
                "profile": FORMAL_PROFILE if self.task=='basic' else 'fixed-sequence-30s',
                "task": self.task,
                "target_assignment": TASK_ASSIGNMENTS[self.task],
                "battery_run_times": list(self.battery_run_times),
                "available_tasks": list(TASK_ASSIGNMENTS),
            }

    def start(self, task: str = 'basic') -> dict[str, Any]:
        with self.lock:
            if task not in TASK_ASSIGNMENTS:
                raise ValueError('任务必须为 basic 或 sequence')
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("任务已经在运行")
            if not PYTHON.is_file():
                raise RuntimeError(f"未找到项目 Python: {PYTHON}")
            if not RUNNER.is_file():
                raise RuntimeError(f"未找到任务入口: {RUNNER}")
            if task=='sequence':
                from .fixed_sequence import load_plan, Setup
                fixed_setup = Setup(SEQUENCE_CONFIG)
                fixed_setup.require_ready('sequence',site_test=True)
                load_plan(fixed_setup)
                # Run the hardware-only gate synchronously before spawning the
                # action process.  This keeps the single-screen console from
                # appearing to start a mission when the camera/robot is absent
                # and guarantees no motion command is created on failure.
                preflight = [str(PYTHON), '-m', 'battery_sorting.fixed_sequence',
                             'preflight', '--config', str(SEQUENCE_CONFIG)]
                probe_env=os.environ.copy()
                probe_env.update({'PYTHONUTF8':'1','PYTHONDONTWRITEBYTECODE':'1'})
                try:
                    result = subprocess.run(preflight, cwd=ROOT, env=probe_env,
                                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True,
                                            encoding='utf-8', errors='replace',timeout=30,
                                            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError('现场预检超时，未执行机械臂动作') from exc
                if result.returncode != 0:
                    detail = (result.stdout or '').strip().splitlines()
                    raise RuntimeError('现场预检未通过，未执行机械臂动作：' + (detail[-1] if detail else '未知错误'))

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = OUTPUT_ROOT / run_id
            if output_dir.exists():
                run_id += f"_{int(time.time_ns() % 1000):03d}"
                output_dir = OUTPUT_ROOT / run_id
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            if task == 'basic':
                # Preserve the established first-task launcher contract.  It
                # owns the legacy timed profile and now shares the same
                # hardware preflight checks as the closed-loop runner.
                command = [str(PYTHON), str(RUNNER), '--execute',
                           '--profile', FORMAL_PROFILE, '--wait-for-start',
                           '--output', str(output_dir)]
            else:
                command = [str(PYTHON), '-m', 'battery_sorting.fixed_sequence',
                           'run', '--config',
                           str(SEQUENCE_CONFIG), '--execute', '--auto-start',
                           '--output', str(output_dir)]
            env = os.environ.copy()
            env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"})
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self.generation += 1
            generation = self.generation
            self.process = process
            self.state = "preparing"
            self.phase = "正在连接设备并准备固定轨迹" if task=='sequence' else "正在连接设备并识别现场"
            self.run_id = run_id
            self.output_dir = output_dir
            self.launched_at = time.time()
            self.cycle_started_at = None
            self.ended_at = None
            self.pick_run_time_s = None
            self.error = None
            self.recent_output = []
            self.start_signal_sent = False
            self.task = task
            self.battery_run_times = []

        threading.Thread(
            target=self._read_process, args=(process, generation), daemon=True
        ).start()
        threading.Thread(
            target=self._watch_audit, args=(process, generation, output_dir), daemon=True
        ).start()
        return self.snapshot()

    def _read_process(self, process: subprocess.Popen[str], generation: int) -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            if line:
                with self.lock:
                    if generation == self.generation:
                        self.recent_output.append(line)
                        self.recent_output = self.recent_output[-40:]
        return_code = process.wait()
        with self.lock:
            if generation != self.generation:
                return
            audit = read_run_audit(self.output_dir or Path(), self.task)
            # The runner's authoritative competition motion result is
            # timing.motion_to_home_s.  Apply it even if the audit watcher did
            # not observe home_arrived before the short-lived child exited.
            if audit and isinstance(audit.get("timing"), dict):
                try:
                    self.pick_run_time_s = run_seconds(audit['timing'],self.task)
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    if self.cycle_started_at is not None:
                        self.ended_at = self.cycle_started_at + self.pick_run_time_s
                    self.battery_run_times=list(audit['timing'].get('battery_run_times',[]))
            completed = bool(audit and audit.get("completed"))
            if completed and return_code == 0:
                self.state = "success"
                self.phase = self.success_phase()
                self.error = None
            else:
                self.state = "failed"
                self.phase = "任务未完成"
                audit_error = audit.get("error") if audit else None
                raw_error = str(audit_error or self._last_error_line() or f"程序退出码 {return_code}")
                self.error = humanize_error(raw_error)
            if self.cycle_started_at is not None and self.ended_at is None:
                self.ended_at = time.time()

    def _last_error_line(self) -> str | None:
        for line in reversed(self.recent_output):
            if line.strip() and not line.lstrip().startswith("File "):
                return line.strip()
        return None

    def success_phase(self) -> str:
        return '任务完成：黄→P1、绿→P2、蓝→P3，已折回 Home' if self.task=='sequence' else '任务完成：红色电池已放入 T0'

    def _watch_audit(
        self, process: subprocess.Popen[str], generation: int, output_dir: Path
    ) -> None:
        seen = 0
        while process.poll() is None:
            audit = read_run_audit(output_dir,self.task)
            if audit is not None:
                events = audit.get("events", [])
                if isinstance(events, list):
                    for event in events[seen:]:
                        if not isinstance(event, dict):
                            continue
                        with self.lock:
                            if generation != self.generation:
                                return
                            self.phase = phase_for_event(event)
                            if (event.get("event") == "prepared_waiting_for_start"
                                    and not self.start_signal_sent):
                                if process.stdin is None:
                                    raise RuntimeError("快速程序启动输入管道不可用")
                                process.stdin.write("\n")
                                process.stdin.flush()
                                self.start_signal_sent = True
                            if event.get("event") == "cycle_started" and self.cycle_started_at is None:
                                self.cycle_started_at = _event_epoch(audit, event) or time.time()
                                self.state = "running"
                            if event.get('event')=='battery_complete':
                                self.battery_run_times.append({k:event[k] for k in ('color','target','run_time_s') if k in event})
                            if event.get("event") == "home_arrived":
                                try:
                                    self.pick_run_time_s = run_seconds(event,self.task)
                                except (KeyError, TypeError, ValueError):
                                    self.pick_run_time_s = max(
                                        0.0, time.time() - (self.cycle_started_at or time.time())
                                    )
                                if self.cycle_started_at is not None:
                                    self.ended_at = self.cycle_started_at + self.pick_run_time_s
                                self.state = "verifying"
                    seen = len(events)
                if audit.get("completed"):
                    with self.lock:
                        if generation == self.generation:
                            if self.ended_at is None:
                                self.ended_at = time.time()
                            self.state = "success"
                            self.phase = self.success_phase()
            time.sleep(0.08)


class ConsoleHandler(BaseHTTPRequestHandler):
    server: "ConsoleServer"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = urlparse(self.path).path
        if route == "/":
            try:
                body = HTML_PATH.read_bytes()
            except OSError as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if route == "/api/status":
            self._json(self.server.console_state.snapshot())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlparse(self.path).path != "/api/start":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4096:
                raise ValueError("请求过大")
            body=json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(body,dict):
                raise ValueError('请求必须为对象')
            task=body.get('task','basic')
            if not isinstance(task,str):
                raise ValueError('任务名称必须为文本')
            result = self.server.console_state.start(task)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        self._json(result, HTTPStatus.ACCEPTED)


class ConsoleServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: ConsoleState):
        super().__init__(address, ConsoleHandler)
        self.console_state = state


def serve(host: str = "127.0.0.1", port: int = 8768, *, open_browser: bool = True, initial_task: str = 'basic') -> None:
    if initial_task not in TASK_ASSIGNMENTS:
        raise ValueError('初始任务必须为 basic 或 sequence')
    server = ConsoleServer((host, port), ConsoleState(task=initial_task))
    url = f"http://{host}:{server.server_port}/"+('?task=sequence' if initial_task=='sequence' else '')
    print(f"比赛单屏控制台：{url}", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="启动本地比赛图像控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--task", choices=tuple(TASK_ASSIGNMENTS), default="basic")
    args = parser.parse_args(argv)
    url = f"http://{args.host}:{args.port}/"
    display_url = url + ("?task=sequence" if args.task == "sequence" else "")
    # Reopening either launcher reuses the console, including during a run.
    try:
        with urlopen(url + "api/status", timeout=1) as response:
            state = json.load(response)
        if isinstance(state, dict) and args.task in state.get("available_tasks", []):
            print(f"比赛单屏控制台：{display_url}", flush=True)
            if not args.no_browser:
                webbrowser.open(display_url)
            return
    except (URLError, OSError, ValueError):
        pass
    serve(args.host, args.port, open_browser=not args.no_browser, initial_task=args.task)


if __name__ == "__main__":
    main()

