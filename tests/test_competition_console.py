from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path

import pytest

import battery_sorting.competition_console as console


def test_event_timestamp_uses_program_monotonic_offset():
    audit = {"timestamp_utc": "2026-09-29T10:00:00+08:00"}
    event = {"program_elapsed_s": 12.375, "utc": "2099-01-01T00:00:00+00:00"}
    expected = datetime(2026, 9, 29, 2, 0, 12, 375000, tzinfo=timezone.utc).timestamp()
    assert console._event_epoch(audit, event) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("event", "phase"),
    [
        ({"event": "cycle_started"}, "自动抓取进行中"),
        ({"event": "move", "label": "home_to_red_hover"}, "正在前往红色电池"),
        ({"event": "smooth_path", "label": "red_place"}, "正在下降放置"),
        ({"event": "released"}, "电池已释放，正在稳定判定"),
    ],
)
def test_event_has_clear_competition_phase(event, phase):
    assert console.phase_for_event(event) == phase


def test_common_depth_error_is_localized():
    assert console.humanize_error('Insufficient depth: red')=='红色区域深度不足'
    assert '主体/把手深度几何持续不稳定' in console.humanize_error(
        'Initial vision failed after 15 attempts: Invalid body/handle geometry: red')
    assert 'RealSense 未连接' in console.humanize_error('未检测到指定 RealSense 234222302865')
    assert 'UR 拒绝' in console.humanize_error('UR 安全限制拒绝目标 yellow:hover')


def test_unverified_placement_is_not_reported_as_verified():
    assert '未复检' in console.phase_for_event({'event':'battery_complete','color':'yellow','target':'P1','placement_verified':False})
    assert '未复检' in console.phase_for_event({'event':'released','color':'yellow','target':'P1','verification_skipped':True})


def test_fixed_preparation_progress_has_visible_segment_count():
    assert console.phase_for_event({'event':'fixed_path_progress','index':3,'total':14})=='固定轨迹检查：3/14 段（尚未计时）'


def test_snapshot_timer_is_zero_until_cycle_event(monkeypatch):
    state = console.ConsoleState()
    monkeypatch.setattr(console.time, "time", lambda: 150.0)
    state.launched_at = 100.0
    assert state.snapshot()["elapsed_s"] == 0.0
    state.cycle_started_at = 120.0
    assert state.snapshot()["elapsed_s"] == 30.0
    state.ended_at = 127.25
    assert state.snapshot()["elapsed_s"] == 7.25


def test_prepared_event_automatically_sends_one_enter(monkeypatch, tmp_path):
    output = tmp_path / 'run'
    output.mkdir()
    (output / 'audit.json').write_text(json.dumps({
        'timestamp_utc':'2026-09-29T10:00:00+08:00',
        'events':[{'event':'prepared_waiting_for_start','program_elapsed_s':1.0}]
    }),encoding='utf-8')
    polls=iter((None,0))
    class FakeProcess:
        stdin=io.StringIO()
        def poll(self): return next(polls)
    state=console.ConsoleState(generation=3,state='preparing')
    monkeypatch.setattr(console.time,'sleep',lambda _:None)
    process=FakeProcess()
    state._watch_audit(process,3,output)
    assert process.stdin.getvalue()=='\n'
    assert state.start_signal_sent is True
    assert state.phase=='设备准备完成，正在自动触发任务'


def test_home_event_locks_exact_pick_runtime_before_verification(monkeypatch, tmp_path):
    output = tmp_path / 'run'
    output.mkdir()
    (output / 'audit.json').write_text(json.dumps({
        'timestamp_utc':'2026-09-29T10:00:00+08:00',
        'events':[
            {'event':'cycle_started','program_elapsed_s':2.0},
            {'event':'home_arrived','program_elapsed_s':10.404,'motion_to_home_s':8.404},
        ],
    }),encoding='utf-8')
    polls=iter((None,0))
    class FakeProcess:
        stdin=io.StringIO()
        def poll(self): return next(polls)
    state=console.ConsoleState(generation=4,state='preparing')
    monkeypatch.setattr(console.time,'sleep',lambda _:None)
    state._watch_audit(FakeProcess(),4,output)
    snapshot=state.snapshot()
    assert snapshot['state']=='verifying'
    assert snapshot['pick_run_time_s']==pytest.approx(8.404)
    assert snapshot['elapsed_s']==pytest.approx(8.404)
    assert snapshot['ended_at']==pytest.approx(snapshot['cycle_started_at']+8.404)


def test_process_exit_recovers_authoritative_runner_timing(tmp_path):
    output=tmp_path/'run';output.mkdir()
    (output/'audit.json').write_text(json.dumps({
        'completed':True,'timing':{'motion_to_home_s':11.031}
    }),encoding='utf-8')
    class FakeProcess:
        stdout=iter(())
        def wait(self):return 0
    state=console.ConsoleState(generation=2,output_dir=output,
        cycle_started_at=100.0,state='running')
    state._read_process(FakeProcess(),2)
    snapshot=state.snapshot()
    assert snapshot['state']=='success'
    assert snapshot['pick_run_time_s']==pytest.approx(11.031)
    assert snapshot['elapsed_s']==pytest.approx(11.031)
    assert snapshot['ended_at']==pytest.approx(111.031)


@pytest.mark.parametrize('task',['basic','sequence'])
def test_start_delegates_to_formal_runner_without_wait_for_enter(monkeypatch, tmp_path, task):
    python = tmp_path / "python.exe"
    runner = tmp_path / "run_smooth_t0.py"
    python.write_text("", encoding="utf-8")
    runner.write_text("", encoding="utf-8")
    monkeypatch.setattr(console, "PYTHON", python)
    monkeypatch.setattr(console, "RUNNER", runner)
    monkeypatch.setattr(console, "OUTPUT_ROOT", tmp_path / "out")

    captured = {}

    class FakeProcess:
        stdout = iter(())
        stdin = None

        def poll(self):
            return None

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    class FakeThread:
        def __init__(self, target, args, daemon):
            captured.setdefault("threads", []).append((target, args, daemon))

        def start(self):
            return None

    monkeypatch.setattr(console.subprocess, "Popen", fake_popen)
    def fake_preflight(command, **kwargs):
        assert command[3]=='preflight' and '--execute' not in command
        return console.subprocess.CompletedProcess(command,0,stdout='preflight ok')
    monkeypatch.setattr(console.subprocess, "run", fake_preflight)
    monkeypatch.setattr(console.threading, "Thread", FakeThread)
    state = console.ConsoleState()
    result = state.start(task)

    command = captured["command"]
    assert command[0] == str(python)
    assert "--execute" in command
    if task=='basic':
        assert command[1]==str(runner)
        assert command[command.index("--profile") + 1] == "ten-second"
        assert "--wait-for-start" in command
    else:
        assert command[1:4]==['-m','battery_sorting.fixed_sequence','run']
        assert command[command.index('--config')+1]==str(console.SEQUENCE_CONFIG)
        assert '--auto-start' in command
        assert '--site-test' not in command
        assert '--colors' not in command
    assert captured["kwargs"]["stdin"] is console.subprocess.PIPE
    assert result["state"] == "preparing"
    assert result["cycle_started_at"] is None
    assert len(captured["threads"]) == 2
    assert result['task']==task


@pytest.mark.parametrize('failure',['rejected','timeout'])
def test_failed_sequence_preflight_never_spawns_action_process(monkeypatch,tmp_path,failure):
    python=tmp_path/'python.exe';python.write_text('')
    runner=tmp_path/'runner.py';runner.write_text('')
    monkeypatch.setattr(console,'PYTHON',python)
    monkeypatch.setattr(console,'RUNNER',runner)
    monkeypatch.setattr(console.subprocess,'Popen',lambda *a,**k:pytest.fail('Motion process must not start'))
    def fail_preflight(command,**kwargs):
        if failure=='timeout': raise console.subprocess.TimeoutExpired(command,30)
        return console.subprocess.CompletedProcess(command,1,stdout='RealSense USB 2.1')
    monkeypatch.setattr(console.subprocess,'run',fail_preflight)
    state=console.ConsoleState()
    with pytest.raises(RuntimeError,match='未执行机械臂动作'):
        state.start('sequence')
    assert state.process is None


def test_html_keeps_required_items_on_one_screen():
    html = console.HTML_PATH.read_text(encoding="utf-8")
    for required in (
        "抓取红色电池放入 T0",
        "自动任务计时",
        "机械臂整体",
        "工作台与任务底图",
        "全部电池",
        "开始任务并自动计时",
        "原速连续",
    ):
        assert required in html
    assert "overflow:hidden" in html
    assert "handOffRobotCamera" in html
    assert "末端 D455 调试模式" in html
    assert "requestAnimationFrame(animateTimer)" in html
    assert "夹取运行时间" in html


def test_sequence_summary_locks_total_and_three_battery_times(tmp_path):
    output=tmp_path/'run';output.mkdir()
    times=[{'color':c,'target':f'P{i+1}','run_time_s':12+i} for i,c in enumerate(('yellow','green','blue'))]
    (output/'events.json').write_text(json.dumps([
        {'event':'cycle_started','utc':'2026-10-11T14:00:00+00:00'},
        {'event':'home_arrived','task_run_time_s':45.125},
    ]),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps({'completed':True,
        'timing':{'task_run_time_s':45.125,'battery_run_times':times}}),encoding='utf-8')
    class FakeProcess:
        stdout=iter(())
        def wait(self):return 0
    state=console.ConsoleState(task='sequence',generation=2,output_dir=output,cycle_started_at=100.)
    state._read_process(FakeProcess(),2)
    result=state.snapshot()
    assert result['state']=='success'
    assert result['pick_run_time_s']==45.125
    assert result['elapsed_s']==45.125 and result['ended_at']==145.125
    assert result['battery_run_times']==times
    assert result['target_assignment']=={'P1':'yellow','P2':'green','P3':'blue'}


def test_sequence_events_update_progress_and_surface_failure(tmp_path,monkeypatch):
    output=tmp_path/'run';output.mkdir()
    (output/'events.json').write_text(json.dumps([
        {'event':'cycle_started','utc':'2026-10-11T14:00:00+00:00'},
        {'event':'battery_complete','color':'yellow','target':'P1','run_time_s':12.5},
        {'event':'failed','error':'Target geometry missing/ambiguous'},
    ]),encoding='utf-8')
    polls=iter((None,0))
    class FakeProcess:
        stdout=iter(())
        def poll(self):return next(polls)
        def wait(self):return 1
    process=FakeProcess()
    state=console.ConsoleState(task='sequence',generation=2,output_dir=output)
    monkeypatch.setattr(console.time,'sleep',lambda _:None)
    state._watch_audit(process,2,output)
    assert state.battery_run_times==[{'color':'yellow','target':'P1','run_time_s':12.5}]
    assert state.cycle_started_at is not None
    state._read_process(process,2)
    assert state.snapshot()['state']=='failed'
    assert state.error=='Target geometry missing/ambiguous'


def test_invalid_task_and_concurrent_start_do_not_launch_process(monkeypatch):
    monkeypatch.setattr(console.subprocess,'Popen',lambda *a,**k:pytest.fail('Unexpected robot process'))
    state=console.ConsoleState()
    with pytest.raises(ValueError):state.start('unknown')
    class Running:
        def poll(self):return None
    state.process=Running()
    with pytest.raises(RuntimeError,match='已经在运行'):state.start('sequence')

