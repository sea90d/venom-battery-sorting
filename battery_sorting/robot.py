"""UR5 安全状态、线性运动封装、FakeRobot 与固定高度路径规划。"""

from __future__ import annotations

import logging
import math
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable

from .config import Pose, RobotConfig

LOGGER = logging.getLogger(__name__)

ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1
RUNTIME_STATE_STOPPED = 1
RUNTIME_STATE_PLAYING = 2

ARRIVAL_TRANSLATION_M = 0.0002
ARRIVAL_ROTATION_RAD = math.radians(0.1)
ARRIVAL_LINEAR_SPEED_MPS = 0.0003
ARRIVAL_ANGULAR_SPEED_RAD_S = 0.005
ARRIVAL_STABLE_SAMPLES = 3
CONTROL_READY_TIMEOUT_S = 3.0


class RobotError(RuntimeError):
    """机器人连接、状态或运动错误。"""


class RobotSafetyError(RobotError):
    """机器人状态不满足安全门禁。"""


class RobotMotionError(RobotError):
    """运动命令被拒绝、超时或未稳定到位。"""


@dataclass(frozen=True)
class RobotSnapshot:
    tcp_pose: tuple[float, ...]
    robot_mode: int
    safety_mode: int
    runtime_state: int
    emergency_stopped: bool
    protective_stopped: bool


@dataclass(frozen=True)
class MotionRecord:
    label: str
    pose: Pose
    speed: float
    acceleration: float


@dataclass(frozen=True)
class Waypoint:
    label: str
    pose: Pose
    profile: str = "transit"


class RobotBase(ABC):
    @property
    @abstractmethod
    def is_real(self) -> bool: ...

    @abstractmethod
    def connect_readonly(self) -> None: ...

    @abstractmethod
    def connect_control(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def snapshot(self) -> RobotSnapshot: ...

    @abstractmethod
    def move_l(self, waypoint: Waypoint, speed: float, acceleration: float) -> bool: ...

    @abstractmethod
    def move_l_without_tcp_force_guard(
        self, waypoint: Waypoint, speed: float, acceleration: float
    ) -> bool: ...

    @abstractmethod
    def move_l_path_without_tcp_force_guard(
        self, path: list[tuple[Waypoint, float, float, float]]
    ) -> bool: ...

    @abstractmethod
    def stop_l(self, acceleration: float = 0.2) -> bool: ...

    def pose_within_safety_limits(self, pose: list[float]) -> bool:
        """Read-only controller safety query; FakeRobot accepts all poses."""
        return True


class FakeRobot(RobotBase):
    """只记录路径的机器人，绝不访问网络。"""

    def __init__(self, initial_pose: Pose) -> None:
        self._pose = initial_pose
        self.records: list[MotionRecord] = []
        self.stop_count = 0
        self.readonly_connected = False
        self.control_connected = False

    @property
    def is_real(self) -> bool:
        return False

    def connect_readonly(self) -> None:
        self.readonly_connected = True

    def connect_control(self) -> None:
        self.control_connected = True

    def disconnect(self) -> None:
        self.readonly_connected = False
        self.control_connected = False

    def snapshot(self) -> RobotSnapshot:
        return RobotSnapshot(
            tuple(self._pose.as_list()),
            ROBOT_MODE_RUNNING,
            SAFETY_MODE_NORMAL,
            RUNTIME_STATE_STOPPED,
            False,
            False,
        )

    def move_l(self, waypoint: Waypoint, speed: float, acceleration: float) -> bool:
        if hasattr(self, '_joints'):
            self._joints = self.joints_for_pose(waypoint.pose, self._joints)
        self.records.append(MotionRecord(waypoint.label, waypoint.pose, speed, acceleration))
        self._pose = waypoint.pose
        return True

    def actual_joints(self):
        if not hasattr(self, '_joints'):
            import json
            from pathlib import Path
            plan=json.loads((Path(__file__).resolve().parents[1]/'config/folded_standby.json').read_text(encoding='utf-8'))
            self._joints=self.joints_for_pose(self._pose,plan['home_joints_rad'])
        return tuple(self._joints)

    def joints_for_pose(self, pose, seed):
        from tools.audit_t0_trajectory import ik
        import numpy as np
        joints,error=ik(pose.as_list(),np.array(seed,float))
        if np.linalg.norm(error[:3])>.0002 or np.linalg.norm(error[3:])>.00025:
            raise RobotMotionError('Fake joint IK failed')
        return joints.tolist()

    def move_j(self, waypoint, joints, speed, acceleration):
        self.records.append(MotionRecord(waypoint.label,waypoint.pose,speed,acceleration))
        self._joints=list(joints)
        self._pose=waypoint.pose
        return True

    def move_l_without_tcp_force_guard(
        self, waypoint: Waypoint, speed: float, acceleration: float
    ) -> bool:
        return self.move_l(waypoint, speed, acceleration)

    def move_l_path_without_tcp_force_guard(
        self, path: list[tuple[Waypoint, float, float, float]]
    ) -> bool:
        if not path or path[-1][3] != 0:
            raise RobotSafetyError("连续路径必须非空且末点 blend=0")
        for waypoint, speed, acceleration, _blend in path:
            self.move_l(waypoint, speed, acceleration)
        return True

    def stop_l(self, acceleration: float = 0.2) -> bool:
        self.stop_count += 1
        return True


class RealUR5(RobotBase):
    """两阶段连接：先只读检查，确认后才创建 RTDE 控制接口。"""

    def __init__(self, config: RobotConfig, *, motion_profile: str = "standard") -> None:
        if motion_profile not in {"standard", "t0_timing_trial"}:
            raise RobotSafetyError("Unknown robot motion profile")
        self.config = config
        self.motion_profile = motion_profile
        self.max_command_speed = 0.5 if motion_profile == "t0_timing_trial" else 0.3
        self.max_command_acceleration = 1.0 if motion_profile == "t0_timing_trial" else 0.5
        self._receiver: Any | None = None
        self._control: Any | None = None

    @property
    def is_real(self) -> bool:
        return True

    def connect_readonly(self) -> None:
        if self._receiver is not None:
            raise RobotError("RTDE Receive 已连接")
        _require_tcp_port(self.config.host, self.config.rtde_port)
        dashboard = read_dashboard(self.config.host, self.config.dashboard_port)
        require_dashboard_state(dashboard, {"STOPPED"}, "只读连接前")
        try:
            import rtde_receive  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RobotError("缺少 ur_rtde==1.6.3，且尚未建立任何控制连接") from exc
        receiver = rtde_receive.RTDEReceiveInterface(self.config.host)
        self._receiver = receiver
        try:
            if not receiver.isConnected():
                raise RobotError(f"无法连接 {self.config.host} 的 RTDE Receive")
            self._require_safe_state({RUNTIME_STATE_STOPPED}, "只读连接后")
        except Exception:
            self.disconnect()
            raise

    def connect_control(self) -> None:
        if self._receiver is None:
            raise RobotError("必须先完成只读连接和人工确认")
        if self._control is not None:
            raise RobotError("RTDE Control 已连接")
        self._require_safe_state({RUNTIME_STATE_STOPPED}, "创建控制接口前")
        require_dashboard_state(
            read_dashboard(self.config.host, self.config.dashboard_port),
            {"STOPPED"},
            "创建控制接口前",
        )
        try:
            import rtde_control  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RobotError("缺少 ur_rtde==1.6.3") from exc
        control = rtde_control.RTDEControlInterface(self.config.host)
        self._control = control
        try:
            if not control.isConnected():
                raise RobotError("RTDE Control 连接失败")
            deadline = time.monotonic() + CONTROL_READY_TIMEOUT_S
            while time.monotonic() < deadline:
                state = self._require_safe_state(
                    {RUNTIME_STATE_STOPPED, RUNTIME_STATE_PLAYING}, "等待 RTDE 控制脚本"
                )
                if state.runtime_state == RUNTIME_STATE_PLAYING:
                    break
                time.sleep(0.05)
            else:
                raise RobotError("RTDE 控制脚本未在 3 秒内进入 PLAYING")
            require_dashboard_state(
                read_dashboard(self.config.host, self.config.dashboard_port),
                {"PLAYING", "STOPPED"},
                "控制接口创建后",
            )
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        control, receiver = self._control, self._receiver
        self._control = None
        self._receiver = None
        if control is not None:
            try:
                control.stopScript()
            except Exception as exc:  # pragma: no cover - hardware cleanup
                LOGGER.warning("stopScript 失败: %s", exc)
            try:
                control.disconnect()
            except Exception as exc:  # pragma: no cover - hardware cleanup
                LOGGER.warning("RTDE Control 断开失败: %s", exc)
        if receiver is not None:
            try:
                receiver.disconnect()
            except Exception as exc:  # pragma: no cover - hardware cleanup
                LOGGER.warning("RTDE Receive 断开失败: %s", exc)

    def snapshot(self) -> RobotSnapshot:
        if self._receiver is None:
            raise RobotError("RTDE Receive 未连接")
        return RobotSnapshot(
            tuple(_finite_vector(self._receiver.getActualTCPPose(), 6, "getActualTCPPose")),
            int(self._receiver.getRobotMode()),
            int(self._receiver.getSafetyMode()),
            int(self._receiver.getRuntimeState()),
            bool(self._receiver.isEmergencyStopped()),
            bool(self._receiver.isProtectiveStopped()),
        )

    def _require_safe_state(self, allowed_runtime: set[int], phase: str) -> RobotSnapshot:
        state = self.snapshot()
        require_snapshot_state(state, allowed_runtime, phase)
        return state

    def active_tcp_offset(self) -> tuple[float, ...]:
        """Read active TCP after control setup; never set or move it."""
        if self._control is None:
            raise RobotError("Active TCP read requires established control interface")
        return tuple(_finite_vector(self._control.getTCPOffset(), 6, "getTCPOffset"))

    def actual_joints(self):
        if self._receiver is None:
            raise RobotError('RTDE Receive 未连接')
        return tuple(_finite_vector(self._receiver.getActualQ(),6,'getActualQ'))

    def joints_for_pose(self, pose, seed):
        if self._control is None:
            raise RobotError('Controller IK requires RTDE Control')
        return _finite_vector(self._control.getInverseKinematics(pose.as_list(),list(seed)),6,'Controller IK')

    def forward_kinematics(self, joints):
        """Return calibrated TCP FK for a joint vector on the active TCP.

        This is deliberately read-only.  It is used when URScript returns a
        numerically valid but different IK branch for the two taught standby
        poses; the configured branch may be selected only after this FK check
        and the normal joint/collision gates pass.
        """
        if self._control is None:
            raise RobotError('Calibrated FK requires RTDE Control')
        target=_finite_vector(joints,6,'FK joints')
        return tuple(_finite_vector(
            self._control.getForwardKinematics(target,list(self.active_tcp_offset())),
            6,'Calibrated FK'))

    def pose_within_safety_limits(self, pose):
        if self._control is None:
            raise RobotError('Controller safety query requires RTDE Control')
        return bool(self._control.isPoseWithinSafetyLimits(_finite_vector(pose,6,'safety pose')))

    def _check_tool_body_path(self, poses, *, start_pose=None, start_joints=None, required_clearance_m=.010):
        """Screen fresh Cartesian paths on the actual joint branch before moveL."""
        if not getattr(self,'tool_body_model_enabled',False):return
        from tools.audit_t0_trajectory import audit_segment
        from tools.simulate_tool_body_collision import check_segment
        from .folded_standby import certify_tool_body_joint_path
        from .config import Pose
        import numpy as np
        q=np.array(self.actual_joints() if start_joints is None else start_joints)
        start=list(self.snapshot().tcp_pose if start_pose is None else start_pose)
        lower_bounds=[]
        controller_ik = bool(
            self._control is not None
            and callable(getattr(self._control, 'getInverseKinematics', None))
            and callable(getattr(self._control, 'getForwardKinematics', None))
        )
        for index,end in enumerate(poses):
            segment,last=audit_segment('runtime_tool_body',start,list(end),q,initial_q=q)
            rows=segment['samples']
            if controller_ik:
                # Use the controller's own branch-continuous IK/FK for a real
                # robot.  The nominal DH solver only supplies the sample
                # poses; it is not authoritative at the calibrated TCP.
                branch_q=q.copy()
                for row_index,row in enumerate(rows):
                    candidate = branch_q if row_index==0 else np.asarray(
                        self.joints_for_pose(Pose(*row['tcp_m_rad']), branch_q), dtype=float)
                    if row_index:
                        step=float(np.max(np.abs(candidate-branch_q)))
                        if step>math.radians(10):
                            raise RobotSafetyError(
                                'Controller IK changed the Cartesian tool-body branch'
                            )
                    fk_pose=self.forward_kinematics(candidate)
                    row['ik_position_error_m']=translation_error(fk_pose,row['tcp_m_rad'])
                    row['ik_rotation_error_rad']=rotation_error(fk_pose[3:],row['tcp_m_rad'][3:])
                    row['q_rad']=candidate.tolist()
                    row['max_joint_step_rad']=float(np.max(np.abs(candidate-branch_q))) if row_index else 0.0
                    branch_q=candidate
                last=branch_q
            else:
                # Offline driver doubles may not expose controller IK/FK. If
                # nominal IK changes branch at a wrist singularity, translate
                # the remaining nominal samples back to the measured branch;
                # clearance and joint-step checks still run on that path.
                branch_offset=None
                for row_index in range(1,len(rows)):
                    previous=np.asarray(rows[row_index-1]['q_rad'],float)
                    candidate=np.asarray(rows[row_index]['q_rad'],float)
                    if np.max(np.abs(candidate-previous))>math.radians(10):
                        offset=previous-candidate
                        for shifted in rows[row_index:]:
                            shifted['q_rad']=(np.asarray(shifted['q_rad'],float)+offset).tolist()
                        branch_offset=offset.tolist()
                if branch_offset is not None:
                    segment['calibrated_branch_offset']=branch_offset
                for row_index,row in enumerate(rows):
                    previous=np.asarray(rows[row_index-1]['q_rad'],float) if row_index else np.asarray(row['q_rad'],float)
                    row['max_joint_step_rad']=float(np.max(np.abs(np.asarray(row['q_rad'],float)-previous))) if row_index else 0.0
                last=np.asarray(rows[-1]['q_rad'],float)
            # Account for nominal versus calibrated robot kinematics at start.
            # This bridge also occurs at grasp/place contact height. The
            # fold/unfold table-clearance rule does not apply to moveL.
            certify_tool_body_joint_path(q,rows[0]['q_rad'])
            result=check_segment(segment, required_clearance_m=required_clearance_m)
            if not result['passed']:
                raise RobotSafetyError(f'Tool/body path intersects or cannot be certified: {result["failures"][:1]}')
            lower_bounds.append(result['swept_clearance_lower_bound_m'] or result['minimum_clearance_m'])
            q=last;start=list(end)
        return dict(end_joints=list(q), swept_clearance_lower_bound_m=min(lower_bounds))

    def move_j(self, waypoint, joints, speed, acceleration):
        if self._control is None or self._receiver is None:
            raise RobotError('RTDE 控制接口未连接')
        target=_finite_vector(joints,6,'moveJ target')
        if not math.isfinite(speed) or not 0<speed<=1.05 or not math.isfinite(acceleration) or not 0<acceleration<=1.4:
            raise RobotSafetyError('moveJ dynamics exceed 1.05 rad/s, 1.4 rad/s²')
        self._require_safe_state({RUNTIME_STATE_PLAYING},waypoint.label+' joint motion')
        if not self._control.isJointsWithinSafetyLimits(target):
            raise RobotSafetyError('UR safety limits reject joint target')
        # ur_rtde 1.6.3 leaves TCP input registers stale when q alone is given.
        # After IK those registers contain qnear, not a TCP pose (upstream #348).
        # Always supply all six active TCP components; retain calibrated FK check.
        tcp_offset=list(self.active_tcp_offset())
        target_tcp=_finite_vector(self._control.getForwardKinematics(target,tcp_offset),6,'Calibrated joint FK')
        assert_pose_close(target_tcp,waypoint.pose,.0002,math.radians(.1),'Calibrated joint target')
        try:
            if not self._control.moveJ(target,speed,acceleration,True):
                raise RobotMotionError('UR5 rejected moveJ: '+waypoint.label)
            stable=0;deadline=time.monotonic()+self.config.move_timeout
            while time.monotonic()<deadline:
                state=self._require_safe_state({RUNTIME_STATE_PLAYING},waypoint.label+' joint monitoring')
                q=self.actual_joints()
                qd=_finite_vector(self._receiver.getActualQd(),6,'getActualQd')
                if (max(abs(a-b) for a,b in zip(q,target))<=math.radians(.1)
                    and max(abs(v) for v in qd)<=.005
                    and translation_error(state.tcp_pose,waypoint.pose.as_list())<=ARRIVAL_TRANSLATION_M
                    and rotation_error(state.tcp_pose[3:],waypoint.pose.as_list()[3:])<=ARRIVAL_ROTATION_RAD):
                    stable+=1
                    if stable>=ARRIVAL_STABLE_SAMPLES:return True
                else:stable=0
                time.sleep(1/self.config.monitor_rate)
            raise RobotMotionError('moveJ did not arrive: '+waypoint.label)
        except (Exception,KeyboardInterrupt):
            self._control.stopJ(self.config.stop_deceleration)
            raise

    def move_l(self, waypoint: Waypoint, speed: float, acceleration: float) -> bool:
        return self._move_l(waypoint, speed, acceleration, monitor_tcp_force=True)

    def move_l_without_tcp_force_guard(
        self, waypoint: Waypoint, speed: float, acceleration: float
    ) -> bool:
        """Move while deliberately ignoring UR's estimated TCP force.

        Robot/safety/runtime state, controller safety limits, speed and arrival
        monitoring remain active. The competition mission deliberately uses
        this path because UR's TCP force estimate is not used.
        """
        return self._move_l(waypoint, speed, acceleration, monitor_tcp_force=False)

    def move_l_path_without_tcp_force_guard(
        self, path: list[tuple[Waypoint, float, float, float]], *, check=None
    ) -> bool:
        """Execute one controller-side blended moveL path and monitor its final arrival."""
        if self._control is None or self._receiver is None:
            raise RobotError("RTDE 控制接口未连接")
        if not isinstance(path, list) or not 1 <= len(path) <= 16:
            raise RobotSafetyError("连续 moveL 路径须包含 1..16 个点")
        command=[]
        for index,item in enumerate(path):
            if not isinstance(item,tuple) or len(item)!=4:
                raise RobotSafetyError("连续路径点格式必须为 (Waypoint,speed,acceleration,blend)")
            waypoint,speed,acceleration,blend=item
            if not isinstance(waypoint,Waypoint):
                raise RobotSafetyError("连续路径目标必须是 Waypoint")
            target=_finite_vector(waypoint.pose.as_list(),6,f"路径目标 {waypoint.label}")
            if not (math.isfinite(speed) and 0<speed<=self.max_command_speed):
                raise RobotSafetyError(f"连续路径 speed 必须在 (0,{self.max_command_speed}] m/s")
            if not (math.isfinite(acceleration) and 0<acceleration<=self.max_command_acceleration):
                raise RobotSafetyError(f"连续路径 acceleration 必须在 (0,{self.max_command_acceleration}] m/s²")
            if not (math.isfinite(blend) and 0<=blend<=.025):
                raise RobotSafetyError("连续路径 blend 必须在 [0,0.025] m")
            if index==len(path)-1 and blend!=0:
                raise RobotSafetyError("连续路径末点 blend 必须为 0")
            if not self._control.isPoseWithinSafetyLimits(target):
                raise RobotSafetyError(f"UR 安全限制拒绝路径目标 {waypoint.label}: {target}")
            command.append([*target,speed,acceleration,blend])
        self._check_tool_body_path([row[:6] for row in command])
        final=command[-1][:6]
        self._require_safe_state({RUNTIME_STATE_PLAYING},"连续路径运动前")
        try:
            if check: check()
            accepted=self._control.moveL(command,True)
            if not accepted:
                raise RobotMotionError("UR5 拒绝连续 moveL 路径")
            deadline=time.monotonic()+self.config.move_timeout
            stable=0;interval=1/self.config.monitor_rate
            while time.monotonic()<deadline:
                state=self._require_safe_state({RUNTIME_STATE_PLAYING},"连续路径运动中")
                if check: check()
                tcp_speed=_finite_vector(self._receiver.getActualTCPSpeed(),6,"getActualTCPSpeed")
                if (translation_error(state.tcp_pose,final)<=ARRIVAL_TRANSLATION_M
                    and rotation_error(state.tcp_pose[3:],final[3:])<=ARRIVAL_ROTATION_RAD
                    and _norm(tcp_speed[:3])<=ARRIVAL_LINEAR_SPEED_MPS
                    and _norm(tcp_speed[3:])<=ARRIVAL_ANGULAR_SPEED_RAD_S):
                    stable+=1
                    if stable>=ARRIVAL_STABLE_SAMPLES:return True
                else:stable=0
                time.sleep(interval)
            raise RobotMotionError(f"连续路径未在 {self.config.move_timeout:.1f}s 内稳定到位")
        except (Exception,KeyboardInterrupt):
            try:self._control.stopL(self.config.stop_deceleration)
            except Exception as stop_exc:LOGGER.critical("stopL 失败，请使用示教器急停: %s",stop_exc)
            raise

    def _move_l(
        self,
        waypoint: Waypoint,
        speed: float,
        acceleration: float,
        *,
        monitor_tcp_force: bool,
    ) -> bool:
        if self._control is None or self._receiver is None:
            raise RobotError("RTDE 控制接口未连接")
        if not (math.isfinite(speed) and 0.0 < speed <= self.max_command_speed):
            raise RobotSafetyError(f"moveL speed 必须在 (0,{self.max_command_speed}] m/s")
        if not (math.isfinite(acceleration) and 0.0 < acceleration <= self.max_command_acceleration):
            raise RobotSafetyError(f"moveL acceleration 必须在 (0,{self.max_command_acceleration}] m/s²")
        if monitor_tcp_force and not (0.0 < self.config.max_tcp_force_delta_n <= 60.0):
            raise RobotSafetyError("max_tcp_force_delta_n 必须在 (0,60] N")
        target = _finite_vector(waypoint.pose.as_list(), 6, f"目标 {waypoint.label}")
        self._require_safe_state({RUNTIME_STATE_PLAYING}, f"{waypoint.label} 运动前")
        if not self._control.isPoseWithinSafetyLimits(target):
            raise RobotSafetyError(f"UR 安全限制拒绝目标 {waypoint.label}: {target}")
        self._check_tool_body_path([target])
        try:
            baseline_force = None
            if monitor_tcp_force:
                baseline_force = _finite_vector(
                    self._receiver.getActualTCPForce(), 6, "getActualTCPForce 基线"
                )
            accepted = self._control.moveL(target, speed, acceleration, True)
            if not accepted:
                raise RobotMotionError(f"UR5 拒绝 moveL: {waypoint.label}")
            deadline = time.monotonic() + self.config.move_timeout
            stable = 0
            interval = 1.0 / self.config.monitor_rate
            while time.monotonic() < deadline:
                state = self._require_safe_state({RUNTIME_STATE_PLAYING}, f"{waypoint.label} 运动中")
                tcp_speed = _finite_vector(
                    self._receiver.getActualTCPSpeed(), 6, "getActualTCPSpeed"
                )
                if monitor_tcp_force:
                    assert baseline_force is not None
                    tcp_force = _finite_vector(
                        self._receiver.getActualTCPForce(), 6, "getActualTCPForce"
                    )
                    force_delta = _norm(
                        [tcp_force[index] - baseline_force[index] for index in range(3)]
                    )
                    if force_delta > self.config.max_tcp_force_delta_n:
                        raise RobotMotionError(
                            f"{waypoint.label} TCP 力变化 {force_delta:.2f}N 超过 "
                            f"{self.config.max_tcp_force_delta_n:.2f}N"
                        )
                translation = translation_error(state.tcp_pose, target)
                rotation = rotation_error(state.tcp_pose[3:], target[3:])
                linear_speed = _norm(tcp_speed[:3])
                angular_speed = _norm(tcp_speed[3:])
                if (
                    translation <= ARRIVAL_TRANSLATION_M
                    and rotation <= ARRIVAL_ROTATION_RAD
                    and linear_speed <= ARRIVAL_LINEAR_SPEED_MPS
                    and angular_speed <= ARRIVAL_ANGULAR_SPEED_RAD_S
                ):
                    stable += 1
                    if stable >= ARRIVAL_STABLE_SAMPLES:
                        return True
                else:
                    stable = 0
                time.sleep(interval)
            raise RobotMotionError(f"{waypoint.label} 未在 {self.config.move_timeout:.1f}s 内稳定到位")
        except (Exception, KeyboardInterrupt):
            try:
                self._control.stopL(self.config.stop_deceleration)
            except Exception as stop_exc:  # pragma: no cover - hardware cleanup
                LOGGER.critical("stopL 失败，请使用示教器急停: %s", stop_exc)
            raise

    def stop_l(self, acceleration: float = 0.2) -> bool:
        if self._control is None:
            return False
        self._control.stopL(acceleration)
        return True


def require_snapshot_state(
    state: RobotSnapshot, allowed_runtime: set[int], phase: str
) -> None:
    if state.emergency_stopped:
        raise RobotSafetyError(f"{phase}: UR5 急停中")
    if state.protective_stopped:
        raise RobotSafetyError(f"{phase}: UR5 保护停中")
    if state.robot_mode != ROBOT_MODE_RUNNING:
        raise RobotSafetyError(f"{phase}: robot_mode={state.robot_mode}，要求 7(RUNNING)")
    if state.safety_mode != SAFETY_MODE_NORMAL:
        raise RobotSafetyError(f"{phase}: safety_mode={state.safety_mode}，要求 1(NORMAL)")
    if state.runtime_state not in allowed_runtime:
        raise RobotSafetyError(
            f"{phase}: runtime_state={state.runtime_state}，允许值 {sorted(allowed_runtime)}"
        )


class PathPlanner:
    """将接触点分解成同 XY 的接近点和统一安全高度转运点。"""

    def __init__(self, config: RobotConfig) -> None:
        self.config = config

    def transit(self, pose: Pose, label: str) -> Waypoint:
        return Waypoint(label, pose.with_z(self.config.safe_z_transit), "transit")

    def approach(self, pose: Pose, label: str) -> Waypoint:
        return Waypoint(label, pose.with_z(self.config.safe_z_approach), "approach")

    @staticmethod
    def contact(pose: Pose, label: str) -> Waypoint:
        return Waypoint(label, pose, "contact")

    def validate_sequence(
        self,
        start: Pose,
        waypoints: Iterable[Waypoint],
        dynamic_contact_poses: Iterable[Pose] = (),
    ) -> None:
        dynamic_contacts = tuple(dynamic_contact_poses)
        self._require_workspace(start, "路径起点")
        previous = start
        previous_profile: str | None = None
        for waypoint in waypoints:
            self._require_profile_integrity(waypoint, dynamic_contacts)
            allowed_next = {
                None: {"transit"},
                "transit": {"transit", "approach"},
                "approach": {"contact", "transit"},
                "contact": {"approach"},
            }[previous_profile]
            if waypoint.profile not in allowed_next:
                prior = "路径起点" if previous_profile is None else previous_profile
                raise RobotSafetyError(
                    f"路径 {waypoint.label} 禁止速度档跳转 {prior}->{waypoint.profile}"
                )
            current = waypoint.pose
            self._require_workspace(current, waypoint.label)
            xy_changed = math.hypot(current.x - previous.x, current.y - previous.y) > 1e-9
            z_changed = abs(current.z - previous.z) > 1e-9
            horizontal_distance = math.hypot(current.x - previous.x, current.y - previous.y)
            vertical_distance = abs(current.z - previous.z)
            if horizontal_distance > self.config.max_horizontal_segment_m:
                raise RobotSafetyError(
                    f"路径 {waypoint.label} 水平段 {horizontal_distance:.3f}m 超过 "
                    f"{self.config.max_horizontal_segment_m:.3f}m"
                )
            if vertical_distance > self.config.max_vertical_segment_m:
                raise RobotSafetyError(
                    f"路径 {waypoint.label} 垂直段 {vertical_distance:.3f}m 超过 "
                    f"{self.config.max_vertical_segment_m:.3f}m"
                )
            if xy_changed and (
                not math.isclose(previous.z, self.config.safe_z_transit, abs_tol=1e-9)
                or not math.isclose(current.z, self.config.safe_z_transit, abs_tol=1e-9)
            ):
                raise RobotSafetyError(
                    f"路径 {waypoint.label} 在非 safe_z_transit 高度改变 XY"
                )
            if xy_changed and z_changed:
                raise RobotSafetyError(f"路径 {waypoint.label} 同时改变 XY 和 Z")
            if (
                previous_profile is not None
                and not (previous_profile == "transit" and waypoint.profile == "transit")
                and rotation_error(previous.as_list()[3:], current.as_list()[3:]) > 1e-9
            ):
                raise RobotSafetyError(
                    f"路径 {waypoint.label} 在非转运段改变工具朝向"
                )
            previous = current
            previous_profile = waypoint.profile

    def _require_profile_integrity(
        self, waypoint: Waypoint, dynamic_contacts: tuple[Pose, ...] = ()
    ) -> None:
        """Bind speed profiles to taught or explicitly validated dynamic geometry."""
        if waypoint.profile not in {"transit", "approach", "contact"}:
            raise RobotSafetyError(f"路径 {waypoint.label} 使用未知速度档 {waypoint.profile!r}")

        taught = {
            name: pose
            for name, pose in self.config.taught_poses.items()
            if pose is not None
        }
        contacts = {name: pose for name, pose in taught.items() if name != "home"}
        if waypoint.profile == "transit":
            candidates = [pose.with_z(self.config.safe_z_transit) for pose in taught.values()]
            candidates.extend(
                pose.with_z(self.config.safe_z_transit) for pose in dynamic_contacts
            )
        elif waypoint.profile == "approach":
            candidates = [pose.with_z(self.config.safe_z_approach) for pose in contacts.values()]
            candidates.extend(
                pose.with_z(self.config.safe_z_approach) for pose in dynamic_contacts
            )
        else:
            candidates = list(contacts.values())
            candidates.extend(dynamic_contacts)

        if not any(self._same_pose(waypoint.pose, candidate) for candidate in candidates):
            raise RobotSafetyError(
                f"路径 {waypoint.label} 的 {waypoint.profile} 位姿不是已示教点或"
                "已验证动态抓取点的对应派生位姿"
            )

    @staticmethod
    def _same_pose(first: Pose, second: Pose) -> bool:
        return (
            translation_error(first.as_list(), second.as_list()) <= 1e-9
            and rotation_error(first.as_list()[3:], second.as_list()[3:]) <= 1e-9
        )

    def _require_workspace(self, pose: Pose, label: str) -> None:
        workspace = self.config.workspace_bounds
        if workspace is None:
            raise RobotSafetyError("workspace_bounds 未配置")
        if not (
            workspace.x_min <= pose.x <= workspace.x_max
            and workspace.y_min <= pose.y <= workspace.y_max
            and workspace.z_min <= pose.z <= workspace.z_max
        ):
            raise RobotSafetyError(f"路径 {label} 超出 workspace_bounds")


def assert_pose_close(
    actual: Iterable[float], expected: Pose, translation_tolerance: float, rotation_tolerance: float,
    phase: str,
) -> None:
    actual_pose = _finite_vector(actual, 6, phase)
    translation = translation_error(actual_pose, expected.as_list())
    rotation = rotation_error(actual_pose[3:], expected.as_list()[3:])
    if translation > translation_tolerance or rotation > rotation_tolerance:
        raise RobotSafetyError(
            f"{phase}: TCP 偏差 {translation * 1000:.3f} mm / "
            f"{math.degrees(rotation):.3f}°，超出允许值"
        )


def assert_pose_unchanged(before: Iterable[float], after: Iterable[float]) -> None:
    first = _finite_vector(before, 6, "确认前 TCP")
    second = _finite_vector(after, 6, "确认后 TCP")
    if translation_error(first, second) > 0.0001 or rotation_error(first[3:], second[3:]) > math.radians(0.05):
        raise RobotSafetyError("人工确认期间 TCP 发生变化，拒绝创建控制接口")


def translation_error(first: Iterable[float], second: Iterable[float]) -> float:
    first_values, second_values = list(first), list(second)
    return _norm([first_values[i] - second_values[i] for i in range(3)])


def rotation_error(first: Iterable[float], second: Iterable[float]) -> float:
    first_q = _rotvec_to_quaternion(list(first))
    second_q = _rotvec_to_quaternion(list(second))
    dot = abs(sum(a * b for a, b in zip(first_q, second_q)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def read_dashboard(host: str, port: int, timeout: float = 2.0) -> dict[str, str]:
    commands = ("robotmode", "safetymode", "programState")
    result: dict[str, str] = {}
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            with sock.makefile("rb") as reader:
                result["banner"] = reader.readline().decode("utf-8", errors="replace").strip()
                for command in commands:
                    sock.sendall((command + "\n").encode("ascii"))
                    result[command] = reader.readline().decode("utf-8", errors="replace").strip()
    except OSError as exc:
        raise RobotError(f"Dashboard {host}:{port} 不可用: {exc}") from exc
    return result


def require_dashboard_state(values: dict[str, str], programs: set[str], phase: str) -> None:
    required = {"robotmode": "RUNNING", "safetymode": "NORMAL"}
    for command, expected in required.items():
        actual = _dashboard_token(command, values.get(command, ""))
        if actual != expected:
            raise RobotSafetyError(f"{phase}: Dashboard {command}={actual or 'EMPTY'}，要求 {expected}")
    actual_program = _dashboard_token("programState", values.get("programState", ""))
    if actual_program not in programs:
        raise RobotSafetyError(
            f"{phase}: Dashboard programState={actual_program or 'EMPTY'}，允许 {sorted(programs)}"
        )


def _dashboard_token(command: str, value: str) -> str:
    tokens = value.upper().replace(":", " ").split()
    if command == "programState":
        if tokens and tokens[0] in {"STOPPED", "PLAYING", "PAUSED"}:
            return tokens[0]
        if len(tokens) >= 2 and tokens[0] == "PROGRAMSTATE":
            return tokens[1]
        return ""
    prefix = "ROBOTMODE" if command == "robotmode" else "SAFETYMODE"
    return tokens[1] if len(tokens) >= 2 and tokens[0] == prefix else ""


def _require_tcp_port(host: str, port: int, timeout: float = 2.0) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        raise RobotError(f"无法连接 {host}:{port}: {exc}") from exc


def _finite_vector(values: Iterable[float], length: int, label: str) -> list[float]:
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RobotError(f"{label} 不是数值向量") from exc
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise RobotError(f"{label} 必须是 {length} 个有限数值")
    return result


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _rotvec_to_quaternion(rotvec: list[float]) -> tuple[float, float, float, float]:
    vector = _finite_vector(rotvec, 3, "旋转向量")
    angle = _norm(vector)
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    scale = math.sin(angle / 2.0) / angle
    return (math.cos(angle / 2.0), *(scale * value for value in vector))
