"""Conservative, explicit motion helper for eye-in-hand calibration.

This is intentionally separate from the read-only sampler.  It only moves to
one numbered pose per invocation, requires ``--execute`` plus physical
confirmation flags, and derives every target from sample 000 in the session.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .config import BatterySortingConfig, Pose
from .handeye import (
    HandEyeError,
    pose_vector_to_transform,
    rotation_distance_deg,
    transform_to_pose_vector,
)
from .robot import RealUR5, RobotError, Waypoint, rotation_error, translation_error


SPEED_MPS = 0.015
ACCELERATION_MPS2 = 0.05
MAX_ORIGIN_TRANSLATION_M = 0.040
MAX_ORIGIN_ROTATION_DEG = 16.0
MAX_SEGMENT_TRANSLATION_M = 0.060
MAX_SEGMENT_ROTATION_DEG = 15.0
START_NEIGHBOURHOOD_M = 0.080
START_NEIGHBOURHOOD_DEG = 25.0


# (base dx,dy,dz metres; local TCP rx,ry degrees).  Consecutive entries are
# ordered to avoid large direct swings between opposite tilts.
OFFSETS = (
    (+0.025, +0.000, +0.000, +8.0, +0.0),
    (+0.025, +0.025, +0.000, +8.0, +6.0),
    (+0.000, +0.025, +0.000, +0.0, +10.0),
    (-0.025, +0.025, +0.000, -8.0, +6.0),
    (-0.025, +0.000, +0.000, -10.0, +0.0),
    (-0.025, -0.025, +0.000, -8.0, -6.0),
    (+0.000, -0.025, +0.000, +0.0, -10.0),
    (+0.025, -0.025, +0.000, +8.0, -6.0),
    (+0.025, +0.000, +0.020, +8.0, +0.0),
    (+0.000, +0.025, +0.020, +0.0, +10.0),
    (-0.025, +0.000, +0.020, -8.0, +0.0),
    (+0.000, -0.025, +0.020, +0.0, -10.0),
    (+0.020, -0.020, -0.015, +6.0, -6.0),
    (+0.020, +0.020, -0.015, +6.0, +6.0),
    (-0.020, +0.020, -0.015, -6.0, +6.0),
    (-0.020, -0.020, -0.015, -6.0, -6.0),
    (+0.000, +0.000, +0.000, -10.0, +0.0),
    (+0.000, +0.000, +0.000, +0.0, -10.0),
    # Extended rotation set.  Run only after returning to the start pose.
    (+0.020, +0.000, +0.000, +15.0, +0.0),
    (+0.020, +0.020, +0.000, +8.0, +12.0),
    (+0.000, +0.020, +0.000, +0.0, +15.0),
    (-0.020, +0.020, +0.000, -8.0, +12.0),
    (-0.020, +0.000, +0.000, -15.0, +0.0),
    (-0.020, -0.020, +0.000, -8.0, -12.0),
    (+0.000, -0.020, +0.000, +0.0, -15.0),
    (+0.020, -0.020, +0.000, +8.0, -12.0),
)


def generate_targets(start_pose: tuple[float, ...]) -> list[Pose]:
    start = pose_vector_to_transform(start_pose)
    targets: list[Pose] = []
    for dx, dy, dz, rx_deg, ry_deg in OFFSETS:
        local_vector = np.radians([rx_deg, ry_deg, 0.0])
        local_rotation, _ = cv2.Rodrigues(local_vector.reshape(3, 1))
        target = start.copy()
        target[:3, :3] = start[:3, :3] @ local_rotation
        target[:3, 3] = start[:3, 3] + np.array([dx, dy, dz])
        vector = transform_to_pose_vector(target)
        targets.append(Pose(*[float(value) for value in vector]))
    validate_plan(start_pose, targets)
    return targets


def validate_plan(start_pose: tuple[float, ...], targets: list[Pose]) -> None:
    start = pose_vector_to_transform(start_pose)
    previous = start
    for index, pose in enumerate(targets):
        if index == 18:
            # Stage two is deliberately started only after return-start.
            previous = start
        target = pose_vector_to_transform(pose.as_list())
        origin_translation = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
        origin_rotation = rotation_distance_deg(start, target)
        segment_translation = float(np.linalg.norm(target[:3, 3] - previous[:3, 3]))
        segment_rotation = rotation_distance_deg(previous, target)
        if origin_translation > MAX_ORIGIN_TRANSLATION_M + 1e-12:
            raise HandEyeError(f"轨迹 {index} 超出起点平移包络")
        if origin_rotation > MAX_ORIGIN_ROTATION_DEG + 1e-9:
            raise HandEyeError(f"轨迹 {index} 超出起点旋转包络")
        if segment_translation > MAX_SEGMENT_TRANSLATION_M + 1e-12:
            raise HandEyeError(f"轨迹 {index} 单段平移过大")
        if segment_rotation > MAX_SEGMENT_ROTATION_DEG + 1e-9:
            raise HandEyeError(f"轨迹 {index} 单段旋转过大")
        previous = target


def _start_pose(session_dir: Path) -> tuple[float, ...]:
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    records = session.get("samples", [])
    if not records:
        raise HandEyeError("会话没有 sample 000，不能定义安全起点")
    first = json.loads((session_dir / records[0]).read_text(encoding="utf-8"))
    pose = tuple(float(value) for value in first["tcp_pose_before"])
    if len(pose) != 6:
        raise HandEyeError("sample 000 TCP 位姿无效")
    return pose


def _require_confirmations(args: argparse.Namespace) -> None:
    if not args.execute:
        raise HandEyeError("未给出 --execute；仅允许打印轨迹")
    if not args.tcp_payload_confirmed:
        raise HandEyeError("必须给出 --tcp-payload-confirmed")
    if args.clearance_mm < 100.0:
        raise HandEyeError("自动标定要求确认至少 100mm 无障碍空间")


def _move(args: argparse.Namespace, return_to_start: bool = False) -> int:
    session_dir = Path(args.session).resolve()
    robot: RealUR5 | None = None
    try:
        _require_confirmations(args)
        start_pose = _start_pose(session_dir)
        targets = generate_targets(start_pose)
        if return_to_start:
            target_pose = Pose(*start_pose)
            label = "handeye_return_start"
        else:
            if args.index < 1 or args.index > len(targets):
                raise HandEyeError(f"--index 必须在 1..{len(targets)}")
            target_pose = targets[args.index - 1]
            label = f"handeye_pose_{args.index:02d}"

        config = BatterySortingConfig.from_json(args.config)
        motion_config = replace(
            config.robot,
            move_timeout=30.0,
            max_tcp_force_delta_n=20.0,
            stop_deceleration=0.20,
        )
        robot = RealUR5(motion_config)
        robot.connect_readonly()
        current = robot.snapshot()
        if translation_error(current.tcp_pose, start_pose) > START_NEIGHBOURHOOD_M:
            raise HandEyeError("当前 TCP 距标定起点超过 80mm，拒绝自动运动")
        if math.degrees(rotation_error(current.tcp_pose[3:], start_pose[3:])) > START_NEIGHBOURHOOD_DEG:
            raise HandEyeError("当前 TCP 姿态距标定起点超过 25°，拒绝自动运动")
        segment_translation = translation_error(current.tcp_pose, target_pose.as_list())
        segment_rotation = math.degrees(
            rotation_error(current.tcp_pose[3:], target_pose.as_list()[3:])
        )
        if segment_translation > MAX_SEGMENT_TRANSLATION_M:
            raise HandEyeError(
                f"当前到目标单段平移 {segment_translation * 1000:.1f}mm 超过 60mm"
            )
        if segment_rotation > MAX_SEGMENT_ROTATION_DEG:
            raise HandEyeError(
                f"当前到目标单段旋转 {segment_rotation:.1f}° 超过 15°"
            )
        robot.connect_control()
        robot.move_l_without_tcp_force_guard(
            Waypoint(label, target_pose, "handeye"), SPEED_MPS, ACCELERATION_MPS2
        )
        final = robot.snapshot()
        print(
            f"ARRIVED label={label} translation_segment={segment_translation * 1000:.1f}mm "
            f"rotation_segment={segment_rotation:.1f}deg "
            f"tcp={[round(value, 6) for value in final.tcp_pose]}"
        )
        return 0
    except (HandEyeError, RobotError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    finally:
        if robot is not None:
            robot.disconnect()


def _plan(args: argparse.Namespace) -> int:
    try:
        start = _start_pose(Path(args.session).resolve())
        targets = generate_targets(start)
        print(f"start={[round(value, 6) for value in start]}")
        for index, target in enumerate(targets, 1):
            print(f"{index:02d}: {[round(value, 6) for value in target.as_list()]}")
        print(
            f"targets={len(targets)} envelope_translation<={MAX_ORIGIN_TRANSLATION_M*1000:.0f}mm "
            f"envelope_rotation<={MAX_ORIGIN_ROTATION_DEG:.0f}deg"
        )
        return 0
    except (HandEyeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


def _batch(args: argparse.Namespace) -> int:
    if args.start_index < 1 or args.end_index < args.start_index or args.end_index > len(OFFSETS):
        print(f"FAIL: 批次索引必须满足 1 <= start <= end <= {len(OFFSETS)}", file=sys.stderr)
        return 2
    from .handeye_cli import capture_once

    for index in range(args.start_index, args.end_index + 1):
        args.index = index
        move_status = _move(args, False)
        if move_status != 0:
            print(f"BATCH_STOP move_index={index}", file=sys.stderr)
            return move_status
        capture_args = argparse.Namespace(
            session=args.session,
            config=args.config,
            serial=args.serial,
            columns=args.columns,
            rows=args.rows,
            square_size_mm=args.square_size_mm,
        )
        capture_status = capture_once(capture_args)
        if capture_status != 0:
            print(f"BATCH_STOP capture_after_index={index}", file=sys.stderr)
            return capture_status
    print(f"BATCH_PASS indices={args.start_index}..{args.end_index}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--session", required=True)
    plan.set_defaults(function=_plan)
    for name in ("move", "return-start"):
        command = subparsers.add_parser(name)
        command.add_argument("--session", required=True)
        command.add_argument("--config", default="config/site_t0_rgbd_20260908.json")
        if name == "move":
            command.add_argument("--index", type=int, required=True)
            command.set_defaults(function=lambda args: _move(args, False))
        else:
            command.set_defaults(function=lambda args: _move(args, True))
        command.add_argument("--execute", action="store_true")
        command.add_argument("--tcp-payload-confirmed", action="store_true")
        command.add_argument("--clearance-mm", type=float, default=0.0)
    batch = subparsers.add_parser("batch")
    batch.add_argument("--session", required=True)
    batch.add_argument("--config", default="config/site_t0_rgbd_20260908.json")
    batch.add_argument("--serial", default="234222302865")
    batch.add_argument("--columns", type=int, default=9)
    batch.add_argument("--rows", type=int, default=6)
    batch.add_argument("--square-size-mm", type=float, default=20.0)
    batch.add_argument("--start-index", type=int, required=True)
    batch.add_argument("--end-index", type=int, required=True)
    batch.add_argument("--execute", action="store_true")
    batch.add_argument("--tcp-payload-confirmed", action="store_true")
    batch.add_argument("--clearance-mm", type=float, default=0.0)
    batch.set_defaults(function=_batch)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
