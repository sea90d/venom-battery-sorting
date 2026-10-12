"""Plan and execute a high-Z eye-in-hand battery pointing validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

from .config import BatterySortingConfig, Pose
from .geometry import transform_point, validate_rigid_transform
from .handeye import HandEyeError, pose_vector_to_transform, transform_to_pose_vector
from .robot import RealUR5, RobotError, Waypoint, translation_error
from .realsense import RealSenseCamera, RealSenseError


MAX_HORIZONTAL_MOVE_M = 0.35
MAX_SEGMENT_M = 0.05
MIN_VERTICAL_CLEARANCE_M = 0.20
SPEED_MPS = 0.020
ACCELERATION_MPS2 = 0.05
ROTATION_SPEED = 0.04


def camera_point_to_base(
    camera_xyz_m: tuple[float, float, float],
    base_T_tcp: np.ndarray,
    tcp_T_camera: np.ndarray,
) -> tuple[float, float, float]:
    validate_rigid_transform(base_T_tcp, "base_T_tcp")
    validate_rigid_transform(tcp_T_camera, "tcp_T_camera")
    return transform_point(camera_xyz_m, base_T_tcp @ tcp_T_camera)


def _load_inputs(args: argparse.Namespace) -> tuple[dict, dict]:
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    diagnostic = json.loads(Path(args.rgbd_report).read_text(encoding="utf-8"))
    if calibration.get("passed") is not True:
        raise HandEyeError("手眼标定结果未通过")
    if diagnostic.get("valid") is not True:
        raise HandEyeError("RGB-D 定位报告未通过")
    return calibration, diagnostic


def _locations(
    calibration: dict,
    diagnostic: dict,
    tcp_pose: tuple[float, ...],
) -> dict[str, tuple[float, float, float]]:
    base_T_tcp = pose_vector_to_transform(tcp_pose)
    tcp_T_camera = np.asarray(calibration["tcp_T_camera"], dtype=np.float64)
    output: dict[str, tuple[float, float, float]] = {}
    for slot in diagnostic["slots"]:
        if not slot.get("valid") or slot.get("camera_xyz_m") is None:
            raise HandEyeError(f"{slot.get('slot_name', '?')} 没有有效相机三维坐标")
        color = str(slot["color"])
        if color in output:
            raise HandEyeError(f"颜色 {color} 出现多次")
        output[color] = camera_point_to_base(
            tuple(float(value) for value in slot["camera_xyz_m"]),
            base_T_tcp,
            tcp_T_camera,
        )
    if set(output) != {"red", "yellow", "blue", "green"}:
        raise HandEyeError(f"颜色集合不完整: {sorted(output)}")
    return output


def _read_robot(args: argparse.Namespace) -> tuple[RealUR5, tuple[float, ...]]:
    config = BatterySortingConfig.from_json(args.config)
    robot = RealUR5(
        replace(config.robot, move_timeout=30.0, stop_deceleration=0.20)
    )
    robot.connect_readonly()
    return robot, robot.snapshot().tcp_pose


def plan(args: argparse.Namespace) -> int:
    robot = None
    try:
        calibration, diagnostic = _load_inputs(args)
        robot, tcp_pose = _read_robot(args)
        locations = _locations(calibration, diagnostic, tcp_pose)
        print(f"tcp_current={[round(value, 6) for value in tcp_pose]}")
        for color, point in locations.items():
            horizontal = math.hypot(point[0] - tcp_pose[0], point[1] - tcp_pose[1])
            clearance = tcp_pose[2] - point[2]
            print(
                f"{color}: base_xyz_m={[round(value, 6) for value in point]} "
                f"horizontal={horizontal * 1000:.1f}mm "
                f"vertical_clearance={clearance * 1000:.1f}mm"
            )
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, HandEyeError, RobotError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    finally:
        if robot is not None:
            robot.disconnect()


def execute(args: argparse.Namespace) -> int:
    raise HandEyeError('Historical report motion disabled; use battery_sorting.closed_loop')
    robot = None
    try:
        if not args.execute or not args.tcp_payload_confirmed or not args.emergency_stop_ready:
            raise HandEyeError(
                "执行需要 --execute --tcp-payload-confirmed --emergency-stop-ready"
            )
        if not args.ignore_tcp_force:
            raise HandEyeError("本次已授权流程要求显式给出 --ignore-tcp-force")
        calibration, diagnostic = _load_inputs(args)
        robot, tcp_pose = _read_robot(args)
        locations = _locations(calibration, diagnostic, tcp_pose)
        if args.color not in locations:
            raise HandEyeError(f"未找到颜色 {args.color}")
        point = locations[args.color]
        horizontal = math.hypot(point[0] - tcp_pose[0], point[1] - tcp_pose[1])
        clearance = tcp_pose[2] - point[2]
        if horizontal > MAX_HORIZONTAL_MOVE_M:
            raise HandEyeError(
                f"水平移动 {horizontal * 1000:.1f}mm 超过 {MAX_HORIZONTAL_MOVE_M * 1000:.0f}mm"
            )
        if clearance < MIN_VERTICAL_CLEARANCE_M:
            raise HandEyeError(
                f"估计垂直净空 {clearance * 1000:.1f}mm 小于 {MIN_VERTICAL_CLEARANCE_M * 1000:.0f}mm"
            )
        segment_count = max(1, math.ceil(horizontal / MAX_SEGMENT_M))
        start = np.asarray(tcp_pose[:3], dtype=np.float64)
        final_xyz = np.array([point[0], point[1], tcp_pose[2]], dtype=np.float64)
        robot.connect_control()
        for index in range(1, segment_count + 1):
            ratio = index / segment_count
            xyz = start + ratio * (final_xyz - start)
            pose = Pose(
                float(xyz[0]), float(xyz[1]), float(xyz[2]),
                *[float(value) for value in tcp_pose[3:]],
            )
            robot.move_l_without_tcp_force_guard(
                Waypoint(f"hover_{args.color}_{index:02d}", pose, "transit"),
                SPEED_MPS,
                ACCELERATION_MPS2,
            )
        actual = robot.snapshot().tcp_pose
        xy_error = math.hypot(actual[0] - point[0], actual[1] - point[1])
        if xy_error > 0.0005:
            raise HandEyeError(f"到位 XY 误差 {xy_error * 1000:.3f}mm 超过 0.5mm")
        report = {
            "passed": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "color": args.color,
            "detected_battery_base_xyz_m": point,
            "start_tcp_pose": tcp_pose,
            "target_tcp_pose": [point[0], point[1], tcp_pose[2], *tcp_pose[3:]],
            "actual_tcp_pose": actual,
            "horizontal_move_m": horizontal,
            "estimated_vertical_clearance_m": clearance,
            "segment_count": segment_count,
            "xy_arrival_error_m": xy_error,
            "tcp_force_ignored_by_operator_request": True,
            "contact_motion": False,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"PASS color={args.color} target_xy=({point[0]:.6f},{point[1]:.6f}) "
            f"actual_xy=({actual[0]:.6f},{actual[1]:.6f}) "
            f"arrival_error={xy_error * 1000:.3f}mm "
            f"z_held={actual[2]:.6f}m clearance={clearance * 1000:.1f}mm"
        )
        print(f"report={output.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, HandEyeError, RobotError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    finally:
        if robot is not None:
            robot.disconnect()


def verify(args: argparse.Namespace) -> int:
    robot = None
    camera = None
    try:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        motion = json.loads(Path(args.motion_report).read_text(encoding="utf-8"))
        if calibration.get("passed") is not True or motion.get("passed") is not True:
            raise HandEyeError("标定结果或悬空运动报告未通过")
        config = BatterySortingConfig.from_json(args.config)
        color_threshold = next(
            (item for item in config.vision.colors if item.name == motion["color"]), None
        )
        if color_threshold is None:
            raise HandEyeError(f"缺少颜色阈值: {motion['color']}")
        robot, before_pose = _read_robot(args)
        camera = RealSenseCamera(warmup_frames=10)
        camera.start(serial=args.serial)
        image, _depth = camera.capture()
        after_pose = robot.snapshot().tcp_pose
        if translation_error(before_pose, after_pose) > 0.0002:
            raise HandEyeError("验证拍摄期间 TCP 发生移动")

        base_T_tcp = pose_vector_to_transform(before_pose)
        tcp_T_camera = np.asarray(calibration["tcp_T_camera"], dtype=np.float64)
        base_T_camera = base_T_tcp @ tcp_T_camera
        camera_T_base = np.linalg.inv(base_T_camera)
        point_base = np.array([*motion["detected_battery_base_xyz_m"], 1.0], dtype=np.float64)
        point_camera = camera_T_base @ point_base
        if point_camera[2] <= 0.0:
            raise HandEyeError("目标点位于当前相机后方")
        predicted = camera.project_color_points(point_camera[:3].reshape(1, 3))[0]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for item in color_threshold.ranges:
            lower = np.array([item.h_min, item.s_min, item.v_min], dtype=np.uint8)
            upper = np.array([item.h_max, item.s_max, item.v_max], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            moments = cv2.moments(contour)
            if area < 500.0 or moments["m00"] <= 0.0:
                continue
            centroid = np.array(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float64,
            )
            candidates.append((float(np.linalg.norm(centroid - predicted)), centroid, area, contour))
        if not candidates:
            raise HandEyeError("新图中未找到足够大的目标颜色实物")
        pixel_error, observed, area, contour = min(candidates, key=lambda item: item[0])
        if pixel_error > args.max_pixel_error:
            raise HandEyeError(
                f"重投影点与新图颜色质心相差 {pixel_error:.1f}px，超过 {args.max_pixel_error:.1f}px"
            )

        annotated = image.copy()
        cv2.drawContours(annotated, [contour], -1, (0, 255, 255), 2)
        cv2.drawMarker(
            annotated,
            tuple(np.rint(predicted).astype(int)),
            (255, 0, 255),
            cv2.MARKER_CROSS,
            28,
            3,
        )
        cv2.circle(annotated, tuple(np.rint(observed).astype(int)), 8, (0, 0, 255), 2)
        cv2.line(
            annotated,
            tuple(np.rint(predicted).astype(int)),
            tuple(np.rint(observed).astype(int)),
            (255, 255, 0),
            2,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / "hover_visual_verification.png"), annotated)
        cv2.imwrite(str(output_dir / "target_color_mask.png"), mask)
        report = {
            "passed": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "color": motion["color"],
            "predicted_pixel": predicted,
            "observed_color_centroid_pixel": observed,
            "pixel_error": pixel_error,
            "max_pixel_error": args.max_pixel_error,
            "component_area_px": area,
            "tcp_pose": before_pose,
        }
        (output_dir / "visual_verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.tolist()) + "\n",
            encoding="utf-8",
        )
        print(
            f"PASS predicted=({predicted[0]:.1f},{predicted[1]:.1f}) "
            f"observed=({observed[0]:.1f},{observed[1]:.1f}) "
            f"pixel_error={pixel_error:.2f}px area={area:.0f}px"
        )
        print(f"output={output_dir.resolve()}")
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        HandEyeError,
        RobotError,
        RealSenseError,
    ) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    finally:
        if camera is not None:
            camera.stop()
        if robot is not None:
            robot.disconnect()


def rotated_tcp_pose(
    initial_pose: tuple[float, ...], local_z_rotation_deg: float
) -> Pose:
    if not math.isfinite(local_z_rotation_deg) or abs(local_z_rotation_deg) > 90.0:
        raise HandEyeError("单次预抓取旋转必须是有限值且不超过 90°")
    transform = pose_vector_to_transform(initial_pose)
    local_rotation = np.array(
        [0.0, 0.0, math.radians(local_z_rotation_deg)], dtype=np.float64
    )
    rotation_matrix, _ = cv2.Rodrigues(local_rotation.reshape(3, 1))
    transform[:3, :3] = transform[:3, :3] @ rotation_matrix
    vector = transform_to_pose_vector(transform)
    return Pose(*[float(value) for value in vector])


def rotate_pregrasp(args: argparse.Namespace) -> int:
    raise HandEyeError('Historical report motion disabled; use battery_sorting.closed_loop')
    robot = None
    try:
        if not args.execute or not args.tcp_payload_confirmed or not args.emergency_stop_ready:
            raise HandEyeError(
                "执行需要 --execute --tcp-payload-confirmed --emergency-stop-ready"
            )
        if not args.ignore_tcp_force:
            raise HandEyeError("本次已授权流程要求显式给出 --ignore-tcp-force")
        motion = json.loads(Path(args.motion_report).read_text(encoding="utf-8"))
        if motion.get("passed") is not True:
            raise HandEyeError("悬空指向运动报告未通过")
        robot, initial = _read_robot(args)
        target_xy = motion["detected_battery_base_xyz_m"][:2]
        xy_error = math.hypot(initial[0] - target_xy[0], initial[1] - target_xy[1])
        if xy_error > 0.001:
            raise HandEyeError(
                f"当前 TCP 已偏离悬空目标 {xy_error * 1000:.2f}mm，拒绝原地旋转"
            )
        steps = max(1, math.ceil(abs(args.rotation_deg) / 30.0))
        robot.connect_control()
        for index in range(1, steps + 1):
            target = rotated_tcp_pose(initial, args.rotation_deg * index / steps)
            robot.move_l_without_tcp_force_guard(
                Waypoint(f"pregrasp_rotate_{index:02d}", target, "pregrasp"),
                ROTATION_SPEED,
                ACCELERATION_MPS2,
            )
        actual = robot.snapshot().tcp_pose
        position_drift = translation_error(initial, actual)
        if position_drift > 0.0005:
            raise HandEyeError(
                f"原地旋转后 TCP 位置漂移 {position_drift * 1000:.3f}mm 超过 0.5mm"
            )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "passed": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rotation_axis": "local_tcp_z",
            "rotation_deg": args.rotation_deg,
            "steps": steps,
            "initial_tcp_pose": initial,
            "target_tcp_pose": rotated_tcp_pose(initial, args.rotation_deg).as_list(),
            "actual_tcp_pose": actual,
            "position_drift_m": position_drift,
            "z_descent_m": 0.0,
            "tcp_force_ignored_by_operator_request": True,
        }
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"PASS rotated_local_z={args.rotation_deg:.1f}deg steps={steps} "
            f"position_drift={position_drift * 1000:.3f}mm "
            f"xyz=({actual[0]:.6f},{actual[1]:.6f},{actual[2]:.6f})"
        )
        print(f"report={output.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, HandEyeError, RobotError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    finally:
        if robot is not None:
            robot.disconnect()


def _common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--calibration", required=True)
    command.add_argument("--rgbd-report", required=True)
    command.add_argument("--config", default="config/site_t0_rgbd_20260908.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    show = subs.add_parser("plan")
    _common(show)
    show.set_defaults(function=plan)
    move = subs.add_parser("execute")
    _common(move)
    move.add_argument("--color", choices=("red", "yellow", "blue", "green"), default="green")
    move.add_argument("--output", required=True)
    move.add_argument("--execute", action="store_true")
    move.add_argument("--tcp-payload-confirmed", action="store_true")
    move.add_argument("--emergency-stop-ready", action="store_true")
    move.add_argument("--ignore-tcp-force", action="store_true")
    move.set_defaults(function=execute)
    check = subs.add_parser("verify")
    check.add_argument("--calibration", required=True)
    check.add_argument("--motion-report", required=True)
    check.add_argument("--config", default="config/site_t0_rgbd_20260908.json")
    check.add_argument("--serial", default="234222302865")
    check.add_argument("--output-dir", required=True)
    check.add_argument("--max-pixel-error", type=float, default=15.0)
    check.set_defaults(function=verify)
    rotate = subs.add_parser("rotate-pregrasp")
    rotate.add_argument("--motion-report", required=True)
    rotate.add_argument("--config", default="config/site_t0_rgbd_20260908.json")
    rotate.add_argument("--rotation-deg", type=float, default=90.0)
    rotate.add_argument("--output", required=True)
    rotate.add_argument("--execute", action="store_true")
    rotate.add_argument("--tcp-payload-confirmed", action="store_true")
    rotate.add_argument("--emergency-stop-ready", action="store_true")
    rotate.add_argument("--ignore-tcp-force", action="store_true")
    rotate.set_defaults(function=rotate_pregrasp)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in ("execute", "rotate-pregrasp"):
        raise SystemExit("Historical report motion disabled; use battery_sorting.closed_loop")
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
