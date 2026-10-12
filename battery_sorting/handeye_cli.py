"""Read-only UR5 + D455 eye-in-hand sampling and offline calibration.

This command never creates ``RTDEControlInterface`` and never moves the robot.
Move to each pose with the teach pendant, stop the robot, then run ``capture``.
Every accepted sample is persisted immediately so an interruption loses no
previous observations.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import BatterySortingConfig
from .handeye import (
    CheckerboardSpec,
    HandEyeError,
    HandEyeSample,
    checkerboard_points_in_camera,
    detect_checkerboard,
    estimate_target_to_camera,
    pose_vector_to_transform,
    rotation_distance_deg,
    solve_eye_in_hand,
)
from .realsense import RealSenseCamera, RealSenseError
from .robot import RealUR5, RobotError, translation_error, rotation_error


MAX_CAPTURE_TRANSLATION_M = 0.0002
MAX_CAPTURE_ROTATION_RAD = math.radians(0.1)
MIN_DUPLICATE_TRANSLATION_M = 0.002
MIN_DUPLICATE_ROTATION_DEG = 2.0
MAX_REPROJECTION_RMSE_PX = 1.0
MIN_MEAN_SQUARE_EDGE_PX = 10.0


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_ready) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("PNG 编码失败")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _load_session(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "session.json"
    if not path.exists():
        raise HandEyeError(f"标定会话不存在: {path}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HandEyeError(f"读取标定会话失败: {exc}") from exc
    if result.get("schema_version") != 1 or result.get("calibration_type") != "eye_in_hand":
        raise HandEyeError("标定会话 schema 或类型不受支持")
    if not isinstance(result.get("samples"), list):
        raise HandEyeError("标定会话 samples 无效")
    return result


def _new_session(
    session_dir: Path,
    spec: CheckerboardSpec,
    robot_host: str,
    serial: str,
    camera: RealSenseCamera,
) -> dict[str, Any]:
    session_dir.mkdir(parents=True, exist_ok=True)
    if (session_dir / "session.json").exists():
        return _load_session(session_dir)
    model, coefficients = camera.color_distortion
    intrinsics = camera.color_intrinsics
    session = {
        "schema_version": 1,
        "calibration_type": "eye_in_hand",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_chain": "base_T_target = base_T_tcp @ tcp_T_camera @ camera_T_target",
        "robot": {
            "host": robot_host,
            "pose_source": "RTDE getActualTCPPose",
            "tcp_must_remain_unchanged": True,
        },
        "camera": {
            "serial": serial,
            "width": 1280,
            "height": 720,
            "intrinsics": {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
            },
            "distortion_model": model,
            "distortion_coefficients": list(coefficients),
            "normalization": "RealSense rs2_deproject_pixel_to_point at depth=1",
        },
        "checkerboard": {
            "inner_columns": spec.inner_columns,
            "inner_rows": spec.inner_rows,
            "square_size_m": spec.square_size_m,
        },
        "samples": [],
    }
    _write_json(session_dir / "session.json", session)
    return session


def _spec_from_session(session: dict[str, Any]) -> CheckerboardSpec:
    board = session["checkerboard"]
    return CheckerboardSpec(
        inner_columns=int(board["inner_columns"]),
        inner_rows=int(board["inner_rows"]),
        square_size_m=float(board["square_size_m"]),
    )


def _sample_from_record(session_dir: Path, relative_path: str) -> HandEyeSample:
    path = session_dir / relative_path
    data = json.loads(path.read_text(encoding="utf-8"))
    return HandEyeSample(
        base_T_gripper=np.asarray(data["base_T_tcp"], dtype=np.float64),
        camera_T_target=np.asarray(data["camera_T_target"], dtype=np.float64),
        reprojection_rmse_px=float(data["reprojection_rmse_px"]),
    )


def _mean_square_edge(corners: np.ndarray, spec: CheckerboardSpec) -> float:
    grid = corners.reshape(spec.inner_rows, spec.inner_columns, 2)
    horizontal = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2)
    vertical = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2)
    return float(np.mean(np.concatenate((horizontal.ravel(), vertical.ravel()))))


def _check_not_duplicate(
    current: np.ndarray, previous: list[HandEyeSample]
) -> None:
    for index, sample in enumerate(previous):
        translation = float(
            np.linalg.norm(current[:3, 3] - sample.base_T_gripper[:3, 3])
        )
        rotation = rotation_distance_deg(current, sample.base_T_gripper)
        if (
            translation < MIN_DUPLICATE_TRANSLATION_M
            and rotation < MIN_DUPLICATE_ROTATION_DEG
        ):
            raise HandEyeError(
                f"当前姿态与样本 {index:03d} 近重复："
                f"{translation * 1000:.2f}mm, {rotation:.2f}°"
            )


def capture_once(args: argparse.Namespace) -> int:
    session_dir = Path(args.session).resolve()
    config = BatterySortingConfig.from_json(args.config)
    spec = CheckerboardSpec(args.columns, args.rows, args.square_size_mm / 1000.0)
    camera = RealSenseCamera(warmup_frames=10)
    robot = RealUR5(config.robot)
    try:
        camera.start(serial=args.serial)
        robot.connect_readonly()
        session = _new_session(session_dir, spec, config.robot.host, args.serial, camera)
        session_spec = _spec_from_session(session)
        if session_spec != spec:
            raise HandEyeError("命令行棋盘格参数与现有会话不一致")
        if session["camera"]["serial"] != args.serial:
            raise HandEyeError("相机序列号与现有会话不一致")
        existing = [
            _sample_from_record(session_dir, record) for record in session["samples"]
        ]

        before = robot.snapshot()
        image, _depth = camera.capture()
        after = robot.snapshot()
        translation_change = translation_error(before.tcp_pose, after.tcp_pose)
        rotation_change = rotation_error(before.tcp_pose[3:], after.tcp_pose[3:])
        if translation_change > MAX_CAPTURE_TRANSLATION_M or rotation_change > MAX_CAPTURE_ROTATION_RAD:
            raise HandEyeError(
                f"拍摄期间 TCP 发生运动：{translation_change * 1000:.3f}mm, "
                f"{math.degrees(rotation_change):.3f}°"
            )
        base_T_tcp = pose_vector_to_transform(before.tcp_pose)
        _check_not_duplicate(base_T_tcp, existing)

        corners = detect_checkerboard(image, spec)
        edge_px = _mean_square_edge(corners, spec)
        if edge_px < MIN_MEAN_SQUARE_EDGE_PX:
            raise HandEyeError(
                f"棋盘格在图中太小：平均格边 {edge_px:.1f}px < {MIN_MEAN_SQUARE_EDGE_PX:.1f}px"
            )
        if existing:
            first_data = json.loads(
                (session_dir / session["samples"][0]).read_text(encoding="utf-8")
            )
            first_corners = np.asarray(first_data["corners_px"], dtype=np.float64)
            current_axis = corners[spec.inner_columns - 1] - corners[0]
            first_axis = first_corners[spec.inner_columns - 1] - first_corners[0]
            if float(np.dot(current_axis, first_axis)) <= 0.0:
                raise HandEyeError("棋盘格角点顺序相对首帧翻转 180°，拒绝该样本")

        normalized = camera.normalize_color_pixels(corners)
        pnp = estimate_target_to_camera(normalized, spec)
        points_camera = checkerboard_points_in_camera(pnp.camera_T_target, spec)
        projected = camera.project_color_points(points_camera)
        pixel_errors = np.linalg.norm(projected - corners, axis=1)
        reprojection_rmse_px = float(np.sqrt(np.mean(pixel_errors**2)))
        if reprojection_rmse_px > MAX_REPROJECTION_RMSE_PX:
            raise HandEyeError(
                f"棋盘格重投影 RMSE={reprojection_rmse_px:.3f}px 超过 "
                f"{MAX_REPROJECTION_RMSE_PX:.3f}px"
            )

        index = len(existing)
        stem = f"sample_{index:03d}"
        annotated = image.copy()
        cv2.drawChessboardCorners(
            annotated,
            (spec.inner_columns, spec.inner_rows),
            corners.reshape(-1, 1, 2).astype(np.float32),
            True,
        )
        _write_png(session_dir / "images" / f"{stem}.png", image)
        _write_png(session_dir / "images" / f"{stem}_corners.png", annotated)
        record = {
            "index": index,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tcp_pose_before": list(before.tcp_pose),
            "tcp_pose_after": list(after.tcp_pose),
            "capture_motion_translation_m": translation_change,
            "capture_motion_rotation_deg": math.degrees(rotation_change),
            "base_T_tcp": base_T_tcp,
            "camera_T_target": pnp.camera_T_target,
            "corners_px": corners,
            "mean_square_edge_px": edge_px,
            "normalized_reprojection_rmse": pnp.normalized_rmse,
            "reprojection_rmse_px": reprojection_rmse_px,
            "reprojection_max_px": float(np.max(pixel_errors)),
            "image": f"images/{stem}.png",
            "annotated_image": f"images/{stem}_corners.png",
        }
        relative_record = f"samples/{stem}.json"
        _write_json(session_dir / relative_record, record)
        session["samples"].append(relative_record)
        _write_json(session_dir / "session.json", session)
        print(
            f"ACCEPT sample={index:03d} corners={spec.corner_count} "
            f"edge={edge_px:.1f}px reproj_rmse={reprojection_rmse_px:.3f}px "
            f"tcp={[round(value, 6) for value in before.tcp_pose]}"
        )
        print(f"session={session_dir} total_samples={len(session['samples'])}")
        return 0
    except (HandEyeError, RealSenseError, RobotError, OSError, ValueError) as exc:
        print(f"REJECT: {exc}", file=sys.stderr)
        return 2
    finally:
        robot.disconnect()
        camera.stop()


def solve_session(args: argparse.Namespace) -> int:
    session_dir = Path(args.session).resolve()
    try:
        session = _load_session(session_dir)
        samples = [
            _sample_from_record(session_dir, record) for record in session["samples"]
        ]
        max_translation_rmse_m = args.max_translation_rmse_mm / 1000.0
        if args.max_rotation_rmse_deg > 0.5 and not args.operator_accepted:
            raise HandEyeError(
                "放宽默认 0.5° 旋转门槛必须同时给出 --operator-accepted"
            )
        result = solve_eye_in_hand(
            samples,
            max_translation_rmse_m=max_translation_rmse_m,
            max_rotation_rmse_deg=args.max_rotation_rmse_deg,
        )
        report = {
            "passed": True,
            "sample_count": result.sample_count,
            "selected_method": result.selected.method,
            "tcp_T_camera": result.selected.gripper_T_camera,
            "translation_span_m": result.translation_span_m,
            "rotation_span_deg": result.rotation_span_deg,
            "second_rotation_span_deg": result.second_rotation_span_deg,
            "selected_metrics": result.selected.__dict__,
            "candidates": [candidate.__dict__ for candidate in result.candidates],
            "coordinate_chain": session["coordinate_chain"],
            "warning": "Result is tied to the active TCP used during every sample.",
            "acceptance_thresholds": {
                "max_translation_rmse_mm": args.max_translation_rmse_mm,
                "max_rotation_rmse_deg": args.max_rotation_rmse_deg,
                "operator_accepted": bool(args.operator_accepted),
            },
        }
        _write_json(session_dir / "handeye_result.json", report)
        print(
            f"PASS method={result.selected.method} samples={result.sample_count} "
            f"translation_rmse={result.selected.translation_rmse_m * 1000:.3f}mm "
            f"rotation_rmse={result.selected.rotation_rmse_deg:.3f}deg"
        )
        print(f"result={session_dir / 'handeye_result.json'}")
        return 0
    except (HandEyeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        failure = {
            "passed": False,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }
        try:
            _write_json(session_dir / "solve_failure.json", failure)
        except OSError:
            pass
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


def status_session(args: argparse.Namespace) -> int:
    try:
        directory = Path(args.session).resolve()
        session = _load_session(directory)
        samples = [_sample_from_record(directory, record) for record in session["samples"]]
        print(f"samples={len(samples)} required=15")
        if samples:
            from .handeye import motion_diversity

            translation, rotation, second = motion_diversity(samples)
            print(
                f"translation_span={translation * 1000:.1f}mm "
                f"rotation_span={rotation:.1f}deg "
                f"second_rotation_span={second:.1f}deg"
            )
        return 0
    except (HandEyeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="只读采集当前静止姿态的一帧")
    capture.add_argument("--session", required=True)
    capture.add_argument("--config", default="config/site_t0_rgbd_20260908.json")
    capture.add_argument("--serial", default="234222302865")
    capture.add_argument("--columns", type=int, default=9)
    capture.add_argument("--rows", type=int, default=6)
    capture.add_argument("--square-size-mm", type=float, default=20.0)
    capture.set_defaults(function=capture_once)
    solve = subparsers.add_parser("solve", help="离线求解并验证手眼外参")
    solve.add_argument("--session", required=True)
    solve.add_argument("--max-translation-rmse-mm", type=float, default=3.0)
    solve.add_argument("--max-rotation-rmse-deg", type=float, default=0.5)
    solve.add_argument("--operator-accepted", action="store_true")
    solve.set_defaults(function=solve_session)
    status = subparsers.add_parser("status", help="查看样本数与运动覆盖")
    status.add_argument("--session", required=True)
    status.set_defaults(function=status_session)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
