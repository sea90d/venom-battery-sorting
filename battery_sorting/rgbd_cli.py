"""Read-only RealSense colour/depth diagnostic for the battery task."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import BatterySortingConfig, ConfigError
from .realsense import DEFAULT_USB3_PROFILE, RealSenseCamera, RealSenseError, StreamProfile
from .rgbd import (
    RGBDError,
    RGBDFrameResult,
    analyse_rgbd_frame,
    annotate_rgbd,
    check_centroid_stability,
    choose_rgbd_consensus,
    create_homography_if_calibrated,
    detection_signature,
    fuse_depth_frames,
)


def _save_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"图像编码失败: {path}")
    written = path.write_bytes(encoded.tobytes())
    if written != encoded.nbytes:
        raise OSError(f"图像未完整写入: {path}")


def _result_dict(result: RGBDFrameResult, consistent_frames: int, total_frames: int) -> dict:
    slots = []
    for slot in result.slots:
        item = asdict(slot)
        for key in ("centroid_px", "camera_xyz_m", "robot_xy_m"):
            if item[key] is not None:
                item[key] = list(item[key])
        slots.append(item)
    detections = []
    for slot in result.detection.slots:
        detections.append(
            {
                "slot_name": slot.slot_name,
                "color": slot.color,
                "coverage": slot.coverage,
                "confidence": slot.confidence,
                "margin": slot.margin,
                "centroid_px": None if slot.centroid is None else list(slot.centroid),
                "contour_area_px": slot.contour_area_px,
                "long_axis_angle_deg": slot.long_axis_angle_deg,
                "valid": slot.valid,
                "reason": slot.reason,
            }
        )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "valid": result.valid,
        "error": result.error,
        "consistent_frames": consistent_frames,
        "frames_collected": total_frames,
        "paper_plane": asdict(result.plane),
        "detections": detections,
        "slots": slots,
        "safety": {
            "robot_connected": False,
            "robot_motion_commands": False,
            "note": "本工具不导入 robot/mission/gripper，不具备运动能力",
        },
    }


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cw, ch, cf, dw, dh, df = DEFAULT_USB3_PROFILE
    parser = argparse.ArgumentParser(description="D455 电池颜色、深度与坐标只读诊断")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--frames", type=_positive_int, default=5)
    parser.add_argument("--required-consistent", type=_positive_int, default=3)
    parser.add_argument("--warmup-frames", type=_positive_int, default=15)
    parser.add_argument("--min-height-mm", type=float, default=15.0)
    parser.add_argument("--max-height-mm", type=float, default=120.0)
    parser.add_argument("--color-width", type=_positive_int, default=cw)
    parser.add_argument("--color-height", type=_positive_int, default=ch)
    parser.add_argument("--color-fps", type=_positive_int, default=cf)
    parser.add_argument("--depth-width", type=_positive_int, default=dw)
    parser.add_argument("--depth-height", type=_positive_int, default=dh)
    parser.add_argument("--depth-fps", type=_positive_int, default=df)
    parser.add_argument(
        "--centroid-stability-px",
        type=float,
        default=10.0,
        help="融合质心相对颜色一致帧中位质心的最大偏差",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.required_consistent > args.frames:
        raise RGBDError("required-consistent 不能大于 frames")
    if not np.isfinite(args.centroid_stability_px) or args.centroid_stability_px <= 0.0:
        raise RGBDError("centroid-stability-px 必须为正的有限数")
    if not (0.0 < args.min_height_mm < args.max_height_mm):
        raise RGBDError("高度阈值必须满足 0 < min < max")
    config = BatterySortingConfig.from_json(args.config)
    config.require_vision_valid()
    profile = StreamProfile(
        args.color_width,
        args.color_height,
        args.color_fps,
        args.depth_width,
        args.depth_height,
        args.depth_fps,
    )
    mapping, mapping_reason = create_homography_if_calibrated(
        config.vision.homography
    )
    camera = RealSenseCamera(profile, warmup_frames=args.warmup_frames)
    colour_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    per_frame_results: list[RGBDFrameResult] = []
    try:
        camera.start(serial=args.serial)
        for _ in range(args.frames):
            colour, depth = camera.capture()
            if depth is None:
                raise RGBDError("对齐深度帧缺失")
            if colour.shape[:2] != depth.shape:
                raise RGBDError("彩色与深度帧未对齐到相同尺寸")
            result = analyse_rgbd_frame(
                colour,
                depth,
                config.vision,
                camera=camera,
                mapping=mapping,
                mapping_unavailable_reason=mapping_reason,
                min_height_m=args.min_height_mm / 1000.0,
                max_height_m=args.max_height_mm / 1000.0,
            )
            colour_frames.append(colour.copy())
            depth_frames.append(depth.copy())
            per_frame_results.append(result)

        fused_depth = fuse_depth_frames(depth_frames)
        # Use a real frame at the temporal centre of the capture window.  A
        # synthetic per-pixel RGB median could create edges that never existed.
        representative_colour = colour_frames[len(colour_frames) // 2]
        fused_result = analyse_rgbd_frame(
            representative_colour,
            fused_depth,
            config.vision,
            camera=camera,
            mapping=mapping,
            mapping_unavailable_reason=mapping_reason,
            min_height_m=args.min_height_mm / 1000.0,
            max_height_m=args.max_height_mm / 1000.0,
        )
    finally:
        camera.stop()

    consensus, count = choose_rgbd_consensus(
        per_frame_results, args.required_consistent
    )
    winner_signature = detection_signature(consensus)
    consensus_frames = [
        result
        for result in per_frame_results
        if result.valid and detection_signature(result) == winner_signature
    ]
    stability_ok, stability_reason = check_centroid_stability(
        fused_result,
        consensus_frames,
        min_samples=args.required_consistent,
        max_deviation_px=args.centroid_stability_px,
    )
    signature_ok = (
        fused_result.valid
        and consensus.valid
        and detection_signature(fused_result) == winner_signature
    )
    errors: list[str] = []
    if not fused_result.valid:
        errors.append(fused_result.error or "融合 RGB-D 结果无效")
    if not consensus.valid:
        errors.append(consensus.error or "逐帧颜色一致性不足")
    if fused_result.valid and consensus.valid and not signature_ok:
        errors.append("融合帧颜色映射与逐帧共识不一致")
    if not stability_ok:
        errors.append(f"质心稳定性失败: {stability_reason}")
    final_valid = fused_result.valid and consensus.valid and signature_ok and stability_ok
    final_slots = fused_result.slots
    if not final_valid:
        # An invalid diagnostic must not expose apparently executable XY even
        # if a homography happens to be configured.
        final_slots = tuple(
            replace(
                slot,
                robot_xy_m=None,
                robot_xy_reason="RGB-D 多帧验证未通过",
            )
            for slot in fused_result.slots
        )
    chosen = RGBDFrameResult(
        fused_result.detection,
        fused_result.plane,
        final_slots,
        fused_result.foreground_mask,
        final_valid,
        "; ".join(errors),
    )
    colour = representative_colour
    depth = fused_depth

    args.out.mkdir(parents=True, exist_ok=True)
    _save_image(args.out / "colour.png", colour)
    _save_image(args.out / "foreground_mask.png", chosen.foreground_mask)
    _save_image(args.out / "annotated.png", annotate_rgbd(colour, chosen))
    np.save(args.out / "depth_m.npy", depth, allow_pickle=False)
    report = _result_dict(chosen, count, len(depth_frames))
    report["homography_calibrated"] = mapping is not None
    report["homography_status"] = "已标定" if mapping is not None else mapping_reason
    report["depth_fusion"] = {"method": "pixelwise_median", "frames": len(depth_frames)}
    report["centroid_stability"] = {
        "valid": stability_ok,
        "max_deviation_px": args.centroid_stability_px,
        "reason": stability_reason,
    }
    (args.out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"RGB-D 诊断: {'PASS' if chosen.valid else 'FAIL'}; "
        f"一致帧 {count}/{len(depth_frames)}"
    )
    print(
        f"纸面拟合: inliers={chosen.plane.inlier_count}/{chosen.plane.sample_count}, "
        f"RMSE={chosen.plane.rmse_m * 1000:.2f} mm"
        if chosen.plane.valid
        else f"纸面拟合失败: {chosen.plane.reason}"
    )
    for slot in chosen.slots:
        xyz = "null" if slot.camera_xyz_m is None else "(" + ", ".join(
            f"{value:.4f}" for value in slot.camera_xyz_m
        ) + ")"
        robot_xy = (
            "null"
            if slot.robot_xy_m is None
            else f"({slot.robot_xy_m[0]:.6f}, {slot.robot_xy_m[1]:.6f})"
        )
        print(
            f"  {slot.slot_name}: {slot.color or '-':6s} "
            f"pixel={slot.centroid_px} height="
            f"{None if slot.height_above_paper_m is None else round(slot.height_above_paper_m * 1000, 1)}mm "
            f"camera_xyz={xyz} robot_xy={robot_xy}"
            + (f" [{slot.robot_xy_reason}]" if slot.robot_xy_m is None else "")
        )
    if not chosen.valid:
        print(f"拒绝原因: {chosen.error}")
    else:
        print(f"质心稳定性: PASS (阈值 {args.centroid_stability_px:.1f}px)")
    print(f"输出目录: {args.out}")
    return 0 if chosen.valid else 2


def main() -> None:
    try:
        raise SystemExit(run())
    except (ConfigError, RGBDError, RealSenseError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
