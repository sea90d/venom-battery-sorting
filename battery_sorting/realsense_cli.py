"""RealSense D455 diagnostic CLI.

Usage (from project root):
  python -m battery_sorting.realsense_cli list
  python -m battery_sorting.realsense_cli probe --frames 5 --out D:/diag
  python -m battery_sorting.realsense_cli probe --color-width 424 --color-height 240 --color-fps 5 --depth-width 480 --depth-height 270 --depth-fps 5 --frames 5 --out D:/diag
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .realsense import (
    DEFAULT_USB3_PROFILE,
    RS_DEPTH_RANGE_M,
    USB2_LOW_PROFILE,
    RealSenseCamera,
    RealSenseDeviceInfo,
    RealSenseError,
    StreamProfile,
)


def _save_image(path: Path, bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise OSError(f"cv2.imencode 失败: {path}")
    written = path.write_bytes(buf.tobytes())
    if written != buf.nbytes:
        raise OSError(f"图像未完整写入: {path} ({written}/{buf.nbytes} bytes)")


def _save_json(path: Path, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    written = path.write_text(text, encoding="utf-8")
    if written != len(text):
        raise OSError(f"JSON 未完整写入: {path} ({written}/{len(text)} chars)")


def _is_usb2(usb_type: str) -> bool:
    return usb_type.strip().startswith("2")


def _select_device(serial: str | None) -> RealSenseDeviceInfo:
    devices = RealSenseCamera.list_devices()
    if serial is not None:
        matches = [device for device in devices if device.serial == serial]
        if not matches:
            raise RealSenseError(f"未找到序列号为 {serial} 的 RealSense 设备")
        return matches[0]
    if not devices:
        raise RealSenseError("未检测到任何 RealSense 设备")
    if len(devices) > 1:
        raise RealSenseError("检测到多个 RealSense 设备，请用 --serial 明确选择")
    return devices[0]


def _depth_colourmap(depth_m: np.ndarray) -> np.ndarray:
    """Colour valid depth in JET; invalid pixels stay black."""
    valid_mask = (depth_m > 0.0) & np.isfinite(depth_m)
    valid_mask &= (depth_m >= RS_DEPTH_RANGE_M[0]) & (depth_m <= RS_DEPTH_RANGE_M[1])
    clipped = np.where(valid_mask, depth_m, 0.0)
    if not np.any(valid_mask):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    d_min = float(np.percentile(clipped[valid_mask], 5))
    d_max = float(np.percentile(clipped[valid_mask], 95))
    d_range = max(d_max - d_min, 1e-6)
    normalized = np.clip((clipped - d_min) / d_range, 0.0, 1.0)
    normalized = np.where(valid_mask, normalized, 0.0)
    grey = (normalized * 255).astype(np.uint8)
    colour = cv2.applyColorMap(grey, cv2.COLORMAP_JET)
    colour[~valid_mask] = 0
    return colour


def cmd_list() -> int:
    try:
        devices = RealSenseCamera.list_devices()
    except RealSenseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if not devices:
        print("未检测到任何 RealSense 设备。")
        return 0
    print(f"发现 {len(devices)} 个 RealSense 设备:")
    for dev in devices:
        print(f"  序列号: {dev.serial}")
        print(f"  名称:   {dev.name}")
        print(f"  USB:    {dev.usb_type}")
        if _is_usb2(dev.usb_type):
            print("  ** WARNING: USB 2.x 带宽不足，必须使用低分辨率 profile **")
        print()
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    cw = args.color_width
    ch = args.color_height
    cf = args.color_fps
    dw = args.depth_width
    dh = args.depth_height
    df = args.depth_fps
    n_frames = args.frames

    if n_frames < 1:
        print("ERROR: --frames 必须 >= 1", file=sys.stderr)
        return 1

    try:
        selected_device = _select_device(args.serial)
        profile = StreamProfile(
            color_width=cw, color_height=ch, color_fps=cf,
            depth_width=dw, depth_height=dh, depth_fps=df,
        )
    except (RealSenseError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    usb2_warning = _is_usb2(selected_device.usb_type)
    print(
        f"设备: {selected_device.name} serial={selected_device.serial} "
        f"USB={selected_device.usb_type or 'unknown'}"
    )
    if usb2_warning:
        print("WARNING: 当前为 USB 2.x；仅建议低带宽诊断，比赛使用前请改接 USB 3.x。")

    cam = RealSenseCamera(profile=profile, warmup_frames=args.warmup_frames)
    try:
        cam.start(serial=selected_device.serial or args.serial)
    except RealSenseError as exc:
        print(f"ERROR: 启动失败: {exc}", file=sys.stderr)
        return 1

    try:
        colour, depth = cam.capture()
        if depth is None:
            print("ERROR: 首帧无对齐深度图", file=sys.stderr)
            return 1

        ch_out, cw_out = colour.shape[:2]
        dh_out, dw_out = depth.shape[:2]
        print(f"对齐后彩色: {cw_out}x{ch_out}")
        print(f"对齐后深度: {dw_out}x{dh_out}")

        intrinsics = cam.color_intrinsics
        print(
            f"内参: fx={intrinsics.fx:.3f} fy={intrinsics.fy:.3f} "
            f"cx={intrinsics.cx:.3f} cy={intrinsics.cy:.3f}"
        )
        print(f"外参标定: {'已标定' if cam.is_extrinsics_calibrated else '未标定（必须现场标定）'}")

        last_colour = colour
        last_depth = depth

        frame_valid_ratios: list[float] = []
        centre_medians: list[float | None] = []
        centre_mads: list[float | None] = []
        centre_counts: list[int] = []
        cx = cw_out // 2
        cy = ch_out // 2

        for i in range(n_frames):
            if i > 0:
                colour, depth = cam.capture()
                if depth is None:
                    print(f"ERROR: 第 {i+1} 帧无对齐深度图", file=sys.stderr)
                    return 1
                last_colour = colour
                last_depth = depth

            total_px = depth.shape[0] * depth.shape[1]
            depth_valid = depth[(depth > 0.0) & np.isfinite(depth)]
            depth_in_range = depth_valid[
                (depth_valid >= RS_DEPTH_RANGE_M[0])
                & (depth_valid <= RS_DEPTH_RANGE_M[1])
            ]
            ratio = len(depth_in_range) / total_px if total_px > 0 else 0.0
            frame_valid_ratios.append(ratio)

            med, mad, cnt = RealSenseCamera._sample_neighbourhood(
                depth, cx, cy, radius=5,
            )
            if cnt > 0:
                centre_medians.append(med)
                centre_mads.append(mad)
            else:
                centre_medians.append(None)
                centre_mads.append(None)
            centre_counts.append(cnt)

        print(f"\n采集 {len(frame_valid_ratios)} 帧:")
        for i, r in enumerate(frame_valid_ratios):
            cm = centre_medians[i]
            cm_str = f"{cm:.4f}m" if cm is not None else "null"
            cma = centre_mads[i]
            cma_str = f"{cma:.4f}m" if cma is not None else "null"
            print(
                f"  帧{i+1}: 有效深度 {r:.2%}  "
                f"中心 median={cm_str} MAD={cma_str} "
                f"valid={centre_counts[i]}"
            )

        ratios = np.array(frame_valid_ratios)
        print(f"\n有效深度比例: min={ratios.min():.2%}  median={float(np.median(ratios)):.2%}  max={ratios.max():.2%}")

        valid_medians = [m for m in centre_medians if m is not None]
        if valid_medians:
            med_arr = np.array(valid_medians, dtype=np.float64)
            agg_median = float(np.median(med_arr))
            agg_mad = float(np.median(np.abs(med_arr - agg_median)))
            print(f"中心邻域深度: median={agg_median:.4f}m  MAD={agg_mad:.4f}m")
        else:
            agg_median = None
            agg_mad = None
            print("中心邻域深度: 无有效深度（所有帧中心邻域均无有效值）")

        if args.out is not None:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)

            _save_image(out_dir / "colour.png", last_colour)
            depth_colour = _depth_colourmap(last_depth)
            _save_image(out_dir / "depth_colour.png", depth_colour)

            report = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "device": {
                    "serial": selected_device.serial,
                    "name": selected_device.name,
                    "usb_type": selected_device.usb_type,
                    "usb2_warning": usb2_warning,
                },
                "profile": {
                    "color": f"{cw}x{ch}@{cf}fps",
                    "depth": f"{dw}x{dh}@{df}fps",
                },
                "actual_output": {
                    "color": f"{cw_out}x{ch_out}",
                    "depth": f"{dw_out}x{dh_out}",
                },
                "intrinsics": {
                    "fx": intrinsics.fx,
                    "fy": intrinsics.fy,
                    "cx": intrinsics.cx,
                    "cy": intrinsics.cy,
                },
                "extrinsics_calibrated": cam.is_extrinsics_calibrated,
                "frames_collected": len(frame_valid_ratios),
                "valid_depth_ratio": {
                    "min": float(ratios.min()),
                    "median": float(np.median(ratios)),
                    "max": float(ratios.max()),
                    "per_frame": frame_valid_ratios,
                },
                "centre_neighbourhood": {
                    "radius_px": 5,
                    "aggregate_median_m": agg_median,
                    "aggregate_mad_m": agg_mad,
                    "per_frame_median_m": centre_medians,
                    "per_frame_mad_m": centre_mads,
                    "per_frame_valid_count": centre_counts,
                },
            }
            _save_json(out_dir / "report.json", report)
            print(f"\n诊断结果已保存到: {out_dir}")
    except (RealSenseError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            cam.stop()
        except RealSenseError:
            pass

    return 0


def _validate_positive_int(value: str, name: str) -> int:
    try:
        v = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{name} 必须是整数")
    if v <= 0:
        raise argparse.ArgumentTypeError(f"{name} 必须是正整数")
    return v


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RealSense D455 诊断工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="列出所有已连接的 RealSense 设备")

    cw, ch, cf, dw, dh, df = DEFAULT_USB3_PROFILE
    probe = sub.add_parser("probe", help="采集帧并输出诊断报告")
    probe.add_argument("--serial", type=str, default=None, help="目标设备序列号")
    probe.add_argument(
        "--frames", type=lambda s: _validate_positive_int(s, "--frames"),
        default=5, help="采集帧数（默认 5）",
    )
    probe.add_argument(
        "--warmup-frames",
        type=lambda s: _validate_positive_int(s, "--warmup-frames"),
        default=10,
        help="正式统计前丢弃的预热帧数（默认 10）",
    )
    probe.add_argument("--out", type=str, default=None, help="输出目录")
    probe.add_argument(
        "--color-width", type=lambda s: _validate_positive_int(s, "--color-width"),
        default=cw, help=f"彩色宽度（默认 {cw}）",
    )
    probe.add_argument(
        "--color-height", type=lambda s: _validate_positive_int(s, "--color-height"),
        default=ch, help=f"彩色高度（默认 {ch}）",
    )
    probe.add_argument(
        "--color-fps", type=lambda s: _validate_positive_int(s, "--color-fps"),
        default=cf, help=f"彩色帧率（默认 {cf}）",
    )
    probe.add_argument(
        "--depth-width", type=lambda s: _validate_positive_int(s, "--depth-width"),
        default=dw, help=f"深度宽度（默认 {dw}）",
    )
    probe.add_argument(
        "--depth-height", type=lambda s: _validate_positive_int(s, "--depth-height"),
        default=dh, help=f"深度高度（默认 {dh}）",
    )
    probe.add_argument(
        "--depth-fps", type=lambda s: _validate_positive_int(s, "--depth-fps"),
        default=df, help=f"深度帧率（默认 {df}）",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(1)
    if args.command == "list":
        raise SystemExit(cmd_list())
    elif args.command == "probe":
        raise SystemExit(cmd_probe(args))
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
