"""Offline command line tool for T0/P printed-target diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .placement_vision import (
    PixelROI,
    PlacementDetection,
    annotate_placement,
    detect_sequence_targets,
    detect_t0,
)


def _read_image(path: Path) -> np.ndarray:
    try:
        payload = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"无法读取图像 {path}: {exc}") from exc
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"图像解码失败: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"图像编码失败: {path}")
    path.write_bytes(encoded.tobytes())


def _result_dict(result: PlacementDetection) -> dict:
    payload = asdict(result)
    payload["kind"] = result.kind.value
    return payload


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线识别比赛底图 T0/P 放置区")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mode", choices=("t0", "sequence"), required=True)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        image = _read_image(args.image)
        roi = None if args.roi is None else PixelROI(*args.roi)
        results = (detect_t0(image, roi),) if args.mode == "t0" else detect_sequence_targets(image, roi)
        annotated = annotate_placement(image, results)
        args.out.mkdir(parents=True, exist_ok=True)
        _write_image(args.out / "annotated.png", annotated)
        valid = all(result.valid for result in results)
        report = {
            "valid": valid,
            "mode": args.mode,
            "source_image": str(args.image.resolve()),
            "roi": args.roi,
            "results": [_result_dict(result) for result in results],
            "safety": {
                "robot_connected": False,
                "gripper_connected": False,
                "camera_opened": False,
                "note": "仅处理已有图片，不具备硬件控制能力",
            },
        }
        (args.out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if valid else 2
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
