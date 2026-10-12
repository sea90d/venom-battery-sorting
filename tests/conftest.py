from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from battery_sorting.config import BatterySortingConfig


ROOT = Path(__file__).resolve().parents[1]


def valid_config_dict() -> dict:
    data = json.loads((ROOT / "config" / "example.json").read_text(encoding="utf-8"))
    orientation = [3.1415926536, 0.0, 0.0]
    data["robot"].update(
        {
            "safe_z_approach": 0.20,
            "safe_z_transit": 0.30,
            "workspace_bounds": {
                "x_min": -0.10,
                "x_max": 0.50,
                "y_min": -0.40,
                "y_max": 0.40,
                "z_min": 0.05,
                "z_max": 0.35
            },
            "pick_xy_bounds": {
                "x_min": 0.05,
                "x_max": 0.25,
                "y_min": 0.05,
                "y_max": 0.25
            },
            "pick_pose_template": {
                "z": 0.10,
                "rx": orientation[0],
                "ry": orientation[1],
                "rz": orientation[2]
            },
            "taught_poses": {
                "home": [0.00, -0.30, 0.30, *orientation],
                "T0": [0.40, 0.00, 0.10, *orientation],
                "P1": [0.40, 0.10, 0.10, *orientation],
                "P2": [0.40, 0.20, 0.10, *orientation],
                "P3": [0.40, 0.30, 0.10, *orientation]
            },
        }
    )
    data["vision"]["homography"] = {
        "pixel_points": [
            [100.0, 100.0],
            [500.0, 100.0],
            [100.0, 380.0],
            [500.0, 380.0]
        ],
        "robot_xy_points_m": [
            [0.05, 0.05],
            [0.25, 0.05],
            [0.05, 0.25],
            [0.25, 0.25]
        ],
        "calibrated": True,
        "max_reprojection_error_m": 0.002
    }
    data["gripper"]["driver"] = "mock"
    return data


def parse_config(data: dict, path: Path) -> BatterySortingConfig:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return BatterySortingConfig.from_json(path)


@pytest.fixture
def valid_config(tmp_path: Path) -> BatterySortingConfig:
    return parse_config(valid_config_dict(), tmp_path / "valid.json")


COLOR_HUES = {"red": 0, "blue": 110, "yellow": 28, "green": 60}
SLOT_COLORS = {"slot1": "red", "slot2": "blue", "slot3": "yellow", "slot4": "green"}


def bgr_from_hsv(hue: int, saturation: int = 220, value: int = 220) -> tuple[int, int, int]:
    hsv = np.uint8([[[hue, saturation, value]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(component) for component in bgr)


def synthetic_frame(
    config: BatterySortingConfig,
    slot_colors: dict[str, str] | None = None,
    hue_overrides: dict[str, int] | None = None,
) -> np.ndarray:
    colors = SLOT_COLORS if slot_colors is None else slot_colors
    hue_overrides = {} if hue_overrides is None else hue_overrides
    frame = np.zeros((config.camera.height, config.camera.width, 3), dtype=np.uint8)
    for roi in config.vision.rois:
        color = colors[roi.slot_name]
        hue = hue_overrides.get(roi.slot_name, COLOR_HUES[color])
        frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = bgr_from_hsv(hue)
    return frame


@pytest.fixture
def frame(valid_config: BatterySortingConfig) -> np.ndarray:
    return synthetic_frame(valid_config)


@pytest.fixture
def config_dict() -> dict:
    return copy.deepcopy(valid_config_dict())
