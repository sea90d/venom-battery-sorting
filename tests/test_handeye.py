from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from battery_sorting.handeye import (
    CheckerboardSpec,
    HandEyeError,
    HandEyeSample,
    estimate_target_to_camera,
    pose_vector_to_transform,
    solve_eye_in_hand,
)


def _transform(rotation_vector, translation) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))[0]
    result[:3, 3] = translation
    return result


def _synthetic_samples(count: int = 20) -> tuple[list[HandEyeSample], np.ndarray]:
    gripper_T_camera = _transform([0.08, -0.04, 0.03], [0.035, -0.012, 0.095])
    base_T_target = _transform([0.0, 0.0, 0.1], [0.35, -0.20, 0.02])
    samples = []
    for index in range(count):
        phase = 2.0 * math.pi * index / count
        base_T_gripper = _transform(
            [
                0.32 * math.sin(phase),
                0.27 * math.cos(phase),
                0.12 * math.sin(2.0 * phase),
            ],
            [
                0.04 * math.sin(phase),
                -0.22 + 0.04 * math.cos(phase),
                0.42 + 0.025 * math.sin(2.0 * phase),
            ],
        )
        camera_T_target = (
            np.linalg.inv(gripper_T_camera)
            @ np.linalg.inv(base_T_gripper)
            @ base_T_target
        )
        samples.append(HandEyeSample(base_T_gripper, camera_T_target, 0.1))
    return samples, gripper_T_camera


def test_pose_vector_to_transform_axis_angle() -> None:
    transform = pose_vector_to_transform([0.1, -0.2, 0.3, 0.0, 0.0, math.pi / 2.0])
    assert transform[:3, 3] == pytest.approx([0.1, -0.2, 0.3])
    assert transform[:3, :3] @ [1.0, 0.0, 0.0] == pytest.approx([0.0, 1.0, 0.0])


def test_pnp_from_normalized_points() -> None:
    spec = CheckerboardSpec()
    expected = _transform([0.08, -0.12, 0.03], [-0.05, 0.02, 0.60])
    camera = spec.object_points() @ expected[:3, :3].T + expected[:3, 3]
    normalized = camera[:, :2] / camera[:, 2:3]
    result = estimate_target_to_camera(normalized, spec)
    assert result.normalized_rmse < 1e-8
    assert result.camera_T_target == pytest.approx(expected, abs=1e-7)


def test_eye_in_hand_recovers_known_camera_pose() -> None:
    samples, expected = _synthetic_samples()
    result = solve_eye_in_hand(samples)
    assert result.selected.gripper_T_camera == pytest.approx(expected, abs=1e-7)
    assert result.selected.translation_rmse_m < 1e-8
    assert result.selected.rotation_rmse_deg < 1e-6


def test_eye_in_hand_rejects_too_few_samples() -> None:
    samples, _expected = _synthetic_samples(5)
    with pytest.raises(HandEyeError, match="至少需要"):
        solve_eye_in_hand(samples)


def test_eye_in_hand_rejects_one_axis_rotations() -> None:
    gripper_T_camera = _transform([0.1, 0.0, 0.0], [0.02, 0.0, 0.1])
    base_T_target = _transform([0.0, 0.0, 0.0], [0.3, -0.2, 0.0])
    samples = []
    for index in range(15):
        base_T_gripper = _transform(
            [0.0, 0.0, math.radians(-20.0 + index * 3.0)],
            [0.003 * index, -0.2, 0.45],
        )
        camera_T_target = np.linalg.inv(gripper_T_camera) @ np.linalg.inv(base_T_gripper) @ base_T_target
        samples.append(HandEyeSample(base_T_gripper, camera_T_target, 0.1))
    with pytest.raises(HandEyeError, match="第二独立旋转方向"):
        solve_eye_in_hand(samples)
