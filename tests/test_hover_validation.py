from __future__ import annotations

import cv2
import numpy as np
import pytest

from battery_sorting.hover_validation import camera_point_to_base, rotated_tcp_pose
from battery_sorting.handeye import pose_vector_to_transform, rotation_distance_deg


def _transform(rvec, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
    result[:3, 3] = translation
    return result


def test_camera_point_to_base_chain_order() -> None:
    base_T_tcp = _transform([0.0, 0.0, np.pi / 2.0], [0.4, -0.2, 0.5])
    tcp_T_camera = _transform([0.0, 0.0, 0.0], [0.0, 0.1, 0.2])
    result = camera_point_to_base((0.1, 0.0, 0.3), base_T_tcp, tcp_T_camera)
    expected = base_T_tcp @ tcp_T_camera @ np.array([0.1, 0.0, 0.3, 1.0])
    assert result == pytest.approx(expected[:3])


def test_rotated_tcp_pose_keeps_position_and_rotates_local_z() -> None:
    initial = (0.1, -0.2, 0.4, 0.0, np.pi, 0.0)
    target = rotated_tcp_pose(initial, 90.0)
    assert target.as_list()[:3] == pytest.approx(initial[:3])
    first = pose_vector_to_transform(initial)
    second = pose_vector_to_transform(target.as_list())
    assert rotation_distance_deg(first, second) == pytest.approx(90.0)
    expected_rotation = first[:3, :3] @ cv2.Rodrigues(
        np.array([0.0, 0.0, np.pi / 2.0])
    )[0]
    assert second[:3, :3] == pytest.approx(expected_rotation)
