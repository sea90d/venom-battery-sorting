"""Tests for battery_sorting.geometry: back-projection, rigid transforms, validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from battery_sorting.geometry import (
    CameraIntrinsics,
    GeometryError,
    solve_rigid_transform,
    back_project,
    transform_point,
    validate_rigid_transform,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rigid(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rotation
    m[:3, 3] = translation
    return m


_INTRINSICS = CameraIntrinsics(fx=600.0, fy=600.0, cx=640.0, cy=360.0)


# ── back_project ─────────────────────────────────────────────────────────────

def test_back_project_optical_centre() -> None:
    x, y, z = back_project(640.0, 360.0, 1.0, _INTRINSICS)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(1.0)


def test_back_project_off_centre() -> None:
    x, y, z = back_project(640.0 + 600.0, 360.0, 2.0, _INTRINSICS)
    assert x == pytest.approx(2.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(2.0)


def test_back_project_rejects_zero_depth() -> None:
    with pytest.raises(GeometryError, match="必须为正"):
        back_project(0.0, 0.0, 0.0, _INTRINSICS)


def test_back_project_rejects_negative_depth() -> None:
    with pytest.raises(GeometryError, match="必须为正"):
        back_project(0.0, 0.0, -0.5, _INTRINSICS)


def test_back_project_rejects_nan_input() -> None:
    with pytest.raises(GeometryError, match="有限数值"):
        back_project(float("nan"), 0.0, 1.0, _INTRINSICS)


def test_back_project_rejects_inf_input() -> None:
    with pytest.raises(GeometryError, match="有限数值"):
        back_project(0.0, float("inf"), 1.0, _INTRINSICS)


def test_back_project_rejects_zero_fx() -> None:
    bad = CameraIntrinsics(fx=0.0, fy=1.0, cx=0.0, cy=0.0)
    with pytest.raises(GeometryError, match="fx/fy 为零"):
        back_project(0.0, 0.0, 1.0, bad)


# ── intrinsics from_matrix ───────────────────────────────────────────────────

def test_intrinsics_from_matrix() -> None:
    m = np.array([[600.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    intr = CameraIntrinsics.from_matrix(m)
    assert intr.fx == 600.0
    assert intr.fy == 400.0
    assert intr.cx == 320.0
    assert intr.cy == 240.0


def test_intrinsics_from_matrix_rejects_skew() -> None:
    m = np.array([[600.0, 0.1, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    with pytest.raises(GeometryError, match="偏斜"):
        CameraIntrinsics.from_matrix(m)


def test_intrinsics_from_matrix_rejects_wrong_shape() -> None:
    with pytest.raises(GeometryError, match="3x3"):
        CameraIntrinsics.from_matrix(np.eye(4))


# ── validate_rigid_transform ─────────────────────────────────────────────────

def test_validate_identity() -> None:
    validate_rigid_transform(_identity())


def test_validate_rotated_translated() -> None:
    m = _rigid(_rotation_x(0.5), (1.0, -2.0, 3.0))
    validate_rigid_transform(m)


def test_validate_rejects_none() -> None:
    with pytest.raises(GeometryError, match="None"):
        validate_rigid_transform(None)


def test_validate_rejects_wrong_shape() -> None:
    with pytest.raises(GeometryError, match="4x4"):
        validate_rigid_transform(np.eye(3))


def test_validate_rejects_nan() -> None:
    m = _identity()
    m[0, 0] = float("nan")
    with pytest.raises(GeometryError, match="非有限"):
        validate_rigid_transform(m)


def test_validate_rejects_bad_bottom_row() -> None:
    m = _identity()
    m[3, 3] = 0.0
    with pytest.raises(GeometryError, match="末行"):
        validate_rigid_transform(m)


def test_validate_rejects_non_orthogonal_rotation() -> None:
    m = _identity()
    m[0, 0] = 0.5
    with pytest.raises(GeometryError, match="非正交"):
        validate_rigid_transform(m)


def test_validate_rejects_reflection() -> None:
    m = _identity()
    m[0, 0] = -1.0
    with pytest.raises(GeometryError, match="det"):
        validate_rigid_transform(m)


def test_validate_rejects_large_translation() -> None:
    m = _rigid(np.eye(3), (20.0, 0.0, 0.0))
    with pytest.raises(GeometryError, match="平移"):
        validate_rigid_transform(m)


# ── transform_point ──────────────────────────────────────────────────────────

def test_transform_identity() -> None:
    x, y, z = transform_point((1.0, 2.0, 3.0), _identity())
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(2.0)
    assert z == pytest.approx(3.0)


def test_transform_translation() -> None:
    m = _rigid(np.eye(3), (0.5, -0.3, 0.1))
    x, y, z = transform_point((1.0, 2.0, 3.0), m)
    assert x == pytest.approx(1.5)
    assert y == pytest.approx(1.7)
    assert z == pytest.approx(3.1)


def test_transform_rotation() -> None:
    m = _rigid(_rotation_x(math.pi), (0.0, 0.0, 0.0))
    x, y, z = transform_point((0.0, 1.0, 0.0), m)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(-1.0)
    assert z == pytest.approx(0.0, abs=1e-9)


def test_transform_rejects_bad_extrinsics() -> None:
    with pytest.raises(GeometryError, match="None"):
        transform_point((0.0, 0.0, 0.0), None)


def test_transform_rejects_nan_point() -> None:
    with pytest.raises(GeometryError, match="非有限"):
        transform_point((float("nan"), 0.0, 0.0), _identity())


def test_transform_rejects_inf_point() -> None:
    with pytest.raises(GeometryError, match="非有限"):
        transform_point((0.0, float("inf"), 0.0), _identity())


def test_solve_rigid_transform_recovers_known_pose() -> None:
    rotation = _rotation_x(0.35)
    translation = np.array([0.12, -0.08, 0.44], dtype=np.float64)
    camera = np.array(
        [[0.0, 0.0, 0.0], [0.20, 0.0, 0.0], [0.0, 0.20, 0.0],
         [0.20, 0.20, 0.0], [0.10, 0.08, 0.0]], dtype=np.float64
    )
    base = camera @ rotation.T + translation
    result = solve_rigid_transform(camera, base)
    assert result.rmse_m == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(result.rotation, rotation, atol=1e-12)
    assert np.allclose(result.translation, translation, atol=1e-12)


def test_solve_rigid_transform_rejects_collinear_points() -> None:
    camera = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]],
        dtype=np.float64,
    )
    with pytest.raises(GeometryError, match="近共线"):
        solve_rigid_transform(camera, camera)


def test_solve_rigid_transform_rejects_bad_correspondence_rmse() -> None:
    camera = np.array(
        [[0.0, 0.0, 0.0], [0.20, 0.0, 0.0], [0.0, 0.20, 0.0],
         [0.20, 0.20, 0.0]], dtype=np.float64
    )
    base = camera.copy()
    base[-1, 0] += 0.02
    with pytest.raises(GeometryError, match="RMSE"):
        solve_rigid_transform(camera, base, max_rmse_m=0.001)
