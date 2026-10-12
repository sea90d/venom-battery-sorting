"""3D geometry utilities: intrinsics back-projection and rigid-body transforms.

This module does NOT depend on pyrealsense2 or any hardware library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class GeometryError(ValueError):
    """Invalid input, degenerate matrix or uncalibrated extrinsics."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole colour-camera intrinsics (fx, fy, cx, cy)."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "CameraIntrinsics":
        if matrix.shape != (3, 3):
            raise GeometryError("内参矩阵必须是 3x3")
        if not np.all(np.isfinite(matrix)):
            raise GeometryError("内参矩阵包含非有限值")
        if matrix[0, 1] != 0.0 or matrix[1, 0] != 0.0:
            raise GeometryError("内参矩阵仅支持无偏斜的针孔模型")
        return cls(
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
        )

    @property
    def matrix(self) -> np.ndarray:
        m = np.eye(3, dtype=np.float64)
        m[0, 0] = self.fx
        m[1, 1] = self.fy
        m[0, 2] = self.cx
        m[1, 2] = self.cy
        return m


@dataclass(frozen=True)
class RigidTransformResult:
    """Rigid transform fit from camera coordinates to robot-base coordinates."""

    matrix: np.ndarray
    rmse_m: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        validate_rigid_transform(self.matrix, "solve_rigid_transform result")
        if not math.isfinite(self.rmse_m) or self.rmse_m < 0.0:
            raise GeometryError("刚体变换 RMSE 必须是非负有限值")
        if self.rotation.shape != (3, 3) or not np.all(np.isfinite(self.rotation)):
            raise GeometryError("刚体变换 rotation 必须是有限 3x3 矩阵")
        if self.translation.shape != (3,) or not np.all(np.isfinite(self.translation)):
            raise GeometryError("刚体变换 translation 必须是有限 3 维向量")


def back_project(u: float, v: float, depth_m: float, intrinsics: CameraIntrinsics) -> tuple[float, float, float]:
    """Use an undistorted pinhole model to back-project a pixel.

    Real D455 frames must use pyrealsense2.rs2_deproject_pixel_to_point so
    that the factory distortion model is respected.
    """
    if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(depth_m)):
        raise GeometryError("back_project 输入坐标必须为有限数值")
    if depth_m <= 0.0:
        raise GeometryError(f"深度值 {depth_m:.4f}m 必须为正")
    if abs(intrinsics.fx) < 1e-9 or abs(intrinsics.fy) < 1e-9:
        raise GeometryError("内参 fx/fy 为零或过小，无法反投影")
    x = (u - intrinsics.cx) * depth_m / intrinsics.fx
    y = (v - intrinsics.cy) * depth_m / intrinsics.fy
    return (float(x), float(y), float(depth_m))


def validate_rigid_transform(matrix: np.ndarray, label: str = "camera_to_base") -> None:
    """Strict 4x4 rigid-body validation, fail-closed on any violation."""
    if matrix is None:
        raise GeometryError(f"{label} 外参矩阵为 None")

    if matrix.shape != (4, 4):
        raise GeometryError(f"{label} 外参矩阵必须是 4x4，实际 {matrix.shape}")

    if not np.all(np.isfinite(matrix)):
        raise GeometryError(f"{label} 外参矩阵包含非有限值")

    bottom_row = matrix[3, :]
    expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if not np.allclose(bottom_row, expected, atol=1e-12):
        raise GeometryError(
            f"{label} 外参矩阵末行必须是 [0,0,0,1]，实际 {bottom_row.tolist()}"
        )

    rotation = matrix[:3, :3]
    r_t_r = rotation.T @ rotation
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(r_t_r, identity, atol=1e-9):
        raise GeometryError(f"{label} 旋转部分 R^T·R 不接近单位矩阵，非正交")

    det = float(np.linalg.det(rotation))
    if not math.isfinite(det):
        raise GeometryError(f"{label} 旋转矩阵行列式为非有限值")
    if abs(det - 1.0) > 1e-6:
        raise GeometryError(f"{label} 旋转矩阵 det={det:.6f}，不接近 +1（可能含反射）")

    translation = matrix[:3, 3]
    if any(abs(v) > 10.0 for v in translation):
        raise GeometryError(
            f"{label} 平移向量 ({translation[0]:.3f},{translation[1]:.3f},{translation[2]:.3f}) "
            "超出 ±10 m 保守边界"
        )


def transform_point(point_cam: tuple[float, float, float], camera_to_base: np.ndarray) -> tuple[float, float, float]:
    """Apply 4x4 rigid transform to a 3D point, returning (x,y,z) in base frame."""
    validate_rigid_transform(camera_to_base, "camera_to_base")
    if not all(math.isfinite(v) for v in point_cam):
        raise GeometryError("待变换点坐标包含非有限值")
    p = np.array([*point_cam, 1.0], dtype=np.float64)
    result = camera_to_base @ p
    if not np.all(np.isfinite(result)):
        raise GeometryError("刚体变换结果包含非有限值")
    return (float(result[0]), float(result[1]), float(result[2]))


def solve_rigid_transform(
    camera_points: np.ndarray,
    base_points: np.ndarray,
    max_rmse_m: float = 0.003,
    *,
    duplicate_tol_m: float = 1e-6,
    min_direction_ratio: float = 1e-3,
) -> RigidTransformResult:
    """Fit ``p_base = R @ p_camera + t`` with the Kabsch algorithm.

    Calibration points may be coplanar, as they normally are when a TCP
    marker is moved over a fixed-height grid.  Only coincident or nearly
    collinear point sets are rejected because those do not constrain a stable
    3-D rigid transform.  A reflection from the SVD is corrected using the
    standard Kabsch last-axis flip; bad correspondences are rejected by RMSE.
    """
    if not (math.isfinite(max_rmse_m) and max_rmse_m >= 0.0):
        raise GeometryError("max_rmse_m 必须是非负有限值")
    if not (math.isfinite(duplicate_tol_m) and duplicate_tol_m > 0.0):
        raise GeometryError("duplicate_tol_m 必须是正的有限值")
    if not (math.isfinite(min_direction_ratio) and 0.0 < min_direction_ratio < 1.0):
        raise GeometryError("min_direction_ratio 必须在 (0,1) 内")

    camera = np.asarray(camera_points, dtype=np.float64)
    base = np.asarray(base_points, dtype=np.float64)
    for label, points in (("camera_points", camera), ("base_points", base)):
        if points.ndim != 2 or points.shape[1] != 3:
            raise GeometryError(f"{label} 必须是 (N,3) 数组，实际 {points.shape}")
        if not np.all(np.isfinite(points)):
            raise GeometryError(f"{label} 包含非有限值")
    if camera.shape[0] != base.shape[0]:
        raise GeometryError(
            f"点数不匹配: camera={camera.shape[0]}, base={base.shape[0]}"
        )
    count = camera.shape[0]
    if count < 4:
        raise GeometryError(f"至少需要 4 个对应点，实际 {count}")

    for label, points in (("camera_points", camera), ("base_points", base)):
        differences = points[:, None, :] - points[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        upper = distances[np.triu_indices(count, k=1)]
        duplicate = np.flatnonzero(upper < duplicate_tol_m)
        if duplicate.size:
            raise GeometryError(f"{label} 存在重复或过近标定点")

    camera_centroid = np.mean(camera, axis=0)
    base_centroid = np.mean(base, axis=0)
    camera_centered = camera - camera_centroid
    base_centered = base - base_centroid
    singular_values = np.linalg.svd(camera_centered, compute_uv=False)
    if singular_values[0] <= np.finfo(np.float64).eps:
        raise GeometryError("camera_points 所有点共位，无法求解")
    # A planar grid has s3 ~= 0 and is valid.  The second direction must be
    # non-negligible to avoid an almost-collinear calibration layout.
    if singular_values[1] / singular_values[0] < min_direction_ratio:
        raise GeometryError(
            f"camera_points 近共线: s2/s1={singular_values[1] / singular_values[0]:.2e}"
        )

    covariance = camera_centered.T @ base_centered
    left, _singular, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transpose[-1, :] *= -1.0
        rotation = right_transpose.T @ left.T
    determinant = float(np.linalg.det(rotation))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise GeometryError(f"Kabsch 旋转无效: det={determinant}")

    translation = base_centroid - rotation @ camera_centroid
    predicted = camera @ rotation.T + translation
    residuals = np.linalg.norm(predicted - base, axis=1)
    rmse_m = float(np.sqrt(np.mean(residuals**2)))
    if rmse_m > max_rmse_m:
        raise GeometryError(
            f"刚体变换 RMSE={rmse_m * 1000:.2f} mm 超过阈值 {max_rmse_m * 1000:.2f} mm"
        )

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return RigidTransformResult(
        matrix=matrix,
        rmse_m=rmse_m,
        rotation=rotation.copy(),
        translation=translation.copy(),
    )
