"""Eye-in-hand calibration math for a D455 rigidly mounted on a UR5 tool.

Coordinate convention (left superscript omitted in code names)::

    base_T_target = base_T_gripper @ gripper_T_camera @ camera_T_target

``cv2.calibrateHandEye`` therefore receives ``base_T_gripper`` and
``camera_T_target`` and returns ``gripper_T_camera``.  This module has no
robot or camera I/O; hardware sampling lives in :mod:`handeye_cli`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .geometry import GeometryError, validate_rigid_transform


class HandEyeError(RuntimeError):
    """Invalid observation, insufficient motion, or failed calibration."""


@dataclass(frozen=True)
class CheckerboardSpec:
    inner_columns: int = 9
    inner_rows: int = 6
    square_size_m: float = 0.020

    def __post_init__(self) -> None:
        if self.inner_columns < 2 or self.inner_rows < 2:
            raise HandEyeError("棋盘格内角点行列数必须 >= 2")
        if not math.isfinite(self.square_size_m) or self.square_size_m <= 0.0:
            raise HandEyeError("棋盘格格长必须是正的有限米值")

    @property
    def corner_count(self) -> int:
        return self.inner_columns * self.inner_rows

    def object_points(self) -> np.ndarray:
        points = np.zeros((self.corner_count, 3), dtype=np.float64)
        grid = np.mgrid[0 : self.inner_columns, 0 : self.inner_rows]
        points[:, :2] = grid.T.reshape(-1, 2) * self.square_size_m
        return points


@dataclass(frozen=True)
class PnPResult:
    camera_T_target: np.ndarray
    normalized_rmse: float

    def __post_init__(self) -> None:
        validate_rigid_transform(self.camera_T_target, "camera_T_target")
        if not math.isfinite(self.normalized_rmse) or self.normalized_rmse < 0.0:
            raise HandEyeError("PnP 归一化重投影 RMSE 无效")


@dataclass(frozen=True)
class HandEyeSample:
    base_T_gripper: np.ndarray
    camera_T_target: np.ndarray
    reprojection_rmse_px: float

    def __post_init__(self) -> None:
        validate_rigid_transform(self.base_T_gripper, "base_T_gripper")
        validate_rigid_transform(self.camera_T_target, "camera_T_target")
        if not math.isfinite(self.reprojection_rmse_px) or self.reprojection_rmse_px < 0.0:
            raise HandEyeError("样本重投影 RMSE 必须是非负有限值")


@dataclass(frozen=True)
class MethodResult:
    method: str
    gripper_T_camera: np.ndarray
    translation_rmse_m: float
    translation_max_m: float
    rotation_rmse_deg: float
    rotation_max_deg: float
    score: float

    def __post_init__(self) -> None:
        validate_rigid_transform(self.gripper_T_camera, "gripper_T_camera")


@dataclass(frozen=True)
class HandEyeResult:
    selected: MethodResult
    candidates: tuple[MethodResult, ...]
    sample_count: int
    translation_span_m: float
    rotation_span_deg: float
    second_rotation_span_deg: float


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def pose_vector_to_transform(pose: Iterable[float]) -> np.ndarray:
    """Convert UR ``[x,y,z,rx,ry,rz]`` axis-angle pose to ``base_T_tcp``."""
    values = np.asarray(tuple(pose), dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise HandEyeError("UR 位姿必须是 6 个有限数")
    rotation, _ = cv2.Rodrigues(values[3:].reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = values[:3]
    validate_rigid_transform(transform, "base_T_gripper")
    return transform


def transform_to_pose_vector(transform: np.ndarray) -> np.ndarray:
    validate_rigid_transform(transform, "rigid transform")
    rotation_vector, _ = cv2.Rodrigues(transform[:3, :3])
    return np.concatenate((transform[:3, 3], rotation_vector.reshape(3)))


def detect_checkerboard(image_bgr: np.ndarray, spec: CheckerboardSpec) -> np.ndarray:
    """Return sub-pixel checkerboard corners as an ``(N,2)`` array."""
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise HandEyeError("棋盘格图像必须是 uint8 BGR")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(
        gray, (spec.inner_columns, spec.inner_rows), flags=flags
    )
    if not found or corners is None or len(corners) != spec.corner_count:
        count = 0 if corners is None else len(corners)
        raise HandEyeError(f"棋盘格检测失败：找到 {count}/{spec.corner_count} 个内角点")
    result = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(result)):
        raise HandEyeError("棋盘格角点包含非有限值")
    return result


def estimate_target_to_camera(
    normalized_image_points: np.ndarray,
    spec: CheckerboardSpec,
) -> PnPResult:
    """Estimate ``camera_T_target`` from undistorted normalized rays.

    The caller must obtain normalized rays using the RealSense SDK factory
    distortion model.  OpenCV is intentionally given identity intrinsics and
    zero distortion here.
    """
    image_points = np.asarray(normalized_image_points, dtype=np.float64)
    if image_points.shape != (spec.corner_count, 2):
        raise HandEyeError(
            f"归一化角点必须是 ({spec.corner_count},2)，实际 {image_points.shape}"
        )
    if not np.all(np.isfinite(image_points)):
        raise HandEyeError("归一化角点包含非有限值")
    object_points = spec.object_points()
    ok, rotation_vector, translation = cv2.solvePnP(
        object_points,
        image_points,
        np.eye(3, dtype=np.float64),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise HandEyeError("solvePnP 求解失败")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    if transform[2, 3] <= 0.0:
        raise HandEyeError("PnP 棋盘格位于相机后方，解无效")
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        np.eye(3, dtype=np.float64),
        None,
    )
    errors = projected.reshape(-1, 2) - image_points
    rmse = float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))
    return PnPResult(transform, rmse)


def checkerboard_points_in_camera(
    camera_T_target: np.ndarray, spec: CheckerboardSpec
) -> np.ndarray:
    validate_rigid_transform(camera_T_target, "camera_T_target")
    object_points = spec.object_points()
    return object_points @ camera_T_target[:3, :3].T + camera_T_target[:3, 3]


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    total = np.sum(np.stack(rotations), axis=0)
    left, _singular, right_transpose = np.linalg.svd(total)
    result = left @ right_transpose
    if np.linalg.det(result) < 0.0:
        left[:, -1] *= -1.0
        result = left @ right_transpose
    return result


def motion_diversity(samples: list[HandEyeSample]) -> tuple[float, float, float]:
    transforms = [sample.base_T_gripper for sample in samples]
    translation_span = 0.0
    rotation_span = 0.0
    for i, first in enumerate(transforms):
        for second in transforms[i + 1 :]:
            translation_span = max(
                translation_span,
                float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
            )
            rotation_span = max(rotation_span, rotation_distance_deg(first, second))
    reference = transforms[0][:3, :3]
    rotation_vectors = []
    for transform in transforms[1:]:
        vector, _ = cv2.Rodrigues(reference.T @ transform[:3, :3])
        rotation_vectors.append(vector.reshape(3))
    second_span_deg = 0.0
    if len(rotation_vectors) >= 2:
        singular = np.linalg.svd(np.stack(rotation_vectors), compute_uv=False)
        if len(singular) >= 2:
            second_span_deg = math.degrees(float(singular[1]))
    return translation_span, rotation_span, second_span_deg


def _validate_candidate(
    name: str, gripper_T_camera: np.ndarray, samples: list[HandEyeSample]
) -> MethodResult:
    validate_rigid_transform(gripper_T_camera, f"{name}.gripper_T_camera")
    base_T_targets = [
        sample.base_T_gripper @ gripper_T_camera @ sample.camera_T_target
        for sample in samples
    ]
    translations = np.stack([transform[:3, 3] for transform in base_T_targets])
    centre = np.median(translations, axis=0)
    translation_errors = np.linalg.norm(translations - centre, axis=1)
    mean_rotation = _mean_rotation([transform[:3, :3] for transform in base_T_targets])
    rotation_errors = np.array(
        [
            rotation_distance_deg(
                np.block([[mean_rotation, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]),
                transform,
            )
            for transform in base_T_targets
        ],
        dtype=np.float64,
    )
    translation_rmse = float(np.sqrt(np.mean(translation_errors**2)))
    rotation_rmse = float(np.sqrt(np.mean(rotation_errors**2)))
    score = translation_rmse / 0.003 + rotation_rmse / 0.5
    return MethodResult(
        method=name,
        gripper_T_camera=gripper_T_camera,
        translation_rmse_m=translation_rmse,
        translation_max_m=float(np.max(translation_errors)),
        rotation_rmse_deg=rotation_rmse,
        rotation_max_deg=float(np.max(rotation_errors)),
        score=score,
    )


def solve_eye_in_hand(
    samples: Iterable[HandEyeSample],
    *,
    min_samples: int = 15,
    max_reprojection_rmse_px: float = 1.0,
    min_translation_span_m: float = 0.030,
    min_rotation_span_deg: float = 15.0,
    min_second_rotation_span_deg: float = 5.0,
    max_translation_rmse_m: float = 0.003,
    max_rotation_rmse_deg: float = 0.5,
) -> HandEyeResult:
    accepted = list(samples)
    if len(accepted) < min_samples:
        raise HandEyeError(f"手眼标定至少需要 {min_samples} 个样本，实际 {len(accepted)}")
    bad = [i for i, sample in enumerate(accepted) if sample.reprojection_rmse_px > max_reprojection_rmse_px]
    if bad:
        raise HandEyeError(f"样本 {bad} 的棋盘格重投影误差超过 {max_reprojection_rmse_px:.2f}px")
    translation_span, rotation_span, second_rotation_span = motion_diversity(accepted)
    if translation_span < min_translation_span_m:
        raise HandEyeError(
            f"平移覆盖仅 {translation_span * 1000:.1f}mm，小于 {min_translation_span_m * 1000:.1f}mm"
        )
    if rotation_span < min_rotation_span_deg:
        raise HandEyeError(
            f"最大相对转角仅 {rotation_span:.1f}°，小于 {min_rotation_span_deg:.1f}°"
        )
    if second_rotation_span < min_second_rotation_span_deg:
        raise HandEyeError(
            f"第二独立旋转方向覆盖仅 {second_rotation_span:.1f}°，小于 {min_second_rotation_span_deg:.1f}°"
        )

    rotations_gripper = [sample.base_T_gripper[:3, :3] for sample in accepted]
    translations_gripper = [sample.base_T_gripper[:3, 3].reshape(3, 1) for sample in accepted]
    rotations_target = [sample.camera_T_target[:3, :3] for sample in accepted]
    translations_target = [sample.camera_T_target[:3, 3].reshape(3, 1) for sample in accepted]
    candidates: list[MethodResult] = []
    failures: list[str] = []
    for name, method in HAND_EYE_METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                rotations_gripper,
                translations_gripper,
                rotations_target,
                translations_target,
                method=method,
            )
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation
            transform[:3, 3] = np.asarray(translation).reshape(3)
            candidates.append(_validate_candidate(name, transform, accepted))
        except (cv2.error, GeometryError, HandEyeError, ValueError) as exc:
            failures.append(f"{name}: {exc}")
    if not candidates:
        raise HandEyeError("所有手眼算法均失败：" + "; ".join(failures))
    selected = min(candidates, key=lambda candidate: candidate.score)
    if selected.translation_rmse_m > max_translation_rmse_m:
        raise HandEyeError(
            f"最佳方法 {selected.method} 平移一致性 RMSE={selected.translation_rmse_m * 1000:.2f}mm，"
            f"超过 {max_translation_rmse_m * 1000:.2f}mm"
        )
    if selected.rotation_rmse_deg > max_rotation_rmse_deg:
        raise HandEyeError(
            f"最佳方法 {selected.method} 旋转一致性 RMSE={selected.rotation_rmse_deg:.3f}°，"
            f"超过 {max_rotation_rmse_deg:.3f}°"
        )
    return HandEyeResult(
        selected=selected,
        candidates=tuple(sorted(candidates, key=lambda candidate: candidate.score)),
        sample_count=len(accepted),
        translation_span_m=translation_span,
        rotation_span_deg=rotation_span,
        second_rotation_span_deg=second_rotation_span,
    )
