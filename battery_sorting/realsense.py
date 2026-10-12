"""Intel RealSense D455 RGB-D camera wrapper (lazy-import pyrealsense2).

Depth Z is NOT connected to robot motion; this module only provides
diagnostic capture and offline depth validation.

Aligned-depth intrinsics are obtained from the depth frame *after*
alignment, so that rs2_deproject_pixel_to_point operates on the
correct pixel grid.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import (
    CameraIntrinsics,
    transform_point,
    validate_rigid_transform,
)

__all__ = [
    "RealSenseError",
    "RealSenseDeviceInfo",
    "StreamProfile",
    "DepthSample",
    "DepthProbe",
    "RealSenseCamera",
    "RS_DEPTH_RANGE_M",
    "DEFAULT_USB3_PROFILE",
    "USB2_LOW_PROFILE",
]

RS_DEPTH_RANGE_M = (0.2, 3.0)

DEFAULT_USB3_PROFILE = (1280, 720, 30, 848, 480, 30)
USB2_LOW_PROFILE = (424, 240, 5, 480, 270, 5)


class RealSenseError(RuntimeError):
    """RealSense hardware access or configuration failure."""


@dataclass(frozen=True)
class RealSenseDeviceInfo:
    serial: str
    name: str
    usb_type: str


@dataclass(frozen=True)
class StreamProfile:
    color_width: int
    color_height: int
    color_fps: int
    depth_width: int
    depth_height: int
    depth_fps: int

    def __post_init__(self) -> None:
        for name in (
            "color_width", "color_height", "color_fps",
            "depth_width", "depth_height", "depth_fps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"StreamProfile.{name}={value} 必须是正整数"
                )


@dataclass(frozen=True)
class DepthSample:
    depth_m: float
    camera_xyz: tuple[float, float, float]
    base_xyz: tuple[float, float, float] | None
    valid_count: int
    mad_m: float
    quality_ok: bool

    def __post_init__(self) -> None:
        if not self.quality_ok:
            return
        if not (RS_DEPTH_RANGE_M[0] <= self.depth_m <= RS_DEPTH_RANGE_M[1]):
            raise RealSenseError(
                f"DepthSample 深度 {self.depth_m:.4f}m 超出量程 "
                f"[{RS_DEPTH_RANGE_M[0]:.1f}, {RS_DEPTH_RANGE_M[1]:.1f}]"
            )


@dataclass(frozen=True)
class DepthProbe:
    """Multi-frame aggregated depth near a pixel centroid."""

    sample: DepthSample | None
    frame_count: int
    converged: bool
    median_depth_m: float | None
    mad_m: float | None


class RealSenseCamera:
    """Lazy-import pyrealsense2; depth Z never connected to robot motion."""

    def __init__(
        self, profile: StreamProfile | None = None, *, warmup_frames: int = 10
    ) -> None:
        if isinstance(warmup_frames, bool) or not isinstance(warmup_frames, int) or warmup_frames < 1:
            raise ValueError("warmup_frames 必须是正整数")
        self._pipeline: Any = None
        self._profile: Any = None
        self._align: Any = None
        self._color_intrinsics: CameraIntrinsics | None = None
        self._color_rs_intrinsics: Any = None
        self._aligned_depth_intrinsics: Any = None
        self._camera_to_base: np.ndarray | None = None
        self._rs: Any = None
        self._stream_profile = profile
        self._warmup_frames = warmup_frames

    # ------------------------------------------------------------------
    # Lazy import
    # ------------------------------------------------------------------
    @staticmethod
    def _import_rs() -> Any:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RealSenseError(
                "未安装 pyrealsense2。请先执行: pip install pyrealsense2\n"
                "若已安装但仍报错，请确认 .venv 中 Python 和 pyrealsense2 "
                "的架构一致（均为 64-bit）。"
            ) from exc
        return rs

    @property
    def rs(self) -> Any:
        if self._rs is None:
            self._rs = self._import_rs()
        return self._rs

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------
    @staticmethod
    def list_devices() -> list[RealSenseDeviceInfo]:
        rs = RealSenseCamera._import_rs()
        ctx = rs.context()
        devices: list[RealSenseDeviceInfo] = []
        for dev in ctx.devices:
            serial = ""
            name = ""
            usb = ""
            if dev.supports(rs.camera_info.serial_number):
                serial = dev.get_info(rs.camera_info.serial_number)
            if dev.supports(rs.camera_info.name):
                name = dev.get_info(rs.camera_info.name)
            if dev.supports(rs.camera_info.usb_type_descriptor):
                usb = dev.get_info(rs.camera_info.usb_type_descriptor)
            devices.append(RealSenseDeviceInfo(serial=serial, name=name, usb_type=usb))
        return devices

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def start(self, serial: str | None = None) -> None:
        """Start pipeline; default USB3 1280x720/848x480@30.  No auto-downgrade."""
        rs = self.rs
        self.stop()

        sp = self._stream_profile
        if sp is None:
            cw, ch, cf, dw, dh, df = DEFAULT_USB3_PROFILE
        else:
            cw, ch, cf = sp.color_width, sp.color_height, sp.color_fps
            dw, dh, df = sp.depth_width, sp.depth_height, sp.depth_fps

        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cf)
        config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, df)

        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            raise RealSenseError(
                f"启动 RealSense 流失败: {exc}\n"
                "请尝试降低分辨率/帧率。若用 USB 2，显式指定 "
                f"--color-width {USB2_LOW_PROFILE[0]} "
                f"--color-height {USB2_LOW_PROFILE[1]} "
                f"--color-fps {USB2_LOW_PROFILE[2]} "
                f"--depth-width {USB2_LOW_PROFILE[3]} "
                f"--depth-height {USB2_LOW_PROFILE[4]} "
                f"--depth-fps {USB2_LOW_PROFILE[5]}"
            ) from exc

        self._pipeline = pipeline
        self._profile = profile

        try:
            self._color_intrinsics = self._read_color_intrinsics()
            align = rs.align(rs.stream.color)
        except Exception as exc:
            self.stop()
            if isinstance(exc, RealSenseError):
                raise
            raise RealSenseError(f"读取内参或创建深度对齐器失败: {exc}") from exc
        self._align = align

        for i in range(self._warmup_frames):
            try:
                _color, depth = self.capture(timeout_ms=5000)
                if depth is None:
                    raise RealSenseError("预热帧缺少对齐深度图")
            except (RuntimeError, RealSenseError) as exc:
                self.stop()
                raise RealSenseError(f"预热取帧失败（第{i + 1}帧）: {exc}") from exc
        if self._aligned_depth_intrinsics is None:
            self.stop()
            raise RealSenseError("预热结束后仍未获得对齐深度内参")

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
        self._pipeline = None
        self._profile = None
        self._align = None
        self._color_intrinsics = None
        self._color_rs_intrinsics = None
        self._aligned_depth_intrinsics = None
        self._last_color_frame = None

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    @property
    def is_active(self) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Intrinsics & extrinsics
    # ------------------------------------------------------------------
    @property
    def color_intrinsics(self) -> CameraIntrinsics:
        if self._color_intrinsics is None:
            raise RealSenseError("相机未启动，无法获取内参")
        return self._color_intrinsics

    @property
    def camera_to_base(self) -> np.ndarray | None:
        return self._camera_to_base

    def set_camera_to_base(self, matrix: np.ndarray) -> None:
        validate_rigid_transform(matrix, "camera_to_base")
        self._camera_to_base = matrix.astype(np.float64).copy()

    @property
    def is_extrinsics_calibrated(self) -> bool:
        return self._camera_to_base is not None

    def _read_color_intrinsics(self) -> CameraIntrinsics:
        if self._profile is None:
            raise RealSenseError("Pipeline 未启动")
        stream = self._profile.get_stream(self.rs.stream.color)
        profile = stream.as_video_stream_profile()
        intr = profile.get_intrinsics()
        self._color_rs_intrinsics = intr
        return CameraIntrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
        )

    @property
    def color_distortion(self) -> tuple[str, tuple[float, ...]]:
        """Return the factory colour distortion model and coefficients.

        The coefficients are diagnostic metadata only.  In particular,
        ``inverse_brown_conrady`` coefficients must not be passed directly to
        OpenCV as forward Brown-Conrady distortion coefficients.
        """
        intr = self._color_rs_intrinsics
        if intr is None:
            raise RealSenseError("相机未启动，无法获取彩色畸变参数")
        return str(intr.model), tuple(float(value) for value in intr.coeffs)

    def normalize_color_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Convert distorted colour pixels to undistorted normalized rays.

        RealSense performs the correct conversion for the device's factory
        distortion model.  Each output row is ``(X/Z, Y/Z)`` for a unit-depth
        camera ray and can be used with OpenCV PnP using ``K=I, dist=0``.
        """
        intr = self._color_rs_intrinsics
        if intr is None:
            raise RealSenseError("相机未启动，无法归一化彩色像素")
        array = np.asarray(pixels, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise RealSenseError(f"pixels 必须是 (N,2) 数组，实际 {array.shape}")
        if not np.all(np.isfinite(array)):
            raise RealSenseError("pixels 包含非有限值")
        width = int(intr.width)
        height = int(intr.height)
        if np.any(array[:, 0] < 0.0) or np.any(array[:, 0] >= width):
            raise RealSenseError("pixels 的 u 坐标超出彩色图像")
        if np.any(array[:, 1] < 0.0) or np.any(array[:, 1] >= height):
            raise RealSenseError("pixels 的 v 坐标超出彩色图像")
        result = np.empty_like(array)
        for index, (u, v) in enumerate(array):
            ray = self.rs.rs2_deproject_pixel_to_point(
                intr, [float(u), float(v)], 1.0
            )
            if ray is None or len(ray) != 3 or not all(math.isfinite(x) for x in ray):
                raise RealSenseError("彩色像素反投影返回非有限值")
            if abs(float(ray[2])) < 1e-12:
                raise RealSenseError("彩色像素反投影返回零 Z 射线")
            result[index] = (float(ray[0]) / float(ray[2]), float(ray[1]) / float(ray[2]))
        return result

    def project_color_points(self, points_camera_m: np.ndarray) -> np.ndarray:
        """Project camera-frame 3-D points with the factory colour model."""
        intr = self._color_rs_intrinsics
        if intr is None:
            raise RealSenseError("相机未启动，无法投影彩色三维点")
        points = np.asarray(points_camera_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise RealSenseError(
                f"points_camera_m 必须是 (N,3) 数组，实际 {points.shape}"
            )
        if not np.all(np.isfinite(points)) or np.any(points[:, 2] <= 0.0):
            raise RealSenseError("彩色投影点必须有限且 Z>0")
        pixels = np.empty((len(points), 2), dtype=np.float64)
        for index, point in enumerate(points):
            pixel = self.rs.rs2_project_point_to_pixel(intr, point.tolist())
            if pixel is None or len(pixel) != 2 or not all(math.isfinite(x) for x in pixel):
                raise RealSenseError("彩色三维点投影返回非有限值")
            pixels[index] = pixel
        return pixels

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------
    def capture(self, timeout_ms: int = 5000) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (colour_bgr, depth_m) aligned pair.

        depth_m is in metres from depth_frame.get_units().  Validates
        that aligned colour and depth have the same resolution.
        Also refreshes _aligned_depth_intrinsics from the aligned depth
        frame profile so that rs2_deproject_pixel_to_point uses the
        correct pixel grid.
        """
        if self._pipeline is None or self._align is None:
            raise RealSenseError("相机未启动")
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as exc:
            raise RealSenseError(f"取帧超时或失败: {exc}") from exc
        try:
            aligned = self._align.process(frames)
        except RuntimeError as exc:
            raise RealSenseError(f"深度对齐失败: {exc}") from exc

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            raise RealSenseError("获取彩色帧失败")

        color = np.asanyarray(color_frame.get_data())
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise RealSenseError(
                f"彩色帧格式异常 dtype={color.dtype} shape={color.shape}"
            )

        depth: np.ndarray | None = None
        if depth_frame:
            units = float(depth_frame.get_units())
            if not (math.isfinite(units) and units > 0.0):
                raise RealSenseError(
                    f"depth_units={units} 无效，必须为正有限值"
                )
            depth = np.asanyarray(depth_frame.get_data(), dtype=np.float32) * units
            if depth.dtype != np.float32:
                raise RealSenseError(f"深度帧 dtype={depth.dtype} 异常")
            ch, cw = color.shape[:2]
            dh, dw = depth.shape[:2]
            if (cw, ch) != (dw, dh):
                raise RealSenseError(
                    f"对齐后彩色 ({cw}x{ch}) 与深度 ({dw}x{dh}) 分辨率不一致"
                )
            self._aligned_depth_intrinsics = (
                depth_frame.profile.as_video_stream_profile().get_intrinsics()
            )

        self._last_color_frame = color_frame
        return color, depth

    def observation_metadata(self) -> dict:
        """Identity and complete factory intrinsics for the last RGB-D pair."""
        frame = getattr(self, "_last_color_frame", None)
        if frame is None:
            raise RealSenseError("No captured frame metadata")
        intr = self._color_rs_intrinsics
        if intr is None:
            raise RealSenseError("Aligned depth intrinsics missing")
        return {
            "serial": str(self._profile.get_device().get_info(self.rs.camera_info.serial_number)),
            "frame_number": int(frame.get_frame_number()),
            "timestamp_ms": float(frame.get_timestamp()),
            "timestamp_domain": str(frame.get_frame_timestamp_domain()),
            "color_settings": self.color_settings(),
            "intrinsics": {
            "width": int(intr.width), "height": int(intr.height),
            "fx": float(intr.fx), "fy": float(intr.fy),
            "ppx": float(intr.ppx), "ppy": float(intr.ppy),
            "model": str(intr.model), "coeffs": list(intr.coeffs),
        }}

    def color_settings(self) -> dict:
        sensor=self._profile.get_device().first_color_sensor()
        return {name:float(sensor.get_option(getattr(self.rs.option,name)))
                for name in ('exposure','gain','white_balance','enable_auto_exposure','enable_auto_white_balance')}

    def configure_color(self, settings: dict) -> None:
        """Apply and read back fixed RGB parameters before production capture."""
        if set(settings) != {"exposure", "gain", "white_balance"}:
            raise RealSenseError("Expected exposure, gain, white_balance")
        sensor = self._profile.get_device().first_color_sensor()
        rs = self.rs
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.enable_auto_white_balance, 0)
        for name, value in settings.items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise RealSenseError(f"Missing fixed color setting: {name}")
            option = getattr(rs.option, name)
            limits = sensor.get_option_range(option)
            if not limits.min <= value <= limits.max:
                raise RealSenseError(f"Color setting outside sensor range: {name}")
            sensor.set_option(option, value)
            # Sensor settings are asynchronous: allow bounded settling instead
            # of treating the first read after set_option as final feedback.
            actual = None
            for _ in range(10):
                time.sleep(0.05)
                actual = sensor.get_option(option)
                if math.isfinite(actual) and abs(actual-value) <= max(limits.step, 1e-6):
                    break
            else:
                raise RealSenseError(f"Color setting readback mismatch: {name}; requested={value}, actual={actual}, tolerance={max(limits.step, 1e-6)} after 10 reads")
        for _ in range(self._warmup_frames):
            self.capture()

    # ------------------------------------------------------------------
    # Robust depth sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_neighbourhood(
        depth_m: np.ndarray,
        cx: int,
        cy: int,
        radius: int = 5,
    ) -> tuple[float, float, int]:
        """Median of valid depth values in a square window, with MAD.

        Returns (median_m, mad_m, valid_count).  Zero, non-finite and
        out-of-range values are treated as invalid.
        """
        if not isinstance(depth_m, np.ndarray) or depth_m.ndim != 2:
            raise RealSenseError("depth_m 必须是 2D numpy 数组")
        if not np.issubdtype(depth_m.dtype, np.number):
            raise RealSenseError("depth_m 必须是数值数组")
        if radius < 1:
            raise RealSenseError("radius 必须 >= 1")
        h, w = depth_m.shape
        if not (0 <= cx < w and 0 <= cy < h):
            raise RealSenseError(
                f"采样中心 ({cx},{cy}) 超出深度图 ({w}x{h})"
            )
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[(patch > 0.0) & np.isfinite(patch)]
        valid = valid[
            (valid >= RS_DEPTH_RANGE_M[0]) & (valid <= RS_DEPTH_RANGE_M[1])
        ]
        count = len(valid)
        if count == 0:
            return (0.0, 0.0, 0)
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))
        return (median, mad, count)

    def _deproject_pixel(
        self, u: float, v: float, depth_m: float,
    ) -> tuple[float, float, float]:
        """Use rs2_deproject_pixel_to_point with aligned-depth intrinsics.

        This preserves the factory distortion model of the D455.
        geometry.back_project is a distortion-free pinhole model used
        only for testing/mock — it is NOT the real hardware path.
        """
        if self._aligned_depth_intrinsics is None:
            raise RealSenseError("对齐深度内参未初始化，请先调用 capture()")
        if not (math.isfinite(u) and math.isfinite(v) and math.isfinite(depth_m)):
            raise RealSenseError("deproject 输入坐标必须为有限数值")
        if depth_m <= 0.0:
            raise RealSenseError("deproject 深度必须为正")
        rs = self.rs
        point = rs.rs2_deproject_pixel_to_point(
            self._aligned_depth_intrinsics, [float(u), float(v)], depth_m,
        )
        if point is None or not all(math.isfinite(p) for p in point):
            raise RealSenseError("rs2_deproject_pixel_to_point 返回非有限值")
        return (float(point[0]), float(point[1]), float(point[2]))

    def deproject_pixel(
        self, u: float, v: float, depth_m: float,
    ) -> tuple[float, float, float]:
        """反投影同一已采集帧中的像素，不触发新的相机采集。"""
        intrinsics = self._aligned_depth_intrinsics
        if intrinsics is None:
            raise RealSenseError("对齐深度内参未初始化，请先调用 capture()")
        width = int(getattr(intrinsics, "width", 0))
        height = int(getattr(intrinsics, "height", 0))
        if width <= 0 or height <= 0:
            raise RealSenseError("对齐深度内参缺少有效 width/height")
        if not (0.0 <= u < width and 0.0 <= v < height):
            raise RealSenseError(
                f"反投影像素 ({u:.1f},{v:.1f}) 超出深度图 ({width}x{height})"
            )
        return self._deproject_pixel(u, v, depth_m)

    # ------------------------------------------------------------------
    # Single-frame depth probe
    # ------------------------------------------------------------------
    def probe_depth(
        self,
        u: float,
        v: float,
        *,
        radius: int = 5,
        min_valid: int = 8,
        max_mad_m: float = 0.05,
    ) -> DepthSample:
        """Robust depth at (u,v) pixel, with neighbourhood median.

        Returns DepthSample.  camera_xyz is in camera frame via
        rs2_deproject_pixel_to_point; base_xyz is in UR base frame
        (None if extrinsics not calibrated).
        """
        if radius < 1:
            raise RealSenseError("radius 必须 >= 1")
        if min_valid < 1:
            raise RealSenseError("min_valid 必须 >= 1")
        if not math.isfinite(max_mad_m) or max_mad_m <= 0.0:
            raise RealSenseError("max_mad_m 必须为正")
        if not (math.isfinite(u) and math.isfinite(v)):
            raise RealSenseError("探测像素坐标必须为有限数值")

        color, depth_m = self.capture()
        if depth_m is None:
            raise RealSenseError("本帧无可用的对齐深度图")

        h, w = depth_m.shape[:2]
        u_int = int(round(u))
        v_int = int(round(v))
        if not (0 <= u_int < w and 0 <= v_int < h):
            raise RealSenseError(
                f"探测像素 ({u_int},{v_int}) 超出深度图 ({w}x{h})"
            )

        median, mad, count = self._sample_neighbourhood(
            depth_m, u_int, v_int, radius=radius,
        )
        quality = count >= min_valid and mad <= max_mad_m and median > 0.0

        cam_xyz = (0.0, 0.0, 0.0)
        base_xyz: tuple[float, float, float] | None = None
        if quality:
            cam_xyz = self._deproject_pixel(u, v, median)
            if self._camera_to_base is not None:
                base_xyz = transform_point(cam_xyz, self._camera_to_base)

        return DepthSample(
            depth_m=median,
            camera_xyz=cam_xyz,
            base_xyz=base_xyz,
            valid_count=count,
            mad_m=mad,
            quality_ok=quality,
        )

    # ------------------------------------------------------------------
    # Multi-frame depth probe
    # ------------------------------------------------------------------
    def probe_depth_multi(
        self,
        u: float,
        v: float,
        *,
        frames: int = 5,
        radius: int = 5,
        min_valid: int = 8,
        max_mad_m: float = 0.05,
        max_frame_mad_m: float = 0.005,
        required_consistent: int = 3,
        settle_time: float = 0.05,
    ) -> DepthProbe:
        """Multi-frame aggregated depth probe.

        Collects *frames* samples, keeps only quality-OK ones.  Computes
        the median of per-frame depth medians.  Rejects if the cross-frame
        MAD exceeds max_frame_mad_m.  The returned sample is re-deprojected
        from the aggregated median depth so that depth_m, camera_xyz and
        base_xyz are consistent.
        """
        if frames < 1:
            raise RealSenseError("frames 必须 >= 1")
        if required_consistent < 1 or required_consistent > frames:
            raise RealSenseError(
                f"required_consistent={required_consistent} 必须在 [1,{frames}]"
            )
        if radius < 1:
            raise RealSenseError("radius 必须 >= 1")
        if min_valid < 1:
            raise RealSenseError("min_valid 必须 >= 1")
        if (
            not math.isfinite(max_mad_m)
            or not math.isfinite(max_frame_mad_m)
            or max_mad_m <= 0.0
            or max_frame_mad_m <= 0.0
        ):
            raise RealSenseError("max_mad_m 和 max_frame_mad_m 必须为正")
        if not math.isfinite(settle_time) or settle_time < 0.0:
            raise RealSenseError("settle_time 不能为负")
        if not (math.isfinite(u) and math.isfinite(v)):
            raise RealSenseError("探测像素坐标必须为有限数值")

        samples: list[DepthSample] = []
        for _ in range(frames):
            sample = self.probe_depth(
                u, v, radius=radius, min_valid=min_valid, max_mad_m=max_mad_m,
            )
            if sample.quality_ok:
                samples.append(sample)
            time.sleep(settle_time)

        if len(samples) < required_consistent:
            return DepthProbe(
                sample=None,
                frame_count=len(samples),
                converged=False,
                median_depth_m=None,
                mad_m=None,
            )

        depths = np.array([s.depth_m for s in samples], dtype=np.float64)
        med = float(np.median(depths))
        mad = float(np.median(np.abs(depths - med)))

        if mad > max_frame_mad_m:
            return DepthProbe(
                sample=None,
                frame_count=len(samples),
                converged=False,
                median_depth_m=med,
                mad_m=mad,
            )

        cam_xyz = self._deproject_pixel(u, v, med)
        base_xyz: tuple[float, float, float] | None = None
        if self._camera_to_base is not None:
            base_xyz = transform_point(cam_xyz, self._camera_to_base)

        aggregated = DepthSample(
            depth_m=med,
            camera_xyz=cam_xyz,
            base_xyz=base_xyz,
            valid_count=min(s.valid_count for s in samples),
            mad_m=max(s.mad_m for s in samples),
            quality_ok=True,
        )

        return DepthProbe(
            sample=aggregated,
            frame_count=len(samples),
            converged=True,
            median_depth_m=med,
            mad_m=mad,
        )
