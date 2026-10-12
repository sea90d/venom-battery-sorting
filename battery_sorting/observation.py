"""Bound RGB-D/TCP observations. Recorded data is never live motion authority."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .handeye import pose_vector_to_transform
from .geometry import validate_rigid_transform
from .robot import rotation_error, translation_error


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Observation:
    colour: np.ndarray
    depth_m: np.ndarray
    metadata: dict
    captured_monotonic: float
    live: bool = False
    _content_digest: str = field(init=False, repr=False)

    def __post_init__(self):
        validate_arrays(self.colour,self.depth_m)
        validate_metadata(self.metadata)
        object.__setattr__(self,"_content_digest",self.content_digest())

    def content_digest(self):
        return hashlib.sha256(self.colour.tobytes()+self.depth_m.tobytes()+digest_json(self.metadata).encode()).hexdigest()

    @property
    def base_T_camera(self):
        tcp = pose_vector_to_transform(self.metadata["tcp_before"])
        return tcp @ np.asarray(self.metadata["tcp_T_camera"], dtype=float)

    def require_live(self, session_id, calibration_sha256, max_age_s):
        if self.content_digest()!=self._content_digest:
            raise ValueError("Observation mutated after capture")
        if not self.live or self.metadata["session_id"] != session_id:
            raise ValueError("Observation is replayed or belongs to another run")
        if self.metadata["calibration_sha256"] != calibration_sha256:
            raise ValueError("Observation calibration mismatch")
        age = time.monotonic() - self.captured_monotonic
        if not 0 <= age <= max_age_s:
            raise ValueError(f"Observation expired: {age:.3f}s")

    def save(self, directory):
        dest = Path(directory)
        dest.mkdir(parents=True, exist_ok=False)
        ok, encoded = cv2.imencode(".png", self.colour)
        if not ok:
            raise ValueError("Cannot encode RGB image")
        (dest / "colour.png").write_bytes(encoded.tobytes())
        np.save(dest / "depth_m.npy", self.depth_m, allow_pickle=False)
        manifest = {"schema_version": 1, "metadata": self.metadata,
                    "files": {name: hashlib.sha256((dest / name).read_bytes()).hexdigest()
                              for name in ("colour.png", "depth_m.npy")}}
        manifest["bundle_sha256"] = digest_json(manifest)
        (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                      indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        dest = Path(directory)
        manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
        expected = manifest.pop("bundle_sha256")
        if expected != digest_json(manifest) or manifest["schema_version"] != 1:
            raise ValueError("Observation manifest integrity mismatch")
        if set(manifest["files"]) != {"colour.png", "depth_m.npy"}:
            raise ValueError("Unexpected observation files")
        for name, digest in manifest["files"].items():
            if hashlib.sha256((dest / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Observation file integrity mismatch: {name}")
        colour = cv2.imdecode(np.fromfile(dest / "colour.png", np.uint8), cv2.IMREAD_COLOR)
        depth = np.load(dest / "depth_m.npy", allow_pickle=False)
        validate_arrays(colour, depth)
        validate_metadata(manifest["metadata"])
        intr=manifest['metadata']['camera']['intrinsics']
        if (intr['height'],intr['width'])!=colour.shape[:2]:
            raise ValueError('Recorded frame and intrinsics dimensions differ')
        return cls(colour, depth, manifest["metadata"], 0.0, False)


def validate_arrays(colour, depth):
    if not isinstance(colour, np.ndarray) or colour.dtype != np.uint8 or colour.ndim != 3 or colour.shape[2] != 3:
        raise ValueError("RGB frame must be uint8 HxWx3")
    if not isinstance(depth, np.ndarray) or depth.ndim != 2 or depth.shape != colour.shape[:2]:
        raise ValueError("Missing or unaligned depth")
    if depth.dtype != np.float32:
        raise ValueError("Depth must be float32 metres")


def validate_metadata(meta):
    for name in ("tcp_before", "tcp_after", "active_tcp_offset"):
        value = np.asarray(meta[name], dtype=float)
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name}")
    validate_rigid_transform(np.asarray(meta["tcp_T_camera"]), "tcp_T_camera")
    if translation_error(meta["tcp_before"], meta["tcp_after"]) > 0.0002 or rotation_error(meta["tcp_before"][3:], meta["tcp_after"][3:]) > np.deg2rad(0.1):
        raise ValueError("TCP moved during capture")
    if not meta["camera"]["serial"] or not meta["calibration_sha256"] or not meta["session_id"]:
        raise ValueError("Observation identity missing")
    camera=meta["camera"]
    if type(camera["frame_number"]) is not int or camera["frame_number"]<0:
        raise ValueError("Invalid camera frame number")
    intr=camera["intrinsics"]
    if any(type(intr[k]) is not int or intr[k]<=0 for k in ("width","height")):
        raise ValueError("Invalid intrinsics dimensions")
    if not np.isfinite([intr[k] for k in ("fx","fy","ppx","ppy")]).all() or min(intr['fx'],intr['fy'])<=0:
        raise ValueError("Invalid camera intrinsics")
    if len(intr['coeffs'])!=5 or not np.isfinite(intr['coeffs']).all():
        raise ValueError("Invalid distortion coefficients")


def capture_observation(camera, robot, tcp_T_camera, calibration_sha256,
                        session_id, active_tcp_offset, expected_serial,
                        *, previous_frame_number=None):
    # Bracket EACH RGB-D pair, never combine old RGB with a new TCP pose.
    # Discard queued frames left over from the previous robot pose.
    camera.capture()
    camera.capture()
    before = tuple(robot.snapshot().tcp_pose)
    started = time.monotonic()
    colour, depth = camera.capture()
    after = tuple(robot.snapshot().tcp_pose)
    validate_arrays(colour, depth)
    camera_meta = camera.observation_metadata()
    if camera_meta["serial"] != expected_serial:
        raise ValueError("Camera serial mismatch")
    number = camera_meta["frame_number"]
    if previous_frame_number is not None and number <= previous_frame_number:
        raise ValueError("Repeated or out-of-order camera frame")
    if time.monotonic() - started > 6:
        raise ValueError("Capture timed out")
    meta = {"observation_id": uuid.uuid4().hex, "session_id": session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tcp_before": list(before), "tcp_after": list(after),
            "active_tcp_offset": list(active_tcp_offset), "camera": camera_meta,
            "tcp_T_camera": np.asarray(tcp_T_camera).tolist(),
            "calibration_sha256": calibration_sha256}
    validate_metadata(meta)
    intr=camera_meta['intrinsics']
    if (intr['height'],intr['width'])!=colour.shape[:2]:
        raise ValueError("Frame and intrinsics dimensions differ")
    return Observation(colour.copy(), depth.copy(), meta, started, True)


def recorded_camera_model(metadata):
    """Factory lens math only: no context, pipeline, device lookup or connection."""
    import pyrealsense2 as rs
    from .realsense import RealSenseCamera
    data=metadata['camera']['intrinsics']
    intr=rs.intrinsics()
    for key in ('width','height','fx','fy','ppx','ppy','coeffs'):
        setattr(intr,key,data[key])
    model=data['model'].split('.')[-1]
    if model not in rs.distortion.__members__:
        raise ValueError('Unsupported recorded distortion model')
    intr.model=rs.distortion.__members__[model]
    result=RealSenseCamera()
    result._rs=rs
    result._color_rs_intrinsics=intr
    return result
