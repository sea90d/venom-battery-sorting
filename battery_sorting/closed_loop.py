"""Eye-in-hand competition entry: readiness / collect / run. Default is offline."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .config import BatterySortingConfig, Pose, WorkspaceBounds
from .competition_start import require_competition_start, competition_observation_pose
from .grasp_lift import GraspLiftConfig, compute_descent_z
from .gripper import create_gripper
from .handeye import pose_vector_to_transform
from .handeye_runtime import HandEyeRuntimeMapper
from .observation import capture_observation, Observation, recorded_camera_model
from .offsite import load_site
from .placement_vision import detect_t0, detect_sequence_targets
from .realsense import RealSenseCamera, RealSenseError
from .robot import RealUR5, Waypoint, assert_pose_close, rotation_error
from .scene import detect_batteries
from .table_view import rectify_table, require_stationary_board
from .tcp_readonly import read_tcp_info


COLORS = {"red", "yellow", "blue", "green"}
OFFICIAL_TASK_COLORS = {"basic": ("red",), "sequence": ("yellow", "green", "blue")}
ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = set("schema_version site_config grasp_config vision_config camera_serial camera_settings active_tcp_offset_m_rad paper_z_m target_view_pose_m_rad pick_view_pose_m_rad sequence_target_view_pose_m_rad transit_high_z_m transit_speed_mps transit_acceleration_mps2 pick_roi target_roi sequence_target_roi grasp_reference_handle_yaw_rad grasp_center_offset_xy_m grip_position_range min_pick_separation_m body_size_m target_containment_margin_m max_observation_age_s max_scene_shift_m max_height_mad_m max_center_spread_m max_yaw_spread_deg frames geometry_and_gripper_verified_by_task".split())


def axis_delta(a, b):
    return (a - b + math.pi / 2) % math.pi - math.pi / 2


def finite_vector(value, n, name):
    if not isinstance(value, (list, tuple)) or len(value) != n or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError(f"{name} must contain {n} finite numbers")
    return tuple(float(v) for v in value)


class Setup:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not set(self.raw).issubset(CONFIG_KEYS) or type(self.raw["schema_version"]) is not int or self.raw["schema_version"] != 1:
            raise ValueError("closed_loop configuration schema mismatch")
        self.site = load_site(self.path.parent / self.raw["site_config"])
        self.grasp = GraspLiftConfig.from_json(self.path.parent / self.raw["grasp_config"])
        self.vision = BatterySortingConfig.from_json(self.path.parent / self.raw["vision_config"]).vision
        calibration_path = Path(self.site.handeye_result_path)
        if not calibration_path.is_absolute():
            calibration_path = ROOT / calibration_path
        self.calibration_path = calibration_path.resolve()
        self.mapper = HandEyeRuntimeMapper.from_json(self.calibration_path)
        self.calibration_sha256 = hashlib.sha256(self.calibration_path.read_bytes()).hexdigest()
        if self.grasp.final_grasp_z_m != self.site.pick_contact_z_m:
            raise ValueError("Grasp Z differs between site and grasp configurations")
        if self.grasp.robot_host != self.site.robot_host or self.grasp.gripper.port != self.site.gripper_port:
            raise ValueError("Robot/gripper identity differs between configurations")
        verified=self.raw["geometry_and_gripper_verified_by_task"]
        if not isinstance(verified,dict) or set(verified)!=set(OFFICIAL_TASK_COLORS) or any(type(v) is not bool for v in verified.values()):
            raise ValueError("geometry_and_gripper_verified_by_task must contain basic/sequence booleans")
        if not isinstance(self.raw["camera_serial"],str) or not self.raw["camera_serial"]:
            raise ValueError("camera_serial missing")
        settings=self.raw["camera_settings"]
        if not isinstance(settings,dict) or set(settings)!={"exposure","gain","white_balance"}:
            raise ValueError("Invalid camera_settings")
        for key,value in settings.items():
            if value is not None:
                finite_vector([value],1,key)
        for key, low, high in (("max_observation_age_s", 1, 120), ("max_scene_shift_m", 0.0001, 0.01),
                               ("max_height_mad_m", 0.0001, 0.01), ("max_center_spread_m", 0.0001, 0.01),
                               ("max_yaw_spread_deg", 0.1, 10), ("target_containment_margin_m", 0, 0.02)):
            value = self.raw[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid {key}")
        if type(self.raw["frames"]) is not int or not 3 <= self.raw["frames"] <= 9:
            raise ValueError("frames must be 3..9")
        for key, n in (("active_tcp_offset_m_rad", 6), ("target_view_pose_m_rad", 6),
                       ("pick_view_pose_m_rad", 6), ("sequence_target_view_pose_m_rad", 6),
                       ("grasp_center_offset_xy_m", 2), ("grip_position_range", 2), ("body_size_m", 2)):
            if self.raw.get(key) is not None:
                finite_vector(self.raw[key], n, key)
        for key in ("paper_z_m", "grasp_reference_handle_yaw_rad", "min_pick_separation_m"):
            if self.raw[key] is not None:
                finite_vector([self.raw[key]], 1, key)
        transit_z=self.raw.get("transit_high_z_m")
        if transit_z is not None:
            finite_vector([transit_z],1,"transit_high_z_m")
            if not self.site.pick_contact_z_m < transit_z <= self.site.safe_high_z_m:
                raise ValueError("transit_high_z_m must be above pick contact and at/below safe high Z")
        for key, low, high in (("transit_speed_mps", .005, .5), ("transit_acceleration_mps2", .005, 1.)):
            value=self.raw.get(key, high)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f"Invalid {key}")
        for key in ("pick_roi", "target_roi", "sequence_target_roi"):
            if self.raw[key] is not None:
                roi = self.raw[key]
                if not isinstance(roi, list) or len(roi) != 4 or any(type(v) is not int for v in roi) or min(roi[:2]) < 0 or min(roi[2:]) <= 0:
                    raise ValueError(f"Invalid {key}")
        if any(v <= 0 or v > 0.2 for v in self.raw["body_size_m"]):
            raise ValueError("Invalid body dimensions")

    def blockers(self, task="basic", *, site_test=False):
        if task not in OFFICIAL_TASK_COLORS:
            raise ValueError("task must be basic or sequence")
        out = []
        target_keys=("target_roi",) if task=="basic" else (
            "sequence_target_view_pose_m_rad","sequence_target_roi")
        shared=("active_tcp_offset_m_rad", "paper_z_m", "pick_roi", "target_view_pose_m_rad",
                    "grasp_reference_handle_yaw_rad", "grip_position_range",
                    "min_pick_separation_m", *target_keys)
        required=shared+(("grasp_center_offset_xy_m",) if task=="basic" else ())
        for key in required:
            if self.raw[key] is None:
                out.append(f"closed_loop.{key}")
        for key in ("placement_contact_z_m", "workspace_bounds_m", "home_pose_m_rad"):
            if getattr(self.site, key) is None:
                out.append(f"site.geometry.{key}")
        for key in ("exposure", "gain", "white_balance"):
            if self.raw["camera_settings"].get(key) is None:
                out.append(f"camera_settings.{key}")
        if not self.raw["geometry_and_gripper_verified_by_task"][task] and not (site_test and task=='sequence'):
            out.append(f"geometry_and_gripper_verified_by_task.{task}=false")
        return out

    def require_ready(self, task="basic", *, site_test=False):
        blockers = self.blockers(task, site_test=site_test)
        if blockers:
            raise ValueError("Missing site data: " + ", ".join(blockers))
        r, s = self.raw, self.site
        self.grasp.require_valid(allow_missing_final_z=False)
        if not (r["paper_z_m"] < s.pick_contact_z_m < s.safe_high_z_m - 0.03 and
                r["paper_z_m"] < s.placement_contact_z_m < s.safe_high_z_m - 0.03):
            raise ValueError("Paper/contact/transit Z ordering invalid")
        low, high = r["grip_position_range"]
        if not 0 < low < high < 1000:
            raise ValueError("Invalid calibrated grip_position_range")
        if task=="basic" and math.hypot(*r["grasp_center_offset_xy_m"]) > 1e-9:
            raise ValueError("Only centered handles supported: an undirected body axis cannot locate an asymmetric handle")
        if not 0.07<=r["min_pick_separation_m"]<=0.3:
            raise ValueError("Measured min_pick_separation_m must be 0.07..0.3")
        if s.gripper_driver!="ag105145_modbus" or self.grasp.gripper.driver!=s.gripper_driver:
            raise ValueError("Production requires AG105145 feedback driver")
        g=self.grasp.gripper
        for name,low,high in (('poll_interval',.01,.5),('serial_timeout',.01,2),('command_timeout',1,30)):
            value=getattr(g,name)
            if not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'Invalid gripper {name}')
        if type(g.read_retries) is not int or not 0<=g.read_retries<=3:
            raise ValueError('Invalid gripper read_retries')
        if not 0<=g.close_position<r['grip_position_range'][0]<r['grip_position_range'][1]<g.open_position<=1000:
            raise ValueError('Gripper positions must bracket calibrated held width')
        if not 0<g.force_percent<=100 or not 0<g.speed_percent<=100:
            raise ValueError('Gripper force/speed must be positive')
        if max(s.safe_high_z_m-s.pick_contact_z_m,s.safe_high_z_m-s.placement_contact_z_m)>self.grasp.max_total_descent_m:
            raise ValueError('Contact descent exceeds configured total limit')
        from .folded_standby import plan_for
        views=(self.pick_view,self.target_view(task)) if plan_for(self) else (self.home,self.pick_view,self.target_view(task))
        for pose in views:
            if abs(pose.z - s.safe_high_z_m) > 1e-6 or not self.in_workspace(pose):
                raise ValueError("Camera view must be within measured workspace at safe high Z")
        require_competition_start(self.home)

    def validate_recorded_scene_bounds(self, analysis_path=None, robot=None, task="sequence"):
        """Check an optional latest RGB-D analysis before live motion.

        This is diagnostic only: live frames remain authoritative.  When a
        saved analysis from the current room contains points outside the
        configured workspace, fail early with the calibration action needed
        instead of reaching a controller rejection during a transfer.
        """
        if analysis_path is None:
            candidates = list((ROOT / 'output').glob('**/analysis.json'))
            if not candidates:
                return
            analysis_path = max(candidates, key=lambda item: item.stat().st_mtime)
        path = Path(analysis_path)
        if not path.is_file():
            return
        try:
            report = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        points=[]
        for item in report.get('instances', {}).values():
            for key in ('center_base_m',):
                value=item.get(key)
                if isinstance(value, list) and len(value) >= 2:
                    points.append((float(value[0]), float(value[1]), key))
            handle=item.get('handle') or {}
            value=handle.get('center_base_m')
            if isinstance(value, list) and len(value) >= 2:
                points.append((float(value[0]), float(value[1]), 'handle'))
        for x,y,kind in points:
            pose=Pose(x,y,self.transit_high_z,*self.pick_view.as_list()[3:])
            if not self.in_workspace(pose):
                raise ValueError(f'Recorded scene {kind} outside configured workspace at ({x:.4f}, {y:.4f}); recapture hand-eye/TCP data')
            # An explicit replay diagnostic may query an established RTDE
            # Control interface. Receive-only preflight and collection cannot
            # make this query. Runtime checks live geometry before motion.
            if robot is not None:
                try:
                    accepted = robot.pose_within_safety_limits(pose.as_list())
                except Exception as exc:
                    raise ValueError(f'Unable to preflight recorded {kind} pose at ({x:.4f}, {y:.4f}): {exc}') from exc
                if not accepted:
                    raise ValueError(
                        f'Recorded scene {kind} is rejected by UR safety limits at '
                        f'({x:.4f}, {y:.4f}, z={self.transit_high_z:.4f}); '
                        'recapture hand-eye/TCP data or adjust the taught observation pose'
                    )

    @property
    def home(self):
        return Pose(*self.site.home_pose_m_rad.as_tuple())

    @property
    def pick_view(self):
        """Downward view reached by coordinated unfolding after the start signal."""
        return competition_observation_pose(self)

    def target_view(self, task):
        key="target_view_pose_m_rad" if task=="basic" else "sequence_target_view_pose_m_rad"
        return Pose(*self.raw[key])

    @property
    def transit_high_z(self):
        """Verified UR-reachable carry height for edge-of-table battery XY."""
        return float(self.raw.get("transit_high_z_m") or self.site.safe_high_z_m)

    @property
    def transit_speed(self):
        return float(self.raw.get("transit_speed_mps", .3))

    @property
    def transit_acceleration(self):
        return float(self.raw.get("transit_acceleration_mps2", .5))

    def target_roi(self, task):
        key="target_roi" if task=="basic" else "sequence_target_roi"
        return self.raw[key]

    def in_workspace(self, pose):
        b = self.site.workspace_bounds_m
        return b is not None and b.x_min <= pose.x <= b.x_max and b.y_min <= pose.y <= b.y_max and b.z_min <= pose.z <= b.z_max

    def robot_config(self):
        config=self.grasp.robot_config(self.home)
        if self.site.workspace_bounds_m is not None:
            config=replace(config,workspace_bounds=WorkspaceBounds(**asdict(self.site.workspace_bounds_m)))
        return config


class Motion:
    """One bounded path executor for rotations, transfer, grasp and place."""
    def __init__(self, setup, robot, log):
        self.setup, self.robot, self.log = setup, robot, log
        self.expected = setup.home
        self.continuous = False

    def continuous_path(self, path, label, check=None, before_segment=None):
        self_offset = self.robot.active_tcp_offset()
        expected_offset = self.setup.raw['active_tcp_offset_m_rad']
        if math.dist(self_offset[:3], expected_offset[:3]) > .0001 or rotation_error(self_offset[3:], expected_offset[3:]) > math.radians(.05):
            raise ValueError('Active TCP changed before continuous motion')
        if not self.setup.in_workspace(self.expected) or any(not self.setup.in_workspace(row[0].pose) for row in path):
            raise ValueError('Outside measured workspace: '+label)
        for waypoint, _, _, _ in path:
            if not self.robot.pose_within_safety_limits(waypoint.pose.as_list()):
                raise ValueError('Controller safety limits reject planned path '+waypoint.label)
        assert_pose_close(self.robot.snapshot().tcp_pose, self.expected, .0005, math.radians(.2), label+' start')
        def monitor():
            if before_segment: before_segment()
            if check: check()
        monitor()
        # Live controller monitors grip/freshness during the single moveL.
        kwargs = {'check': monitor} if self.robot.is_real else {}
        if not self.robot.move_l_path_without_tcp_force_guard(path, **kwargs):
            raise RuntimeError('Robot rejected '+label)
        final = path[-1][0].pose
        assert_pose_close(self.robot.snapshot().tcp_pose, final, .001, math.radians(.5), label+' arrival')
        self.expected = final
        monitor()
        self.log('smooth_path', label=label, poses=[row[0].pose.as_list() for row in path],
                 speeds=[row[1] for row in path], accelerations=[row[2] for row in path],
                 blends_m=[row[3] for row in path])

    def _is_high(self, z):
        return abs(z-self.setup.site.safe_high_z_m)<=1e-6 or abs(z-self.setup.transit_high_z)<=1e-6

    def move(self, target, label, speed=None):
        s, g = self.setup.site, self.setup.grasp
        start = self.expected
        if not self.setup.in_workspace(start) or not self.setup.in_workspace(target):
            raise ValueError(f"Outside measured workspace: {label}")
        horizontal = math.hypot(target.x-start.x, target.y-start.y)
        rotating = rotation_error(start.as_list()[3:], target.as_list()[3:]) > 1e-7
        dz = abs(target.z-start.z)
        offset=self.robot.active_tcp_offset()
        if math.dist(offset[:3],self.setup.raw['active_tcp_offset_m_rad'][:3])>0.0001 or rotation_error(offset[3:],self.setup.raw['active_tcp_offset_m_rad'][3:])>math.radians(.05):
            raise ValueError('Active TCP changed before motion')
        if (horizontal > 1e-8 or rotating) and (not self._is_high(start.z) or not self._is_high(target.z)):
            raise ValueError("Horizontal motion and rotation require a verified high Z")
        if rotating and horizontal > 1e-8:
            raise ValueError("Rotate separately from transfer")
        if dz > g.max_z_segment_m + 1e-9 or horizontal > 0.05 + 1e-9 or rotation_error(start.as_list()[3:],target.as_list()[3:])>math.radians(30)+1e-8:
            raise ValueError("Motion segment exceeds limit")
        assert_pose_close(self.robot.snapshot().tcp_pose, start, 0.0005, math.radians(0.2), label)
        command_speed = speed if speed is not None else (self.setup.transit_speed if horizontal > 1e-8 or rotating else g.approach_speed_mps)
        command_acceleration = g.acceleration_mps2 if speed is not None else (self.setup.transit_acceleration if horizontal > 1e-8 or rotating else g.acceleration_mps2)
        if not self.robot.move_l_without_tcp_force_guard(Waypoint(label, target, "approach"), command_speed, command_acceleration):
            raise RuntimeError(f"Robot rejected {label}")
        assert_pose_close(self.robot.snapshot().tcp_pose, target, 0.001, math.radians(0.5), label+":arrival")
        self.expected = target
        self.log("move", label=label, pose=target.as_list())

    def high(self, target, label, check=None):
        if not self._is_high(self.expected.z):
            raise ValueError("Transfer must start high")
        start = self.expected
        if abs(target.z-start.z) > 1e-6:
            raise ValueError("Transfer must end high")
        n = max(1, math.ceil(math.hypot(target.x-start.x, target.y-start.y)/0.05))
        planned=[]
        for i in range(1, n+1):
            planned.append(Pose(start.x+(target.x-start.x)*i/n, start.y+(target.y-start.y)*i/n,
                                start.z, start.rx,start.ry,start.rz))
        # SO(3) interpolation keeps local rotation segments <=30 degrees.
        first = pose_vector_to_transform(self.expected.as_list())[:3,:3]
        last = pose_vector_to_transform(target.as_list())[:3,:3]
        delta = cv2.Rodrigues(first.T @ last)[0].reshape(3)
        count = max(1, math.ceil(np.linalg.norm(delta)/math.radians(30)))
        for i in range(1, count+1):
            vec = cv2.Rodrigues(first @ cv2.Rodrigues(delta*i/count)[0])[0].reshape(3)
            planned.append(Pose(target.x,target.y,target.z,*vec))
        for i,pose in enumerate(planned,1):
            if not self.robot.pose_within_safety_limits(pose.as_list()):
                raise ValueError(f"Controller safety limits reject planned path {label}:{i}: {pose.as_list()}")
        if self.continuous:
            # Collinear transfer has one endpoint; rotation stays a separate
            # command, so blending cannot cut a corner through nearby blocks.
            xy_end = planned[n-1]
            if math.dist(start.as_list()[:3], xy_end.as_list()[:3]) > 1e-8:
                self.continuous_path([(Waypoint(label+':xy', xy_end), self.setup.transit_speed,
                                       self.setup.transit_acceleration, 0.)], label+':xy', check)
            if rotation_error(self.expected.as_list()[3:], target.as_list()[3:]) > 1e-7:
                self.continuous_path([(Waypoint(label+':rotate', target), self.setup.transit_speed,
                                       self.setup.transit_acceleration, 0.)], label+':rotate', check)
            return
        for i,pose in enumerate(planned,1):
            if check:
                check()
            self.move(pose, f"{label}:{'xy' if i<=n else 'rotate'}:{i if i<=n else i-n}")
            if check:
                check()

    def vertical(self, z, label, check=None, before_segment=None):
        g = self.setup.grasp
        start = self.expected
        if self.continuous:
            from .smooth_dynamics import CONTACT_SPEED_MPS, CONTACT_ACCELERATION_MPS2, BLEND_M
            distance = abs(z-start.z)
            if distance > g.max_total_descent_m:
                raise ValueError('Continuous vertical distance exceeds limit')
            if distance < 1e-8:
                if before_segment: before_segment()
                if check: check()
                return
            contact = z < start.z and z <= max(self.setup.site.pick_contact_z_m, self.setup.site.placement_contact_z_m) + 1e-6
            path = []
            if contact and distance > g.final_slow_distance_m:
                fast = start.with_z(z+g.final_slow_distance_m)
                path.append((Waypoint(label+':fast', fast), self.setup.transit_speed,
                             self.setup.transit_acceleration, min(BLEND_M, g.final_slow_distance_m/3)))
            path.append((Waypoint(label+(':contact' if contact else ':end'), start.with_z(z)),
                         CONTACT_SPEED_MPS if contact else self.setup.transit_speed,
                         CONTACT_ACCELERATION_MPS2 if contact else self.setup.transit_acceleration, 0.))
            self.continuous_path(path, label, check, before_segment)
            return
        if z < start.z:
            steps = compute_descent_z(start.z,z,g.max_z_segment_m,g.final_slow_distance_m)
        else:
            count = max(1,math.ceil((z-start.z)/g.max_z_segment_m))
            steps = [(start.z+(z-start.z)*i/count,False) for i in range(1,count+1)]
        for i,(height,slow) in enumerate(steps):
            if check:
                check()
            if before_segment:
                before_segment()
            speed=g.lift_speed_mps if z>=start.z else (g.contact_speed_mps if slow else g.approach_speed_mps)
            self.move(start.with_z(height),f"{label}:{i}",speed)
            if check:
                check()


class Runtime:
    def __init__(self, setup, camera, robot, gripper, output, *, sleep=time.sleep, clock=time.monotonic):
        self.setup, self.camera, self.robot, self.gripper = setup,camera,robot,gripper
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.session_id=uuid.uuid4().hex
        self.events=[]
        self.last_frame=None
        self.sleep,self.clock=sleep,clock
        self.motion=Motion(setup,robot,self.log)
        self.targets={}
        self.board_reference=None
        self.placed={}
        self.task="basic"
        self.skip_placement_verification = False

    def log(self,event,**detail):
        self.events.append({"event":event,"utc":datetime.now(timezone.utc).isoformat(),**detail})
        (self.output/"events.json").write_text(json.dumps(self.events,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

    def frame(self,label):
        s=self.setup
        offset=self.robot.active_tcp_offset()
        expected=s.raw["active_tcp_offset_m_rad"]
        if math.dist(offset[:3],expected[:3])>0.0001 or rotation_error(offset[3:],expected[3:])>math.radians(0.05):
            raise ValueError("Active TCP changed since calibration")
        if hashlib.sha256(s.calibration_path.read_bytes()).hexdigest()!=s.calibration_sha256:
            raise ValueError("Calibration file changed during run")
        observation=capture_observation(self.camera,self.robot,s.mapper.tcp_T_camera,s.calibration_sha256,
            self.session_id,offset,s.raw["camera_serial"],previous_frame_number=self.last_frame)
        self.last_frame=observation.metadata["camera"]["frame_number"]
        settings=observation.metadata['camera'].get('color_settings')
        if settings is None or settings['enable_auto_exposure'] or settings['enable_auto_white_balance']:
            raise ValueError('Automatic RGB settings changed during run')
        if any(abs(settings[k]-v)>1.0 for k,v in s.raw['camera_settings'].items()):
            raise ValueError('Fixed RGB settings changed during run')
        observation.save(self.output/f"{len(self.events):04d}_{label}_{self.last_frame}")
        self.log("capture",label=label,observation_id=observation.metadata["observation_id"])
        return observation

    def scene(self,label,roi=None,include_colors=None):
        s=self.setup
        thresholds=tuple(t for t in s.vision.colors if include_colors is None or t.name in include_colors)
        # The wide pick view must contain every remaining battery.  At a
        # close hover the camera intentionally sees only a local subset, so
        # that frame is used to validate the selected colour only.
        # At a pick-view frame, the ROI can legitimately contain only the
        # selected/local subset in simulation and after a previous release.
        # Enforce exact color completeness at the grasp hover and verification
        # frames, where the selected object and placed set are safety-critical.
        require_exact=include_colors is not None and label in {'grasp_hover','released','settled'}
        required=max(1,int(s.raw["frames"]))
        # RGB-D depth holes are transient on this USB camera.  Retry at the
        # same TCP pose, but only accept a complete, mutually consistent
        # window; no search motion is initiated by a failed frame.
        accepted=[]
        errors=[]
        for attempt in range(required*3):
            obs=self.frame(label)
            try:
                found=detect_batteries(obs.colour,obs.depth_m,thresholds,obs.base_T_camera,
                    self.camera.normalize_color_pixels,s.raw["paper_z_m"],roi=roi,
                    max_height_mad_m=s.raw["max_height_mad_m"])
                if not found:
                    raise ValueError('No requested battery instances visible')
                if label == 'before_pick' and include_colors is not None:
                    # The pick-view ROI is deliberately wide enough for a
                    # whole initial scene on the real D455, but offline
                    # drivers may crop that view. Require only the selected
                    # color here; grasp_hover performs the strict same-pose
                    # complete-scene gate before descent.
                    found = {name: item for name, item in found.items() if name in include_colors}
                    if not found:
                        raise ValueError('No requested battery instances visible')
                expected={t.name for t in thresholds} if require_exact else set(found)
                if require_exact and set(found)!=expected:
                    missing=sorted(expected-set(found))
                    raise ValueError('Missing battery instances: '+', '.join(missing))
                if require_exact and any(item.handle is None for item in found.values()):
                    missing=next(item.color for item in found.values() if item.handle is None)
                    raise ValueError(f'Battery handle missing: {missing}')
            except (ValueError, RuntimeError) as exc:
                errors.append(str(exc))
                self.log('scene_frame_rejected',label=label,attempt=attempt+1,error=str(exc))
                continue
            accepted.append((obs,found))
            if len(accepted)>=required:
                reference=accepted[-1][1]
                stable=True
                for _,candidate in accepted[:-1]:
                    if set(candidate)!=set(reference):
                        stable=False
                        break
                    for name in set(reference):
                        a,b=candidate[name],reference[name]
                        if (math.dist(a.center_base_m,b.center_base_m)>s.raw['max_center_spread_m']
                            or abs(axis_delta(a.yaw_rad,b.yaw_rad))>math.radians(s.raw['max_yaw_spread_deg'])):
                            stable=False
                            break
                    if not stable: break
                if stable:
                    break
                # Keep a sliding window while continuing to reacquire.
                accepted=accepted[-(required-1):]
        if len(accepted)<required:
            reason=errors[-1] if errors else 'no complete scene frame'
            raise ValueError(f'{label} vision failed after {required*3} attempts: {reason}')
        chosen=accepted[-1][1]
        self.log("scene",label=label,accepted_frames=len(accepted),instances={k:asdict(v) for k,v in chosen.items()})
        return accepted[-1]

    def plane_points(self,obs,pixels,z):
        rays=self.camera.normalize_color_pixels(np.asarray(pixels,float))
        rays=np.column_stack((rays,np.ones(len(rays))))
        transform=obs.base_T_camera
        directions=rays@transform[:3,:3].T
        if np.any(abs(directions[:,2])<1e-8):
            raise ValueError("Target ray parallel to table")
        scales=(z-transform[2,3])/directions[:,2]
        if np.any(scales<=0):
            raise ValueError("Target plane behind camera")
        return transform[:3,3]+directions*scales[:,None]

    def scan_targets(self,task):
        required=max(1,int(self.setup.raw["frames"]))
        accepted=[]
        errors=[]
        # A ring edge can be absent from an individual RGB-D frame while the
        # robot is already stationary. Retry only at this exact pose; a bad
        # frame must never cause a camera search motion.
        first_frame_live=None
        attempt_limit=required*3
        accepted_required=required
        for attempt in range(attempt_limit):
            obs=self.frame("targets")
            if first_frame_live is None:
                first_frame_live=bool(getattr(obs,'live',True))
                if not first_frame_live:
                    # Offline/replay observations cannot provide a fresh
                    # stability window. Keep replay diagnostics deterministic;
                    # live hardware still receives the full retry window.
                    attempt_limit=1
                    accepted_required=1
            view=self.table(obs)
            cv2.imencode('.png',view.image)[1].tofile(self.output/f'target_view_{attempt+1}.png')
            try:
                detected=(detect_t0(view.image),) if task=="basic" else detect_sequence_targets(view.image)
                if not all(d.valid for d in detected):
                    raise ValueError("Target geometry missing/ambiguous: "+str([d.reason for d in detected]))
                targets={}
                for d in detected:
                    points=view.to_base(d.outer_points_px)
                    center=view.to_base([d.center_px])[0]
                    target_pose=Pose(*center,self.setup.site.safe_high_z_m,0,0,0)
                    if not self.setup.in_workspace(target_pose):
                        bounds=self.setup.site.workspace_bounds_m
                        raise ValueError(
                            "Target outside measured workspace: "
                            f"{d.label} center=({center[0]:.4f}, {center[1]:.4f}) m, "
                            f"safe_z={self.setup.site.safe_high_z_m:.4f} m, "
                            f"bounds=({bounds.x_min:.4f}..{bounds.x_max:.4f}, "
                            f"{bounds.y_min:.4f}..{bounds.y_max:.4f}, "
                            f"{bounds.z_min:.4f}..{bounds.z_max:.4f})"
                        )
                    edges=np.roll(points,-1,axis=0)-points
                    lengths=np.linalg.norm(edges,axis=1)
                    expected=(0.1,0.2) if task=="basic" else (0.15,0.15)
                    if any(abs(a-b)>0.009 for a,b in zip((min(lengths),max(lengths)),expected)):
                        raise ValueError("Printed target metric dimensions mismatch")
                    edge=edges[np.argmax(np.linalg.norm(edges,axis=1))]
                    targets[d.label]={"center":center.tolist(),"points":points.tolist(),
                        "yaw":math.atan2(edge[1],edge[0]),"kind":d.kind.value}
            except (ValueError, RuntimeError) as exc:
                errors.append(str(exc))
                self.log('target_frame_rejected',attempt=attempt+1,error=str(exc))
                if first_frame_live is False:
                    break
                continue
            accepted.append((targets,view))
            if len(accepted)>=accepted_required:
                reference=accepted[-1][0]
                stable=True
                for candidate,_ in accepted[:-1]:
                    if set(candidate)!=set(reference):
                        stable=False
                        break
                    for key in reference:
                        if (math.dist(candidate[key]["center"],reference[key]["center"])>self.setup.raw["max_center_spread_m"]
                            or (task=="basic" and abs(axis_delta(candidate[key]["yaw"],reference[key]["yaw"]))>math.radians(2))):
                            stable=False
                            break
                    if not stable:
                        break
                if stable:
                    break
                accepted=accepted[-(required-1):]
        if len(accepted)<accepted_required:
            reason=errors[-1] if errors else 'no complete target frame'
            raise ValueError(f'Target vision failed after {attempt_limit} attempts: {reason}')
        if self.board_reference is None:
            self.board_reference=accepted[-1][1]
            cv2.imencode('.png',self.board_reference.image)[1].tofile(self.output/'table_reference.png')
        self.targets=accepted[-1][0]
        self.log("targets",targets=self.targets)

    def table(self,obs):
        if self.task not in OFFICIAL_TASK_COLORS:
            raise ValueError("Runtime task is not selected")
        return rectify_table(obs.colour,obs.base_T_camera,self.camera.normalize_color_pixels,
            self.camera.project_color_points,self.setup.raw["paper_z_m"],self.setup.target_roi(self.task),self.board_reference)

    def verify_board(self):
        obs=self.frame("board_check")
        inliers=require_stationary_board(self.board_reference,self.table(obs))
        self.log("board_stationary",inliers=inliers)

    def grip_check(self):
        if self.gripper.has_object() is not True:
            raise RuntimeError("Grasp missing/unknown or object dropped")
        pos=self.gripper.position()
        lo,hi=self.setup.raw["grip_position_range"]
        if not lo<=pos<=hi:
            raise RuntimeError(f"Gripped width differs from calibrated handle: {pos}")

    def handle_yaw(self,instance):
        if instance.handle is None:
            raise ValueError(f"Battery handle missing: {instance.color}")
        return float(instance.handle['yaw_rad'])

    def orientation(self,handle_yaw):
        delta=axis_delta(handle_yaw,self.setup.raw["grasp_reference_handle_yaw_rad"])
        reference=cv2.Rodrigues(np.asarray(self.setup.site.tool_rotation_vector_rad))[0]
        rotation=cv2.Rodrigues(np.array([0.,0.,delta]))[0]@reference
        return tuple(cv2.Rodrigues(rotation)[0].reshape(3))

    def grip_xy(self,instance):
        self.handle_yaw(instance)
        return tuple(map(float,instance.handle['center_base_m'][:2]))

    def placement_xy(self,instance,target,task,place_yaw):
        center=np.asarray(target["center"],float)
        if task=="sequence":
            # Sequence placement preserves the observed handle axis. Carry the
            # same-frame handle-to-body offset so the BODY lands on the circle.
            handle=np.asarray(instance.handle['center_base_m'][:2],float)
            body=np.asarray(instance.center_base_m[:2],float)
            return tuple(map(float,center+handle-body))
        dx,dy=self.setup.raw["grasp_center_offset_xy_m"]
        return (center[0]+math.cos(place_yaw)*dx-math.sin(place_yaw)*dy,
                center[1]+math.sin(place_yaw)*dx+math.cos(place_yaw)*dy)

    def validate_planned_xy(self, label, xy, z):
        """Reject an out-of-envelope vision point before issuing motion."""
        pose = Pose(float(xy[0]), float(xy[1]), float(z), *self.motion.expected.as_list()[3:])
        if not self.setup.in_workspace(pose):
            raise ValueError(f"Vision point outside measured workspace: {label} ({pose.x:.4f}, {pose.y:.4f})")
        if not self.robot.pose_within_safety_limits(pose.as_list()):
            raise ValueError(f"Controller safety limits reject planned point: {label} ({pose.x:.4f}, {pose.y:.4f})")
        return pose

    def validate_target_path(self, label, target):
        """Validate a target-derived path before the first transfer segment."""
        center = target.get('center') if isinstance(target, dict) else None
        if center is None or len(center) != 2 or not np.isfinite(center).all():
            raise ValueError(f'Invalid target geometry: {label}')
        self.validate_planned_xy(label, center, self.setup.transit_high_z)

    def inside_target(self,instance,target):
        center=np.asarray(instance.center_base_m[:2])
        if math.dist(center,target["center"])>self.setup.raw["max_scene_shift_m"]:
            return False
        yaw=self.handle_yaw(instance)
        if target["kind"]!="p_ring" and abs(axis_delta(yaw,target["yaw"]))>math.radians(self.setup.raw["max_yaw_spread_deg"]):
            return False
        nominal=sorted(self.setup.raw["body_size_m"])
        measured=sorted(instance.size_m)
        short,long=[max(a,b) for a,b in zip(nominal,measured)]
        u=np.array([math.cos(yaw),math.sin(yaw)])
        v=np.array([-u[1],u[0]])
        corners=np.array([center+a*long/2*u+b*short/2*v for a,b in ((-1,-1),(-1,1),(1,1),(1,-1))])
        margin=self.setup.raw["target_containment_margin_m"]
        if target["kind"]=="p_ring":
            # Official outer circle: 150 mm diameter. Inner-circle scoring is
            # deliberately not claimed; 70x80mm footprint cannot fit D=80mm.
            return bool(np.all(np.linalg.norm(corners-np.array(target["center"]),axis=1)<=0.075-margin))
        poly=np.asarray(target["points"],np.float32)
        return all(cv2.pointPolygonTest(poly,tuple(map(float,p)),True)>=margin for p in corners)

    def run(self,task,colors):
        self.setup.require_ready(task, site_test=getattr(self.setup,'site_test',False))
        if task not in ("basic","sequence") or len(colors)!=(1 if task=="basic" else 3) or len(set(colors))!=len(colors) or not set(colors)<=COLORS:
            raise ValueError("Invalid task colors")
        s=self.setup
        self.task=task
        self.motion.continuous = task == 'sequence'
        skip_verification = self.skip_placement_verification
        if skip_verification and task != 'sequence':
            raise ValueError('Skipping placement verification is only available for sequence')
        last_release_deadline = None
        task_started=None
        battery_times=[]
        try:
            assert_pose_close(self.robot.snapshot().tcp_pose,s.home,0.001,math.radians(0.5),"start home")
            if not self.gripper.initialize(recalibrate=False) or not self.gripper.open():
                raise RuntimeError("Gripper initialization/open failed")
            self.sleep(s.grasp.grip_settle_s)
            if self.gripper.has_object() is not False:
                raise RuntimeError("Initial gripper is not empty/open")
            self.log('competition_start_checked',**require_competition_start(Pose(*self.robot.snapshot().tcp_pose)))
            task_started=self.clock()
            self.log('cycle_started',task=task,colors=colors)
            from .folded_standby import plan_for, transition
            folded=plan_for(s)
            if folded:
                self.motion.expected=transition(s,self.robot,'observation',self.log)
                if getattr(self.camera,'_deferred_competition_start',False):
                    self.camera.start(serial=s.raw['camera_serial'])
                    self.camera.configure_color(s.raw['camera_settings'])
            else:self.motion.high(s.pick_view,"start_observation_pose")
            self.log('target_view_plan',task=task,start_pose=self.motion.expected.as_list(),
                     target_pose=s.target_view(task).as_list(),search_motion=False)
            self.motion.high(s.target_view(task),"target_view")
            self.scan_targets(task)
            for index,color in enumerate(colors):
                target=self.targets["T0" if task=="basic" else f"P{index+1}"]
                self.validate_target_path(f'{color}:target', target)
                battery_started=self.clock()
                self.motion.high(s.pick_view,"pick_view")
                remaining=set(colors)-set(colors[:index])
                # Restrict the initial scan to the official colors for this task.
                # Unrelated colored objects near the expanded pick ROI must not
                # create duplicate-instance failures (for example red clutter
                # beside the yellow/green/blue battery group in sequence mode).
                obs,scene=self.scene("before_pick",s.raw["pick_roi"],include_colors=remaining)
                if color not in scene:
                    raise ValueError(f"Selected battery not visible in initial zone: {color}")
                item=scene[color]
                obs.require_live(self.session_id,s.calibration_sha256,s.raw['max_observation_age_s'])
                # Camera and TCP observe the current scene again at grasp hover.
                pickxy=self.grip_xy(item)
                self.validate_planned_xy(f'{color}:pick', pickxy, s.transit_high_z)
                item_yaw=self.handle_yaw(item)
                orient=self.orientation(item_yaw)
                # The close transit height is deliberately below the wide
                # camera ROI.  Reacquire the RGB-D frame at the current wide
                # pick view before lowering; the object cannot move while the
                # robot executes the subsequent bounded path.
                fresh,new_scene=self.scene("grasp_hover",s.raw["pick_roi"],include_colors=remaining)
                if color not in new_scene:
                    raise ValueError("Selected battery not visible at hover")
                fresh_item=new_scene[color]
                if math.dist(item.center_base_m,fresh_item.center_base_m)>s.raw["max_scene_shift_m"] or abs(axis_delta(item_yaw,self.handle_yaw(fresh_item)))>math.radians(s.raw["max_yaw_spread_deg"]):
                    raise ValueError("Battery shifted before grasp")
                for other in (*scene.values(),*new_scene.values()):
                    if other.color!=color and math.dist(pickxy,other.center_base_m[:2])<s.raw["min_pick_separation_m"]:
                        raise ValueError("Insufficient separation from non-target battery")
                # The left edge of the real table is outside the controller's
                # reachable envelope at the camera height.  Lowering to the
                # measured transit height preserves ample tool clearance while
                # keeping the XY hover inside the UR safety envelope.
                if abs(self.motion.expected.z-s.transit_high_z)>1e-6:
                    self.motion.vertical(s.transit_high_z,f"{color}:hover_lower")
                hover=Pose(*pickxy,s.transit_high_z,*orient)
                self.motion.high(hover,f"{color}:hover")
                def fresh_gate():
                    fresh.require_live(self.session_id,s.calibration_sha256,s.raw["max_observation_age_s"])
                self.motion.vertical(s.site.pick_contact_z_m,f"{color}:pick",before_segment=fresh_gate)
                if not self.gripper.close():
                    raise RuntimeError("Gripper close failed")
                self.sleep(s.grasp.grip_settle_s)
                self.grip_check()
                self.motion.vertical(s.transit_high_z,f"{color}:lift",check=self.grip_check)
                self.sleep(max(2.0,s.site.timing.lifted_hold_s))
                self.grip_check()
                place_yaw=target["yaw"] if task=="basic" else item_yaw
                place_xy=self.placement_xy(fresh_item,target,task,place_yaw)
                self.validate_planned_xy(f'{color}:place', place_xy, s.transit_high_z)
                high=Pose(*place_xy,s.transit_high_z,*self.orientation(place_yaw))
                self.motion.high(high,f"{color}:place_high",check=self.grip_check)
                self.grip_check()
                self.motion.vertical(s.site.placement_contact_z_m,f"{color}:place",check=self.grip_check)
                if not self.gripper.open():
                    raise RuntimeError("Gripper release failed")
                self.sleep(s.grasp.grip_settle_s)
                if self.gripper.has_object() is not False:
                    raise RuntimeError("Release feedback missing")
                released_at=self.clock()
                self.log("released",color=color,target="T0" if task=="basic" else f"P{index+1}",verification_skipped=skip_verification)
                self.motion.vertical(s.transit_high_z,f"{color}:retreat")
                if skip_verification:
                    # The next pick still needs a fresh view, but there is no
                    # return to the target camera, containment test or photo
                    # after placing. Release settling overlaps the next pick.
                    corridor=Pose(s.pick_view.x,s.pick_view.y,s.transit_high_z,
                                  *self.motion.expected.as_list()[3:])
                    self.motion.high(corridor,f'{color}:next_pick_corridor')
                    self.motion.vertical(s.site.safe_high_z_m,f'{color}:next_pick_raise')
                    self.motion.high(s.pick_view,f'{color}:next_pick_view')
                    self.placed[color]=target
                    last_release_deadline=released_at+max(3.,s.site.timing.release_settle_s)
                    battery_time={'color':color,'target':f'P{index+1}',
                                  'run_time_s':self.clock()-battery_started,
                                  'placement_verified':False}
                    battery_times.append(battery_time)
                    self.log('placement_verification_skipped',color=color,target=f'P{index+1}')
                    self.log('battery_complete',**battery_time)
                    continue
                # Edge targets are reachable at the verified transit height
                # but not necessarily at the camera height.  Leave the target
                # along the transit plane, raise over the central observation
                # corridor, then rotate/translate to the camera pose.
                corridor=Pose(s.target_view(task).x,s.target_view(task).y,
                              s.transit_high_z,*self.motion.expected.as_list()[3:])
                self.motion.high(corridor,f"{color}:verify_corridor")
                self.motion.vertical(s.site.safe_high_z_m,f"{color}:verify_raise")
                self.motion.high(s.target_view(task),"verify_view")
                self.verify_board()
                # At the verification view restrict RGB-D detection to the
                # target ROI and colours that have actually been placed.  A
                # full-frame colour scan sees unrelated red/yellow objects in
                # the room and would turn those into duplicate instances.
                placed_colors=set(self.placed)|{color}
                _,placed_before=self.scene("released",s.target_roi(task),include_colors=placed_colors)
                self.placed[color]=target
                for placed_color,placed_target in self.placed.items():
                    if placed_color not in placed_before or not self.inside_target(placed_before[placed_color],placed_target):
                        raise ValueError("Battery not wholly within target after release")
                remaining=released_at+max(3.0,s.site.timing.release_settle_s)-self.clock()
                if remaining>0:
                    self.sleep(remaining)
                _,placed_after=self.scene("settled",s.target_roi(task),include_colors=placed_colors)
                for placed_color,placed_target in self.placed.items():
                    if placed_color not in placed_after or not self.inside_target(placed_after[placed_color],placed_target) or math.dist(placed_before[placed_color].center_base_m,placed_after[placed_color].center_base_m)>s.raw["max_center_spread_m"] or abs(axis_delta(self.handle_yaw(placed_before[placed_color]),self.handle_yaw(placed_after[placed_color])))>math.radians(s.raw["max_yaw_spread_deg"]):
                        raise ValueError("Placement not stable after three seconds")
                battery_time={"color":color,"target":"T0" if task=="basic" else f"P{index+1}",
                              "run_time_s":self.clock()-battery_started}
                battery_times.append(battery_time)
                self.log("battery_complete",**battery_time)
            if folded:
                self.motion.high(s.pick_view,'return_to_fold_entry')
                self.motion.expected=transition(s,self.robot,'home',self.log)
            else:self.motion.high(s.home,"return_home")
            require_competition_start(Pose(*self.robot.snapshot().tcp_pose))
            if last_release_deadline is not None:
                remaining=last_release_deadline-self.clock()
                if remaining>0:self.sleep(remaining)
            timing={"task_run_time_s":self.clock()-task_started,"battery_run_times":battery_times,
                    "timing_scope":("first observation motion through fresh vision, commanded placements and home; placement verification skipped"
                                    if skip_verification else "first observation motion through fresh vision, three placements, verification and sideways home")}
            self.log("home_arrived",task_run_time_s=timing['task_run_time_s'])
            summary={"completed":True,"task":task,"colors":colors,"timing":timing,
                     "placement_verified":not skip_verification,
                     "final_pose_m_rad":list(self.robot.snapshot().tcp_pose)}
            (self.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
            print(json.dumps({"TASK_RUN_TIME_S":timing['task_run_time_s'],"battery_run_times":battery_times},ensure_ascii=False),flush=True)
            self.log("complete",task=task,colors=colors,timing=timing)
            return timing
        except BaseException as exc:
            # Communication failure is not evidence of an empty or safe gripper.
            # Stop in place; never release or start an unverified recovery lift.
            stop_errors=[]
            try:
                if not self.robot.stop_l(0.2):
                    stop_errors.append('robot stop not acknowledged')
            except Exception as stop_exc:
                stop_errors.append(f'robot: {stop_exc}')
            try:
                self.gripper.stop()
            except Exception as stop_exc:
                stop_errors.append(f'gripper: {stop_exc}')
            self.log("failed",error=str(exc),stop_errors=stop_errors)
            raise


def resolve_task_colors(task, colors=None):
    """Notice defaults; explicit color overrides remain available for bench tests."""
    chosen=list(OFFICIAL_TASK_COLORS[task] if colors is None else colors)
    if len(chosen)!=(1 if task=='basic' else 3) or len(set(chosen))!=len(chosen) or not set(chosen)<=COLORS:
        raise ValueError('Invalid task colors')
    return chosen


def preflight_realsense(camera, serial):
    """Check the selected RGB-D device before any robot motion.

    The production calibration and ROIs are tied to the USB3 stream profile;
    silently falling back to the USB2 diagnostic profile would invalidate the
    pixel geometry. Test doubles without device discovery deliberately skip
    this hardware-only check.
    """
    discover = getattr(type(camera), "list_devices", None)
    if not callable(discover):
        return
    devices = list(discover())
    matches = [device for device in devices if str(getattr(device, "serial", "")) == serial]
    if not matches:
        found = ", ".join(
            f"{getattr(device, 'serial', '?')} ({getattr(device, 'usb_type', '?')})"
            for device in devices
        ) or "无"
        raise RealSenseError(f"未检测到指定 RealSense {serial}；当前设备: {found}")
    usb_type = str(getattr(matches[0], "usb_type", ""))
    if usb_type and not usb_type.startswith("3"):
        raise RealSenseError(
            f"RealSense {serial} 当前为 USB {usb_type}；当前标定需要 USB3 1280x720/30，"
            "请改用 USB3 端口后再启动"
        )


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("readiness","preflight","tcp-read","collect","analyse","run"),nargs="?",default="readiness")
    parser.add_argument("--config",default=None,help="Default: task-specific closed-loop configuration")
    parser.add_argument("--task",choices=("basic","sequence"),default="basic")
    parser.add_argument("--colors",nargs="+",default=None,
        help="Bench-test override; official defaults: basic=red, sequence=yellow green blue")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--auto-start",action="store_true",help="Start from the clicked console button without an additional terminal phrase")
    parser.add_argument("--site-test",action="store_true",help="Sequence field test before acceptance; retains live geometry, TCP and collision checks")
    parser.add_argument('--skip-placement-verification',action='store_true',help='Sequence only: no photos or containment checks after release')
    parser.add_argument("--output",type=Path)
    parser.add_argument("--bundle",type=Path)
    parser.add_argument("--fixed-color",action="store_true",help="collect with configured fixed RGB settings")
    args=parser.parse_args(argv)
    if args.skip_placement_verification and (args.command!='run' or args.task!='sequence'):
        raise ValueError('--skip-placement-verification requires sequence run')
    if args.auto_start and (args.command!='run' or not args.execute):
        raise ValueError('--auto-start requires run --execute')
    if args.site_test and (args.task!='sequence' or args.command not in ('run','readiness','preflight')):
        raise ValueError('--site-test is only available for sequence run/readiness')
    if args.config is None:
        args.config=str(ROOT/'config'/('closed_loop_sequence.json' if args.task=='sequence' else 'closed_loop.json'))
    args.colors=resolve_task_colors(args.task,args.colors)
    setup=Setup(args.config)
    setup.site_test=args.site_test
    if args.command=='tcp-read':
        if args.output and args.output.exists():
            raise ValueError('TCP report already exists')
        report=read_tcp_info(setup.site.robot_host)
        report['timestamp_utc']=datetime.now(timezone.utc).isoformat()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x',encoding='utf-8') as handle:
                json.dump(report,handle,indent=2,allow_nan=False)
        print(json.dumps(report,indent=2,allow_nan=False))
        return 0
    if args.command == 'preflight':
        # A hardware probe does not certify a mission or set its acceptance
        # flags. In particular the smooth T0 runner measures the handle offset
        # from live frames instead of using closed-loop's fixed offset.
        setup.grasp.require_valid(allow_missing_final_z=False)
        expected = finite_vector(setup.raw['active_tcp_offset_m_rad'], 6, 'active_tcp_offset_m_rad')
        report = {'task': args.task, 'camera_serial': setup.raw['camera_serial'],
                  'config': 'ok', 'camera': 'pending', 'robot': 'pending', 'gripper': 'pending',
                  'mission_blockers': setup.blockers(args.task, site_test=args.site_test),
                  'controller_path_check': 'deferred to live run before each move'}
        camera = RealSenseCamera()
        preflight_realsense(camera, setup.raw['camera_serial'])
        report['camera'] = 'ok'
        robot = RealUR5(setup.robot_config())
        try:
            robot.connect_readonly()
            snap = robot.snapshot()
            report['robot'] = {'ok': True, 'tcp_pose': list(snap.tcp_pose)}
            # getTCPOffset / isPoseWithinSafetyLimits need RTDE Control.
            # Read the controller broadcast instead, without uploading a
            # control script or sending any motion commands.
            tcp_info = read_tcp_info(setup.site.robot_host)
            offset = finite_vector(tcp_info['active_tcp_offset_m_rad'], 6, 'broadcast TCP offset')
            if (math.dist(offset[:3], expected[:3]) > .0001 or
                    rotation_error(offset[3:], expected[3:]) > math.radians(.05)):
                raise ValueError('Active TCP offset differs from hand-eye setup')
            report['tcp_offset_m_rad'] = list(offset)
            report['tcp_offset_source'] = 'primary_readonly_cartesian_info'
        finally:
            robot.disconnect()
        gripper = create_gripper(setup.grasp.gripper)
        try:
            gripper.connect()
            report['gripper'] = {'ok': True, 'position': gripper.position(),
                                 'has_object': gripper.has_object()}
        finally:
            gripper.disconnect()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command=="readiness":
        blockers=setup.blockers(args.task,site_test=args.site_test)
        if not blockers:
            setup.require_ready(args.task,site_test=args.site_test)
        print(json.dumps({"ready":not blockers,"blockers":blockers,
            "task":args.task,"colors":args.colors,"site_test":args.site_test,
            "site_verified":setup.raw['geometry_and_gripper_verified_by_task'][args.task],
            "matches_official_task":tuple(args.colors)==OFFICIAL_TASK_COLORS[args.task],
            "target_assignment":dict(zip(['T0'] if args.task=='basic' else ['P1','P2','P3'],args.colors)),
            "config":str(setup.path),
            "workspace_bounds_m":asdict(setup.site.workspace_bounds_m),
            "confirmed_pick_z_m":setup.grasp.final_grasp_z_m,
            "standby_pose_m_rad":setup.home.as_list(),
            "pick_view_pose_m_rad":setup.raw.get('pick_view_pose_m_rad', setup.raw['target_view_pose_m_rad']),
            "sequence_target_view_pose_m_rad":setup.raw['sequence_target_view_pose_m_rad'],
            "standby_rule":"outside battery zone; gripper longitudinal axis horizontal; observe only after start",
            "target_xy_source":"current RGB + same-frame TCP + table plane"},indent=2,ensure_ascii=False))
        return 0
    if args.command=='analyse':
        if args.bundle is None or setup.raw['paper_z_m'] is None:
            raise ValueError('analyse requires --bundle and measured paper_z_m')
        obs=Observation.load(args.bundle)
        if obs.metadata['calibration_sha256']!=setup.calibration_sha256 or not np.allclose(obs.metadata['tcp_T_camera'],setup.mapper.tcp_T_camera,atol=1e-10):
            raise ValueError('Recorded calibration differs from selected configuration')
        camera=recorded_camera_model(obs.metadata)
        task_colors=set(args.colors)
        thresholds=tuple(t for t in setup.vision.colors if t.name in task_colors)
        found=detect_batteries(obs.colour,obs.depth_m,thresholds,obs.base_T_camera,camera.normalize_color_pixels,setup.raw['paper_z_m'],roi=setup.raw['pick_roi'],max_height_mad_m=setup.raw['max_height_mad_m'])
        view=rectify_table(obs.colour,obs.base_T_camera,camera.normalize_color_pixels,camera.project_color_points,setup.raw['paper_z_m'],setup.target_roi(args.task))
        targets=(detect_t0(view.image),) if args.task=='basic' else detect_sequence_targets(view.image)
        report={'replay_only':True,'instances':{k:asdict(v) for k,v in found.items()},
                'targets':{t.label:{'valid':t.valid,'reason':t.reason,'base_xy_m':view.to_base([t.center_px])[0].tolist() if t.valid else None} for t in targets}}
        if args.output:
            args.output.mkdir(parents=True,exist_ok=False)
            cv2.imencode('.png',view.image)[1].tofile(args.output/'table_view.png')
            annotated=obs.colour.copy()
            x,y,w,h=setup.raw['pick_roi']
            cv2.rectangle(annotated,(x,y),(x+w,y+h),(255,255,255),1)
            def project_base(points):
                points=np.asarray(points,float)
                transform=obs.base_T_camera
                camera_points=(points-transform[:3,3])@transform[:3,:3]
                if np.any(camera_points[:,2]<=0):
                    raise ValueError('Annotation behind camera')
                return np.rint(camera.project_color_points(camera_points)).astype(np.int32)
            for name,item in found.items():
                body=np.column_stack((item.corners_base_xy,np.full(4,item.center_base_m[2])))
                cv2.polylines(annotated,[project_base(body)],True,(255,255,255),2)
                center=tuple(project_base([item.center_base_m])[0])
                cv2.circle(annotated,center,4,(255,255,255),1)
                cv2.putText(annotated,name,(center[0]-25,center[1]-38),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                if item.handle is not None:
                    handle=item.handle
                    corners=np.column_stack((handle['corners_base_xy'],np.full(4,handle['center_base_m'][2])))
                    cv2.polylines(annotated,[project_base(corners)],True,(255,0,255),2)
                    point=tuple(project_base([handle['center_base_m']])[0])
                    cv2.drawMarker(annotated,point,(255,0,255),cv2.MARKER_CROSS,14,2)
            cv2.imencode('.png',annotated)[1].tofile(args.output/'annotated.png')
            (args.output/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(report,indent=2,ensure_ascii=False))
        return 0
    if args.command=="run":
        setup.require_ready(args.task,site_test=args.site_test)  # Fail before factories, connections or writes to devices.
        if not args.execute:
            raise ValueError("run requires explicit --execute")
        if len(args.colors)!=(1 if args.task=="basic" else 3) or len(set(args.colors))!=len(args.colors) or not set(args.colors)<=COLORS:
            raise ValueError("Invalid task colors")
    output=args.output or ROOT/"output/closed_loop"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    if output.exists():
        raise ValueError('Output directory already exists')
    if args.command=='collect' and args.fixed_color and any(v is None for v in setup.raw['camera_settings'].values()):
        raise ValueError('Fill fixed RGB settings before --fixed-color collection')
    robot=(RealUR5(replace(setup.robot_config(),stop_deceleration=1.), motion_profile='t0_timing_trial')
           if args.command=='run' and args.task=='sequence' else RealUR5(setup.robot_config()))
    camera=RealSenseCamera()
    gripper=None
    runtime=None
    try:
        if args.command in ('collect', 'run'):
            # Fail before startup-home control or gripper motion when the
            # calibrated RGB-D device is absent or connected at USB2.
            preflight_realsense(camera, setup.raw['camera_serial'])
        robot.connect_readonly()
        # Collection must remain read-only and usable for recalibration.
        # Saved analyses may belong to a different task or previous scene;
        # live geometry is checked by Runtime on the actual control interface
        # before motion. A receive-only connection cannot query controller IK.
        from .folded_standby import plan_for
        deferred=args.command=='run' and plan_for(setup) is not None
        camera._deferred_competition_start=deferred
        if not deferred:camera.start(serial=setup.raw["camera_serial"])
        if args.command=="collect":
            if args.fixed_color:
                camera.configure_color(setup.raw['camera_settings'])
            # Read controller broadcasts; no control interface or script upload.
            offset=read_tcp_info(setup.site.robot_host)['active_tcp_offset_m_rad']
            expected=setup.raw['active_tcp_offset_m_rad']
            if expected is not None and (math.dist(offset[:3],expected[:3])>0.0001 or
                    rotation_error(offset[3:],expected[3:])>math.radians(.05)):
                raise ValueError('Current TCP offset differs from declared calibration offset')
            obs=capture_observation(camera,robot,setup.mapper.tcp_T_camera,setup.calibration_sha256,
                uuid.uuid4().hex,offset,setup.raw["camera_serial"])
            after_offset=read_tcp_info(setup.site.robot_host)['active_tcp_offset_m_rad']
            if math.dist(offset[:3],after_offset[:3])>0.0001 or rotation_error(offset[3:],after_offset[3:])>math.radians(.05):
                raise ValueError('Active TCP offset changed during collection')
            obs.metadata['active_tcp_offset_source']='primary_readonly_cartesian_info'
            obs.metadata['matches_historical_calibration']=None
            obs=replace(obs,live=False)
            obs.save(output)
            print(json.dumps({'bundle':str(output),'tcp_base_m_rad':obs.metadata['tcp_before'],
                'active_tcp_offset_m_rad':offset,
                'camera_settings':obs.metadata['camera'].get('color_settings')},indent=2))
            return 0
        if not deferred:camera.configure_color(setup.raw["camera_settings"])
        phrase=f"EXECUTE {setup.site.robot_host} {args.task} {' '.join(args.colors)} Z={setup.grasp.final_grasp_z_m:.6f}"
        if tuple(args.colors)!=OFFICIAL_TASK_COLORS[args.task]:
            print('BENCH TEST: color selection differs from the official competition task')
        if not args.auto_start and input(phrase+"\n> ").strip()!=phrase:
            raise ValueError("Start confirmation mismatch")
        gripper=create_gripper(setup.grasp.gripper)
        gripper.connect()
        runtime=Runtime(setup,camera,robot,gripper,output)
        runtime.skip_placement_verification=args.skip_placement_verification
        runtime.log('field_test_started',task=args.task,site_test=args.site_test,
                    site_verified=setup.raw['geometry_and_gripper_verified_by_task'][args.task])
        from .folded_standby import prepare_startup_home
        if not prepare_startup_home(setup,robot,gripper,runtime.log):
            robot.connect_control()
        assert_pose_close(robot.snapshot().tcp_pose,setup.home,0.001,math.radians(0.5),'home after startup return')
        require_competition_start(Pose(*robot.snapshot().tcp_pose))
        actual=robot.active_tcp_offset()
        expected=setup.raw["active_tcp_offset_m_rad"]
        if math.dist(actual[:3],expected[:3])>0.0001 or rotation_error(actual[3:],expected[3:])>math.radians(0.05):
            raise ValueError("Active TCP offset differs from hand-eye setup")
        runtime.run(args.task,args.colors)
        print(f"Complete. Evidence: {output}")
        return 0
    except BaseException as exc:
        if runtime is not None and (not getattr(runtime,'events',[]) or runtime.events[-1].get('event')!='failed'):
            runtime.log('failed',error=str(exc),phase='startup')
        raise
    finally:
        # Cleanup neither opens fingers nor commands a recovery motion.
        try:
            camera.stop()
        finally:
            try:
                robot.disconnect()
            finally:
                if gripper is not None:
                    gripper.disconnect()


if __name__=="__main__":
    raise SystemExit(main())
