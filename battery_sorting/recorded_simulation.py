"""Export recorded smooth-T0 evidence for browser playback; no hardware I/O."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .closed_loop import Setup
from .observation import Observation, recorded_camera_model
from .placement_vision import detect_t0
from .scene import detect_batteries
from .simulation_export import _quat, _vector, write_plan
from .table_view import rectify_table


def _translation_seconds(start, end, speed, acceleration):
    distance = math.dist(start[:3], end[:3])
    return (distance / speed + speed / acceleration if distance >= speed**2 / acceleration
            else 2 * math.sqrt(distance / acceleration))


def export_recorded_t0(directory, config, *, allow_mechanical_complete=False):
    directory = Path(directory)
    audit_path = directory / 'audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    post_status_path = directory / 'post_run_status.json'
    post_status = (json.loads(post_status_path.read_text(encoding='utf-8'))
                   if post_status_path.exists() else {})
    mechanically_complete = (allow_mechanical_complete
                             and post_status.get('mechanical_cycle_completed') is True)
    if audit.get('executed') is not True or not (audit.get('completed') is True or mechanically_complete):
        raise ValueError('Only a completed, executed recording can be exported')
    setup = Setup(config)
    initial_paths = sorted(directory.glob('initial_*/manifest.json'),
                           key=lambda p: int(p.parent.name.split('_')[-1]))
    if not initial_paths:
        raise ValueError('Initial RGB-D evidence is missing')
    observation = Observation.load(initial_paths[-1].parent)
    if observation.metadata['calibration_sha256'] != setup.calibration_sha256:
        raise ValueError('Recorded calibration differs from the selected configuration')
    camera = recorded_camera_model(observation.metadata)
    instances = detect_batteries(observation.colour, observation.depth_m, setup.vision.colors,
        observation.base_T_camera, camera.normalize_color_pixels, setup.raw['paper_z_m'],
        roi=setup.raw['pick_roi'], max_height_mad_m=setup.raw['max_height_mad_m'])
    if set(instances) != {'red', 'yellow', 'blue', 'green'}:
        raise ValueError('Recording must show the four initial batteries')
    view = rectify_table(observation.colour, observation.base_T_camera,
        camera.normalize_color_pixels, camera.project_color_points,
        setup.raw['paper_z_m'], setup.raw['target_roi'])
    target = detect_t0(view.image)
    if not target.valid:
        raise ValueError('Initial T0 evidence is invalid: ' + target.reason)
    corners = view.to_base(target.outer_points_px)
    target_xy = view.to_base([target.center_px])[0]
    origin = datetime.fromisoformat(audit['timestamp_utc'])
    initial_pose = _vector(observation.metadata['tcp_before'])
    current = initial_pose.copy()
    elapsed = 0.
    steps = []

    def append(kind, label, duration, end=None, speed=None, acceleration=None,
               object_id=None, target_id=None, **extra):
        nonlocal current, elapsed
        if not math.isfinite(duration) or duration < -1e-6:
            raise ValueError('Recording timestamps are out of order: ' + label)
        end = current.copy() if end is None else _vector(end)
        duration = max(0., duration)
        steps.append(dict(kind=kind, label=label, start_pose=current.copy(), end_pose=end,
            duration_s=duration, profile='recorded' if kind == 'move' else None,
            speed_mps=speed, acceleration_mps2=acceleration,
            object_id=object_id, target_id=target_id, **extra))
        current = end
        elapsed += duration

    names = {'normalize_high_z': '高位微调', 'home_to_red_hover': 'home → 红色电池上方',
             'red_lift': '提升红色电池', 'red_to_t0': '搬运至 T0 上方',
             't0_retreat': '释放后竖直撤离', 'return_home': '返回 home'}
    for event in audit['events']:
        at = (datetime.fromisoformat(event['utc']) - origin).total_seconds()
        duration = at - elapsed
        kind = event['event']
        if kind == 'pick_full_open':
            append('dwell', '设备准备、实时识别、夹爪全开（记录时间）', duration)
        elif kind == 'move':
            append('move', names.get(event['label'], event['label']), duration,
                   event['pose'], event['speed'], event['acceleration'],
                   recorded_event_utc=event['utc'])
        elif kind == 'smooth_path':
            poses = [_vector(p) for p in event['poses']]
            accelerations = event.get('accelerations',
                                      [audit['motion_profile']['acceleration_mps2']] * len(poses))
            starts = [current.copy(), *poses[:-1]]
            weights = [_translation_seconds(a, b, v, acc) for a, b, v, acc in
                       zip(starts, poses, event['speeds'], accelerations)]
            if sum(weights) <= 0:
                raise ValueError('Empty recorded motion path')
            label = '下降抓取' if event['label'] == 'red_pick' else '下降放置'
            for i, (pose, speed, acceleration, weight) in enumerate(
                    zip(poses, event['speeds'], accelerations, weights)):
                append('move', f'{label} · {"接近" if i == 0 else "慢速接触"}',
                    duration * weight / sum(weights), pose, speed, acceleration,
                    recorded_event_utc=event['utc'], reconstructed_path=event['label'],
                    blend_radius_m=event['blend_m'], subsegment_time_is_estimated=True)
        elif kind == 'grasped':
            append('close', '闭合、稳定等待、确认抓住红色电池', duration, object_id='red')
            append('check_object', '持物反馈通过', 0., object_id='red')
        elif kind == 'released':
            # The historical runner logged release only AFTER its fixed wait.
            # The boundary is inferred from that wait, not an encoder sample.
            settle = max(3., setup.site.timing.release_settle_s)
            if duration < settle:
                if not mechanically_complete:
                    raise ValueError('Release interval cannot contain the historical stable wait')
                # Current runner logs release immediately. Its remaining stable
                # interval overlaps the direct return and is represented by the
                # following recorded events.
                append('open', '松开红色电池', duration,
                       object_id='red', target_id='T0')
            else:
                append('open', '松开红色电池（释放时点按固定等待推算）', duration - settle,
                       object_id='red', target_id='T0', release_time_is_inferred=True)
                append('dwell', '释放后稳定等待', settle)

    verification = Observation.load(directory / 'verification')
    capture_at = (datetime.fromisoformat(verification.metadata['timestamp_utc']) - origin).total_seconds()
    verification_passed = audit.get('completed') is True
    append('visual_accept', ('回 home 后采集复检帧 · 入框验收通过' if verification_passed
                             else '回 home 后采集复检帧 · 线缆遮挡导致严格复检未通过'),
           capture_at - elapsed,
           object_id='red', target_id='T0')
    paper_z = setup.raw['paper_z_m']
    return dict(schema='ur5-simulation', version=1, frame='ur_base',
        units=dict(position='m', rotation='rad'), initial_pose=initial_pose,
        initial_gripper_open=True, scene=dict(table_z_m=paper_z),
        source=dict(planner='tools.run_smooth_t0', label='真实运行 · 红色 → T0 → home · ' + origin.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M'),
            data_kind='recorded', timing='recorded_event_intervals',
            audit_path=audit_path.as_posix(), audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            initial_observation_sha256=observation.content_digest(),
            verification=audit.get('verification'),
            actual_final_pose=_vector(audit.get('final_pose', setup.home.as_list())),
            mechanical_cycle_completed=(audit.get('completed') is True or mechanically_complete),
            strict_visual_acceptance=verification_passed,
            timing_note='阶段总时长来自日志；连续下降内部时长按运动模型分配，释放时点按固定等待推算。缺少连续 TCP 和关节采样，段内轨迹为插值重建。',
            geometry='四色 XY/把手方向来自记录的 RGB-D；高度及外形采用 70×80×80 mm CAD 示意，不是物块三维实测。',
            captured_frame_offset_s=capture_at),
        objects=[dict(id=color, color=color, center=[*item.center_base_m[:2], paper_z + .04],
                      # The nearly square body can swap its longest edge by 90°.
                      # CAD handle axis is local X: use the narrow handle axis.
                      quaternion=_quat([0., 0., (item.handle['yaw_rad'] + math.pi/2) % math.pi - math.pi/2]),
                      size_m=[.07, .08, .08]) for color, item in instances.items()],
        targets=[dict(id='T0', pose=[*map(float, target_xy), paper_z, 0., 0., 0.],
                      outline_base_xy=np.asarray(corners).tolist())], steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/closed_loop.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = export_recorded_t0(args.recording, args.config)
    write_plan(plan, args.out)
    print(json.dumps({'output': str(args.out), 'steps': len(plan['steps']),
                      'recorded_interval_s': sum(s['duration_s'] for s in plan['steps']),
                      'no_hardware_connected': True}, indent=2))


if __name__ == '__main__':
    main()
