"""Read-only incident collection and offline 6-D nominal UR5 trajectory audit.

No RTDE Control, motion, gripper commands, unlocking or safety reset.
Nominal DH and capsule geometry are screening models, NOT collision certification.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battery_sorting.handeye import pose_vector_to_transform
from battery_sorting.robot import rotation_error
from tools.export_smooth_t0_simulation import export_trial

A = [0, -.425, -.3922, 0, 0, 0]
D = [.089159, 0, 0, .10915, .09465, .0823]
ALPHA = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]


def fk(q):
    t = np.eye(4)
    points = [t[:3, 3].copy()]
    for angle, a, d, alpha in zip(q, A, D, ALPHA):
        c, s, ca, sa = math.cos(angle), math.sin(angle), math.cos(alpha), math.sin(alpha)
        t = t@np.array([[c, -s*ca, s*sa, a*c], [s, c*ca, -c*sa, a*s],
                        [0, sa, ca, d], [0, 0, 0, 1]])
        points.append(t[:3, 3].copy())
    return t, np.array(points)


def ik(pose, seed):
    target = pose_vector_to_transform(pose)
    target[:3, 3] -= target[:3, 2]*.246
    q = np.array(seed, float).copy()
    def residual(value):
        t, _ = fk(value)
        return np.r_[t[:3, 3]-target[:3, 3],
                     cv2.Rodrigues(target[:3, :3].T@t[:3, :3])[0].ravel()*.25]
    for _ in range(120):
        error = residual(q)
        if np.linalg.norm(error[:3]) < .0001 and np.linalg.norm(error[3:]) < .00005:
            break
        eps = 1e-5
        jac = np.column_stack([(residual(q+np.eye(6)[i]*eps)-error)/eps for i in range(6)])
        delta = np.linalg.solve(jac.T@jac+np.eye(6)*1e-7, -jac.T@error)
        q += delta*min(1., .15/max(np.max(abs(delta)), 1e-9))
    return q, residual(q)


def segment_distance(a, b, c, d):
    a, b, c, d = map(np.asarray, (a, b, c, d))
    u, v = b-a, d-c
    def point_segment(p, x, delta):
        fraction = np.clip((p-x)@delta/max(delta@delta, 1e-20), 0, 1)
        return np.linalg.norm(p-x-fraction*delta)
    distances = [point_segment(p, c, v) for p in (a, b)] + [point_segment(p, a, u) for p in (c, d)]
    fractions = np.linalg.lstsq(np.column_stack((u, -v)), c-a, rcond=None)[0]
    if np.all(fractions >= 0) and np.all(fractions <= 1):
        distances.append(np.linalg.norm(a+fractions[0]*u-c-fractions[1]*v))
    return float(min(distances))


def interpolate(start, end, fraction):
    first, last = pose_vector_to_transform(start), pose_vector_to_transform(end)
    delta = cv2.Rodrigues(first[:3, :3].T@last[:3, :3])[0].ravel()
    rotation = first[:3, :3]@cv2.Rodrigues(delta*fraction)[0]
    return np.r_[first[:3, 3]+fraction*(last[:3, 3]-first[:3, 3]), cv2.Rodrigues(rotation)[0].ravel()]


def audit_segment(label, start, end, seed, *, initial_q=None):
    distance = float(np.linalg.norm(np.array(end[:3])-start[:3]))
    angle = rotation_error(start[3:], end[3:])
    count = max(2, math.ceil(distance/.005), math.ceil(angle/math.radians(2)))+1
    q = np.array(seed if initial_q is None else initial_q, float).copy()
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('initial joint seed must contain six finite values')
    rows = []
    for sample_index, fraction in enumerate(np.linspace(0, 1, count)):
        pose = interpolate(start, end, fraction)
        previous = q.copy()
        if sample_index == 0 and initial_q is not None:
            # The controller's measured joint state is authoritative at the
            # segment boundary.  Nominal DH IK can differ from a calibrated
            # TCP FK by a small branch/offset; do not replace the measured
            # state before checking the first tool envelope sample.
            q = np.array(initial_q, float).copy()
            _, error = ik(pose, q)
            initial_joint_authoritative = True
        else:
            q, error = ik(pose, q)
            initial_joint_authoritative = False
        flange, points = fk(q)
        normal = flange[:3, 2]
        # Exclude the first 50 mm at the flange mounting interface.
        tool_start, tool_end = flange[:3, 3]+normal*.05, flange[:3, 3]+normal*.246
        distances = {name: segment_distance(tool_start, tool_end, points[a], points[b])
                     for name, a, b in [('upper_arm', 1, 2), ('forearm', 2, 3), ('wrist_1', 3, 4)]}
        nearest = min(distances, key=distances.get)
        rows.append(dict(fraction=float(fraction), tcp_m_rad=pose.tolist(), q_rad=q.tolist(),
                         initial_joint_authoritative=initial_joint_authoritative,
                         ik_position_error_m=float(np.linalg.norm(error[:3])),
                         ik_rotation_error_rad=float(np.linalg.norm(error[3:])/.25),
                         max_joint_step_rad=float(max(abs(q-previous))),
                         nearest_link=nearest, tool_link_centerline_distance_m=distances[nearest],
                         tool_axis_min_z_m=float(min(tool_start[2], tool_end[2])),
                         wrist2_abs_sin=float(abs(math.sin(q[4])))))
    worst = min(rows, key=lambda r: r['tool_link_centerline_distance_m'])
    return dict(label=label, start_pose=start, end_pose=end, translation_m=distance,
                rotation_deg=math.degrees(angle), samples=rows,
                min_tool_link_centerline_distance_m=worst['tool_link_centerline_distance_m'],
                nearest_link_at_minimum=worst['nearest_link'], minimum_at_fraction=worst['fraction'],
                max_ik_position_error_m=max(r['ik_position_error_m'] for r in rows),
                max_ik_rotation_error_rad=max(r['ik_rotation_error_rad'] for r in rows),
                max_joint_step_deg=math.degrees(max(r['max_joint_step_rad'] for r in rows)),
                min_wrist2_abs_sin=min(r['wrist2_abs_sin'] for r in rows)), q


def collect_snapshot(host):
    import rtde_receive
    receiver = rtde_receive.RTDEReceiveInterface(host)
    try:
        return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), host=host,
                    q_rad=receiver.getActualQ(), tcp_m_rad=receiver.getActualTCPPose(),
                    safety_mode=receiver.getSafetyMode(), runtime_state=receiver.getRuntimeState(),
                    emergency_stop=receiver.isEmergencyStopped(), protective_stop=receiver.isProtectiveStopped())
    finally:
        receiver.disconnect()


def run(snapshot, incident, *, motion_config=None, output=None, revised=False, from_snapshot=False):
    out = Path(output or ROOT/'output/trajectory_audit_20261011')
    out.mkdir(parents=True,exist_ok=True)
    actual = json.loads(Path(snapshot).read_text(encoding='utf-8'))
    log = json.loads(Path(incident).read_text(encoding='utf-8'))
    failure = next(e for e in log['events'] if e['event'] == 'startup_return_plan')
    # Reconstruct the nominal branch backwards from today's actual q, not the
    # browser's arbitrary position-only inverse-kinematics seed.
    seed = np.array(actual['q_rad'])
    if from_snapshot:
        initial_q=seed
        pose=actual['tcp_m_rad']
    else:
        lifted = failure['targets'][0]
        for fraction in np.linspace(0, 1, 81):
            seed, _ = ik(interpolate(actual['tcp_m_rad'], lifted, fraction), seed)
        initial_q, _ = ik(failure['actual_start'], seed)
        pose=failure['actual_start']
    segments = []
    q = initial_q
    startup=list(zip(('startup_vertical_clearance', 'startup_return_home'), failure['targets']))
    if revised:
        from battery_sorting.closed_loop import Setup
        from battery_sorting.config import Pose
        from battery_sorting.competition_start import competition_observation_pose
        from tools.run_smooth_t0 import startup_home_targets, startup_move_label
        setup=Setup(motion_config or ROOT/'config/closed_loop.json')
        targets=startup_home_targets(Pose(*pose),setup.home,setup.site.safe_high_z_m,competition_observation_pose(setup))
        previous=Pose(*pose)
        startup=[]
        for target in targets:
            startup.append((startup_move_label(previous,target,setup.home),target.as_list()))
            previous=target
    for label, target in startup:
        segment, q = audit_segment(label, pose, target, q)
        segments.append(segment)
        pose = target
    plan = export_trial(ROOT/'output/smooth_t0/new_room_20260929_111616', ROOT/'config/closed_loop_scene_20260929.json',motion_config=motion_config)
    if np.linalg.norm(np.array(segments[-1]['end_pose'])-plan['initial_pose'])>1e-6:
        raise ValueError('Incident Home differs from current motion configuration: use --revised or an archived --motion-config')
    for step in plan['steps']:
        if step['kind'] == 'move':
            segment, q = audit_segment(step['label'], step['start_pose'], step['end_pose'], q)
            segments.append(segment)
    config = json.loads(Path(motion_config or ROOT/'config/closed_loop.json').read_text(encoding='utf-8'))
    observation = config['target_view_pose_m_rad']
    for label, target in [('verification_observation_pose', observation), ('verification_return_sideways_home', plan['initial_pose'])]:
        segment, q = audit_segment(label, segments[-1]['end_pose'], target, q)
        segments.append(segment)
    flange, _ = fk(actual['q_rad'])
    nominal_tcp = flange.copy()
    nominal_tcp[:3, 3] += nominal_tcp[:3, 2]*.246
    real_tcp = pose_vector_to_transform(actual['tcp_m_rad'])
    report = dict(hardware_motion_sent=False, incident_path=str(incident), incident_error=log.get('startup_home_error'),
                  reported_interference='User reported gripper contacting robot wrist/link during startup return',
                  suspect_home_pose_m_rad=log['start_pose_m_rad'], audited_home_pose_m_rad=plan['initial_pose'],
                  revised_trajectory=revised, actual_collision_cause_certified=False,
                  startup_from_cached_snapshot=from_snapshot,
                  actual_joint_motion_recorded=False,
                  actual_snapshot=actual,
                  model='Nominal UR5 DH from web_sim/app.js; 6-D branch-continuous numerical IK. TCP=246 mm.',
                  model_at_actual_position_error_m=float(np.linalg.norm(real_tcp[:3, 3]-nominal_tcp[:3, 3])),
                  model_at_actual_rotation_error_deg=math.degrees(float(np.linalg.norm(cv2.Rodrigues(real_tcp[:3, :3].T@nominal_tcp[:3, :3])[0]))),
                  limitations=['No joint samples in incident log: historical controller IK branch is unknown.',
                               'Future pick/place inputs come from Sep29 recording, not the current scene.',
                               'Nominal DH is not controller-calibrated; capsule centerline gaps are screening only.',
                               'Tool mount, open jaws, camera and all robot surface meshes are not available.',
                               'Browser solves only TCP position and rotates tool mesh independently: cannot validate self-collision.',
                               'blend corner geometry is approximated as individual segments.'],
                  total_samples=sum(len(s['samples']) for s in segments), segments=segments)
    (out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# 第一题轨迹遍历：事故后离线检查'+('（修改后的轨迹）' if revised else ''), '',
             '真实事故发生在启动前 startup_return_home：先抬升，再同时平移并转腕。尚未开始夹取。', '',
             '下面是名义运动学筛查，不是整机碰撞验收。事故日志没有关节轨迹，不能重建真实分支；后续抓放复用旧场景坐标。', '',
             '| 段 | 平移 mm | 转动 ° | 最小工具-连杆轴线距 mm | 最近连杆 | IK最大误差 mm |',
             '| --- | ---: | ---: | ---: | --- | ---: |']
    for s in segments:
        lines.append(f"| {s['label']} | {s['translation_m']*1000:.1f} | {s['rotation_deg']:.1f} | {s['min_tool_link_centerline_distance_m']*1000:.1f} | {s['nearest_link_at_minimum']} | {s['max_ik_position_error_m']*1000:.2f} |")
    lines += ['', f"遍历 {report['total_samples']} 个位姿；名义模型与当前实机位姿差 {report['model_at_actual_position_error_m']*1000:.2f} mm、{report['model_at_actual_rotation_error_deg']:.2f}°。", '',
              '轴线距离没有扣除真实外壳、安装座、手指和相机包络，因此不能作为净空。离线 IK 误差或关节跳变处不能用于实机执行。', '',
              '当前网页动画只求位置逆解，工具姿态另行旋转，不能据此判断夹爪是否碰腕部或连杆。']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'segments'}, ensure_ascii=False, indent=2))
    for s in segments:
        print(s['label'], 'distance_mm', round(s['translation_m']*1000, 1), 'angle_deg', round(s['rotation_deg'], 1),
              'nearest_mm', round(s['min_tool_link_centerline_distance_m']*1000, 1), 'ik_mm', round(s['max_ik_position_error_m']*1000, 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--read-current', action='store_true')
    parser.add_argument('--snapshot', type=Path, default=ROOT/'output/trajectory_audit_20261011/current_snapshot.json')
    parser.add_argument('--incident', type=Path, default=ROOT/'output/smooth_t0/20261011_155213/audit.json')
    parser.add_argument('--motion-config',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--revised',action='store_true',help='Audit current startup planner instead of the recorded unsafe return')
    parser.add_argument('--from-snapshot',action='store_true',help='Use cached TCP/q directly as the startup seed')
    args = parser.parse_args()
    if args.read_current:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(collect_snapshot('192.168.1.10'), indent=2), encoding='utf-8')
    run(args.snapshot,args.incident,motion_config=args.motion_config,output=args.output,revised=args.revised,from_snapshot=args.from_snapshot)
