"""One-shot paper-contact retreat and high return; never operates the gripper."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from battery_sorting.config import Pose, WorkspaceBounds
from battery_sorting.grasp_lift import GraspLiftConfig
from battery_sorting.robot import RealUR5, Waypoint, assert_pose_close, rotation_error
from battery_sorting.tcp_readonly import read_tcp_info

ROOT = Path(__file__).resolve().parents[1]


def plan(start, home):
    if not .010 < start.z < .025 or home.z < .44:
        raise ValueError('Unexpected recorded contact or home height')
    result = []
    z = start.z
    while z < home.z - 1e-10:
        z = min(home.z, z + (.005 if not result else .025))
        result.append(start.with_z(z))
    count = max(1, math.ceil(math.dist(start.as_list()[:2], home.as_list()[:2])/.025))
    for i in range(1, count+1):
        result.append(Pose(start.x+(home.x-start.x)*i/count,
                           start.y+(home.y-start.y)*i/count, home.z,
                           start.rx, start.ry, start.rz))
    first = cv2.Rodrigues(np.array(start.as_list()[3:]))[0]
    last = cv2.Rodrigues(np.array(home.as_list()[3:]))[0]
    delta = cv2.Rodrigues(first.T @ last)[0].reshape(3)
    count = max(1, math.ceil(float(np.linalg.norm(delta))/math.radians(5)))
    for i in range(1, count+1):
        vec = cv2.Rodrigues(first @ cv2.Rodrigues(delta*i/count)[0])[0].reshape(3)
        result.append(Pose(home.x, home.y, home.z, *map(float, vec)))
    assert_pose_close(result[-1].as_list(), home, 1e-9, 1e-6, 'Planned home')
    previous = start
    for target in result:
        if math.dist(previous.as_list()[:3], target.as_list()[:3]) > .025001:
            raise ValueError('Oversized translation')
        if rotation_error(previous.as_list()[3:], target.as_list()[3:]) > math.radians(5.001):
            raise ValueError('Oversized rotation')
        if target.z < home.z-1e-9 and target.as_list()[:2] != start.as_list()[:2]:
            raise ValueError('Lateral motion below clearance')
        previous = target
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Audit already exists')
    cfg = GraspLiftConfig.from_json(ROOT/'config/grasp_lift.json')
    reference = json.loads((ROOT/'output/site_20260915_initial/paper_contact_tcp.json').read_text())
    home = Pose(*json.loads((ROOT/'config/competition_template.json').read_text())['geometry']['home_pose_m_rad'])
    start = Pose(*reference['tcp_base_m_rad'])
    targets = plan(start, home)
    audit = dict(start=start.as_list(), home=home.as_list(), targets=[p.as_list() for p in targets],
                 completed_segments=[], completed=False, executed=False,
                 authorization='User requested read paper-contact Base Z then return home',
                 force_guard='Disabled per explicit user instruction; controller safety retained')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    save()
    print(f'Plan: {len(targets)} segments, contact Z={start.z:.6f}, clearance={home.z:.6f}', flush=True)
    if not args.execute:
        return
    assert_pose_close(read_tcp_info(cfg.robot_host)['tcp_base_m_rad'], start, .001, math.radians(.5), 'Recorded contact')
    rc = replace(cfg.robot_config(home), workspace_bounds=WorkspaceBounds(
        min(start.x,home.x)-.002,max(start.x,home.x)+.002,
        min(start.y,home.y)-.002,max(start.y,home.y)+.002,start.z-.001,home.z+.002))
    robot = RealUR5(rc)
    previous = start
    try:
        robot.connect_readonly()
        assert_pose_close(robot.snapshot().tcp_pose,start,.001,math.radians(.5),'Before control')
        before = Pose(*robot.snapshot().tcp_pose)
        robot.connect_control()
        assert_pose_close(robot.snapshot().tcp_pose,before,.0002,math.radians(.1),'Control handover')
        for index,target in enumerate(targets):
            offset = robot.active_tcp_offset()
            if math.dist(offset[:3],(0,0,.246))>.0001 or rotation_error(offset[3:],(0,0,0))>math.radians(.05):
                raise ValueError('TCP offset changed')
            assert_pose_close(robot.snapshot().tcp_pose,previous,.001 if index==0 else .0002,math.radians(.5 if index==0 else .1),'Segment start')
            audit['executed'] = True
            robot.move_l_without_tcp_force_guard(Waypoint(f'paper_return_{index+1}',target,'approach'),.005 if index==0 else .015,.05)
            actual = robot.snapshot().tcp_pose
            assert_pose_close(actual,target,.0002,math.radians(.1),'Segment arrival')
            audit['completed_segments'].append(dict(target=target.as_list(),actual=list(actual)))
            save()
            print(f'{index+1}/{len(targets)} XYZ={actual[:3]}',flush=True)
            previous = target
        audit['completed'] = True
    except BaseException as exc:
        audit['error'] = str(exc)
        try:
            robot.stop_l(.2)
        except Exception as stop_exc:
            audit['stop_error'] = str(stop_exc)
        raise
    finally:
        save()
        robot.disconnect()


if __name__ == '__main__':
    main()
