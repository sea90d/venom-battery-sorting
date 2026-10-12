"""One explicitly authorized empty-tool vertical clearance; no gripper I/O."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from battery_sorting.config import Pose, WorkspaceBounds
from battery_sorting.grasp_lift import GraspLiftConfig
from battery_sorting.robot import RealUR5, Waypoint, assert_pose_close, rotation_error
from battery_sorting.tcp_readonly import read_tcp_info

ROOT = Path(__file__).resolve().parents[1]
TARGET_Z = .447856


def plan(start):
    if abs(start.z-.069207)>.002:
        raise ValueError('Start is outside the user-confirmed low pose')
    previous=start
    result=[]
    z=start.z
    while z<TARGET_Z-1e-10:
        z=min(TARGET_Z,z+(.01 if not result else .025))
        target=start.with_z(z)
        assert 0<target.z-previous.z<=.025+1e-10
        assert target.as_list()[:2]==start.as_list()[:2]
        assert target.as_list()[3:]==start.as_list()[3:]
        result.append(target)
        previous=target
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('Audit path already exists')
    cfg=GraspLiftConfig.from_json(ROOT/'config/grasp_lift.json')
    reference=json.loads((ROOT/'output/site_20260915_initial/tcp.json').read_text())
    start=Pose(*read_tcp_info(cfg.robot_host)['tcp_base_m_rad'])
    assert_pose_close(start.as_list(),Pose(*reference['tcp_base_m_rad']),.001,math.radians(.5),'Authorized start')
    targets=plan(start)
    audit={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'start':start.as_list(),
           'targets':[p.as_list() for p in targets], 'executed':False,'completed':False,
           'authorization':'User confirmed empty tool, no contact, unobstructed Base +Z and free cables',
           'completed_segments':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    save()
    print(f'UP ONLY: {start.z:.6f} -> {TARGET_Z:.6f} m, {len(targets)} segments',flush=True)
    if not args.execute:
        return
    # Corridor is ONLY the confirmed vertical column; not a competition workspace.
    rc=replace(cfg.robot_config(targets[-1]),workspace_bounds=WorkspaceBounds(
        start.x-.002,start.x+.002,start.y-.002,start.y+.002,start.z-.001,TARGET_Z+.001))
    robot=RealUR5(rc)
    previous=start
    try:
        robot.connect_readonly()
        assert_pose_close(robot.snapshot().tcp_pose,start,.0002,math.radians(.1),'Before control')
        robot.connect_control()
        assert_pose_close(robot.snapshot().tcp_pose,start,.0002,math.radians(.1),'After control')
        audit['executed']=True
        for index,target in enumerate(targets):
            offset=robot.active_tcp_offset()
            if math.dist(offset[:3],(0,0,.246))>.0001 or rotation_error(offset[3:],(0,0,0))>math.radians(.05):
                raise ValueError('Active TCP offset changed')
            assert_pose_close(robot.snapshot().tcp_pose,previous,.0002,math.radians(.1),'Segment start')
            b=rc.workspace_bounds
            if not (b.x_min<=target.x<=b.x_max and b.y_min<=target.y<=b.y_max and b.z_min<=target.z<=b.z_max):
                raise ValueError('Outside authorized vertical corridor')
            robot.move_l_without_tcp_force_guard(Waypoint(f'clearance_{index+1}',target,'approach'),
                .005 if index==0 else .015,.05)
            actual=robot.snapshot().tcp_pose
            assert_pose_close(actual,target,.0002,math.radians(.1),'Segment arrival')
            audit['completed_segments'].append({'target':target.as_list(),'actual':list(actual)})
            save()
            print(f'{index+1}/{len(targets)} Z={actual[2]:.6f}',flush=True)
            previous=target
        audit['completed']=True
    except BaseException as exc:
        audit['error']=str(exc)
        try:
            robot.stop_l(.2)
        except Exception as stop_exc:
            audit['stop_error']=str(stop_exc)
        raise
    finally:
        save()
        robot.disconnect()


if __name__=='__main__':
    main()
