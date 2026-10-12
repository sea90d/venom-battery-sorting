"""Authorized empty-gripper baseline at verified home; no robot control connection."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from battery_sorting.config import Pose
from battery_sorting.grasp_lift import GraspLiftConfig
from battery_sorting.gripper import create_gripper
from battery_sorting.robot import assert_pose_close, rotation_error, read_dashboard, require_dashboard_state
from battery_sorting.tcp_readonly import read_tcp_info

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Audit already exists')
    cfg = GraspLiftConfig.from_json(ROOT/'config/grasp_lift.json')
    home = Pose(*json.loads((ROOT/'config/competition_template.json').read_text())['geometry']['home_pose_m_rad'])
    audit = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), events=[], completed=False,
                 force_percent=cfg.gripper.force_percent, speed_percent=cfg.gripper.speed_percent,
                 note='Empty baseline only; not a verified held-object range')
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    def check_home():
        require_dashboard_state(read_dashboard(cfg.robot_host,cfg.dashboard_port),{'STOPPED'},'Empty baseline')
        report = read_tcp_info(cfg.robot_host)
        assert_pose_close(report['tcp_base_m_rad'],home,.001,math.radians(.5),'Empty baseline home')
        offset = report['active_tcp_offset_m_rad']
        if math.dist(offset[:3],(0,0,.246))>.0001 or rotation_error(offset[3:],(0,0,0))>math.radians(.05):
            raise ValueError('TCP offset changed')
        return report
    audit['initial_tcp'] = check_home()
    audit['dashboard'] = read_dashboard(cfg.robot_host,cfg.dashboard_port)
    save()
    if not args.execute:
        print('PLAN ONLY: initialize -> open -> empty close -> open, no robot motion')
        return
    gripper = create_gripper(cfg.gripper)
    try:
        gripper.connect()
        for label,action in [('initialize',lambda:gripper.initialize(recalibrate=False)),
                             ('open',gripper.open),('empty_close',gripper.close),('reopen',gripper.open)]:
            check_home()
            if not action():
                raise RuntimeError(f'{label} did not complete')
            samples = []
            for _ in range(3):
                samples.append(dict(position=gripper.position(),has_object=gripper.has_object(),state=gripper.last_grip_state))
            audit['events'].append(dict(action=label,samples=samples))
            save()
            print(label,json.dumps(samples),flush=True)
            if any(s['has_object'] is not False for s in samples):
                raise RuntimeError('Unexpected object/moving state in declared empty baseline; no further actuation')
        audit['final_tcp'] = check_home()
        audit['completed'] = True
    except BaseException as exc:
        audit['error'] = str(exc)
        raise
    finally:
        save()
        gripper.disconnect()


if __name__ == '__main__':
    main()
