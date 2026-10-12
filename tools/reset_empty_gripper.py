"""Initialize/open an empty gripper at the current pose. Never connects to UR5."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from battery_sorting.grasp_lift import GraspLiftConfig
from battery_sorting.gripper import create_gripper


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Confirm empty gripper, stationary robot and clearance to open; send initialization/open commands')
    parser.add_argument('--config',type=Path,default=ROOT/'config/grasp_lift.json')
    parser.add_argument('--open-position',type=int,default=400,help='0..1000; default 400 matches smooth T0')
    args=parser.parse_args(argv)
    cfg=GraspLiftConfig.from_json(args.config)
    cfg.require_valid(allow_missing_final_z=True)
    if not cfg.gripper.close_position<args.open_position<=1000:
        raise ValueError('open-position must exceed close_position and be <=1000')
    if not args.execute:
        print(f'PLAN ONLY: initialize if needed -> open to {args.open_position} -> verify state 1 and opening three times; no UR5 connection')
        return 0
    gripper=create_gripper(replace(cfg.gripper,open_position=args.open_position))
    try:
        gripper.connect()
        print(json.dumps(gripper.reset_empty_open(empty_and_clear_confirmed=True),indent=2))
        return 0
    finally:
        gripper.disconnect()


if __name__=='__main__':
    raise SystemExit(main())
