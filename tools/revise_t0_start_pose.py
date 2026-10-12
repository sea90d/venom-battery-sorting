"""Stage/apply the incident trajectory correction without connecting hardware."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battery_sorting.handeye import pose_vector_to_transform
from battery_sorting.observation import Observation, recorded_camera_model
from battery_sorting.competition_start import require_competition_start
from battery_sorting.config import Pose

OUT = ROOT/'output/trajectory_forward_20261011'
# Forward is Base -X, toward the battery-zone side of the robot base.
# Use a 90-degree tilt from the downward view, without an added axial twist.
HOME = [-.25, .16, .447856, -1.2088991213571405, -1.2093924493197987, 1.2088991213571405]
OBSERVATION = HOME[:3]+[2.2209882520372397, 2.2218945936749397, 0.]
ACTIVE = ['competition_template.json', 'competition_sequence.json',
          'closed_loop.json', 'closed_loop_sequence.json', 't0_start_pose.json']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def stage():
    backup, preview = OUT/'backup', OUT/'preview'
    backup.mkdir(parents=True, exist_ok=True)
    preview.mkdir(exist_ok=True)
    for path in (ROOT/'config').glob('*.json'):
        if not (backup/path.name).exists():
            shutil.copy2(path, backup/path.name)
        shutil.copy2(path, preview/path.name)
    # Keep the newer joint standby and its real-image ROIs intact. This script
    # remains available for historical Cartesian-plan reconstruction only.
    if (ROOT/'config/folded_standby.json').exists():
        from battery_sorting.closed_loop import Setup
        from battery_sorting.folded_standby import plan_for
        if plan_for(Setup(ROOT/'config/closed_loop.json')):
            return dict(hardware_connected=False,change='Current explicit folded standby preserved',
                        configuration='config/folded_standby.json',old_projection_skipped=True)
    for relative in ['tools/run_smooth_t0.py', 'tools/audit_t0_trajectory.py', 'web_sim/plans/accelerated-red-t0.json']:
        destination = backup/Path(relative).name
        if not destination.exists():
            shutil.copy2(ROOT/relative, destination)
    raw = read(preview/'closed_loop.json')
    observation = Observation.load(ROOT/'output/new_room_high_20260929_111517')
    camera = recorded_camera_model(observation.metadata)
    transform = pose_vector_to_transform(OBSERVATION)@np.array(observation.metadata['tcp_T_camera'])
    survey = read(ROOT/'output/sequence_estimate/report.json')
    plan = read(ROOT/'web_sim/plans/accelerated-red-t0.json')
    polygons = {'pick': survey['zone_corners_base_xy_m'], 'target': plan['targets'][0]['outline_base_xy']}
    rois = {}
    for name, polygon in polygons.items():
        cloud = np.array([[*xy, height] for xy in polygon for height in (raw['paper_z_m'], raw['paper_z_m']+.12)])
        uv = camera.project_color_points((cloud-transform[:3, 3])@transform[:3, :3])
        lower, upper = np.floor(uv.min(0)-8).astype(int), np.ceil(uv.max(0)+8).astype(int)
        if min(lower) < 0 or upper[0] >= 1280 or upper[1] >= 720:
            raise ValueError('Revised observation does not fit full zone/target')
        rois[name] = [*map(int, lower), *map(int, upper-lower+1)]
    for name in ('competition_template.json', 'competition_sequence.json'):
        site = read(preview/name)
        site['geometry']['home_pose_m_rad'] = HOME
        write(preview/name, site)
    for name in ('closed_loop.json', 'closed_loop_sequence.json'):
        config = read(preview/name)
        config['target_view_pose_m_rad'] = OBSERVATION
        config['pick_roi'], config['target_roi'] = rois['pick'], rois['target']
        write(preview/name, config)
    rules = read(preview/'t0_start_pose.json')
    rules['parallel_definition'] = 'Gripper tool +Z is horizontal forward along Base -X toward the battery-zone side at standby; turn down in place after start'
    rules['live_verified'] = False
    write(preview/'t0_start_pose.json', rules)
    start_check = require_competition_start(Pose(*HOME), rules)
    report = dict(hardware_connected=False, home_pose_m_rad=HOME, observation_pose_m_rad=OBSERVATION,
                  pick_roi=rois['pick'], target_roi=rois['target'], start_geometry=start_check,
                  home_live_verified=False, model_only=True,
                  change='Home at [-250,160,447.856] mm, tool axis forward -X toward the battery-zone side. Separate startup lift, X=-400 mm corridor, downward turn, Y transfer, X transfer and forward horizontal turn.',
                  scope='Active first and second task configs share the revised Home; second task Y upper bound stays 370 mm.',
                  projection_source='Sep29 calibrated scene, projected using new view; not a new live observation.')
    write(OUT/'configuration_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    report = stage()
    if args.apply:
        for name in ACTIVE:
            shutil.copy2(OUT/'preview'/name, ROOT/'config'/name)
    print(json.dumps(dict(report, active_configuration_applied=args.apply), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
