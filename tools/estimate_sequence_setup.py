"""Estimate task-2 geometry from the official map and archived calibrated photo.

Offline only. Estimates never set live verification or widen production workspace.
The replacement task-2 sheet is assumed to keep the same battery-zone alignment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from battery_sorting.handeye import pose_vector_to_transform
from battery_sorting.observation import Observation, recorded_camera_model
from battery_sorting.placement_vision import detect_sequence_targets, detect_t0
from battery_sorting.table_view import rectify_table

PHOTO_ZONE_CORNERS = np.array([[458, 226], [746, 237], [737, 469], [449, 455]], float)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def image(path):
    result = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f'Cannot read map render: {path}')
    return result


def zone_corners(im):
    # Largest grey component other than white canvas is the battery-zone fill.
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.inRange(im, (232, 232, 232), (242, 242, 242)))
    x, y, w, h, _ = max((s for s in stats[1:] if s[2] > 100), key=lambda s: s[4])
    x, y, w, h = x-4, max(0, y-4), w+8, h+8
    return np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], float)


def fit_similarity(source, destination):
    # The image axes and Base XY have opposite handedness; allow reflection.
    a, b = source-source.mean(0), destination-destination.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    q = u @ vt
    scale = float(np.sum((a @ q)*b)/np.sum(a*a))
    linear = scale*q
    offset = destination.mean(0)-source.mean(0)@linear
    return linear, offset


def estimate(*, apply=False):
    out = ROOT/'output/sequence_estimate'
    out.mkdir(parents=True, exist_ok=True)
    raw = read_json(ROOT/'config/closed_loop.json')
    site = read_json(ROOT/'config'/raw['site_config'])
    obs = Observation.load(ROOT/'output/new_room_high_20260929_111517')
    camera = recorded_camera_model(obs.metadata)
    first, second = (image(ROOT/f'tmp/pdfs/board-{i}.png') for i in (1, 2))
    z = raw['paper_z_m']
    rays = np.column_stack((camera.normalize_color_pixels(PHOTO_ZONE_CORNERS), np.ones(4))) @ obs.base_T_camera[:3, :3].T
    base_corners = obs.base_T_camera[:3, 3]+rays*((z-obs.base_T_camera[2, 3])/rays[:, 2])[:, None]
    source = zone_corners(first)
    linear, offset = fit_similarity(source, base_corners[:, :2])
    fitted = source@linear+offset
    residual = np.linalg.norm(fitted-base_corners[:, :2], axis=1)
    # Both pages have the same battery zone but a different page origin.
    page_shift = source.mean(0)-zone_corners(second).mean(0)
    rings = detect_sequence_targets(second)
    if not all(r.valid for r in rings):
        raise ValueError('Official map rings were not uniquely detected')
    targets = {r.label: ((np.array(r.center_px)+page_shift)@linear+offset).tolist() for r in rings}
    t0 = detect_t0(first)
    predicted_t0 = np.array(t0.center_px)@linear+offset
    recorded = read_json(ROOT/'output/new_room_high_20260929_111517_analysis/analysis.json')
    t0_error = float(np.linalg.norm(predicted_t0-recorded['targets']['T0']['base_xy_m']))
    # Camera is horizontal/downward, within the current production TCP bounds.
    view = [-.10, .08, site['geometry']['safe_high_z_m'], *raw['target_view_pose_m_rad'][3:]]
    transform = pose_vector_to_transform(view)@np.array(obs.metadata['tcp_T_camera'])
    theta = np.linspace(0, 2*np.pi, 181)
    # Include 20 mm layout uncertainty, the 75 mm outer radius and body height.
    cloud = np.concatenate([np.column_stack((np.array(xy)+.095*np.column_stack((np.cos(theta), np.sin(theta))), np.full(len(theta), height)))
                            for xy in targets.values() for height in (z, z+.12)])
    uv = camera.project_color_points((cloud-transform[:3, 3])@transform[:3, :3])
    lower = np.floor(uv.min(0)-12).astype(int)
    upper = np.ceil(uv.max(0)+12).astype(int)
    if min(lower) < 0 or upper[0] >= 1280 or upper[1] >= 720:
        raise ValueError('Estimated circles do not fit camera view; never clip ROI')
    roi = [*map(int, lower), *map(int, upper-lower+1)]
    overlay = np.full((720, 1280, 3), 255, np.uint8)
    for label, xy in targets.items():
        for radius in (.075, .04):
            circle = np.column_stack((np.array(xy)+radius*np.column_stack((np.cos(theta), np.sin(theta))), np.full(len(theta), z)))
            points = camera.project_color_points((circle-transform[:3, 3])@transform[:3, :3])
            cv2.polylines(overlay, [np.rint(points).astype(np.int32)], True, (30, 30, 30), 2)
        center = np.column_stack((np.array(xy).reshape(1, 2), [z]))
        pixel = camera.project_color_points((center-transform[:3, 3])@transform[:3, :3])[0]
        cv2.putText(overlay, label, tuple(np.rint(pixel).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .7, (80, 80, 80), 2)
    # Exercise the real target rectification/detection with projected map rings.
    rectified = rectify_table(overlay, transform, camera.normalize_color_pixels, camera.project_color_points, z, roi)
    detected = detect_sequence_targets(rectified.image)
    if not all(r.valid for r in detected):
        raise ValueError('Candidate ROI fails real target detector')
    recovered = {r.label: rectified.to_base([r.center_px])[0].tolist() for r in detected}
    if max(np.linalg.norm(np.array(recovered[k])-targets[k]) for k in targets) > .003:
        raise ValueError('Candidate target labels or projection mismatch')
    cv2.rectangle(overlay, tuple(lower), tuple(upper), (0, 150, 0), 2)
    cv2.putText(overlay, 'ESTIMATED MAP / NOT A LIVE PHOTO', (20, 680), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 200), 2)
    (out/'candidate_camera_view.png').write_bytes(cv2.imencode('.png', overlay)[1].tobytes())
    (out/'rectified_targets.png').write_bytes(cv2.imencode('.png', rectified.image)[1].tobytes())
    widths = {c: float(min(v['handle']['size_m'])) for c, v in recorded['instances'].items()}
    red_width = widths['red']
    # Relative aperture model anchored to red's actual 232..239 feedback.
    # Other colours are estimates, never represented as live grip calibration.
    aperture = {c: 235.5*w/red_width for c, w in widths.items()}
    grip_range = [120, 330]
    centers = [v['center_base_m'][:2] for v in recorded['instances'].values()]
    min_spacing = min(float(np.linalg.norm(np.array(a)-b)) for i, a in enumerate(centers) for b in centers[i+1:])
    production_bounds = site['geometry']['workspace_bounds_m']
    outside = [k for k, xy in targets.items() if not (production_bounds[0] <= xy[0] <= production_bounds[1] and production_bounds[2] <= xy[1] <= production_bounds[3])]
    candidate_bounds = production_bounds.copy()
    candidate_bounds[3] = round(max(xy[1]+.10 for xy in targets.values()), 2)
    updated = dict(raw, sequence_target_view_pose_m_rad=view, sequence_target_roi=roi,
                   grip_position_range=grip_range, min_pick_separation_m=.070)
    updated['geometry_and_gripper_verified_by_task'] = dict(raw['geometry_and_gripper_verified_by_task'], sequence=False)
    candidate_site = json.loads(json.dumps(site))
    candidate_site['geometry']['workspace_bounds_m'] = candidate_bounds
    candidate_site['geometry']['target_centers_base_xy_m'].update(targets)
    write_json(ROOT/'config/competition_sequence_estimate.json', candidate_site)
    candidate = dict(updated, site_config='competition_sequence_estimate.json')
    write_json(ROOT/'config/closed_loop_sequence_estimate.json', candidate)
    if apply:
        # Preserve a recoverable original; do not overwrite the first backup.
        backup = out/'closed_loop_before_estimate.json'
        if not backup.exists():
            write_json(backup, raw)
        write_json(ROOT/'config/closed_loop.json', updated)
    sources = [ROOT/'task/《2026年挑战赛任务底图》.pdf', ROOT/'tmp/pdfs/board-1.png', ROOT/'tmp/pdfs/board-2.png', ROOT/'output/new_room_high_20260929_111517/manifest.json']
    report = dict(estimate_only=True, hardware_connected=False, live_verified=False,
                  assumption='Task-2 sheet replaces task-1 sheet with the battery zone at the same position and angle; separately cut P circles must be photographed.',
                  sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  reviewed_photo_zone_corners_px=PHOTO_ZONE_CORNERS.tolist(),
                  zone_corners_base_xy_m=base_corners[:, :2].tolist(),
                  map_to_base_linear=linear.tolist(), map_to_base_offset=offset.tolist(),
                  fit_residual_max_m=float(max(residual)), t0_independent_crosscheck_error_m=t0_error,
                  estimated_layout_tolerance_m=.020, tolerance_note='Planning allowance, not a statistical accuracy guarantee. Sheet alignment changes invalidate estimates.',
                  target_centers_base_xy_m=targets, sequence_target_view_pose_m_rad=view,
                  sequence_target_roi=roi, projected_detector_centers_base_xy_m=recovered,
                  grip_position_range=grip_range, estimated_aperture_positions=aperture,
                  grip_range_note='Visual handle widths scaled from actual red position; yellow/green/blue still require live grip feedback.',
                  measured_archived_min_center_spacing_m=min_spacing, min_pick_separation_m=.070,
                  targets_outside_production_workspace=outside, production_workspace_bounds_m=production_bounds,
                  candidate_workspace_bounds_m=candidate_bounds, production_workspace_expanded=False,
                  production_fields_applied=apply)
    write_json(out/'report.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Apply candidate view/ROI/aperture/spacing; never enable live execution')
    estimate(apply=parser.parse_args().apply)
