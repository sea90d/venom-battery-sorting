"""Check solid tool/body collision and export FK-consistent joint playback."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_t0_trajectory import fk, A, D
from tools.tool_body_collision import tool_body_clearances, geometry_description, REQUIRED_CLEARANCE_M, TOOL_LENGTH_M
from tools.export_smooth_t0_simulation import export_trial
from battery_sorting.simulation_export import nominal_motion_seconds


def clearance(q):
    flange, points = fk(q)
    gaps = tool_body_clearances(flange, points)
    link = min(gaps, key=gaps.get)
    return gaps[link], link, gaps


def check_interval(first, last, first_gap, last_gap, *, depth=0, required_clearance_m=REQUIRED_CLEARANCE_M):
    """Certify a joint-linear interval by conservative swept-displacement bound.

    Every point on either solid moves at most reach*sum(|dq|). Relative
    clearance changes by at most twice that, covering between-frame collision.
    """
    reach = sum(abs(v) for v in A)+sum(abs(v) for v in D)+TOOL_LENGTH_M+.2
    bound = 2*reach*float(np.sum(abs(last-first)))
    lower = min(first_gap, last_gap)-bound
    if lower >= required_clearance_m:
        return lower, 0, None
    middle = (first+last)/2
    gap, link, _ = clearance(middle)
    if gap < required_clearance_m:
        return min(gap, first_gap, last_gap), 1, dict(q_rad=middle.tolist(), clearance_m=gap, link=link)
    if depth >= 16:
        return lower, 1, dict(reason='sweep bound could not establish clearance', clearance_lower_bound_m=lower)
    left, count_l, fail_l = check_interval(first, middle, first_gap, gap, depth=depth+1, required_clearance_m=required_clearance_m)
    right, count_r, fail_r = check_interval(middle, last, gap, last_gap, depth=depth+1, required_clearance_m=required_clearance_m)
    return min(left, right), 1+count_l+count_r, fail_l or fail_r


def check_segment(segment, *, required_clearance_m=REQUIRED_CLEARANCE_M):
    rows = copy.deepcopy(segment['samples'])
    # A runtime segment may carry a calibrated branch offset when the nominal
    # DH model starts at a controller-calibrated TCP pose.  In that fallback
    # mode the joint samples are still screened for clearance and step size,
    # while the nominal IK residual is intentionally not treated as a motion
    # failure.  Real UR5 runs use controller IK/FK when those APIs are
    # available; this flag is primarily for offline driver tests.
    calibrated_branch = bool(segment.get('calibrated_branch_offset'))
    minimum = float('inf')
    failures, checks, swept_lower = [], 0, float('inf')
    for row in rows:
        gap, link, gaps = clearance(row['q_rad'])
        row.update(tool_body_clearance_m=gap, collision_link=link, tool_body_gaps_m=gaps)
        minimum = min(minimum, gap)
        if gap < required_clearance_m:
            failures.append(dict(fraction=row['fraction'], clearance_m=gap, link=link))
        if ((not calibrated_branch and
             (row['ik_position_error_m'] > .0002 or row['ik_rotation_error_rad'] > .001))
                or row['max_joint_step_rad'] > math.radians(10)):
            failures.append(dict(fraction=row['fraction'], reason='IK residual or joint discontinuity'))
    for a, b in zip(rows, rows[1:]):
        if min(a['tool_body_clearance_m'], b['tool_body_clearance_m']) < required_clearance_m:
            continue
        lower, count, failure = check_interval(np.array(a['q_rad']), np.array(b['q_rad']), a['tool_body_clearance_m'], b['tool_body_clearance_m'], required_clearance_m=required_clearance_m)
        a['next_swept_clearance_lower_bound_m'] = lower
        swept_lower, checks = min(swept_lower, lower), checks+count
        if failure:
            failures.append(dict(interval=[a['fraction'], b['fraction']], **failure))
    return dict(label=segment['label'], samples=rows, minimum_clearance_m=minimum,
                swept_clearance_lower_bound_m=swept_lower if math.isfinite(swept_lower) else None,
                extra_sweep_checks=checks, failures=failures, passed=not failures)


def make_playback(audit, checked, plan):
    plan = copy.deepcopy(plan)
    motions = iter(zip(audit['segments'], checked))
    startup = []
    for segment, result in zip(audit['segments'], checked):
        if not segment['label'].startswith('startup_'):
            break
        rotating = segment['translation_m'] < 1e-9
        speed = .04 if rotating else .1
        startup.append(dict(kind='move', label=segment['label'], start_pose=segment['start_pose'], end_pose=segment['end_pose'],
                            duration_s=nominal_motion_seconds(segment['start_pose'], segment['end_pose'], speed, .1, 1.),
                            speed_mps=speed, acceleration_mps2=.1, profile='preparation', object_id=None, target_id=None))
    end = []
    for segment in audit['segments'][-2:]:
        end.append(dict(kind='move', label=segment['label'], start_pose=segment['start_pose'], end_pose=segment['end_pose'],
                        duration_s=nominal_motion_seconds(segment['start_pose'], segment['end_pose'], .04, .1, 1.),
                        speed_mps=.04, acceleration_mps2=.1, profile='verification', object_id=None, target_id=None))
    plan['initial_pose'] = audit['segments'][0]['start_pose']
    plan['steps'] = startup+plan['steps']+end
    previous = checked[0]['samples'][0]
    for step in plan['steps']:
        if step['kind'] == 'move':
            segment, result = next(motions)
            if np.linalg.norm(np.array(segment['end_pose'])-step['end_pose']) > 1e-8:
                raise ValueError('Collision samples do not match current motion plan')
            step['joint_samples'] = [{k:r[k] for k in ('fraction','q_rad','tool_body_clearance_m','collision_link','next_swept_clearance_lower_bound_m') if k in r} for r in result['samples']]
            previous = result['samples'][-1]
        else:
            step['joint_samples'] = [{k:previous[k] for k in ('q_rad','tool_body_clearance_m','collision_link')} | {'fraction':f} for f in (0.,1.)]
    if next(motions, None) is not None:
        raise ValueError('Missing collision playback segment')
    plan['collision_geometry'] = geometry_description()
    plan['collision_check'] = dict(passed=all(s['passed'] for s in checked),
                                  minimum_clearance_m=min(s['minimum_clearance_m'] for s in checked),
                                  swept_clearance_lower_bound_m=min(s['swept_clearance_lower_bound_m'] for s in checked if s['swept_clearance_lower_bound_m'] is not None),
                                  mode='nominal full-6D IK, joint-interpolated continuous solid clearance bound', hardware_executed=False)
    plan['source']['label'] = '夹爪实体碰撞仿真 · 准备 → 红色 T0 → Home → 复检'
    plan['source']['timing_note'] += ' 此计划另含准备与复检动作；使用离线六维逆解的关节采样，夹爪按法兰至 TCP 全长、连杆直径 1.5 倍建模。'
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=ROOT/'output/trajectory_forward_20261011/from_snapshot/report.json')
    parser.add_argument('--out', type=Path, default=ROOT/'web_sim/plans/tool-body-collision.json')
    parser.add_argument('--cycle-out', type=Path, default=ROOT/'web_sim/plans/accelerated-red-t0.json')
    parser.add_argument('--report', type=Path, default=ROOT/'output/tool_body_collision_20261011/report.json')
    parser.add_argument('--check-only', action='store_true', help='Check archived geometry without exporting a current-cycle playback')
    args = parser.parse_args()
    if not args.check_only and args.audit==ROOT/'output/trajectory_forward_20261011/from_snapshot/report.json':
        from battery_sorting.closed_loop import Setup
        from battery_sorting.folded_standby import plan_for
        if plan_for(Setup(ROOT/'config/closed_loop.json')):
            from tools.export_folded_t0_simulation import export
            playback=export()
            for destination in (args.out,args.cycle_out):
                destination.write_text(json.dumps(playback,ensure_ascii=False,indent=2),encoding='utf-8')
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text((ROOT/'output/folded_standby_20261011/collision_report.json').read_text(encoding='utf-8'),encoding='utf-8')
            return
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    checked = [check_segment(s) for s in audit['segments']]
    result = dict(geometry=geometry_description(), passed=all(s['passed'] for s in checked),
                  source_audit=str(args.audit), hardware_executed=False,
                  minimum_clearance_m=min(s['minimum_clearance_m'] for s in checked),
                  sample_count=sum(len(s['samples']) for s in checked),
                  extra_sweep_checks=sum(s['extra_sweep_checks'] for s in checked),
                  segments=[{k:v for k,v in s.items() if k!='samples'} for s in checked])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.check_only:
        plan = export_trial(ROOT/'output/smooth_t0/new_room_20260929_111616', ROOT/'config/closed_loop_scene_20260929.json', motion_config=ROOT/'config/closed_loop.json')
        playback = make_playback(audit, checked, plan)
        args.out.write_text(json.dumps(playback, ensure_ascii=False, indent=2), encoding='utf-8')
        # Keep the existing accelerated button on the same audited joint branch;
        # it shows only the timed cycle, without preparation/verification.
        startup_count=sum(s['label'].startswith('startup_') for s in audit['segments'])
        plan['steps']=copy.deepcopy(playback['steps'][startup_count:-2])
        plan['collision_geometry']=playback['collision_geometry']
        cycle_results=checked[startup_count:-2]
        plan['collision_check']=dict(playback['collision_check'],
            passed=all(s['passed'] for s in cycle_results),
            minimum_clearance_m=min(s['minimum_clearance_m'] for s in cycle_results),
            swept_clearance_lower_bound_m=min(s['swept_clearance_lower_bound_m'] for s in cycle_results))
        plan['source']['timing_note']+=' 同步采用实体碰撞检查的完整姿态关节轨迹；工具直径为连杆的 1.5 倍。'
        args.cycle_out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='segments'}, ensure_ascii=False, indent=2))
    for s in checked:
        print(s['label'], 'min_mm', round(s['minimum_clearance_m']*1000,2), 'passed', s['passed'], 'failures',len(s['failures']))


if __name__ == '__main__':
    main()
