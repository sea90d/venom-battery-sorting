"""Export the four-phase joint standby cycle with the Oct11 recorded scene."""
import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.export_smooth_t0_simulation import export_trial
from tools.audit_t0_trajectory import audit_segment,fk
from tools.simulate_tool_body_collision import check_segment
from tools.tool_body_collision import geometry_description


def export():
    plan=export_trial(ROOT/'output/smooth_t0/new_room_20260929_111616',ROOT/'config/closed_loop_scene_20260929.json',
        scene_recording=ROOT/'output/smooth_t0/20261011_213309')
    return audit_plan(plan)


def audit_plan(plan,publish=True):
    cfg=json.loads((ROOT/'config/folded_standby.json').read_text(encoding='utf-8'))
    q=np.array(cfg['home_joints_rad']);results=[];segments=[]
    last=None
    for step in plan['steps']:
        if step['kind']=='move':
            if step.get('motion_space')=='joint':
                end=np.array(step['joint_end_rad'])
                if max(abs(q-np.array(step['joint_start_rad'])))>.01:
                    raise ValueError('Joint execution branch differs from continuous Cartesian branch')
                rows=[]
                for f in np.linspace(0,1,257):
                    value=q+(end-q)*f;T,_=fk(value)
                    pose=np.r_[T[:3,3]+T[:3,2]*.246,cv2.Rodrigues(T[:3,:3])[0].ravel()]
                    rows.append(dict(fraction=float(f),q_rad=value.tolist(),tcp_m_rad=pose.tolist(),
                        ik_position_error_m=0.,ik_rotation_error_rad=0.,max_joint_step_rad=float(max(abs(end-q))/256)))
                segment=dict(label=step['label'],samples=rows)
                q=end
            else:
                segment,q=audit_segment(step['label'],step['start_pose'],step['end_pose'],q)
            result=check_segment(segment)
            results.append(result);segments.append(segment)
            step['joint_samples']=result['samples'];last=result['samples'][-1]
            print(step['label'],result['passed'],'min_mm',round(result['minimum_clearance_m']*1000,2),flush=True)
        else:
            step['joint_samples']=[dict(last,fraction=f) for f in (0.,1.)]
    check=dict(passed=all(r['passed'] for r in results),minimum_clearance_m=min(r['minimum_clearance_m'] for r in results),
        swept_clearance_lower_bound_m=min(r['swept_clearance_lower_bound_m'] for r in results),
        mode='nominal UR5 full tool cylinder, continuous swept bound; explicit joint fold/unfold',hardware_executed=False)
    plan.update(collision_geometry=geometry_description(),collision_check=check)
    out=ROOT/'output/folded_standby_20261011';out.mkdir(parents=True,exist_ok=True)
    (out/'collision_report.json').write_text(json.dumps(dict(check,segments=[{k:v for k,v in r.items() if k!='samples'} for r in results]),ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'audit_segments.json').write_text(json.dumps(segments,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    if not check['passed']:raise ValueError('New trajectory did not pass; browser plans preserved')
    if publish:
        for name in ('accelerated-red-t0.json','tool-body-collision.json'):
            (ROOT/'web_sim/plans'/name).write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(check,nominal_duration_s=sum(s['duration_s'] for s in plan['steps'])),ensure_ascii=False))
    return plan


if __name__=='__main__':export()
