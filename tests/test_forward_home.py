import json
from pathlib import Path
import cv2
import numpy as np
from tools.revise_t0_start_pose import HOME as LEGACY_HOME, OBSERVATION
HOME=json.loads((Path(__file__).resolve().parents[1]/"config/folded_standby.json").read_text(encoding="utf-8"))["home_pose_m_rad"]


def test_active_standby_points_forward_and_is_shared_by_both_tasks():
    root=Path(__file__).resolve().parents[1]
    for name in ('competition_template.json','competition_sequence.json'):
        site=json.loads((root/'config'/name).read_text(encoding='utf-8'))
        pose=site['geometry']['home_pose_m_rad']
        np.testing.assert_allclose(pose,HOME)
        axis=cv2.Rodrigues(np.array(pose[3:]))[0][:,2]
        np.testing.assert_allclose(axis,[-1,0,0],atol=1e-9)
    down=cv2.Rodrigues(np.array(OBSERVATION[3:]))[0]
    front=cv2.Rodrigues(np.array(HOME[3:]))[0]
    assert abs(np.dot(down[:,2],front[:,2]))<1e-9


def test_both_simulations_end_at_forward_home_with_collision_checked_joints():
    root=Path(__file__).resolve().parents[1]
    for name in ('accelerated-red-t0.json','tool-body-collision.json'):
        plan=json.loads((root/'web_sim/plans'/name).read_text(encoding='utf-8'))
        np.testing.assert_allclose(plan['steps'][-1]['end_pose'],HOME)
        assert plan['collision_check']['passed']
        assert plan['collision_check']['swept_clearance_lower_bound_m']>=.01


def test_previous_left_facing_home_can_follow_new_preparation_in_proxy_model():
    from battery_sorting.config import Pose
    from tools.run_smooth_t0 import startup_home_targets
    from tools.audit_t0_trajectory import audit_segment
    from tools.simulate_tool_body_collision import check_segment
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'output/trajectory_revision_20261011/from_snapshot/report.json').read_text(encoding='utf-8'))['segments'][5]
    previous=old['end_pose']
    q=np.array(old['samples'][-1]['q_rad'])
    targets=startup_home_targets(Pose(*previous),Pose(*LEGACY_HOME),LEGACY_HOME[2],Pose(*OBSERVATION))
    for target in targets:
        segment,q=audit_segment('left_to_forward',previous,target.as_list(),q)
        assert check_segment(segment)['passed']
        previous=target.as_list()
    np.testing.assert_allclose(previous,LEGACY_HOME)
