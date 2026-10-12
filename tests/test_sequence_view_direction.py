"""Use the real failed view to check which Base direction faces the paper."""
import json
import numpy as np
import pytest
from battery_sorting.closed_loop import Setup, Runtime, ROOT
from battery_sorting.config import Pose
from battery_sorting.handeye import pose_vector_to_transform
from battery_sorting.observation import Observation, recorded_camera_model
from battery_sorting.robot import FakeRobot
from tools.audit_t0_trajectory import audit_segment
from tools.simulate_tool_body_collision import check_segment


def recorded_view():
    obs=Observation.load(ROOT/'output/competition_console/20261011_224338/0014_targets_393')
    return obs,recorded_camera_model(obs.metadata)


def test_corrected_view_brings_recorded_visible_paper_into_camera():
    setup=Setup(ROOT/'config/closed_loop_sequence.json')
    obs,camera=recorded_view()
    # Visible P3 lower rim, taken from the real image (not old map estimates).
    pixels=np.array([[770.,90.],[810.,98.],[850.,90.]])
    rays=np.c_[camera.normalize_color_pixels(pixels),np.ones(3)]@obs.base_T_camera[:3,:3].T
    paper=obs.base_T_camera[:3,3]+rays*((setup.raw['paper_z_m']-obs.base_T_camera[2,3])/rays[:,2])[:,None]
    view=setup.target_view('sequence')
    assert view.x<setup.pick_view.x
    transform=pose_vector_to_transform(view.as_list())@np.array(obs.metadata['tcp_T_camera'])
    projected=camera.project_color_points((paper-transform[:3,3])@transform[:3,:3])
    assert np.all(projected[:,1]>pixels[:,1]+250)
    x,y,w,h=setup.target_roi('sequence')
    assert np.all((projected[:,0]>x)&(projected[:,0]<x+w))
    assert np.all((projected[:,1]>y)&(projected[:,1]<y+h))
    setup.require_ready('sequence',site_test=True)
    assert setup.site.workspace_bounds_m.y_max==.37


def test_corrected_camera_move_keeps_full_tool_body_clearance():
    setup=Setup(ROOT/'config/closed_loop_sequence.json')
    fold=json.loads((ROOT/'config/folded_standby.json').read_text(encoding='utf-8'))
    segment,_=audit_segment('corrected_sequence_view',setup.pick_view.as_list(),
                            setup.target_view('sequence').as_list(),np.array(fold['observation_joints_rad']))
    result=check_segment(segment)
    assert result['passed'] is True and result['swept_clearance_lower_bound_m']>=.01


def test_failed_ring_scan_never_moves_to_search(tmp_path):
    setup=Setup(ROOT/'config/closed_loop_sequence.json')
    obs,camera=recorded_view()
    robot=FakeRobot(Pose(*obs.metadata['tcp_after']))
    runtime=Runtime(setup,camera,robot,None,tmp_path/'run')
    runtime.task='sequence';captures=[]
    runtime.frame=lambda label:captures.append(label) or obs
    with pytest.raises(ValueError,match='Target geometry missing'):
        runtime.scan_targets('sequence')
    assert captures==['targets'] and robot.records==[]
    assert (runtime.output/'target_view_1.png').is_file()
