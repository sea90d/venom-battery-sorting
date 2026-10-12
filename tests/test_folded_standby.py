import json
import math
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from battery_sorting.closed_loop import Setup
from battery_sorting.config import Pose
from battery_sorting.folded_standby import plan_for,certify_joint_path,transition
from battery_sorting.robot import FakeRobot,Waypoint,RobotMotionError
from .test_robot import wired_robot

ROOT=Path(__file__).resolve().parents[1]


def test_photo_fold_has_back_lean_horizontal_forearm_and_forward_tool():
    from tools.audit_t0_trajectory import fk
    setup=Setup(ROOT/'config/closed_loop.json');plan=plan_for(setup)
    flange,points=fk(plan['home_joints_rad'])
    upper=points[2]-points[1];fore=points[3]-points[2]
    assert math.degrees(math.acos(upper[2]/np.linalg.norm(upper)))==pytest.approx(20.)
    assert abs(fore[2])<1e-9
    np.testing.assert_allclose(flange[:3,2],[-1,0,0],atol=1e-9)
    for first,last in ((plan['home_joints_rad'],plan['observation_joints_rad']),
                       (plan['observation_joints_rad'],plan['home_joints_rad'])):
        assert certify_joint_path(first,last)['swept_clearance_lower_bound_m']>=.01


def test_unsafe_observation_ik_branch_is_rejected_before_command():
    setup=Setup(ROOT/'config/closed_loop.json');plan=plan_for(setup)
    robot=FakeRobot(setup.home);robot.active_tcp_offset=lambda:setup.raw['active_tcp_offset_m_rad']
    robot.joints_for_pose=lambda *args:np.radians([-54.195,-78.161,34.447,133.714,90,125.781]).tolist()
    with pytest.raises(ValueError,match='joint branch'):transition(setup,robot,'observation',lambda *a,**k:None)
    assert robot.records==[]


def test_live_scene_new_roi_accepts_three_frames_without_disabling_clip_guard():
    from battery_sorting.observation import Observation,recorded_camera_model
    from battery_sorting.scene import detect_batteries,SceneError
    from tools.run_smooth_t0 import initial_red_estimate
    setup=Setup(ROOT/'config/closed_loop.json');scenes=[]
    for number in (2,3,4):
        obs=Observation.load(ROOT/f'output/smooth_t0/20261011_213309/initial_{number}')
        camera=recorded_camera_model(obs.metadata)
        args=(obs.colour,obs.depth_m,setup.vision.colors,obs.base_T_camera,camera.normalize_color_pixels,setup.raw['paper_z_m'])
        scenes.append(detect_batteries(*args,roi=setup.raw['pick_roi'],max_height_mad_m=setup.raw['max_height_mad_m']))
        with pytest.raises(SceneError,match='Clipped instance: red'):
            detect_batteries(*args,roi=[70,216,415,314],max_height_mad_m=setup.raw['max_height_mad_m'])
    xy,yaw=initial_red_estimate(scenes)
    assert xy[0]==pytest.approx(-.495,abs=.002)
    assert xy[1]==pytest.approx(-.102,abs=.002)


def test_joint_monitor_requires_joints_tcp_and_stopped_feedback(valid_config,monkeypatch):
    robot,control=wired_robot(valid_config)
    target=valid_config.robot.taught_poses['home'];q=[0.,-.5,.5,0.,-.5,0.]
    control.isJointsWithinSafetyLimits=lambda _:True
    control.getTCPOffset=lambda:[0.,0.,.246,0.,0.,0.]
    control.getForwardKinematics=lambda q,tcp:target.as_list()
    control.moveJ=lambda *args:control.move_calls.append(args) or True
    robot._receiver.getActualQ=lambda:q
    robot._receiver.getActualQd=lambda:[0.]*6
    monkeypatch.setattr('battery_sorting.robot.time.sleep',lambda _:None)
    assert robot.move_j(Waypoint('fold',target),q,1.,1.4)
    assert control.move_calls==[(q,1.,1.4,True)]


def test_joint_monitor_stops_j_when_tcp_arrives_but_joints_do_not(valid_config,monkeypatch):
    robot,control=wired_robot(valid_config);target=valid_config.robot.taught_poses['home']
    control.isJointsWithinSafetyLimits=lambda _:True
    control.getTCPOffset=lambda:[0.,0.,.246,0.,0.,0.]
    control.getForwardKinematics=lambda q,tcp:target.as_list()
    control.moveJ=lambda *args:True
    stops=[];control.stopJ=lambda deceleration:stops.append(deceleration)
    robot._receiver.getActualQ=lambda:[1.]*6
    robot._receiver.getActualQd=lambda:[0.]*6
    robot.config=replace(robot.config,move_timeout=.1)
    ticks=iter([0.,.01,.2]);monkeypatch.setattr('battery_sorting.robot.time.monotonic',lambda:next(ticks))
    monkeypatch.setattr('battery_sorting.robot.time.sleep',lambda _:None)
    with pytest.raises(RobotMotionError,match='did not arrive'):robot.move_j(Waypoint('fold',target),[0.]*6,.15,.3)
    assert stops==[robot.config.stop_deceleration]


def test_joint_fk_explicit_tcp_prevents_recorded_ur_rtde_163_stale_register_bug(valid_config,monkeypatch):
    from battery_sorting.handeye import pose_vector_to_transform
    import cv2
    from battery_sorting.robot import translation_error,rotation_error
    setup=Setup(ROOT/'config/closed_loop.json');plan=plan_for(setup)
    audit=json.loads((ROOT/'output/smooth_t0/20261011_220725/audit.json').read_text(encoding='utf-8'))
    checked=next(e for e in audit['events'] if e['event']=='joint_path_checked')
    q=checked['q_end_rad'];target=setup.home;offset=setup.raw['active_tcp_offset_m_rad']
    flange=pose_vector_to_transform(target.as_list())@np.linalg.inv(pose_vector_to_transform(offset))
    calls=[]
    def fk_with_163_registers(joints,tcp=()):
        calls.append((list(joints),list(tcp)))
        # The previous IK left its qnear in input registers 6..11.
        pose=flange@pose_vector_to_transform(list(tcp) if tcp else plan['home_joints_rad'])
        return np.r_[pose[:3,3],cv2.Rodrigues(pose[:3,:3])[0].ravel()].tolist()
    wrong=fk_with_163_registers(q)
    assert translation_error(wrong,target.as_list())*1000==pytest.approx(2904.692213,abs=.001)
    assert math.degrees(rotation_error(wrong[3:],target.as_list()[3:]))==pytest.approx(10.)
    calls.clear()
    robot,control=wired_robot(valid_config)
    control.isJointsWithinSafetyLimits=lambda _:True
    control.getTCPOffset=lambda:offset.copy()
    control.getForwardKinematics=fk_with_163_registers
    control.moveJ=lambda *args:control.move_calls.append(args) or True
    robot._receiver.getActualTCPPose=lambda:target.as_list()
    robot._receiver.getActualQ=lambda:q.copy()
    robot._receiver.getActualQd=lambda:[0.]*6
    monkeypatch.setattr('battery_sorting.robot.time.sleep',lambda _:None)
    assert robot.move_j(Waypoint('fold_regression',target),q,.15,.3)
    assert calls==[(q,offset)]
    assert control.move_calls==[(q,.15,.3,True)]


def test_joint_fk_mismatch_still_rejects_before_motion(valid_config):
    from battery_sorting.robot import RobotSafetyError
    robot,control=wired_robot(valid_config);target=valid_config.robot.taught_poses['home']
    control.isJointsWithinSafetyLimits=lambda _:True
    control.getTCPOffset=lambda:[0.,0.,.246,0.,0.,0.]
    control.getForwardKinematics=lambda q,tcp:replace(target,x=target.x+.001).as_list()
    control.moveJ=lambda *args:control.move_calls.append(args) or True
    with pytest.raises(RobotSafetyError,match='Calibrated joint target'):
        robot.move_j(Waypoint('incorrect_fk',target),[0.]*6,.15,.3)
    assert control.move_calls==[]


def test_folded_cycle_has_one_unfold_one_fold_and_verifies_before_return():
    plan=json.loads((ROOT/'web_sim/plans/accelerated-red-t0.json').read_text(encoding='utf-8'))
    steps=plan['steps'];joint=[s for s in steps if s.get('motion_space')=='joint']
    assert len(joint)==2 and steps[0] is joint[0] and steps[-1] is joint[-1]
    assert all('原地' not in s['label'] for s in steps)
    release=next(i for i,s in enumerate(steps) if s['kind']=='open')
    verify=next(i for i,s in enumerate(steps) if s['kind']=='visual_accept')
    assert sum(s['duration_s'] for s in steps[release+1:verify])>=3.1-1e-9
    assert verify<len(steps)-1
    assert plan['source']['scene_recording_path'].endswith('20261011_213309')
