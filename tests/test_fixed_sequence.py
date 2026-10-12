import copy
import json
import math
from dataclasses import replace
import pytest

from battery_sorting import fixed_sequence as fixed
from battery_sorting.config import Pose
from battery_sorting.robot import FakeRobot, RobotSafetyError


def setup_plan():
    setup = fixed.Setup(fixed.CONFIG)
    return setup, fixed.load_plan(setup)


def test_taught_points_match_actual_completed_run_and_budget():
    s, p = setup_plan()
    source = json.loads((fixed.ROOT/p['source']).read_text(encoding='utf-8'))
    for step in p['steps']:
        for kind in ('pick', 'place'):
            event = next(e for e in source if e['event']=='smooth_path' and e['label']==step['color']+':'+kind)
            assert step[kind+'_pose_m_rad'] == event['poses'][-1]
    commands, budget = fixed.compile_plan(s, p)
    assert budget['nominal_seconds'] < 30
    assert sum(c['kind']=='joint' for c in commands)==2
    contacts = [pose for cmd in commands if cmd['kind']=='cartesian' for pose in cmd['poses'] if pose[2]==s.site.pick_contact_z_m]
    assert len(contacts)==6


def test_fixed_run_never_uses_camera_or_waits_between_batteries(tmp_path):
    s,p=setup_plan()
    class Robot(FakeRobot):
        def active_tcp_offset(self): return s.raw['active_tcp_offset_m_rad']
        def move_fixed_joint(self, target, label): self._pose=target; return True
    robot=Robot(s.home)
    class Gripper:
        last_grip_state=1
        held=False
        actions=[]
        def initialize(self, **kwargs): return True
        def close(self): self.actions.append('close');self.held=True;return True
        def open(self): self.actions.append('open');self.held=False;return True
        def has_object(self): return self.held
        def position(self): return 239 if self.held else 1000
        def stop(self): self.actions.append('stop');return True
    sleeps=[]
    class ForbiddenCamera:
        def __getattr__(self, name): raise AssertionError('Camera used: '+name)
    rt=fixed.FixedRuntime(s,ForbiddenCamera(),robot,Gripper(),tmp_path/'run',clock=lambda:0.,sleep=sleeps.append)
    rt.run_fixed(p)
    assert rt.gripper.actions==['open','close','open','close','open','close','open']
    assert sleeps==[s.site.timing.release_settle_s]  # Only final release interval, overlaps real return.
    assert robot.snapshot().tcp_pose==pytest.approx(s.home.as_list())
    summary=json.loads((rt.output/'summary.json').read_text(encoding='utf-8'))
    assert summary['completed'] and not summary['vision_enabled'] and not summary['placement_verified']


def test_lost_grip_stops_without_opening_or_recovery(tmp_path):
    s,p=setup_plan()
    class Robot(FakeRobot):
        def active_tcp_offset(self): return s.raw['active_tcp_offset_m_rad']
        def move_fixed_joint(self,target,label): self._pose=target;return True
    class Gripper:
        last_grip_state=1
        actions=[]
        def initialize(self, **kwargs):return True
        def open(self):self.actions.append('open');return True
        def close(self):self.actions.append('close');return True
        def has_object(self):return False
        def position(self):return 1000
        def stop(self):self.actions.append('stop');return True
    robot=Robot(s.home);gripper=Gripper()
    rt=fixed.FixedRuntime(s,None,robot,gripper,tmp_path/'run')
    with pytest.raises(RuntimeError,match='Grasp missing'):
        rt.run_fixed(p)
    assert gripper.actions==['open','close','stop'] and robot.stop_count==1
    assert not any(e.get('label')=='yellow:place' for e in rt.events)


def test_prepared_certificate_rejects_changed_path_and_joint_branch():
    s,p=setup_plan()
    r=fixed.FixedRobot(s.robot_config(),motion_profile='t0_timing_trial')
    start=Pose(*p['steps'][0]['pick_pose_m_rad']).with_z(s.transit_high_z)
    target=Pose(*p['steps'][0]['pick_pose_m_rad']).as_list()
    r.snapshot=lambda:type('Snapshot',(),{'tcp_pose':start.as_list()})()
    r.actual_joints=lambda:[0.]*6
    r.fixed_certificate_index=0
    r.fixed_certificates=[dict(kind='cartesian',start=start.as_list(),start_joints=[0.]*6,
                               poses=[target],swept_clearance_lower_bound_m=.015)]
    altered=target.copy();altered[0]+=.01
    with pytest.raises(RobotSafetyError,match='differs'):
        r._check_tool_body_path([altered])
    assert r.fixed_certificate_index==0
    r.actual_joints=lambda:[.1]*6
    with pytest.raises(RobotSafetyError,match='branch'):
        r._check_tool_body_path([target])
    assert r.fixed_certificate_index==0
    r.actual_joints=lambda:[0.]*6
    r._check_tool_body_path([target])
    assert r.fixed_certificate_index==1


def test_default_plan_never_constructs_hardware(monkeypatch,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('Hardware constructed')
    monkeypatch.setattr(fixed,'FixedRobot',forbidden)
    monkeypatch.setattr(fixed,'create_gripper',forbidden)
    assert fixed.main([])==0
    assert json.loads(capsys.readouterr().out)['timing_budget']['nominal_seconds']<30


def test_skipped_trajectory_has_no_collision_or_joint_scan(monkeypatch):
    s,p=setup_plan()
    commands,_=fixed.compile_plan(s,p)
    robot=fixed.FixedRobot(s.robot_config(),motion_profile='t0_timing_trial')
    def forbidden(*args,**kwargs):raise AssertionError('Slow trajectory check called')
    robot.actual_joints=forbidden
    robot.joints_for_pose=forbidden
    monkeypatch.setattr('tools.simulate_tool_body_collision.check_interval',forbidden)
    monkeypatch.setattr('tools.simulate_tool_body_collision.check_segment',forbidden)
    assert robot.prepare_fixed_paths(commands,check_trajectory=False)==commands
    assert robot.tool_body_model_enabled is False
    robot.snapshot=lambda:type('Snapshot',(),{'tcp_pose':commands[0]['start']})()
    assert robot._prepared('joint')==commands[0]


def test_unchecked_already_home_has_no_control_or_collision_calls():
    s,p=setup_plan()
    robot=FakeRobot(s.home)
    robot.connect_control=lambda:pytest.fail('Already-home preparation must not create control')
    logs=[]
    assert fixed.startup_home_without_trajectory_check(s,robot,None,lambda event,**d:logs.append(event)) is False
    assert logs==['startup_home_complete']
