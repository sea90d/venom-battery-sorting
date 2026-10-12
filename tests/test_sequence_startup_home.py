import json
from dataclasses import replace
import pytest
from battery_sorting.closed_loop import Setup, ROOT
from battery_sorting.config import Pose
from battery_sorting.folded_standby import prepare_startup_home
from battery_sorting.robot import FakeRobot, RobotSafetyError


class EmptyProbe:
    last_grip_state=1
    def has_object(self):return False
    def open(self):pytest.fail('Startup must not open fingers')
    def initialize(self,**kwargs):pytest.fail('Startup must not initialize fingers')


def hardware_recorded_start():
    setup=Setup(ROOT/'config/closed_loop_sequence.json')
    state=json.loads((ROOT/'output/sequence_start_check_20261011/state.json').read_text(encoding='utf-8'))
    robot=FakeRobot(Pose(*state['rtde']['tcp_m_rad']))
    robot._joints=state['rtde']['q_rad']
    robot.active_tcp_offset=lambda:setup.raw['active_tcp_offset_m_rad']
    audit=json.loads((ROOT/'output/smooth_t0/20261011_221228/audit.json').read_text(encoding='utf-8'))
    target=next(e['q_end_rad'] for e in audit['events'] if e.get('label')=='return_folded_standby')
    robot.joints_for_pose=lambda *args:target.copy()
    return setup,robot


def test_recorded_second_view_returns_once_before_task_without_gripper_writes():
    setup,robot=hardware_recorded_start();events=[]
    def log(event,**detail):events.append({'event':event,**detail})
    assert prepare_startup_home(setup,robot,EmptyProbe(),log) is True
    assert robot.control_connected is True
    assert len(robot.records)==1
    assert robot.records[0].label=='return_folded_standby'
    assert robot.records[0].speed==.15 and robot.records[0].acceleration==.3
    assert robot.snapshot().tcp_pose==pytest.approx(setup.home.as_list())
    certificate=next(e for e in events if e['event']=='joint_path_checked')
    assert certificate['swept_clearance_lower_bound_m']>.015
    assert events[-1]['event']=='startup_home_complete'
    assert not any(e['event']=='cycle_started' for e in events)
    robot.records.clear()
    assert prepare_startup_home(setup,robot,EmptyProbe(),log) is False
    assert robot.records==[]


@pytest.mark.parametrize('held,state',[(True,2),(None,None),(False,0),(False,3)])
def test_held_moving_unknown_or_dropped_feedback_prevents_startup(held,state):
    setup,robot=hardware_recorded_start()
    probe=EmptyProbe();probe.has_object=lambda:held;probe.last_grip_state=state
    with pytest.raises(ValueError,match='empty stationary'):
        prepare_startup_home(setup,robot,probe,lambda *a,**k:None)
    assert robot.records==[] and not robot.control_connected


def test_changed_tcp_prevents_startup_before_control():
    setup,robot=hardware_recorded_start()
    robot.active_tcp_offset=lambda:[0,0,.25,0,0,0]
    with pytest.raises(ValueError,match='TCP differs before startup'):
        prepare_startup_home(setup,robot,EmptyProbe(),lambda *a,**k:None)
    assert robot.records==[] and not robot.control_connected


def test_stopped_robot_is_required_before_startup():
    setup,robot=hardware_recorded_start();snapshot=robot.snapshot
    robot.snapshot=lambda:replace(snapshot(),runtime_state=2)
    with pytest.raises(RobotSafetyError):
        prepare_startup_home(setup,robot,EmptyProbe(),lambda *a,**k:None)
    assert robot.records==[] and not robot.control_connected


def test_old_tool_body_collision_is_still_rejected_before_move():
    setup,robot=hardware_recorded_start()
    old=json.loads((ROOT/'output/trajectory_audit_20261011/report.json').read_text(encoding='utf-8'))
    sample=old['segments'][1]['samples'][-1]
    robot._pose=Pose(*sample['tcp_m_rad']);robot._joints=sample['q_rad']
    with pytest.raises(ValueError,match='collision check failed'):
        prepare_startup_home(setup,robot,EmptyProbe(),lambda *a,**k:None)
    assert robot.records==[]
