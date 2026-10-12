"""Exercise the real driver's preflight offline, including low contact moves."""
import json
from pathlib import Path

import numpy as np
import pytest

from battery_sorting.config import Pose
from battery_sorting.folded_standby import certify_joint_path, certify_tool_body_joint_path
from battery_sorting.robot import Waypoint, RobotSafetyError
from tools.audit_t0_trajectory import ik
from .test_robot import wired_robot

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_pick_place_and_straight_retreat_pass_real_driver(valid_config, monkeypatch):
    audit = json.loads((ROOT/'output/smooth_t0/20261011_221228/audit.json').read_text(encoding='utf-8'))
    hover = next(e['pose'] for e in audit['events'] if e.get('label')=='home_to_red_hover')
    vision = next(e for e in audit['events'] if e['event']=='vision_pass')
    unfolded = next(e for e in audit['events'] if e.get('label')=='unfold_to_observation')
    observation = json.loads((ROOT/'config/folded_standby.json').read_text(encoding='utf-8'))['observation_pose_m_rad']
    q, _ = ik(hover, np.array(unfolded['q_end_rad']))
    contact = hover.copy(); contact[2] = .069207
    # Reproduce the rejected second bridge at the 79.207 mm fast endpoint.
    fast = contact.copy(); fast[2] += .010
    low_q, _ = ik(fast, q)
    with pytest.raises(ValueError, match='approaches the table'):
        certify_joint_path(low_q, low_q)
    assert certify_tool_body_joint_path(low_q, low_q)['swept_clearance_lower_bound_m']>=.01

    robot, control = wired_robot(valid_config)
    robot.tool_body_model_enabled = True
    state = dict(pose=observation, q=np.array(unfolded['q_end_rad']))
    robot._receiver.getActualTCPPose = lambda: list(state['pose'])
    robot._receiver.getActualQ = lambda: list(state['q'])
    def move_l(*args):
        control.move_calls.append(args)
        target = args[0][-1][:6] if isinstance(args[0][0], list) else args[0]
        state['q'], _ = ik(target, state['q'])
        state['pose'] = list(target)
        return True
    control.moveL = move_l
    monkeypatch.setattr('battery_sorting.robot.time.sleep', lambda _: None)
    def descend(pose):
        near = pose.copy(); near[2] += .010
        assert robot.move_l_path_without_tcp_force_guard([
            (Waypoint('fast', Pose(*near)), .05, .1, .004),
            (Waypoint('contact', Pose(*pose)), .04, .1, 0.)])
    def move(label, pose):
        assert robot.move_l_without_tcp_force_guard(Waypoint(label, Pose(*pose)), .05, .1)
    move('approach_hover', hover)
    descend(contact)
    move('lift', hover)
    place_hover = hover.copy(); place_hover[:2] = vision['place_tcp_xy_m']
    move('transfer', place_hover)
    place = place_hover.copy(); place[2] = .069207
    descend(place)
    move('straight_retreat', observation)
    assert len(control.move_calls)==6
    assert state['pose']==observation


def test_runtime_still_rejects_tool_body_collision_before_command(valid_config):
    old = json.loads((ROOT/'output/trajectory_audit_20261011/report.json').read_text(encoding='utf-8'))
    sample = old['segments'][1]['samples'][-1]
    q = sample['q_rad']; pose = sample['tcp_m_rad']
    robot, control = wired_robot(valid_config)
    robot.tool_body_model_enabled = True
    robot._receiver.getActualQ = lambda: q
    robot._receiver.getActualTCPPose = lambda: pose
    with pytest.raises((ValueError, RobotSafetyError), match='collision|intersects'):
        robot.move_l_without_tcp_force_guard(Waypoint('unsafe', Pose(*pose)), .05, .1)
    assert control.move_calls==[]


def test_calibrated_observation_joint_bridge_preserves_body_check(valid_config):
    audit = json.loads((ROOT/'output/smooth_t0/20261011_221228/audit.json').read_text(encoding='utf-8'))
    unfolded = next(e for e in audit['events'] if e.get('label')=='unfold_to_observation')
    q = unfolded['q_end_rad']
    pose = json.loads((ROOT/'config/folded_standby.json').read_text(encoding='utf-8'))['observation_pose_m_rad']
    nominal, _ = ik(pose, np.array(q))
    assert max(abs(nominal-np.array(q))) > .001
    robot, _ = wired_robot(valid_config)
    robot.tool_body_model_enabled = True
    robot._receiver.getActualQ = lambda: q
    robot._receiver.getActualTCPPose = lambda: pose
    robot._check_tool_body_path([pose])
