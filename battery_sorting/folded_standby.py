"""Explicit joint standby and coordinated unfolding; no hardware at import time."""
import json
import math
from pathlib import Path
import numpy as np
from .config import Pose
from .robot import Waypoint, assert_pose_close

ROOT = Path(__file__).resolve().parents[1]


def plan_for(setup):
    path = ROOT / 'config/folded_standby.json'
    if not path.exists():
        return None
    plan = json.loads(path.read_text(encoding='utf-8'))
    # Historical scene configs retain their original motion convention.
    if not np.allclose(setup.home.as_list(), plan['home_pose_m_rad'], atol=1e-8, rtol=0):
        return None
    if not np.allclose(setup.raw['target_view_pose_m_rad'], plan['observation_pose_m_rad'], atol=1e-8, rtol=0):
        raise ValueError('Folded standby observation configuration differs')
    return plan


def joint_seconds(first, last, speed, acceleration):
    distance = max(abs(a-b) for a,b in zip(first,last))
    return distance/speed+speed/acceleration if distance>=speed*speed/acceleration else 2*math.sqrt(distance/acceleration)


def certify_tool_body_joint_path(first, last):
    """Check tool/robot separation without a fold-only table height rule."""
    from tools.simulate_tool_body_collision import clearance, check_interval
    first, last = np.array(first,float), np.array(last,float)
    if first.shape!=(6,) or last.shape!=(6,) or not np.isfinite([first,last]).all():
        raise ValueError('Invalid joint path')
    lower,count,failure = check_interval(first,last,clearance(first)[0],clearance(last)[0])
    if failure or lower < .010:
        raise ValueError(f'Tool-body collision check failed: {failure}, lower={lower}')
    return dict(swept_clearance_lower_bound_m=lower, extra_sweep_checks=count,
                model='nominal UR5 / full 246 mm tool, radius 75 mm')


def certify_joint_path(first, last):
    """Fold/unfold additionally requires clearance above the table."""
    from tools.audit_t0_trajectory import fk
    certificate = certify_tool_body_joint_path(first, last)
    first, last = np.array(first,float), np.array(last,float)
    # Even a horizontal tool must stay above the paper over the entire path.
    lowest = min(min((t:=fk(first+(last-first)*f)[0])[2,3],(t[:3,3]+t[:3,2]*.246)[2])
                 for f in np.linspace(0,1,257))
    if lowest-.075 < .030:
        raise ValueError('Fold/unfold tool envelope approaches the table')
    return dict(certificate, lowest_tool_axis_z_m=float(lowest))


def prepare_startup_home(setup, robot, gripper, log):
    """Return an empty, stopped robot to standby before the timed task.

    Returns whether RTDE Control was connected here. No gripper opening or
    initialization is allowed during this return; the timed runner does that
    after arrival. Every move uses the calibrated IK and full tool sweep.
    """
    from .robot import require_snapshot_state, RUNTIME_STATE_STOPPED, rotation_error
    before = robot.snapshot()
    if math.dist(before.tcp_pose[:3],setup.home.as_list()[:3])<=.001 and rotation_error(before.tcp_pose[3:],setup.home.as_list()[3:])<=math.radians(.5):
        log('startup_home_complete',pose=list(before.tcp_pose),already_home=True)
        return False
    if plan_for(setup) is None:
        assert_pose_close(before.tcp_pose,setup.home,.001,math.radians(.5),'home before start')
    require_snapshot_state(before,{RUNTIME_STATE_STOPPED},'Startup folded return')
    if robot.is_real:
        from .tcp_readonly import read_tcp_info
        offset=read_tcp_info(setup.site.robot_host)['active_tcp_offset_m_rad']
    else:
        offset=robot.active_tcp_offset()
    expected=setup.raw['active_tcp_offset_m_rad']
    if math.dist(offset[:3],expected[:3])>.0001 or rotation_error(offset[3:],expected[3:])>math.radians(.05):
        raise ValueError('Active TCP differs before startup return')
    # Feedback reads only: never open a possibly loaded gripper to pass a gate.
    if gripper.has_object() is not False or gripper.last_grip_state!=1:
        raise ValueError('Startup return requires an empty stationary gripper')
    checked=robot.snapshot()
    require_snapshot_state(checked,{RUNTIME_STATE_STOPPED},'Startup folded return checked')
    assert_pose_close(checked.tcp_pose,Pose(*before.tcp_pose),.0002,math.radians(.1),'Stationary before startup return')
    log('startup_return_plan',actual_start=list(before.tcp_pose),targets=[setup.home.as_list()],motion_space='joint')
    robot.connect_control()
    assert_pose_close(robot.snapshot().tcp_pose,Pose(*before.tcp_pose),.0002,math.radians(.1),'Startup return handover')
    transition(setup,robot,'home',log,slow=True)
    log('startup_home_complete',pose=list(robot.snapshot().tcp_pose),already_home=False)
    return True


def transition(setup, robot, destination, log, *, slow=False):
    plan = plan_for(setup)
    if plan is None:
        raise ValueError('Explicit folded standby plan unavailable')
    robot.tool_body_model_enabled=True
    offset=robot.active_tcp_offset()
    expected=setup.raw['active_tcp_offset_m_rad']
    from .robot import rotation_error
    if math.dist(offset[:3],expected[:3])>.0001 or rotation_error(offset[3:],expected[3:])>math.radians(.05):
        raise ValueError('Active TCP differs from the folded tool model')
    key = 'home' if destination=='home' else 'observation'
    target = Pose(*plan[key+'_pose_m_rad'])
    seed = plan[key+'_joints_rad']
    joints = robot.joints_for_pose(target, seed)
    current = robot.actual_joints()
    branch_delta=max(abs(a-b) for a,b in zip(joints,seed))
    if branch_delta > math.radians(10):
        # UR's numerical IK can occasionally return an alternate wrist branch
        # even with the taught joints as seed.  Never accept that branch.  On
        # a real controller, use the explicitly taught joints only when its
        # calibrated FK reproduces the requested TCP exactly; FakeRobot keeps
        # the rejection path deterministic for offline tests.
        if not robot.is_real or not hasattr(robot,'forward_kinematics'):
            raise ValueError('Controller IK changed the specified fold/unfold joint branch')
        configured_fk=robot.forward_kinematics(seed)
        assert_pose_close(configured_fk,target,.0002,math.radians(.1),'Configured fold/unfold target')
        joints=list(seed)
        log('folded_ik_branch_fallback',destination=destination,
            controller_branch_delta_deg=math.degrees(branch_delta),source='configured_taught_joints')
    if max(abs(a-b) for a,b in zip(current,joints)) < math.radians(.1):
        assert_pose_close(robot.snapshot().tcp_pose,target,.001,math.radians(.5),'Folded standby already reached')
        return target
    certificate = certify_joint_path(current,joints)
    speed = .15 if slow else plan['joint_speed_rad_s']
    acceleration = .3 if slow else plan['joint_acceleration_rad_s2']
    label = 'return_folded_standby' if key=='home' else 'unfold_to_observation'
    log('joint_path_checked',label=label,q_start_rad=list(current),q_end_rad=list(joints),**certificate)
    if not robot.move_j(Waypoint(label,target),joints,speed,acceleration):
        raise RuntimeError('Robot rejected '+label)
    assert_pose_close(robot.snapshot().tcp_pose,target,.001,math.radians(.5),label+' arrival')
    log('joint_move',label=label,pose=target.as_list(),joints_rad=list(joints),
        speed_rad_s=speed,acceleration_rad_s2=acceleration)
    return target
