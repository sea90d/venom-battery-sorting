"""Fixed taught sequence; default prints a plan without connecting hardware."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import uuid
import tempfile

from .closed_loop import ROOT, Setup, Runtime, finite_vector
from .config import Pose
from .folded_standby import prepare_startup_home, plan_for, joint_seconds
from .gripper import create_gripper
from .robot import RealUR5, FakeRobot, Waypoint, assert_pose_close, rotation_error, RobotSafetyError, require_snapshot_state, RUNTIME_STATE_STOPPED
from .simulation_export import nominal_motion_seconds
from .smooth_dynamics import BLEND_M, CONTACT_SPEED_MPS, CONTACT_ACCELERATION_MPS2
from .tcp_readonly import read_tcp_info

PLAN = ROOT / 'config/fixed_sequence.json'
CONFIG = ROOT / 'config/closed_loop_sequence.json'


def load_plan(setup, path=PLAN):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or data.get('mode') != 'fixed_sequence_open_loop':
        raise ValueError('Invalid fixed sequence schema')
    offset = finite_vector(data['active_tcp_offset_m_rad'], 6, 'fixed TCP')
    require_tcp(offset, setup.raw['active_tcp_offset_m_rad'])
    transit_z = finite_vector([data['transit_z_m']], 1, 'fixed transit Z')[0]
    if not setup.site.pick_contact_z_m+.10 <= transit_z <= setup.site.safe_high_z_m:
        raise ValueError('Fixed transit height must leave at least 100 mm above contact')
    setup.raw['transit_high_z_m'] = transit_z
    steps = data['steps']
    if [(s['color'], s['target']) for s in steps] != [('yellow', 'P1'), ('green', 'P2'), ('blue', 'P3')]:
        raise ValueError('Fixed sequence must be yellow/P1, green/P2, blue/P3')
    for step in steps:
        for kind, height in (('pick', setup.site.pick_contact_z_m), ('place', setup.site.placement_contact_z_m)):
            pose = Pose(*finite_vector(step[kind+'_pose_m_rad'], 6, kind))
            if not setup.in_workspace(pose) or not setup.in_workspace(pose.with_z(setup.transit_high_z)):
                raise ValueError('Fixed point outside workspace: '+step['color']+':'+kind)
            if abs(pose.z-height) > 1e-6:
                raise ValueError('Fixed contact Z differs from configured Z')
        if rotation_error(step['pick_pose_m_rad'][3:], step['place_pose_m_rad'][3:]) > 1e-6:
            raise ValueError('Fixed carry must preserve grasp orientation')
    return data


def require_tcp(actual, expected):
    if math.dist(actual[:3], expected[:3]) > .0001 or rotation_error(actual[3:], expected[3:]) > math.radians(.05):
        raise ValueError('Active TCP differs from fixed taught TCP')


class FixedRuntime(Runtime):
    def fold_transition(self, destination):
        s = self.setup
        if destination == 'home':
            target = s.home
        else:
            target = Pose(s.pick_view.x, s.pick_view.y, s.transit_high_z,
                          *self.fixed_plan['steps'][0]['pick_pose_m_rad'][3:])
        label = 'return_folded_standby' if destination == 'home' else 'unfold_to_fixed_entry'
        self.robot.move_fixed_joint(target, label)
        self.log('joint_move', label=label, pose=target.as_list())
        return target

    def path_to_contact(self, pose, label, check=None, lift=False):
        s = self.setup
        rows = []
        if lift:
            rows.append((Waypoint(label+':lift', self.motion.expected.with_z(s.transit_high_z)),
                         s.transit_speed, s.transit_acceleration, BLEND_M))
        rows.extend([
            (Waypoint(label+':hover', pose.with_z(s.transit_high_z)), s.transit_speed, s.transit_acceleration, BLEND_M),
            (Waypoint(label+':approach', pose.with_z(pose.z+s.grasp.final_slow_distance_m)),
             s.transit_speed, s.transit_acceleration, min(BLEND_M, s.grasp.final_slow_distance_m/3)),
            (Waypoint(label+':contact', pose), CONTACT_SPEED_MPS, CONTACT_ACCELERATION_MPS2, 0.),
        ])
        self.motion.continuous_path(rows, label, check=check)

    def empty_check(self):
        if self.gripper.has_object() is not False or self.gripper.last_grip_state != 1:
            raise RuntimeError('Release/empty stationary feedback missing')

    def run_fixed(self, plan):
        s = self.setup
        self.task = 'sequence'
        self.motion.continuous = True
        self.fixed_plan = plan
        started = None
        times = []
        try:
            assert_pose_close(self.robot.snapshot().tcp_pose, s.home, .001, math.radians(.5), 'fixed start home')
            if not self.gripper.initialize(recalibrate=False) or not self.gripper.open():
                raise RuntimeError('Gripper initialization/open failed')
            self.empty_check()
            started = self.clock()
            self.log('cycle_started', task='sequence', mode=plan['mode'], vision_enabled=False)
            self.motion.expected = self.fold_transition('observation')
            for i, step in enumerate(plan['steps']):
                battery_started = self.clock()
                pick, place = (Pose(*step[k+'_pose_m_rad']) for k in ('pick', 'place'))
                color = step['color']
                # Rotate only at the transit plane, never beside the other handles.
                if i:
                    self.motion.vertical(s.transit_high_z, color+':retreat', check=self.empty_check)
                orient = replace(self.motion.expected, rx=pick.rx, ry=pick.ry, rz=pick.rz)
                self.motion.high(orient, color+':orient', check=self.empty_check)
                self.path_to_contact(pick, color+':pick', self.empty_check)
                if not self.gripper.close():
                    raise RuntimeError('Gripper close failed')
                self.grip_check()
                self.log('grasped', color=color, position=self.gripper.position())
                self.path_to_contact(place, color+':place', self.grip_check, lift=True)
                if not self.gripper.open():
                    raise RuntimeError('Gripper release failed')
                self.empty_check()
                released_at = self.clock()
                self.log('released', color=color, target=step['target'], verification_skipped=True)
                item = dict(color=color, target=step['target'], run_time_s=self.clock()-battery_started,
                            placement_verified=False)
                times.append(item)
                self.log('battery_complete', **item)
            self.motion.vertical(s.transit_high_z, 'fixed:final_retreat', check=self.empty_check)
            corridor = Pose(s.pick_view.x, s.pick_view.y, s.transit_high_z, *self.motion.expected.as_list()[3:])
            self.motion.high(corridor, 'fixed:return_corridor', check=self.empty_check)
            self.motion.expected = self.fold_transition('home')
            # The final release interval overlaps return; no camera or visual verdict.
            remaining = released_at+s.site.timing.release_settle_s-self.clock()
            if remaining > 0:
                self.sleep(remaining)
            self.empty_check()
            timing = dict(task_run_time_s=self.clock()-started, battery_run_times=times,
                          timing_scope='unfold through fixed commanded placements and folded home; no vision')
            summary = dict(completed=True, task='sequence', mode=plan['mode'], vision_enabled=False,
                           placement_verified=False, timing=timing, source=plan['source'],
                           within_30_seconds=bool(timing['task_run_time_s'] <= 30.),
                           final_pose_m_rad=list(self.robot.snapshot().tcp_pose))
            (self.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            self.log('home_arrived', task_run_time_s=timing['task_run_time_s'])
            self.log('complete', **summary)
            print(json.dumps(dict(TASK_RUN_TIME_S=timing['task_run_time_s'], battery_run_times=times)), flush=True)
            return timing
        except BaseException as exc:
            errors = []
            for name, stop in (('robot', lambda: self.robot.stop_l(1.)), ('gripper', self.gripper.stop)):
                try:
                    if stop() is False:
                        errors.append(name+': stop not acknowledged')
                except Exception as error:
                    errors.append(name+': '+str(error))
            self.log('failed', error=str(exc), stop_errors=errors)
            raise


def compile_plan(setup, plan, *, grip_command_s=.6):
    """Run exactly the same scheduler on virtual devices, with stop-at-point bounds."""
    commands = []
    now = [0.]
    class Robot(FakeRobot):
        def __init__(self, pose):
            super().__init__(pose)
            self._joints = plan_for(setup)['home_joints_rad']
        def move_fixed_joint(self, target, label):
            from tools.audit_t0_trajectory import ik
            import numpy as np
            seed = plan_for(setup)['home_joints_rad'] if label=='return_folded_standby' else self._joints
            q, error = ik(target.as_list(), np.asarray(seed))
            if np.linalg.norm(error) > .0002:
                raise ValueError('Fixed entry nominal IK failed')
            commands.append(dict(kind='joint', start=self.snapshot().tcp_pose,
                                 target=target.as_list(), seed=q.tolist(), label=label))
            folded = plan_for(setup)
            now[0] += float(joint_seconds(self._joints, q, folded['joint_speed_rad_s'], folded['joint_acceleration_rad_s2']))+.08
            self._joints=q.tolist()
            self._pose=target
            return True
        def active_tcp_offset(self):
            return setup.raw['active_tcp_offset_m_rad']
        def move_l_path_without_tcp_force_guard(self, path):
            commands.append(dict(kind='cartesian', start=self.snapshot().tcp_pose,
                                 poses=[r[0].pose.as_list() for r in path]))
            previous = self.snapshot().tcp_pose
            for waypoint, speed, acceleration, blend in path:
                now[0] += nominal_motion_seconds(previous, waypoint.pose.as_list(), speed, acceleration)
                previous = waypoint.pose.as_list()
            now[0] += .08  # five stable samples at the existing monitor rate
            result = super().move_l_path_without_tcp_force_guard(path)
            from tools.audit_t0_trajectory import ik
            import numpy as np
            q, error = ik(previous, np.asarray(self._joints))
            if np.linalg.norm(error) > .0002:
                raise ValueError('Fixed Cartesian endpoint nominal IK failed')
            self._joints=q.tolist()
            return result
    class Gripper:
        last_grip_state = 1
        held = False
        def initialize(self, **kwargs): return True
        def open(self): self.held=False; now[0]+=grip_command_s; return True
        def close(self): self.held=True; now[0]+=grip_command_s; return True
        def has_object(self): return self.held
        def position(self): return 239 if self.held else 1000
        def stop(self): return True
    class Planner(FixedRuntime):
        def log(self, event, **details): pass
    robot = Robot(setup.home)
    with tempfile.TemporaryDirectory(prefix='fixed_sequence_plan_') as temp:
        runtime = Planner(setup, None, robot, Gripper(), Path(temp)/'run',
                          clock=lambda:now[0], sleep=lambda sec:now.__setitem__(0,now[0]+sec))
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            timing = runtime.run_fixed(plan)
    return commands, dict(nominal_seconds=timing['task_run_time_s'], target_seconds=30.,
                          grip_command_assumption_s=grip_command_s,
                          excludes='preparation and communication; not a measured result')


class FixedRobot(RealUR5):
    """Certify the fixed Cartesian commands before timing; consume them in order."""
    def prepare_fixed_paths(self, commands, progress=None, *, check_trajectory=True):
        self.trajectory_checks_enabled = check_trajectory
        if not check_trajectory:
            self.tool_body_model_enabled = False
            self.fixed_certificates = [dict(cmd) for cmd in commands]
            self.fixed_certificate_index = 0
            return self.fixed_certificates
        self.tool_body_model_enabled = True
        q = self.actual_joints()
        certificates = []
        for index, cmd in enumerate(commands, 1):
            if progress:
                progress(index, len(commands))
            if cmd['kind']=='joint':
                from tools.simulate_tool_body_collision import check_interval, clearance
                from tools.audit_t0_trajectory import fk
                import numpy as np
                last = self.joints_for_pose(Pose(*cmd['target']), cmd['seed'])
                if max(abs(a-b) for a,b in zip(last,cmd['seed'])) > math.radians(10):
                    assert_pose_close(self.forward_kinematics(cmd['seed']), Pose(*cmd['target']), .0002, math.radians(.1), 'Fixed joint branch')
                    last = cmd['seed']
                lower, count, failure = check_interval(np.asarray(q), np.asarray(last), clearance(q)[0], clearance(last)[0], required_clearance_m=.020)
                lowest = min(min((t:=fk(np.asarray(q)+(np.asarray(last)-q)*f)[0])[2,3],
                                 (t[:3,3]+t[:3,2]*.246)[2]) for f in np.linspace(0,1,257))
                if failure or lowest-.075 < .030:
                    raise RobotSafetyError('Fixed fold/entry collision or table clearance check failed')
                certificates.append(dict(**cmd, start_joints=list(q), end_joints=list(last),
                                         swept_clearance_lower_bound_m=lower))
                q = last
                continue
            for pose in cmd['poses']:
                if not self.pose_within_safety_limits(pose):
                    raise RobotSafetyError('Controller rejects fixed prepared point')
            result = super()._check_tool_body_path(cmd['poses'], start_pose=cmd['start'],
                                                  start_joints=q, required_clearance_m=.020)
            certificates.append(dict(**cmd, start_joints=list(q), **result))
            q = result['end_joints']
        self.fixed_certificates = certificates
        self.fixed_certificate_index = 0
        return certificates

    def _prepared(self, kind):
        # Changed plans/branches fail closed instead of doing slow replanning in the timed loop.
        index = self.fixed_certificate_index
        if index >= len(self.fixed_certificates):
            raise RobotSafetyError('Unprepared fixed motion')
        cert = self.fixed_certificates[index]
        if kind != cert['kind']:
            raise RobotSafetyError('Fixed motion order differs from prepared path')
        assert_pose_close(self.snapshot().tcp_pose, Pose(*cert['start']), .0005, math.radians(.2), 'Prepared fixed start')
        if not getattr(self, 'trajectory_checks_enabled', True):
            return cert
        from tools.audit_t0_trajectory import A, D
        reach = sum(abs(v) for v in A)+sum(abs(v) for v in D)+.246+.2
        deviation = sum(abs(a-b) for a,b in zip(self.actual_joints(),cert['start_joints']))
        if 2*reach*deviation > cert['swept_clearance_lower_bound_m']-.010:
            raise RobotSafetyError('Actual joint branch exceeds prepared clearance margin')
        return cert

    def _check_tool_body_path(self, poses):
        cert = self._prepared('cartesian')
        if poses != cert['poses']:
            raise RobotSafetyError('Fixed motion differs from prepared path')
        self.fixed_certificate_index += 1

    def move_fixed_joint(self, target, label):
        require_tcp(self.active_tcp_offset(), self.fixed_tcp_offset)
        cert = self._prepared('joint')
        if target.as_list() != cert['target'] or label != cert['label']:
            raise RobotSafetyError('Fixed joint target differs from prepared path')
        folded = json.loads((ROOT/'config/folded_standby.json').read_text(encoding='utf-8'))
        if 'end_joints' not in cert:
            cert['end_joints'] = self.joints_for_pose(target, cert['seed'])
            if max(abs(a-b) for a,b in zip(cert['end_joints'],cert['seed'])) > math.radians(10):
                assert_pose_close(self.forward_kinematics(cert['seed']), target, .0002, math.radians(.1), 'Fixed joint branch')
                cert['end_joints'] = cert['seed']
        if not self.move_j(Waypoint(label, target), cert['end_joints'],
                           folded['joint_speed_rad_s'], folded['joint_acceleration_rad_s2']):
            raise RuntimeError('Fixed joint command rejected')
        self.fixed_certificate_index += 1


def startup_home_without_trajectory_check(setup, robot, gripper, log):
    before = robot.snapshot()
    if (math.dist(before.tcp_pose[:3], setup.home.as_list()[:3]) <= .001 and
            rotation_error(before.tcp_pose[3:], setup.home.as_list()[3:]) <= math.radians(.5)):
        log('startup_home_complete', pose=list(before.tcp_pose), already_home=True)
        return False
    require_snapshot_state(before, {RUNTIME_STATE_STOPPED}, 'Fixed startup return')
    if gripper.has_object() is not False or gripper.last_grip_state != 1:
        raise ValueError('Startup return requires an empty stationary gripper')
    require_tcp(read_tcp_info(setup.site.robot_host)['active_tcp_offset_m_rad'], setup.raw['active_tcp_offset_m_rad'])
    assert_pose_close(robot.snapshot().tcp_pose, Pose(*before.tcp_pose), .0002, math.radians(.1), 'Stationary startup return')
    robot.connect_control()
    assert_pose_close(robot.snapshot().tcp_pose, Pose(*before.tcp_pose), .0002, math.radians(.1), 'Startup control handover')
    require_tcp(robot.active_tcp_offset(), setup.raw['active_tcp_offset_m_rad'])
    seed = plan_for(setup)['home_joints_rad']
    joints = robot.joints_for_pose(setup.home, seed)
    if max(abs(a-b) for a,b in zip(joints,seed)) > math.radians(10):
        assert_pose_close(robot.forward_kinematics(seed), setup.home, .0002, math.radians(.1), 'Startup home branch')
        joints = seed
    log('startup_return_plan', trajectory_checked=False, targets=[setup.home.as_list()])
    if not robot.move_j(Waypoint('return_folded_standby', setup.home), joints, .15, .3):
        raise RuntimeError('Startup return rejected')
    log('startup_home_complete', pose=list(robot.snapshot().tcp_pose), already_home=False)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'preflight', 'run'), nargs='?', default='plan')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--plan', type=Path, default=PLAN)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--auto-start', action='store_true')
    parser.add_argument('--check-trajectory', action='store_true', help='Opt in to slow full trajectory/collision certification; default skips it')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    setup = Setup(args.config)
    setup.require_ready('sequence', site_test=True)
    plan = load_plan(setup, args.plan)
    commands, budget = compile_plan(setup, plan)
    if args.command == 'plan':
        print(json.dumps(dict(**plan, timing_budget=budget), indent=2, ensure_ascii=False))
        return 0
    if args.command == 'run' and not args.execute:
        raise ValueError('run requires --execute')
    if args.command == 'run' and not args.auto_start:
        phrase = 'EXECUTE FIXED SEQUENCE'
        if input(phrase+'\n> ').strip() != phrase:
            raise ValueError('Start confirmation mismatch')
    output = args.output or ROOT/'output/fixed_sequence'/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    if args.command == 'run' and output.exists():
        raise ValueError('Output directory already exists')
    robot = FixedRobot(replace(setup.robot_config(), stop_deceleration=1.), motion_profile='t0_timing_trial')
    gripper = create_gripper(setup.grasp.gripper)
    runtime = None
    try:
        robot.connect_readonly()
        require_tcp(read_tcp_info(setup.site.robot_host)['active_tcp_offset_m_rad'], plan['active_tcp_offset_m_rad'])
        gripper.connect()
        if args.command == 'preflight':
            print(json.dumps(dict(robot='ok', tcp='ok', gripper_position=gripper.position(),
                                  has_object=gripper.has_object(), camera_required=False)))
            return 0
        runtime = FixedRuntime(setup, None, robot, gripper, output)
        runtime.log('fixed_plan_loaded', **plan)
        home_preparer = prepare_startup_home if args.check_trajectory else startup_home_without_trajectory_check
        if not home_preparer(setup, robot, gripper, runtime.log):
            robot.connect_control()
        robot.tool_body_model_enabled = True
        require_tcp(robot.active_tcp_offset(), plan['active_tcp_offset_m_rad'])
        robot.fixed_tcp_offset = plan['active_tcp_offset_m_rad']
        if args.check_trajectory:
            runtime.log('fixed_paths_preparing', timing_budget=budget)
        else:
            runtime.log('fixed_trajectory_checks_skipped', trajectory_checked=False)
            print('Full trajectory/collision checks skipped; executing fixed taught points.', flush=True)
        def progress(index, total):
            runtime.log('fixed_path_progress', index=index, total=total)
            print(f'Preparing fixed path {index}/{total}...', flush=True)
        certificates = robot.prepare_fixed_paths(commands, progress=progress, check_trajectory=args.check_trajectory)
        runtime.log('fixed_paths_prepared' if args.check_trajectory else 'fixed_points_ready',
                    count=len(certificates), timing_budget=budget, trajectory_checked=args.check_trajectory)
        runtime.run_fixed(plan)
        return 0
    except BaseException as exc:
        if runtime is not None and (not runtime.events or runtime.events[-1]['event'] != 'failed'):
            runtime.log('failed', error=str(exc), phase='startup')
        raise
    finally:
        try:
            robot.disconnect()
        finally:
            gripper.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
