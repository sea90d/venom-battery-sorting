"""Check actual command grouping, contact dynamics and pre-move rejection."""
from dataclasses import replace
import pytest
from battery_sorting.closed_loop import Setup, Motion, ROOT
from battery_sorting.config import Pose
from battery_sorting.robot import FakeRobot
from tools.run_smooth_t0 import PROFILES, CONTACT_SPEED, CONTACT_ACCELERATION


def motion_at(z=.35):
    setup = Setup(ROOT/'config/closed_loop_sequence.json')
    pose = Pose(-.4,.15,z,*setup.pick_view.as_list()[3:])
    class Robot(FakeRobot):
        def active_tcp_offset(self): return setup.raw['active_tcp_offset_m_rad']
        def move_l_path_without_tcp_force_guard(self, path):
            self.paths.append(path)
            return super().move_l_path_without_tcp_force_guard(path)
    robot = Robot(pose)
    robot.paths = []
    events = []
    motion = Motion(setup,robot,lambda event,**data:events.append((event,data)))
    motion.expected=pose
    motion.continuous=True
    return setup,robot,motion,events


def test_sequence_dynamics_match_actual_first_task():
    setup,_,_,_=motion_at()
    profile=PROFILES['ten-second']
    assert (setup.transit_speed,setup.transit_acceleration)==(profile.speed,profile.acceleration)


def test_descent_is_one_blended_command_with_slow_contact():
    s,r,m,events=motion_at()
    gates=[]
    m.vertical(s.site.pick_contact_z_m,'yellow:pick',before_segment=lambda:gates.append('fresh'))
    assert len(r.paths)==1 and len(r.paths[0])==2
    first,last=r.paths[0]
    assert first[0].pose.z==pytest.approx(s.site.pick_contact_z_m+.01)
    assert first[1:3]==(.5,1.) and first[3]>0
    assert last[1:]==(CONTACT_SPEED,CONTACT_ACCELERATION,0.)
    assert m.expected.z==s.site.pick_contact_z_m
    assert gates==['fresh','fresh']


def test_lift_and_horizontal_transfer_have_single_endpoints():
    s,r,m,_=motion_at(.069207)
    checks=[]
    m.vertical(.35,'yellow:lift',check=lambda:checks.append('held'))
    target=replace(m.expected,x=-.6,y=.25)
    m.high(target,'yellow:place_high',check=lambda:checks.append('held'))
    assert [len(path) for path in r.paths]==[1,1]
    assert all(path[0][1:]==(.5,1.,0.) for path in r.paths)
    assert len(checks)==4
    assert m.expected==target


def test_workspace_and_stale_observation_reject_before_command():
    s,r,m,_=motion_at()
    with pytest.raises(ValueError,match='workspace'):
        m.vertical(.01,'bad')
    assert r.paths==[]
    def stale(): raise ValueError('stale observation')
    with pytest.raises(ValueError,match='stale'):
        m.vertical(s.site.pick_contact_z_m,'yellow:pick',before_segment=stale)
    assert r.paths==[]


def test_sequence_skips_all_post_release_photos_and_containment(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from battery_sorting.closed_loop import Runtime
    from battery_sorting.gripper import MockGripper
    import battery_sorting.folded_standby as folded
    s,r,_,_=motion_at()
    s.site_test=True
    r._pose=s.home
    class Gripper(MockGripper):
        def position(self):return 239
    clock=[0.]
    rt=Runtime(s,SimpleNamespace(),r,Gripper(True),tmp_path/'run',
               sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds),clock=lambda:clock[0])
    rt.skip_placement_verification=True
    def transition(setup,robot,destination,log):
        robot._pose=setup.home if destination=='home' else setup.pick_view
        return robot._pose
    monkeypatch.setattr(folded,'transition',transition)
    colors=['yellow','green','blue']
    objects={color:SimpleNamespace(color=color,center_base_m=[-.5+i*.1,.02,.04],
                                  handle={'yaw_rad':0.,'center_base_m':[-.5+i*.1,.02,.05]})
             for i,color in enumerate(colors)}
    labels=[]
    def scene(label,roi,include_colors):
        labels.append(label)
        assert label in ('before_pick','grasp_hover')
        return SimpleNamespace(require_live=lambda *args:None),{c:objects[c] for c in include_colors}
    rt.scene=scene
    rt.scan_targets=lambda task:rt.targets.update({f'P{i+1}':{'center':[-.55,.10+i*.08],
                                                               'yaw':0.,'kind':'p_ring'} for i in range(3)})
    rt.validate_target_path=lambda *args:None
    rt.placement_xy=lambda item,target,task,yaw:target['center']
    def forbidden(*args):raise AssertionError('Placement verification called')
    rt.verify_board=forbidden
    rt.inside_target=forbidden
    timing=rt.run('sequence',colors)
    assert labels==['before_pick','grasp_hover']*3
    assert all(item['placement_verified'] is False for item in timing['battery_run_times'])
    summary=json.loads((rt.output/'summary.json').read_text(encoding='utf-8'))
    assert summary['completed'] is True and summary['placement_verified'] is False
    assert r.snapshot().tcp_pose==pytest.approx(s.home.as_list())
