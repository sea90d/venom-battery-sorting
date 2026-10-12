import pytest
import math
from dataclasses import replace

from battery_sorting.config import Pose
from battery_sorting.offsite import Pose6
from tools.run_smooth_t0 import SmoothMotion,BOUNDS,TRANSIT_SPEED,CONTACT_SPEED,ACCELERATION,BLEND_RADIUS,startup_home_targets


def test_low_speed_plan_prints_matching_execution_command(monkeypatch,tmp_path,capsys):
    import json
    import sys
    from tools.run_smooth_t0 import main
    monkeypatch.setattr(sys,'argv',['run_smooth_t0.py','--profile','trajectory-check','--output',str(tmp_path/'plan')])
    main()
    plan=json.loads(capsys.readouterr().out)
    assert plan['motion_profile']['transit_speed_mps']==.05
    assert '--profile trajectory-check' in plan['command']
from tools.run_smooth_t0 import execute_startup_return,START_BOUNDS
from tools.run_smooth_t0 import descend_to_pick,PICK_OPEN_POSITION
from tools.run_smooth_t0 import transfer_setup,DEFAULT_TRANSFER_LIFT_M
from tools.run_smooth_t0 import require_competition_start


def test_production_home_clears_whole_printed_zone_and_is_sideways():
    from battery_sorting.closed_loop import Setup
    from tools.run_smooth_t0 import ROOT
    setup=Setup(ROOT/'config/closed_loop.json')
    check=require_competition_start(setup.home)
    assert check['tool_envelope_margin_m']>=.02
    assert check['parallel_tilt_deg']<1e-5
    assert setup.raw['target_view_pose_m_rad'][:3]!=setup.home.as_list()[:3]
    assert setup.raw['target_view_pose_m_rad'][3:]!=setup.home.as_list()[3:]


@pytest.mark.parametrize('x',[-.52,-.45,-.36])
def test_start_rejects_tcp_or_tool_footprint_above_battery_zone(x):
    import math
    with pytest.raises(ValueError,match='overlaps battery zone'):
        require_competition_start(Pose(x,-.15,.447856,math.pi/2,0.,0.))


def test_start_rejects_tilted_and_upward_tool_planes():
    import math
    for orientation in [(0.,0.,0.),(math.pi,0.,0.),(math.pi/2-.02,0.,0.)]:
        with pytest.raises(ValueError,match='not parallel'):
            require_competition_start(Pose(-.06,-.15,.447856,*orientation))


def test_start_checks_flange_to_tcp_body_instead_of_tcp_point_only():
    import math
    # TCP is outside, but the horizontal tool body extends back over the zone.
    with pytest.raises(ValueError,match='overlaps battery zone'):
        require_competition_start(Pose(-.06,-.15,.447856,0.,math.pi/2,0.))


def test_default_loaded_lift_is_150mm(tmp_path):
    rt,_,_=environment(tmp_path)
    carry=transfer_setup(rt.setup,None)
    assert DEFAULT_TRANSFER_LIFT_M==.15
    assert carry.site.safe_high_z_m==pytest.approx(.219207)
    assert carry.site.safe_high_z_m-rt.setup.site.pick_contact_z_m==pytest.approx(.15)
    assert rt.setup.site.safe_high_z_m==pytest.approx(.447856)
from battery_sorting.config import GripperConfig
from .closed_loop_fakes import environment


def motion(tmp_path):
    rt,_,_=environment(tmp_path)
    start=Pose(0.,-.25,.447856,3.1415,0.,0.)
    rt.robot._pose=start
    rt.setup.site=replace(rt.setup.site,home_pose_m_rad=Pose6(*start.as_list()),workspace_bounds_m=BOUNDS)
    events=[]
    return SmoothMotion(rt.setup,rt.robot,start,lambda event,**detail:events.append((event,detail))),rt.robot,events


def test_smooth_pick_uses_one_high_leg_and_one_two_waypoint_path(tmp_path):
    m,robot,events=motion(tmp_path)
    hover=Pose(.20,-.34,.447856,3.0,.2,0.)
    m.single(hover,'hover')
    m.descend(.069207,'pick')
    assert [r.label for r in robot.records]==['hover','pick:fast','pick:contact']
    assert [r.speed for r in robot.records]==[TRANSIT_SPEED,TRANSIT_SPEED,CONTACT_SPEED]
    assert all(r.acceleration==ACCELERATION for r in robot.records)
    assert events[-1][1]['blend_m']==BLEND_RADIUS


def test_smooth_motion_rejects_lateral_low_move(tmp_path):
    m,_,_=motion(tmp_path)
    m.descend(.069207,'pick')
    with pytest.raises(ValueError,match='Lateral/rotation'):
        m.single(Pose(.01,-.25,.069207,3.1415,0,0),'bad')


def test_fast_v5_300mmps_transfer_profile():
    assert TRANSIT_SPEED==pytest.approx(.300)
    assert CONTACT_SPEED==pytest.approx(.020*2)
    assert ACCELERATION==pytest.approx(.100)


def test_tiny_vertical_normalization_allows_next_high_transfer(tmp_path):
    m,robot,_=motion(tmp_path)
    actual=replace(m.expected,z=m.expected.z+.00004)
    robot._pose=actual;m.expected=actual
    m.single(actual.with_z(.447856),'normalize',CONTACT_SPEED)
    m.single(Pose(.20,-.34,.447856,3.0,.2,0.),'high_transfer')
    assert [r.label for r in robot.records]==['normalize','high_transfer']


def test_startup_home_plan_lifts_before_lateral_motion():
    home=Pose(-.046,-.22,.447856,0.,-3.14,0.)
    low=Pose(.20,-.34,.069207,-2.2,-2.1,0.)
    plan=startup_home_targets(low,home,.447856,home)
    assert plan[0]==low.with_z(.447856)
    assert plan[-1]==home
    assert plan[1].x==-.40
    for a,b in zip([low]+plan,plan):
        assert math.dist(a.as_list()[:3],b.as_list()[:3])<1e-9 or a.as_list()[3:]==b.as_list()[3:]
    assert startup_home_targets(home,home,.447856,home)==[]


def test_startup_home_rejects_unknown_outside_pose():
    home=Pose(-.046,-.22,.447856,0.,-3.14,0.)
    with pytest.raises(ValueError,match='outside'):
        startup_home_targets(Pose(.50,-.34,.069207,0,0,0),home,.447856,home)


def test_startup_replans_exact_vertical_from_post_control_pose(tmp_path):
    rt,_,_=environment(tmp_path)
    home=Pose(-.046,-.22,.447856,0.,-math.pi,0.)
    original=Pose(.20,-.34,.069207,-2.2,-2.1,0.)
    before=replace(original,x=original.x+.00004,ry=original.ry+.00002)
    actual=replace(before,y=before.y-.00003,z=before.z+.00002,rx=before.rx+.00001)
    rt.setup.site=replace(rt.setup.site,home_pose_m_rad=Pose6(*home.as_list()),workspace_bounds_m=START_BOUNDS)
    rt.setup.raw['target_view_pose_m_rad']=home.as_list()
    rt.setup.raw['pick_view_pose_m_rad']=home.as_list()
    rt.robot._pose=actual
    # Reproduce the original error with the stale, pre-connection plan.
    old=SmoothMotion(rt.setup,rt.robot,before,lambda *a,**k:None)
    with pytest.raises(ValueError,match='Lateral/rotation'):
        old.single(startup_home_targets(original,home,.447856,home)[0],'stale')
    execute_startup_return(rt.setup,rt.robot,before,lambda *a,**k:None)
    first,last=rt.robot.records[0],rt.robot.records[-1]
    assert first.label=='startup_vertical_clearance'
    assert first.pose==actual.with_z(.447856)
    assert last.pose==home
    assert len(rt.robot.records)>2


@pytest.mark.parametrize('change',[{'x':.201},{'rx':-2.18}])
def test_startup_real_handover_drift_still_blocks(tmp_path,change):
    rt,_,_=environment(tmp_path)
    before=Pose(.20,-.34,.069207,-2.2,-2.1,0.)
    rt.robot._pose=replace(before,**change)
    with pytest.raises(Exception,match='Startup return handover'):
        execute_startup_return(rt.setup,rt.robot,before,lambda *a,**k:None)
    assert rt.robot.records==[]


@pytest.mark.parametrize('position,state',[(400,1),(970,1),(1000,3),(1000,0),(1000,2)])
def test_insufficient_opening_blocks_descent_before_robot_command(tmp_path,position,state):
    m,robot,_=motion(tmp_path)
    class Gripper:
        config=GripperConfig()
        last_grip_state=state
        def has_object(self):return False if state in (1,3) else None
        def position(self):return position
    with pytest.raises(RuntimeError,match='not fully open'):
        descend_to_pick(m,Gripper(),.069207)
    assert robot.records==[]


def test_full_open_feedback_allows_pick_descent(tmp_path):
    m,robot,_=motion(tmp_path)
    class Gripper:
        config=GripperConfig()
        last_grip_state=1
        def has_object(self):return False
        def position(self):return 1000
    assert PICK_OPEN_POSITION==1000
    descend_to_pick(m,Gripper(),.069207)
    assert robot.records[-1].pose.z==.069207


def test_low_carry_plane_is_separate_from_camera_home(tmp_path):
    m,robot,_=motion(tmp_path)
    original=m.s.site.safe_high_z_m
    reduced=transfer_setup(m.s,.17)
    assert m.s.site.safe_high_z_m==original
    robot._pose=m.expected.with_z(.069207)
    carry=SmoothMotion(reduced,robot,robot._pose,lambda *a,**k:None)
    carry.single(robot._pose.with_z(.17),'lift')
    carry.single(Pose(.1,-.3,.17,*robot._pose.as_list()[3:]),'transfer')
    carry.descend(.069207,'place')
    returning=SmoothMotion(m.s,robot,carry.expected,lambda *a,**k:None)
    returning.single(carry.expected.with_z(original),'retreat')
    assert robot.records[1].pose.z==.17
    assert robot.records[-1].pose.z==original


@pytest.mark.parametrize('height',[float('nan'),.069207,.079207,.5])
def test_invalid_transfer_plane_rejected(tmp_path,height):
    m,_,_=motion(tmp_path)
    with pytest.raises(ValueError,match='Transfer Z'):transfer_setup(m.s,height)


class Clock:
    def __init__(self):self.now=0.;self.waits=[]
    def time(self):return self.now
    def sleep(self,seconds):self.waits.append(seconds);self.now+=seconds


def timed_cycle(tmp_path,*,drop_during_hold=False,bad_release=False,profile='ten-second'):
    import math
    from battery_sorting.robot import FakeRobot
    from tools.run_smooth_t0 import execute_cycle,PROFILES
    rt,_,_=environment(tmp_path)
    from battery_sorting.closed_loop import Setup
    from tools.run_smooth_t0 import ROOT
    configured=Setup(ROOT/'config/closed_loop.json')
    start=Pose(-.046,-.22,.447856,*configured.home.as_list()[3:])
    rt.setup.site=replace(rt.setup.site,home_pose_m_rad=Pose6(*start.as_list()),workspace_bounds_m=BOUNDS)
    rt.setup.raw['target_view_pose_m_rad']=start.as_list()[:3]+configured.raw['target_view_pose_m_rad'][3:]
    rt.setup.raw['pick_view_pose_m_rad']=rt.setup.raw['target_view_pose_m_rad']
    clock=Clock()
    class Robot(FakeRobot):
        def active_tcp_offset(self):return [0.]*6
        def move_l(self,waypoint,speed,acceleration):
            from battery_sorting.simulation_export import nominal_motion_seconds
            clock.now+=nominal_motion_seconds(self._pose.as_list(),waypoint.pose.as_list(),speed,acceleration,1.)
            return super().move_l(waypoint,speed,acceleration)
    robot=Robot(start)
    class Gripper:
        config=GripperConfig(open_position=1000)
        last_grip_state=1
        def position(self):return 239 if self.last_grip_state==2 else self.config.open_position
        def has_object(self):
            if drop_during_hold and robot.records and robot.records[-1].label=='red_lift':
                self.last_grip_state=3
            return self.last_grip_state==2
        def close(self):clock.now+=.8;self.last_grip_state=2;return True
        def open(self):clock.now+=.35;self.last_grip_state=3 if bad_release else 1;return True
    gripper=Gripper();events=[]
    def log(event,**details):events.append((event,clock.now,details))
    pick=Pose(.21,-.34,.447856,-2.24,-2.18,0.)
    place=Pose(-.115,-.298,.447856,*pick.as_list()[3:])
    def run():
        return execute_cycle(rt.setup,transfer_setup(rt.setup,None),robot,gripper,start,pick,place,log,PROFILES[profile],clock=clock.time,sleep=clock.sleep)
    return run,robot,events,clock


def test_entire_cycle_overlaps_release_wait_and_restores_home(tmp_path,capsys):
    run,robot,events,clock=timed_cycle(tmp_path)
    timing=run()
    release=next(t for e,t,_ in events if e=='released')
    home=next(t for e,t,_ in events if e=='home_arrived')
    assert [r.label for r in robot.records]==[
        'start_observation_pose','home_to_red_hover','red_pick:fast','red_pick:contact','red_lift',
        'red_to_t0','red_place:fast','red_place:contact','direct_return_home','home_sideways_pose']
    assert home>release
    assert clock.now==pytest.approx(max(home,release+3.1))
    assert timing['motion_to_home_s']==pytest.approx(home)
    assert timing['motion_to_home_s']>10.  # Two required 90-degree turns count.
    printed=capsys.readouterr().out
    assert '"PICK_RUN_TIME_S"' in printed
    assert timing['release_stable_s']>=3.1-1e-9
    lifted=next(t for e,t,_ in events if e=='move' and _['label']=='red_lift')
    hold=next(t for e,t,_ in events if e=='lifted_hold_complete')
    assert hold-lifted>=0
    contact=[r for r in robot.records if r.label.endswith(':contact')]
    assert all(r.speed==.04 and r.acceleration==.1 for r in contact)
    direct=robot.records[-2]
    assert direct.label=='direct_return_home'
    assert direct.pose.as_list()[:3]==[-.046,-.22,.447856]
    require_competition_start(robot.records[-1].pose)
    assert direct.speed==pytest.approx(.5)
    assert direct.acceleration==pytest.approx(1.)


def test_drop_during_lifted_hover_prevents_transfer_and_release(tmp_path):
    run,robot,events,_=timed_cycle(tmp_path,drop_during_hold=True)
    with pytest.raises(RuntimeError,match='missing/dropped'):run()
    assert robot.records[-1].label=='red_lift'
    assert not any(e=='released' for e,_,_ in events)


def test_missing_release_feedback_prevents_retreat(tmp_path):
    run,robot,events,_=timed_cycle(tmp_path,bad_release=True)
    with pytest.raises(RuntimeError,match='Release feedback'):run()
    assert robot.records[-1].label=='red_place:contact'
    assert not any(e=='home_arrived' for e,_,_ in events)


def test_new_cycle_translation_model_is_faster_than_previous_profile(tmp_path):
    fast,*_=timed_cycle(tmp_path/'fast')
    previous,*_=timed_cycle(tmp_path/'previous',profile='previous')
    fast_timing=fast()
    previous_timing=previous()
    assert fast_timing['motion_and_stability_s']<previous_timing['motion_and_stability_s']*.6
