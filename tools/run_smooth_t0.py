"""Smooth official task 1: live red-handle pick, place in T0, verify, home."""
from dataclasses import asdict,dataclass,replace
from datetime import datetime,timezone
import argparse,copy,json,math,sys,time,uuid
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from battery_sorting.closed_loop import Setup,axis_delta
from battery_sorting.competition_start import require_competition_start, competition_observation_pose, segment_zone_distance_xy
from battery_sorting.config import Pose,WorkspaceBounds
from battery_sorting.gripper import create_gripper
from battery_sorting.observation import capture_observation
from battery_sorting.offsite import Workspace
from battery_sorting.placement_vision import detect_t0,PlacementDetection,TargetKind
from battery_sorting.realsense import RealSenseCamera
from battery_sorting.closed_loop import preflight_realsense
from battery_sorting.robot import RealUR5,Waypoint,assert_pose_close,rotation_error,read_dashboard,require_dashboard_state
from battery_sorting.scene import detect_batteries,SceneError
from battery_sorting.table_view import rectify_table
from battery_sorting.tcp_readonly import read_tcp_info
from battery_sorting.folded_standby import plan_for as folded_plan, transition as folded_transition
from battery_sorting.smooth_dynamics import SPEED_MPS, ACCELERATION_MPS2, CONTACT_SPEED_MPS, CONTACT_ACCELERATION_MPS2, BLEND_M

ROOT=Path(__file__).resolve().parents[1]
BOUNDS=Workspace(-.52,.27,-.40,.27,.068,.451)
START_BOUNDS=Workspace(-.48,.27,-.40,.18,.040,.451)
STARTUP_CORRIDOR_X_M=-.40
STARTUP_TRANSIT_SPEED=.100
OPEN_POSITION=400        # nominal opening about 58 mm
PICK_OPEN_POSITION=1000  # full opening for clearance around the handle
DEFAULT_TRANSFER_LIFT_M=.150  # measured scene leaves >150 mm TCP/table clearance
TRANSIT_SPEED=.300       # operator-requested transfer/lift speed: 300 mm/s
CONTACT_SPEED=CONTACT_SPEED_MPS
ACCELERATION=.100
BLEND_RADIUS=BLEND_M
HELD_POSITION_RANGE=(199,279)  # successful live sample 239, provisional +/-40
MAX_INITIAL_ATTEMPTS=15
# The printed T0 border can contribute roughly 150-200 red HSV pixels. A real
# battery contributes several thousand at the validated camera pose. Use a
# target-only instance floor; pick and post-place checks remain unchanged.
TARGET_OCCUPANCY_MIN_PIXELS=500


@dataclass(frozen=True)
class MotionProfile:
    speed: float
    acceleration: float
    stop_deceleration: float
    controller_profile: str = 'standard'


PROFILES={
    'trajectory-check': MotionProfile(.05,.1,.2),
    'previous': MotionProfile(TRANSIT_SPEED,ACCELERATION,.2),
    'accelerated': MotionProfile(.3,.5,.5),
    'ten-second': MotionProfile(SPEED_MPS,ACCELERATION_MPS2,1.,'t0_timing_trial'),
}
# Contact motion retains the previous acceleration as well as its slow speed.
CONTACT_ACCELERATION=CONTACT_ACCELERATION_MPS2


def checked_wait(seconds,check,*,clock=time.monotonic,sleep=time.sleep):
    """Wait while reading object feedback, including at the end of the hold."""
    deadline=clock()+seconds
    while True:
        check()
        remaining=deadline-clock()
        if remaining<=0:return
        sleep(min(.05,remaining))


def wait_release_deadline(released_at,seconds,check,*,clock=time.monotonic,sleep=time.sleep):
    """Retreat time counts toward the release stability interval."""
    checked_wait(max(0.,released_at+seconds-clock()),check,clock=clock,sleep=sleep)


def execute_cycle(setup,carry_setup,robot,gripper,start,pick,place,log,profile,
                  *,clock=time.monotonic,sleep=time.sleep,cycle_started_at=None,verify_at_observation=None):
    """The timed cycle excludes preparation; folded mode verifies before return."""
    started=clock() if cycle_started_at is None else cycle_started_at
    motion=SmoothMotion(setup,robot,start,log,profile)
    folded=folded_plan(setup)
    if cycle_started_at is None:
        require_competition_start(start)
        log('cycle_started',profile=asdict(profile))
        if folded:
            motion.expected=folded_transition(setup,robot,'observation',log,slow=profile.speed<=.05)
        elif abs(start.z-setup.site.safe_high_z_m)>1e-6:
            motion.single(start.with_z(setup.site.safe_high_z_m),'normalize_high_z',CONTACT_SPEED)
        if not folded:motion.single(competition_observation_pose(setup),'start_observation_pose')
    else:
        assert_pose_close(start.as_list(),competition_observation_pose(setup),.001,math.radians(.5),'Observed start')
        if abs(start.z-setup.site.safe_high_z_m)>1e-6:
            motion.single(start.with_z(setup.site.safe_high_z_m),'normalize_high_z',CONTACT_SPEED)
    # Descend toward the pick while translating, but never below the verified
    # carrying plane. This removes a redundant high horizontal + long vertical
    # pair while retaining full clearance over every detected object.
    motion.clearance_diagonal(pick.with_z(carry_setup.site.safe_high_z_m),'home_to_red_hover',
                              carry_setup.site.safe_high_z_m)
    motion=SmoothMotion(carry_setup,robot,motion.expected,log,profile)
    descend_to_pick(motion,gripper,setup.site.pick_contact_z_m)
    if not gripper.close():raise RuntimeError('Close failed')
    def held():
        if gripper.has_object() is not True:raise RuntimeError('Held feedback missing/dropped')
        pos=gripper.position()
        if not HELD_POSITION_RANGE[0]<=pos<=HELD_POSITION_RANGE[1]:
            raise RuntimeError(f'Held feedback invalid: {pos}')
    checked_wait(min(.05,setup.grasp.grip_settle_s),held,clock=clock,sleep=sleep)
    log('grasped',position=gripper.position())
    motion.single(pick.with_z(carry_setup.site.safe_high_z_m),'red_lift',check=held)
    # The close command already waits for stationary grip feedback. Recheck at
    # lift completion and throughout transfer instead of idling for 2.1 s.
    held()
    log('lifted_hold_complete',duration_s=0.)
    motion.single(place.with_z(carry_setup.site.safe_high_z_m),'red_to_t0',check=held)
    motion.descend(setup.site.placement_contact_z_m,'red_place',held)
    gripper.config=replace(gripper.config,open_position=OPEN_POSITION)
    if not gripper.open():raise RuntimeError('Release failed')
    def empty():
        if gripper.has_object() is not False or gripper.last_grip_state!=1:
            raise RuntimeError('Release feedback missing/not stationary')
    empty()
    released_at=clock()
    log('released',position=gripper.position())
    motion=SmoothMotion(setup,robot,motion.expected,log,profile)
    # After a confirmed release, return in one straight Cartesian segment.
    # This deliberately replaces the former vertical-retreat + high return.
    # Keep fingers downward during the straight retreat. Turn sideways only
    # after arriving outside the battery zone at full clearance height.
    motion.direct_home(competition_observation_pose(setup),check=empty)
    if folded:
        wait_release_deadline(released_at,max(3.,setup.site.timing.release_settle_s),empty,clock=clock,sleep=sleep)
        if verify_at_observation:verify_at_observation()
        empty()
        folded_transition(setup,robot,'home',log,slow=profile.speed<=.05)
    else:
        motion.single(setup.home,'home_sideways_pose',check=empty)
    home_at=clock()
    log('home_arrived',motion_to_home_s=home_at-started)
    print(json.dumps({'PICK_RUN_TIME_S':home_at-started},ensure_ascii=False),flush=True)
    wait_release_deadline(released_at,max(3.,setup.site.timing.release_settle_s),empty,clock=clock,sleep=sleep)
    return {'motion_to_home_s':home_at-started,
            'motion_and_stability_s':clock()-started,
            'release_stable_s':clock()-released_at}


def mean_axis(values):
    return math.atan2(sum(math.sin(2*v) for v in values),sum(math.cos(2*v) for v in values))/2%math.pi


def detect_t0_for_smooth(image):
    """Accept the normal closed contour or a dimension-checked occluded border.

    The fallback is specific to the metric 1 mm/pixel table view used here. It
    still requires four independent border lines and a 200 x 100 mm rectangle;
    a cable may interrupt a border but cannot define the rectangle by itself.
    """
    detected=detect_t0(image)
    if detected.valid:
        return detected
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(gray,60,160)
    raw=cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength=35,maxLineGap=20)
    if raw is None:
        return detected
    h,w=image.shape[:2]
    lines=[]
    for values in raw[:,0]:
        x1,y1,x2,y2=map(float,values)
        length=math.hypot(x2-x1,y2-y1)
        angle=abs(math.degrees(math.atan2(y2-y1,x2-x1)))%180
        lines.append(dict(q=np.array([x1,y1,x2,y2]),length=length,angle=angle,
                          mx=(x1+x2)/2,my=(y1+y2)/2))
    horizontal=[v for v in lines if (v['angle']<8 or v['angle']>172) and v['length']>=.60*200 and 4<v['my']<h-4]
    vertical=[v for v in lines if 82<v['angle']<98 and v['length']>=.45*100 and 4<v['mx']<w-4]
    def coeff(line):
        x1,y1,x2,y2=line['q'];a=y1-y2;b=x2-x1;c=x1*y2-x2*y1
        norm=math.hypot(a,b)
        return np.array([a/norm,b/norm,c/norm])
    def intersection(first,second):
        a,b,c=coeff(first);d,e,f=coeff(second)
        return np.linalg.solve(np.array([[a,b],[d,e]]),-np.array([c,f]))
    candidates=[]
    for top in horizontal:
        for bottom in horizontal:
            if bottom['my']<=top['my'] or not 90<=bottom['my']-top['my']<=112:
                continue
            for left in vertical:
                for right in vertical:
                    if right['mx']<=left['mx'] or not 188<=right['mx']-left['mx']<=212:
                        continue
                    try:
                        corners=np.array([intersection(top,left),intersection(top,right),
                                          intersection(bottom,right),intersection(bottom,left)])
                    except np.linalg.LinAlgError:
                        continue
                    sides=np.array([math.dist(corners[i],corners[(i+1)%4]) for i in range(4)])
                    long=float(np.mean(sides[::2]));short=float(np.mean(sides[1::2]))
                    if not (190<=long<=210 and 92<=short<=108 and np.all(corners[:,0]>2) and
                            np.all(corners[:,0]<w-2) and np.all(corners[:,1]>2) and np.all(corners[:,1]<h-2)):
                        continue
                    score=abs(long-200)+abs(short-100)
                    candidates.append((score,corners,long,short))
    if not candidates:
        return detected
    candidates.sort(key=lambda item:item[0])
    if len(candidates)>1 and abs(candidates[1][0]-candidates[0][0])<.25 and np.linalg.norm(candidates[1][1].mean(axis=0)-candidates[0][1].mean(axis=0))>3:
        return detected
    _,corners,long,short=candidates[0]
    center=corners.mean(axis=0)
    edge=corners[1]-corners[0]
    return PlacementDetection('T0',TargetKind.T0_RECTANGLE,True,tuple(map(float,center)),
        tuple(tuple(map(float,p)) for p in corners),(),math.degrees(math.atan2(edge[1],edge[0])),
        min(.95,.70-.002*(abs(long-200)+abs(short-100))),'occluded border fit')


def initial_red_estimate(scenes):
    """Median center and axial median; every accepted sample must agree."""
    for found in scenes:
        if set(found)!={'red','yellow','blue','green'}:
            raise ValueError(f'Initial zone requires four batteries; found={sorted(found)}')
        if found['red'].handle is None:
            raise SceneError('Red handle missing')
    handles=[s['red'].handle for s in scenes]
    centers=np.asarray([h['center_base_m'][:2] for h in handles])
    xy=np.median(centers,axis=0)
    angles=[h['yaw_rad'] for h in handles]
    anchor=min(angles,key=lambda a:sum(abs(axis_delta(v,a)) for v in angles))
    yaw=(anchor+float(np.median([axis_delta(v,anchor) for v in angles])))%math.pi
    shift=max(math.dist(xy,c) for c in centers)
    spread=max(abs(axis_delta(yaw,a)) for a in angles)
    if shift>.002 or spread>math.radians(5):
        raise SceneError(f'Red handle unstable: center={shift*1000:.3f}mm/2mm, angle={math.degrees(spread):.3f}deg/5deg')
    return xy,yaw


def detect_target_occupancy(observation,setup,camera):
    """Find a real battery in T0 without treating its printed border as one."""
    return detect_batteries(
        observation.colour,observation.depth_m,setup.vision.colors,
        observation.base_T_camera,camera.normalize_color_pixels,
        setup.raw['paper_z_m'],roi=setup.raw['target_roi'],
        max_height_mad_m=setup.raw['max_height_mad_m'],
        min_pixels=TARGET_OCCUPANCY_MIN_PIXELS)


def startup_home_targets(current,home,safe_z,observation_pose):
    """Lift, reach an X corridor, turn down, translate, then turn sideways.

    Each segment changes translation or orientation separately. The corridor
    avoids interpolating across the inner region near the base. This is a
    configured route, not a collision certificate for arbitrary start poses.
    """
    if not (START_BOUNDS.x_min<=current.x<=START_BOUNDS.x_max and START_BOUNDS.y_min<=current.y<=START_BOUNDS.y_max and START_BOUNDS.z_min<=current.z<=START_BOUNDS.z_max):
        raise ValueError('Current TCP is outside the permitted startup-return workspace')
    if math.dist(current.as_list()[:3],home.as_list()[:3])<=.001 and rotation_error(current.as_list()[3:],home.as_list()[3:])<=math.radians(.5):
        return []
    if abs(home.z-safe_z)>1e-6:
        raise ValueError('Startup Home must share the configured high Z')
    targets=[]
    expected=current
    def append(target):
        nonlocal expected
        if math.dist(expected.as_list()[:3],target.as_list()[:3])>1e-9 or rotation_error(expected.as_list()[3:],target.as_list()[3:])>1e-9:
            targets.append(target)
            expected=target
    append(current.with_z(safe_z))
    append(Pose(STARTUP_CORRIDOR_X_M,expected.y,safe_z,*expected.as_list()[3:]))
    append(Pose(expected.x,expected.y,safe_z,*observation_pose.as_list()[3:]))
    append(Pose(expected.x,home.y,safe_z,*observation_pose.as_list()[3:]))
    append(observation_pose)
    # Keep translation and wrist rotation separate.  The folded Home pose is
    # intentionally distinct from the camera observation pose.
    append(Pose(observation_pose.x, observation_pose.y, observation_pose.z,
                *home.as_list()[3:]))
    append(home)
    return targets


def startup_move_label(start,target,home):
    translating=math.dist(start.as_list()[:3],target.as_list()[:3])>1e-8
    rotating=rotation_error(start.as_list()[3:],target.as_list()[3:])>1e-7
    if translating and rotating:
        raise ValueError('Startup translation and rotation must be separate')
    if rotating:
        return 'startup_sideways_home' if target==home else 'startup_turn_down'
    if abs(target.z-start.z)>1e-8:
        return 'startup_vertical_clearance'
    return 'startup_return_home' if target.x==home.x and target.y==home.y else 'startup_move_to_corridor'


def prepare_startup_gripper(config, reset_requested, log, factory=create_gripper):
    """No robot control connection; failures prevent the caller returning home."""
    probe=factory(replace(config,open_position=OPEN_POSITION))
    try:
        probe.connect()
        if reset_requested:
            log('startup_empty_reset_requested',open_position=OPEN_POSITION)
            log('startup_empty_reset_complete',**probe.reset_empty_open(empty_and_clear_confirmed=True))
        if probe.last_grip_state!=1:
            raise RuntimeError(f'Startup return requires empty stationary gripper; state={probe.last_grip_state}. If physically empty and opening is clear, use --reset-empty-gripper')
    except BaseException as exc:
        log('startup_gripper_failed',error=str(exc))
        raise
    finally:
        probe.disconnect()


def execute_startup_return(setup, robot, before_control, log, *,transit_speed=STARTUP_TRANSIT_SPEED):
    """Rebuild from the checked post-handover pose, not an older socket sample."""
    actual=Pose(*robot.snapshot().tcp_pose)
    assert_pose_close(actual.as_list(),before_control,.0002,math.radians(.1),'Startup return handover')
    targets=startup_home_targets(actual,setup.home,setup.site.safe_high_z_m,competition_observation_pose(setup))
    motion=SmoothMotion(setup,robot,actual,log)
    log('startup_return_plan',actual_start=actual.as_list(),targets=[p.as_list() for p in targets])
    for target in targets:
        label=startup_move_label(motion.expected,target,setup.home)
        rotation=label in ('startup_turn_down','startup_sideways_home')
        motion.single(target,label,min(CONTACT_SPEED,transit_speed) if rotation else transit_speed)
    return motion


def require_pick_open(gripper):
    """Read back actual opening; an empty/drop flag alone is insufficient."""
    empty=gripper.has_object()
    position=gripper.position()
    if empty is not False or gripper.last_grip_state!=1 or abs(position-PICK_OPEN_POSITION)>gripper.config.position_tolerance:
        raise RuntimeError(f'Gripper not fully open for descent: state={gripper.last_grip_state}, position={position}, required={PICK_OPEN_POSITION}')


def descend_to_pick(motion,gripper,z):
    # This check runs at hover, immediately before the robot descent command.
    require_pick_open(gripper)
    motion.descend(z,'red_pick',check=lambda:require_pick_open(gripper))


def transfer_setup(setup, transfer_z):
    """Separate a site-verified carrying plane from the camera/home plane."""
    height=setup.site.pick_contact_z_m+DEFAULT_TRANSFER_LIFT_M if transfer_z is None else transfer_z
    if not math.isfinite(height) or not max(setup.site.pick_contact_z_m,setup.site.placement_contact_z_m)+.010<height<=setup.site.safe_high_z_m:
        raise ValueError('Transfer Z must exceed contact Z + final 10mm and not exceed home Z')
    result=copy.copy(setup)
    result.site=replace(setup.site,safe_high_z_m=height)
    return result


class SmoothMotion:
    def __init__(self,setup,robot,start,log,profile=None):
        self.s,self.robot,self.expected,self.log=setup,robot,start,log
        self.profile=profile or PROFILES['previous']
    def _offset(self):
        actual=self.robot.active_tcp_offset();expected=self.s.raw['active_tcp_offset_m_rad']
        if math.dist(actual[:3],expected[:3])>.0001 or rotation_error(actual[3:],expected[3:])>math.radians(.05):raise ValueError('Active TCP changed')
    def _inside(self,p):
        b=self.s.site.workspace_bounds_m
        return b is not None and b.x_min<=p.x<=b.x_max and b.y_min<=p.y<=b.y_max and b.z_min<=p.z<=b.z_max
    def single(self,target,label,speed=None,check=None):
        speed=self.profile.speed if speed is None else speed
        acceleration=CONTACT_ACCELERATION if speed<=CONTACT_SPEED else self.profile.acceleration
        if not self._inside(self.expected) or not self._inside(target):raise ValueError('Outside smooth T0 workspace: '+label)
        if (math.hypot(target.x-self.expected.x,target.y-self.expected.y)>1e-8 or rotation_error(target.as_list()[3:],self.expected.as_list()[3:])>1e-7) and (abs(target.z-self.s.site.safe_high_z_m)>1e-6 or abs(self.expected.z-self.s.site.safe_high_z_m)>1e-6):raise ValueError('Lateral/rotation only allowed at high Z')
        if check:check()
        self._offset();assert_pose_close(self.robot.snapshot().tcp_pose,self.expected,.0005,math.radians(.2),label+' start')
        if not self.robot.move_l_without_tcp_force_guard(Waypoint(label,target),speed,acceleration):raise RuntimeError('Robot rejected '+label)
        assert_pose_close(self.robot.snapshot().tcp_pose,target,.001,math.radians(.5),label+' arrival');self.expected=target
        if check:check()
        self.log('move',label=label,pose=target.as_list(),speed=speed,acceleration=acceleration)
    def clearance_diagonal(self,target,label,min_z,check=None):
        """Translate/rotate while descending only within a verified clear slab."""
        if not self._inside(self.expected) or not self._inside(target):
            raise ValueError('Outside smooth T0 workspace: '+label)
        if min(self.expected.z,target.z)<min_z or min_z<max(self.s.site.pick_contact_z_m,self.s.site.placement_contact_z_m)+.10:
            raise ValueError('Insufficient diagonal clearance: '+label)
        if check:check()
        self._offset();assert_pose_close(self.robot.snapshot().tcp_pose,self.expected,.0005,math.radians(.2),label+' start')
        if not self.robot.move_l_without_tcp_force_guard(Waypoint(label,target),self.profile.speed,self.profile.acceleration):
            raise RuntimeError('Robot rejected '+label)
        assert_pose_close(self.robot.snapshot().tcp_pose,target,.001,math.radians(.5),label+' arrival')
        self.expected=target
        if check:check()
        self.log('move',label=label,pose=target.as_list(),speed=self.profile.speed,
                 acceleration=self.profile.acceleration,minimum_clearance_z_m=min_z)
    def direct_home(self,target,check=None):
        """One post-release linear move from the placement contact to home."""
        label='direct_return_home'
        if not self._inside(self.expected) or not self._inside(target):
            raise ValueError('Outside smooth T0 workspace: '+label)
        if target != competition_observation_pose(self.s):
            raise ValueError('Direct post-release target must be the configured observation pose')
        if check:check()
        self._offset()
        assert_pose_close(self.robot.snapshot().tcp_pose,self.expected,.0005,math.radians(.2),label+' start')
        if not self.robot.move_l_without_tcp_force_guard(
                Waypoint(label,target),self.profile.speed,self.profile.acceleration):
            raise RuntimeError('Robot rejected '+label)
        assert_pose_close(self.robot.snapshot().tcp_pose,target,.001,math.radians(.5),label+' arrival')
        self.expected=target
        if check:check()
        self.log('move',label=label,pose=target.as_list(),speed=self.profile.speed,
                 acceleration=self.profile.acceleration)
    def descend(self,z,label,check=None):
        if abs(self.expected.z-self.s.site.safe_high_z_m)>1e-6 or not BOUNDS.z_min<=z<self.expected.z:raise ValueError('Invalid smooth descent')
        if check:check()
        fast=self.expected.with_z(z+.010);final=self.expected.with_z(z)
        if not all(self._inside(p) for p in (self.expected,fast,final)):
            raise ValueError('Outside smooth T0 workspace: '+label)
        self._offset();assert_pose_close(self.robot.snapshot().tcp_pose,self.expected,.0005,math.radians(.2),label+' start')
        path=[(Waypoint(label+':fast',fast),self.profile.speed,self.profile.acceleration,BLEND_RADIUS),(Waypoint(label+':contact',final),CONTACT_SPEED,CONTACT_ACCELERATION,0.)]
        if not self.robot.move_l_path_without_tcp_force_guard(path):raise RuntimeError('Robot rejected '+label)
        assert_pose_close(self.robot.snapshot().tcp_pose,final,.001,math.radians(.5),label+' arrival');self.expected=final
        if check:check()
        self.log('smooth_path',label=label,poses=[fast.as_list(),final.as_list()],speeds=[self.profile.speed,CONTACT_SPEED],accelerations=[self.profile.acceleration,CONTACT_ACCELERATION],blend_m=BLEND_RADIUS)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--profile',choices=tuple(PROFILES),default='ten-second',help='ten-second is an unvalidated timing trial at 0.5 m/s, 1.0 m/s²; previous retains the old dynamics')
    p.add_argument('--wait-for-start',action='store_true',help='Prepare folded standby and open gripper, then press Enter; unfold and start camera after trigger')
    p.add_argument('--reset-empty-gripper',action='store_true',help='Confirm gripper is empty, robot is stationary and fingers have clearance; initialize/open before startup return')
    p.add_argument('--transfer-z-mm',type=float,help='Override loaded transfer Base Z in mm; default is 150 mm above grasp Z')
    a=p.parse_args();profile=PROFILES[a.profile];program_started=time.monotonic();trigger_at=None
    output=a.output or ROOT/'output/smooth_t0'/datetime.now().strftime('%Y%m%d_%H%M%S')
    if output.exists():raise ValueError('Output already exists')
    output.mkdir(parents=True)
    s=Setup(ROOT/'config/closed_loop.json');home=s.home;s.site=replace(s.site,workspace_bounds_m=BOUNDS)
    start_constraints=require_competition_start(home)
    observation_pose=competition_observation_pose(s)
    carry_setup=transfer_setup(s,None if a.transfer_z_mm is None else a.transfer_z_mm/1000.)
    if s.grasp.final_grasp_z_m!=.069207 or s.site.placement_contact_z_m!=.069207:raise ValueError('Confirmed grasp/place Z changed')
    audit={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'mode':'smooth_t0_v7_folded_standby','executed':False,'completed':False,
           'profile_live_verified':False,'selected_profile':a.profile,
           'start_pose_m_rad':home.as_list(),'start_constraints':start_constraints,
           'motion_profile':{'open_position':OPEN_POSITION,'pick_open_position':PICK_OPEN_POSITION,'transit_speed_mps':profile.speed,
                             'contact_speed_mps':CONTACT_SPEED,'acceleration_mps2':profile.acceleration,
                             'contact_acceleration_mps2':CONTACT_ACCELERATION,'stop_deceleration_mps2':profile.stop_deceleration,
                             'blend_radius_m':BLEND_RADIUS,'transfer_z_m':carry_setup.site.safe_high_z_m},'events':[]}
    def save():(output/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    def log(event,**detail):audit['events'].append({'event':event,'utc':datetime.now(timezone.utc).isoformat(),'program_elapsed_s':time.monotonic()-program_started,**detail});save()
    save()
    if not a.execute:
        print(json.dumps({'ready_to_connect':False,'reason':'plan only','selected_profile':a.profile,'motion_profile':audit['motion_profile'],
            'standby_pose_m_rad':home.as_list(),'observation_pose_m_rad':observation_pose.as_list(),
            'start_constraints':start_constraints,
            'timing_target_s':10,'timing_target_is_measured':False,
            'command':r'.\.venv\Scripts\python.exe tools\run_smooth_t0.py --execute --wait-for-start --profile '+a.profile+
                ('' if a.transfer_z_mm is None else f' --transfer-z-mm {a.transfer_z_mm}')},ensure_ascii=False,indent=2));return
    # Discover the calibrated D455 before any robot or gripper command.
    # A missing/USB2 camera must never leave the arm repositioned.
    preflight_realsense(RealSenseCamera(), s.raw['camera_serial'])
    current=Pose(*read_tcp_info(s.site.robot_host)['tcp_base_m_rad'])
    folded=folded_plan(s)
    startup=([home] if math.dist(current.as_list()[:3],home.as_list()[:3])>.001 or rotation_error(current.as_list()[3:],home.as_list()[3:])>math.radians(.5) else []) if folded else startup_home_targets(current,home,s.site.safe_high_z_m,observation_pose)
    if startup or a.reset_empty_gripper:
        require_dashboard_state(read_dashboard(s.site.robot_host,s.grasp.dashboard_port),{'STOPPED'},'Startup gripper reset')
        assert_pose_close(read_tcp_info(s.site.robot_host)['tcp_base_m_rad'],current,.0002,math.radians(.1),'Stationary before reset')
        prepare_startup_gripper(s.grasp.gripper,a.reset_empty_gripper,log)
        assert_pose_close(read_tcp_info(s.site.robot_host)['tcp_base_m_rad'],current,.0002,math.radians(.1),'Stationary after reset')
    if startup:
        return_setup=copy.copy(s);return_setup.site=replace(s.site,workspace_bounds_m=START_BOUNDS)
        return_rc=replace(s.grasp.robot_config(current),workspace_bounds=WorkspaceBounds(**asdict(START_BOUNDS)),move_timeout=45.)
        return_robot=RealUR5(return_rc)
        try:
            return_robot.connect_readonly();before=Pose(*return_robot.snapshot().tcp_pose)
            assert_pose_close(before.as_list(),current,.0005,math.radians(.2),'Startup return initial')
            return_robot.connect_control()
            audit['startup_home_executed']=True;save()
            if folded:folded_transition(s,return_robot,'home',log,slow=a.profile=='trajectory-check')
            else:execute_startup_return(return_setup,return_robot,before,log,transit_speed=min(STARTUP_TRANSIT_SPEED,profile.speed))
            log('startup_home_complete',pose=list(return_robot.snapshot().tcp_pose))
        except BaseException as exc:
            audit['startup_home_error']=str(exc)
            try:return_robot.stop_l(.2)
            except Exception as stop:audit['startup_home_stop_error']=str(stop)
            save();raise
        finally:return_robot.disconnect()
        current=Pose(*read_tcp_info(s.site.robot_host)['tcp_base_m_rad'])
    assert_pose_close(current.as_list(),home,.001,math.radians(.5),'Smooth start home')
    rc=replace(s.grasp.robot_config(home),workspace_bounds=WorkspaceBounds(**asdict(BOUNDS)),move_timeout=45.,stop_deceleration=profile.stop_deceleration)
    robot=RealUR5(rc,motion_profile=profile.controller_profile) if profile.controller_profile!='standard' else RealUR5(rc)
    camera=RealSenseCamera();gripper=None
    grip_cfg=replace(s.grasp.gripper,open_position=PICK_OPEN_POSITION)
    def open_pick_gripper():
        device=create_gripper(grip_cfg)
        try:
            device.connect()
            if not device.initialize(recalibrate=False):raise RuntimeError('Initialization failed')
            if not device.open():raise RuntimeError('Full opening failed')
            require_pick_open(device)
            log('pick_full_open',position=device.position(),required=PICK_OPEN_POSITION)
            return device
        except BaseException:
            device.disconnect()
            raise
    try:
        robot.connect_readonly();assert_pose_close(robot.snapshot().tcp_pose,home,.001,math.radians(.5),'Smooth start')
        if not folded:
            camera.start(serial=s.raw['camera_serial']);camera.configure_color(s.raw['camera_settings'])
        if a.wait_for_start:
            actual_offset=read_tcp_info(s.site.robot_host)['active_tcp_offset_m_rad']
            expected_offset=s.raw['active_tcp_offset_m_rad']
            if math.dist(actual_offset[:3],expected_offset[:3])>.0001 or rotation_error(actual_offset[3:],expected_offset[3:])>math.radians(.05):
                raise ValueError('TCP offset differs from hand-eye setup before preparation')
            # Only gripper IO during preparation; RTDE Control is still absent.
            gripper=open_pick_gripper()
            actual_start=Pose(*robot.snapshot().tcp_pose)
            assert_pose_close(actual_start.as_list(),home,.001,math.radians(.25),'Prepared standby home')
            log('competition_start_checked',**require_competition_start(actual_start))
            log('prepared_waiting_for_start')
            input('设备已准备。按 Enter 开始本轮（随后重新采图识别）\n> ')
            trigger_at=time.monotonic()
            log('start_trigger')
            require_dashboard_state(read_dashboard(s.site.robot_host,s.grasp.dashboard_port),{'STOPPED'},'Prepared start')
            assert_pose_close(robot.snapshot().tcp_pose,home,.001,math.radians(.5),'Prepared start home')
            require_pick_open(gripper)
        # The competition signal precedes every task motion. A sideways-mounted
        # camera cannot see the table; rotate down at the outside-zone standby
        # point, then acquire fresh RGB-D. Both motion and recognition are timed.
        if gripper is None:gripper=open_pick_gripper()
        actual_offset=read_tcp_info(s.site.robot_host)['active_tcp_offset_m_rad']
        expected_offset=s.raw['active_tcp_offset_m_rad']
        if math.dist(actual_offset[:3],expected_offset[:3])>.0001 or rotation_error(actual_offset[3:],expected_offset[3:])>math.radians(.05):
            raise ValueError('TCP offset differs from hand-eye setup before observation')
        before=Pose(*robot.snapshot().tcp_pose)
        require_competition_start(before)
        robot.connect_control()
        assert_pose_close(robot.snapshot().tcp_pose,before,.0002,math.radians(.1),'Control handover')
        audit['executed']=True;save()
        cycle_at=time.monotonic()
        log('cycle_started',profile=asdict(profile))
        if folded:
            folded_transition(s,robot,'observation',log,slow=a.profile=='trajectory-check')
            camera.start(serial=s.raw['camera_serial']);camera.configure_color(s.raw['camera_settings'])
        else:
            observing=SmoothMotion(s,robot,before,log,profile)
            if abs(before.z-s.site.safe_high_z_m)>1e-6:
                observing.single(before.with_z(s.site.safe_high_z_m),'normalize_high_z',CONTACT_SPEED)
            observing.single(observation_pose,'start_observation_pose')
        session=uuid.uuid4().hex;last=None;scenes=[];targets=[];target_points=None
        last_vision_error='No accepted observations'
        for i in range(MAX_INITIAL_ATTEMPTS):
            offset=read_tcp_info(s.site.robot_host)['active_tcp_offset_m_rad'];expected=s.raw['active_tcp_offset_m_rad']
            if math.dist(offset[:3],expected[:3])>.0001 or rotation_error(offset[3:],expected[3:])>math.radians(.05):raise ValueError('TCP offset differs from hand-eye setup')
            obs=capture_observation(camera,robot,s.mapper.tcp_T_camera,s.calibration_sha256,session,offset,s.raw['camera_serial'],previous_frame_number=last);last=obs.metadata['camera']['frame_number'];obs.save(output/f'initial_{i+1}')
            try:
                found=detect_batteries(obs.colour,obs.depth_m,s.vision.colors,obs.base_T_camera,camera.normalize_color_pixels,s.raw['paper_z_m'],roi=s.raw['pick_roi'],max_height_mad_m=s.raw['max_height_mad_m'])
                initial_red_estimate([found])
            except SceneError as exc:
                last_vision_error=str(exc);log('initial_frame_rejected',attempt=i+1,error=str(exc))
                continue
            try:
                occupied=detect_target_occupancy(obs,s,camera)
            except SceneError as exc:
                last_vision_error=f'T0 occupancy frame rejected: {exc}'
                log('target_frame_rejected',attempt=i+1,error=last_vision_error)
                continue
            if occupied:raise ValueError('T0 must be empty before starting')
            view=rectify_table(obs.colour,obs.base_T_camera,camera.normalize_color_pixels,camera.project_color_points,s.raw['paper_z_m'],s.raw['target_roi']);t=detect_t0_for_smooth(view.image)
            if not t.valid:
                last_vision_error='T0 invalid: '+t.reason
                log('target_frame_rejected',attempt=i+1,error=last_vision_error)
                continue
            points=view.to_base(t.outer_points_px);edges=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
            if abs(min(edges)-.1)>.009 or abs(max(edges)-.2)>.009:
                last_vision_error='T0 dimensions mismatch'
                log('target_frame_rejected',attempt=i+1,error=last_vision_error)
                continue
            target_points=points;targets.append(view.to_base([t.center_px])[0]);scenes.append(found)
            scenes=scenes[-3:];targets=targets[-3:]
            if len(scenes)==3:
                try:
                    pickxy,yaw=initial_red_estimate(scenes)
                except SceneError as exc:
                    last_vision_error=str(exc);log('initial_window_rejected',attempt=i+1,error=str(exc))
                    continue
                candidate_target=np.median(targets,axis=0)
                target_spread=max(math.dist(candidate_target,c) for c in targets)
                if target_spread>.002:
                    last_vision_error=f'T0 unstable: center={target_spread*1000:.3f}mm/2mm'
                    log('initial_window_rejected',attempt=i+1,error=last_vision_error)
                    continue
                log('initial_consensus',attempts=i+1,accepted_frames=3)
                break
        else:
            raise ValueError(f'Initial vision failed after {MAX_INITIAL_ATTEMPTS} attempts: {last_vision_error}')
        targetxy=np.median(targets,axis=0)
        if max(math.dist(targetxy,c) for c in targets)>.002:raise AssertionError('Accepted T0 consensus changed')
        nearest=min(math.dist(pickxy,item.center_base_m[:2]) for scene in scenes for color,item in scene.items() if color!='red')
        if nearest<.070:raise ValueError('Neighbor spacing below validated trial condition')
        body=np.median([x['red'].center_base_m[:2] for x in scenes],axis=0);comp=pickxy-body;placexy=targetxy+comp
        reference=cv2.Rodrigues(np.asarray(s.site.tool_rotation_vector_rad))[0];orient=tuple(cv2.Rodrigues(cv2.Rodrigues(np.array([0.,0.,axis_delta(yaw,0.)]))[0]@reference)[0].reshape(3))
        pick=Pose(*map(float,pickxy),s.site.safe_high_z_m,*orient);place=Pose(*map(float,placexy),s.site.safe_high_z_m,*orient)
        log('vision_pass',pick_xy_m=pickxy.tolist(),target_xy_m=targetxy.tolist(),place_tcp_xy_m=placexy.tolist(),nearest_m=nearest)
        obs.require_live(session,s.calibration_sha256,s.raw['max_observation_age_s'])
        if gripper is None:gripper=open_pick_gripper()
        before=Pose(*robot.snapshot().tcp_pose)
        assert_pose_close(before.as_list(),observation_pose,.001,math.radians(.5),'Fresh-vision observation pose')
        require_pick_open(gripper)
        obs.require_live(session,s.calibration_sha256,s.raw['max_observation_age_s'])
        audit['executed']=True;save()
        def verify_at_observation():
            # Keep the camera streaming; capture_observation discards queued frames
            # before binding a new image to the stationary home TCP pose.
            final_motion=SmoothMotion(s,robot,Pose(*robot.snapshot().tcp_pose),log,profile)
            if not folded:final_motion.single(observation_pose,'verification_observation_pose')
            obs=capture_observation(camera,robot,s.mapper.tcp_T_camera,s.calibration_sha256,session,robot.active_tcp_offset(),s.raw['camera_serial']);obs.save(output/'verification')
            if not folded:final_motion.single(home,'verification_return_sideways_home')
            red_only=tuple(t for t in s.vision.colors if t.name=='red');found=detect_batteries(obs.colour,obs.depth_m,red_only,obs.base_T_camera,camera.normalize_color_pixels,s.raw['paper_z_m'],roi=s.raw['target_roi'],max_height_mad_m=s.raw['max_height_mad_m'])
            if set(found)!={'red'}:raise ValueError('Red not found in T0 after release')
            assert target_points is not None;item=found['red'];margins=[cv2.pointPolygonTest(np.asarray(target_points,np.float32),tuple(map(float,p)),True) for p in item.corners_base_xy];error=math.dist(item.center_base_m[:2],targetxy)
            if error>.010 or min(margins)<=s.raw['target_containment_margin_m']:raise ValueError('Red placement outside acceptance')
            audit['verification']={'center_error_m':error,'min_corner_margin_m':min(margins)}
        audit['timing']=execute_cycle(s,carry_setup,robot,gripper,before,pick,place,log,profile,cycle_started_at=cycle_at,verify_at_observation=verify_at_observation if folded else None)
        audit['timing']['preparation_to_cycle_s']=cycle_at-program_started
        if trigger_at is not None:audit['timing']['trigger_to_home_s']=cycle_at-trigger_at+audit['timing']['motion_to_home_s']
        if not folded:verify_at_observation()
        error=audit['verification']['center_error_m'];margins=[audit['verification']['min_corner_margin_m']]
        audit['timing']['program_to_complete_s']=time.monotonic()-program_started
        if trigger_at is not None:audit['timing']['trigger_to_complete_s']=time.monotonic()-trigger_at
        audit['completed']=True;audit['verification']={'center_error_m':error,'min_corner_margin_m':min(margins)};audit['final_pose']=list(robot.snapshot().tcp_pose);save();print(json.dumps({'SMOOTH_T0_COMPLETE':True,'center_error_mm':error*1000,'min_margin_mm':min(margins)*1000,'timing':audit['timing']},indent=2))
    except BaseException as exc:
        audit['error']=str(exc)
        try:robot.stop_l(profile.stop_deceleration)
        except Exception as stop:audit['stop_error']=str(stop)
        if gripper is not None:
            try:gripper.stop()
            except Exception as stop:audit['gripper_stop_error']=str(stop)
        save();raise
    finally:
        try:
            camera.stop()
        finally:
            try:
                robot.disconnect()
            finally:
                if gripper is not None:gripper.disconnect()

if __name__=='__main__':main()
