"""Run the current accelerated T0 cycle on fake devices, using archived scene data."""
from dataclasses import replace
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from battery_sorting.closed_loop import Setup, axis_delta
from battery_sorting.config import Pose
from battery_sorting.recorded_simulation import export_recorded_t0
from battery_sorting.observation import Observation, recorded_camera_model
from battery_sorting.scene import detect_batteries
from battery_sorting.robot import FakeRobot
from battery_sorting.simulation_export import nominal_motion_seconds, write_plan
from tools.run_smooth_t0 import BOUNDS, PROFILES, execute_cycle, transfer_setup, initial_red_estimate, detect_t0_for_smooth
from battery_sorting.folded_standby import plan_for, joint_seconds
from battery_sorting.table_view import rectify_table
from battery_sorting.simulation_export import _quat


def export_trial(recording, config, profile_name='ten-second', angular_speed=1., motion_config=None, scene_recording=None):
    """Execute the same cycle implementation; no camera, serial or robot connection."""
    plan = export_recorded_t0(recording, config, allow_mechanical_complete=True)
    audit = json.loads((Path(recording) / 'audit.json').read_text(encoding='utf-8'))
    setup = Setup(motion_config or ROOT/'config/closed_loop.json')
    setup.site = replace(setup.site, workspace_bounds_m=BOUNDS)
    carry = transfer_setup(setup, None)
    profile = PROFILES[profile_name]
    # Reuse the archived accepted grasp/placement inputs, never the old motion times.
    pick = Pose(*next(e['pose'] for e in audit['events']
                      if e['event'] == 'move' and e['label'] == 'home_to_red_hover'))
    place = Pose(*next(e['pose'] for e in audit['events']
                       if e['event'] == 'move' and e['label'] == 'red_to_t0'))
    # Match live planning: preserve the archived handle yaw, with the current
    # level reference orientation instead of replaying the old tilted wrist.
    obs=Observation.load(sorted(Path(recording).glob('initial_*/manifest.json'))[-1].parent)
    camera=recorded_camera_model(obs.metadata)
    scene_setup=Setup(config)
    found=detect_batteries(obs.colour,obs.depth_m,scene_setup.vision.colors,
        obs.base_T_camera,camera.normalize_color_pixels,scene_setup.raw['paper_z_m'],
        roi=scene_setup.raw['pick_roi'],max_height_mad_m=scene_setup.raw['max_height_mad_m'])
    yaw=found['red'].handle['yaw_rad']
    if scene_recording:
        accepted=[];centers=[];digests=[]
        for bundle in sorted(Path(scene_recording).glob('initial_*'),key=lambda p:int(p.name.split('_')[-1])):
            obs=Observation.load(bundle);camera=recorded_camera_model(obs.metadata)
            if obs.metadata['calibration_sha256']!=setup.calibration_sha256:
                raise ValueError('Live scene calibration differs')
            try:
                found=detect_batteries(obs.colour,obs.depth_m,setup.vision.colors,obs.base_T_camera,
                    camera.normalize_color_pixels,setup.raw['paper_z_m'],roi=setup.raw['pick_roi'],max_height_mad_m=setup.raw['max_height_mad_m'])
                initial_red_estimate([found])
            except ValueError:continue
            view=rectify_table(obs.colour,obs.base_T_camera,camera.normalize_color_pixels,camera.project_color_points,
                setup.raw['paper_z_m'],setup.raw['target_roi'])
            t=detect_t0_for_smooth(view.image)
            if not t.valid:continue
            accepted.append(found);centers.append(view.to_base([t.center_px])[0]);digests.append(obs.content_digest())
            if len(accepted)<3:continue
            try:xy,yaw=initial_red_estimate(accepted[-3:])
            except ValueError:continue
            targetxy=np.median(centers[-3:],axis=0)
            if max(math.dist(targetxy,v) for v in centers[-3:])>.002:continue
            found=accepted[-1]
            body=np.median([s['red'].center_base_m[:2] for s in accepted[-3:]],axis=0)
            place_xy=targetxy+xy-body
            pick=Pose(*map(float,xy),setup.site.safe_high_z_m,*pick.as_list()[3:])
            place=Pose(*map(float,place_xy),setup.site.safe_high_z_m,*place.as_list()[3:])
            plan['objects']=[dict(id=c,color=c,center=[*item.center_base_m[:2],setup.raw['paper_z_m']+.04],
                quaternion=_quat([0.,0.,(item.handle['yaw_rad']+math.pi/2)%math.pi-math.pi/2]),size_m=[.07,.08,.08]) for c,item in found.items()]
            plan['targets']=[dict(id='T0',pose=[*map(float,targetxy),setup.raw['paper_z_m'],0.,0.,0.],outline_base_xy=view.to_base(t.outer_points_px).tolist())]
            break
        else:raise ValueError('No stable three-frame scene for new simulation')
    reference=cv2.Rodrigues(np.asarray(setup.site.tool_rotation_vector_rad))[0]
    orient=cv2.Rodrigues(cv2.Rodrigues(np.array([0.,0.,axis_delta(yaw,0.)]))[0]@reference)[0].reshape(3)
    pick=Pose(pick.x,pick.y,setup.site.safe_high_z_m,*map(float,orient))
    place=Pose(place.x,place.y,setup.site.safe_high_z_m,*map(float,orient))
    steps = []
    now = 0.
    current = setup.home.as_list()

    def append(kind, label, seconds, end=None, speed=None, acceleration=None,
               object_id=None, target_id=None, **extra):
        nonlocal now, current
        end = current.copy() if end is None else end
        if kind == 'dwell' and steps and steps[-1]['kind'] == kind and steps[-1]['label'] == label:
            steps[-1]['duration_s'] += seconds
        else:
            steps.append(dict(kind=kind, label=label, start_pose=current.copy(), end_pose=end.copy(),
                duration_s=seconds, speed_mps=speed, acceleration_mps2=acceleration,
                profile=profile_name if kind == 'move' else None,
                object_id=object_id, target_id=target_id, **extra))
        current = end.copy()
        now += seconds

    class Clock:
        def time(self): return now
        def sleep(self, seconds):
            label = ('夹持反馈确认' if robot.records[-1].label == 'red_pick:contact' else
                     '提升后持物反馈确认' if robot.records[-1].label == 'red_lift' else
                     '回 home 后补足释放稳定时间')
            append('dwell', label, seconds)

    names = {'start_observation_pose':'开始信号后在区外转为朝下观察姿态',
        'home_sideways_pose':'回到区外高位后恢复侧向 Home',
        'home_to_red_hover': 'home → 红色电池上方', 'red_pick:fast': '下降抓取 · 接近',
        'red_pick:contact': '下降抓取 · 最后 10 mm 慢速接触', 'red_lift': '提升至抓取上方 15 cm',
        'red_to_t0': '在 15 cm 搬运平面移动至 T0', 'red_place:fast': '下降放置 · 接近',
        'red_place:contact': '下降放置 · 最后 10 mm 慢速接触',
        'direct_return_home': '放置点 → home 直线返回（稳定计时继续）'}

    class Robot(FakeRobot):
        def active_tcp_offset(self): return setup.raw['active_tcp_offset_m_rad']
        def move_l(self, waypoint, speed, acceleration):
            end = waypoint.pose.as_list()
            seconds = nominal_motion_seconds(current, end, speed, acceleration, angular_speed)
            append('move', names.get(waypoint.label, waypoint.label), seconds, end, speed, acceleration)
            return super().move_l(waypoint, speed, acceleration)
        def move_j(self, waypoint, joints, speed, acceleration):
            first=self.actual_joints()
            seconds=joint_seconds(first,joints,speed,acceleration)
            append('move', '大小臂整体展开 → 夹爪朝下观察' if waypoint.label=='unfold_to_observation' else '大小臂折回 → 夹爪朝前待机',
                seconds,waypoint.pose.as_list(),None,None,motion_space='joint',joint_start_rad=list(first),joint_end_rad=list(joints),
                speed_rad_s=speed,acceleration_rad_s2=acceleration)
            return super().move_j(waypoint,joints,speed,acceleration)

    robot = Robot(setup.home)
    clock = Clock()

    class Gripper:
        config = replace(setup.grasp.gripper, open_position=1000)
        last_grip_state = 1
        def has_object(self): return self.last_grip_state == 2
        def position(self): return 239 if self.last_grip_state == 2 else self.config.open_position
        def close(self):
            append('close', '闭合夹爪（假设 0.8 秒）', .8, object_id='red')
            self.last_grip_state = 2
            return True
        def open(self):
            append('open', '松开夹爪（假设 0.35 秒）', .35, object_id='red', target_id='T0')
            self.last_grip_state = 1
            return True

    def log(event, **details):
        if event == 'grasped': append('check_object', '持物反馈通过（模拟）', 0., object_id='red')

    def verify():append('visual_accept','观察位复检（耗时未计入，须实测）',0.,object_id='red',target_id='T0')
    timing = execute_cycle(setup, carry, robot, Gripper(), setup.home, pick, place,
                           log, profile, clock=clock.time, sleep=clock.sleep,
                           verify_at_observation=verify if plan_for(setup) else None)
    old_source = plan['source']
    plan.update(initial_pose=setup.home.as_list(), initial_gripper_open=True, steps=steps)
    plan['source'] = dict(planner='tools.run_smooth_t0',
        planner_sha256=hashlib.sha256((ROOT / 'tools/run_smooth_t0.py').read_bytes()).hexdigest(),
        label='新提速方案 · 红色 → T0 → home · ' + profile_name,
        data_kind='configured', timing='nominal_current_cycle', simulation_variant='timing_trial',
        profile_name=profile_name, speed_mps=profile.speed, acceleration_mps2=profile.acceleration,
        transfer_z_m=carry.site.safe_high_z_m, angular_speed_rad_s=angular_speed,
        assumed_close_s=.8, assumed_open_s=.35, predicted_timing=timing,
        scene_audit_sha256=old_source['audit_sha256'], scene_audit_path=old_source['audit_path'],
        geometry=old_source['geometry'],
        fresh_vision_duration_s=None,
        timing_note='当前 execute_cycle 在假设备上运行；复用历史场景和抓放坐标，使用当前侧向 Home，包含开始后朝下观察和结束恢复侧向的原地转动。运动时间按加减速及假设角速度 1 rad/s 估算；不含新鲜视觉识别、通信与最后复检。实机 PICK_RUN_TIME_S 还会计入开始后的新鲜识别，不是真机实测。')
    if plan_for(setup):
        plan['source']['standby_configuration_sha256']=hashlib.sha256((ROOT/'config/folded_standby.json').read_bytes()).hexdigest()
        plan['source']['label']='图示折叠待机 → 展开观察 → 红色 T0 → 折回'
        plan['source']['timing_note']='实际 execute_cycle 的关节展开/折回和直线抓放路径；含释放后3秒稳定要求。关节运动按1 rad/s、1.4 rad/s²估算，夹爪开闭耗时是假设；相机启动、识别、复检、通信及轨迹计算耗时未知，未计入，不能当实测总时间。'
    if scene_recording:
        plan['source'].update(scene_recording_path=str(scene_recording),initial_observation_sha256=digests[-3:],
            scene_audit_path=str(Path(scene_recording)/'audit.json'),scene_audit_sha256=hashlib.sha256((Path(scene_recording)/'audit.json').read_bytes()).hexdigest(),
            geometry='Oct11真实RGB-D三帧共识：四色XY/把手角度及T0轮廓；物块外形高度采用CAD示意')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, default=ROOT / 'output/smooth_t0/new_room_20260929_111616')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/closed_loop_scene_20260929.json', help='Frozen scene/calibration configuration for the recording')
    parser.add_argument('--motion-config', type=Path, default=ROOT / 'config/closed_loop.json', help='Current competition home and tool orientation')
    parser.add_argument('--profile', choices=tuple(PROFILES), default='ten-second')
    parser.add_argument('--scene-recording',type=Path,default=ROOT/'output/smooth_t0/20261011_213309')
    parser.add_argument('--out', type=Path, default=ROOT / 'web_sim/plans/accelerated-red-t0.json')
    args = parser.parse_args()
    plan = export_trial(args.recording, args.config, args.profile, motion_config=args.motion_config,scene_recording=args.scene_recording)
    if any(s.get('motion_space')=='joint' for s in plan['steps']):
        from tools.export_folded_t0_simulation import audit_plan
        plan=audit_plan(plan,publish=False)
    write_plan(plan, args.out)
    print(json.dumps(dict(output=str(args.out), nominal_duration_s=sum(s['duration_s'] for s in plan['steps']),
                         timing=plan['source']['predicted_timing'], no_hardware_connected=True), indent=2))


if __name__ == '__main__': main()
