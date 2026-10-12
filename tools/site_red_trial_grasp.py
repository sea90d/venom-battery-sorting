"""One authorized red-handle trial grasp: live RGB-D, grasp, lift, then hold."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import argparse, hashlib, json, math, sys, time, uuid
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from battery_sorting.closed_loop import Setup, Motion, axis_delta
from battery_sorting.config import Pose, WorkspaceBounds
from battery_sorting.gripper import create_gripper
from battery_sorting.observation import capture_observation
from battery_sorting.offsite import Workspace
from battery_sorting.realsense import RealSenseCamera
from battery_sorting.robot import RealUR5, assert_pose_close, rotation_error
from battery_sorting.scene import detect_batteries
from battery_sorting.tcp_readonly import read_tcp_info

ROOT=Path(__file__).resolve().parents[1]
PARTIAL_OPEN_POSITION=350  # ~50.75 mm of the nominal 145 mm stroke.
PICK_BOUNDS=Workspace(-.06,.27,-.40,-.20,.068,.451)


def mean_axis(values):
    return math.atan2(sum(math.sin(2*v) for v in values),sum(math.cos(2*v) for v in values))/2%math.pi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('Audit directory already exists')
    args.output.mkdir(parents=True)
    setup=Setup(ROOT/'config/closed_loop.json')
    if setup.raw['paper_z_m'] is None or setup.grasp.final_grasp_z_m!=.069207:
        raise ValueError('Confirmed paper/grasp geometry missing')
    setup.site=replace(setup.site,workspace_bounds_m=PICK_BOUNDS)
    home=setup.home
    rc=replace(setup.grasp.robot_config(home),workspace_bounds=WorkspaceBounds(**asdict(PICK_BOUNDS)))
    audit={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'executed':False,'completed':False,
           'scope':'red handle trial: hold at high Z; no T0 transfer/release',
           'partial_open_position':PARTIAL_OPEN_POSITION,'events':[]}
    def save():
        (args.output/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    def log(event,**detail):
        audit['events'].append({'event':event,'utc':datetime.now(timezone.utc).isoformat(),**detail});save()
    save()
    if not args.execute:
        print('PLAN ONLY: live 3-frame red handle -> high hover -> Z=.069207 -> close -> high hold')
        return
    robot=RealUR5(rc);camera=RealSenseCamera();gripper=None;motion=None
    try:
        robot.connect_readonly()
        assert_pose_close(robot.snapshot().tcp_pose,home,.001,math.radians(.5),'Trial start home')
        camera.start(serial=setup.raw['camera_serial']);camera.configure_color(setup.raw['camera_settings'])
        frames=[];last=None;session=uuid.uuid4().hex
        for index in range(3):
            offset=read_tcp_info(setup.site.robot_host)['active_tcp_offset_m_rad']
            expected=setup.raw['active_tcp_offset_m_rad']
            if math.dist(offset[:3],expected[:3])>.0001 or rotation_error(offset[3:],expected[3:])>math.radians(.05):
                raise ValueError('Active TCP differs from hand-eye setup')
            obs=capture_observation(camera,robot,setup.mapper.tcp_T_camera,setup.calibration_sha256,
                                    session,offset,setup.raw['camera_serial'],previous_frame_number=last)
            last=obs.metadata['camera']['frame_number'];obs.save(args.output/f'frame_{index+1}')
            found=detect_batteries(obs.colour,obs.depth_m,setup.vision.colors,obs.base_T_camera,
                camera.normalize_color_pixels,setup.raw['paper_z_m'],roi=setup.raw['pick_roi'],
                max_height_mad_m=setup.raw['max_height_mad_m'])
            if set(found)!={'red','yellow','blue','green'} or found['red'].handle is None:
                raise ValueError('Initial zone is not four unambiguous handled batteries')
            frames.append(found)
        red_handles=[f['red'].handle for f in frames]
        centers=np.asarray([h['center_base_m'][:2] for h in red_handles],float)
        pickxy=tuple(map(float,np.mean(centers,axis=0)))
        yaw=mean_axis([h['yaw_rad'] for h in red_handles])
        spread=max(math.dist(pickxy,c) for c in centers)
        yaw_spread=max(abs(axis_delta(yaw,h['yaw_rad'])) for h in red_handles)
        if spread>.002 or yaw_spread>math.radians(2): raise ValueError('Red handle is unstable')
        if not (PICK_BOUNDS.x_min<=pickxy[0]<=PICK_BOUNDS.x_max and PICK_BOUNDS.y_min<=pickxy[1]<=PICK_BOUNDS.y_max):
            raise ValueError('Red handle outside trial workspace')
        nearest=min(math.dist(pickxy,item.handle['center_base_m'][:2])
                    for color,item in frames[-1].items() if color!='red' and item.handle is not None)
        if nearest<.070: raise ValueError(f'Neighbor center separation too small: {nearest}')
        reference=cv2.Rodrigues(np.asarray(setup.site.tool_rotation_vector_rad))[0]
        orient=tuple(cv2.Rodrigues(cv2.Rodrigues(np.array([0.,0.,axis_delta(yaw,0.)]))[0]@reference)[0].reshape(3))
        hover=Pose(*pickxy,setup.site.safe_high_z_m,*orient)
        log('vision_pass',pick_xy_m=pickxy,handle_yaw_rad=yaw,spread_m=spread,
            yaw_spread_rad=yaw_spread,nearest_handle_center_m=nearest,hover_pose=hover.as_list(),
            instances={k:asdict(v) for k,v in frames[-1].items()})
        camera.stop()
        before=Pose(*robot.snapshot().tcp_pose);robot.connect_control()
        assert_pose_close(robot.snapshot().tcp_pose,before,.0002,math.radians(.1),'Control handover')
        grip_cfg=replace(setup.grasp.gripper,open_position=PARTIAL_OPEN_POSITION)
        gripper=create_gripper(grip_cfg);gripper.connect();gripper.initialize(recalibrate=False)
        if not gripper.open() or gripper.has_object() is not False: raise RuntimeError('Partial opening failed')
        log('partial_open',position=gripper.position(),has_object=gripper.has_object())
        motion=Motion(setup,robot,log);motion.high(hover,'red_trial_hover')
        motion.vertical(setup.site.pick_contact_z_m,'red_trial_pick')
        if not gripper.close(): raise RuntimeError('Gripper close failed')
        time.sleep(setup.grasp.grip_settle_s)
        held=gripper.has_object();position=gripper.position()
        log('closed',position=position,has_object=held)
        if held is True:
            motion.vertical(setup.site.safe_high_z_m,'red_trial_lift',check=lambda: (_ for _ in ()).throw(RuntimeError('Object dropped during lift')) if gripper.has_object() is not True else None)
        else:
            motion.vertical(setup.site.safe_high_z_m,'red_trial_empty_retreat')
            raise RuntimeError(f'No object after close; position={position}')
        final_held=gripper.has_object();final_position=gripper.position()
        if final_held is not True: raise RuntimeError('Object absent after lift')
        audit['completed']=True;audit['final_pose']=list(robot.snapshot().tcp_pose)
        audit['held_feedback']={'has_object':final_held,'position':final_position};save()
        print(json.dumps({'TRIAL_GRASP_COMPLETE':True,'holding_red_for_user_check':True,
                          'position':final_position,'tcp':audit['final_pose']},ensure_ascii=False,indent=2))
    except BaseException as exc:
        audit['error']=str(exc)
        if motion is not None and motion.expected.z<setup.site.safe_high_z_m-.001:
            try:
                motion.vertical(setup.site.safe_high_z_m,'error_vertical_retreat')
                audit['error_retreat_completed']=True
            except Exception as recovery:
                audit['error_retreat_error']=str(recovery)
        try: robot.stop_l(.2)
        except Exception as stop: audit['stop_error']=str(stop)
        save();raise
    finally:
        camera.stop();robot.disconnect()
        if gripper is not None: gripper.disconnect()


if __name__=='__main__': main()
