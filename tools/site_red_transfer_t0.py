"""Transfer the already-held red battery to live-detected T0, then return home."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import argparse,json,math,sys,time,uuid
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from battery_sorting.closed_loop import Setup,Motion,axis_delta
from battery_sorting.config import Pose,WorkspaceBounds
from battery_sorting.gripper import create_gripper
from battery_sorting.observation import capture_observation,Observation,recorded_camera_model
from battery_sorting.offsite import Workspace
from battery_sorting.placement_vision import detect_t0
from battery_sorting.realsense import RealSenseCamera
from battery_sorting.robot import RealUR5,assert_pose_close,rotation_error
from battery_sorting.scene import detect_batteries
from battery_sorting.table_view import rectify_table
from battery_sorting.tcp_readonly import read_tcp_info

ROOT=Path(__file__).resolve().parents[1]
CURRENT_T0_ROI=[570,20,190,260]
TRANSFER_BOUNDS=Workspace(-.14,.27,-.40,-.20,.068,.451)
PARTIAL_OPEN_POSITION=350


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Output already exists')
    a.output.mkdir(parents=True)
    s=Setup(ROOT/'config/closed_loop.json');historic_home=s.home
    held_audit=json.loads((ROOT/'output/site_20260915_initial/red_trial_execution_retry/audit.json').read_text())
    if not held_audit.get('completed') or held_audit.get('held_feedback',{}).get('has_object') is not True:raise ValueError('Successful held trial evidence missing')
    held_reference=Pose(*held_audit['final_pose'])
    if s.grasp.final_grasp_z_m!=.069207:raise ValueError('Placement Z changed')
    s.site=replace(s.site,workspace_bounds_m=TRANSFER_BOUNDS)
    current=Pose(*read_tcp_info(s.site.robot_host)['tcp_base_m_rad'])
    assert_pose_close(current.as_list(),held_reference,.001,math.radians(.5),'Held start')
    rc=replace(s.grasp.robot_config(current),workspace_bounds=WorkspaceBounds(**asdict(TRANSFER_BOUNDS)))
    audit={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'executed':False,'completed':False,'events':[],'start':current.as_list()}
    def save():(a.output/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    def log(event,**detail):audit['events'].append({'event':event,'utc':datetime.now(timezone.utc).isoformat(),**detail});save()
    save()
    if not a.execute:print('PLAN ONLY: verify held -> live T0 -> high transfer -> Z=.069207 -> release -> high -> home -> RGB-D verify');return
    robot=RealUR5(rc);camera=RealSenseCamera();gripper=None;motion=None
    try:
        robot.connect_readonly();assert_pose_close(robot.snapshot().tcp_pose,current,.0005,math.radians(.2),'Transfer start')
        camera.start(serial=s.raw['camera_serial']);camera.configure_color(s.raw['camera_settings'])
        centers=[];target_records=[];target_points=None;last=None;session=uuid.uuid4().hex
        for i in range(3):
            offset=read_tcp_info(s.site.robot_host)['active_tcp_offset_m_rad'];expected=s.raw['active_tcp_offset_m_rad']
            if math.dist(offset[:3],expected[:3])>.0001 or rotation_error(offset[3:],expected[3:])>math.radians(.05):raise ValueError('TCP offset changed')
            obs=capture_observation(camera,robot,s.mapper.tcp_T_camera,s.calibration_sha256,session,offset,s.raw['camera_serial'],previous_frame_number=last);last=obs.metadata['camera']['frame_number'];obs.save(a.output/f'target_frame_{i+1}')
            view=rectify_table(obs.colour,obs.base_T_camera,camera.normalize_color_pixels,camera.project_color_points,s.raw['paper_z_m'],CURRENT_T0_ROI)
            target=detect_t0(view.image)
            if not target.valid:raise ValueError('Current T0 invalid: '+target.reason)
            points=view.to_base(target.outer_points_px);edges=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
            if abs(min(edges)-.1)>.009 or abs(max(edges)-.2)>.009:raise ValueError('Current T0 metric dimensions mismatch')
            center=view.to_base([target.center_px])[0];target_points=points;centers.append(center);target_records.append({'center':center.tolist(),'edges':edges.tolist(),'confidence':target.confidence})
        target_xy=np.mean(centers,axis=0);spread=max(math.dist(target_xy,c) for c in centers)
        if spread>.002:raise ValueError('T0 center unstable')
        # The held item's pre-grasp body-to-handle offset is in Base XY because
        # the tool has not rotated after grasp. Center the body, not the TCP.
        instances=held_audit['events'][0]['instances'];red=instances['red'];handle=np.asarray(red['handle']['center_base_m'][:2]);body=np.asarray(red['center_base_m'][:2]);comp=handle-body
        place_xy=target_xy+comp
        if not (TRANSFER_BOUNDS.x_min<=place_xy[0]<=TRANSFER_BOUNDS.x_max and TRANSFER_BOUNDS.y_min<=place_xy[1]<=TRANSFER_BOUNDS.y_max):raise ValueError('Placement outside transfer workspace')
        log('target_pass',target_frames=target_records,target_center_xy_m=target_xy.tolist(),spread_m=spread,body_center_compensation_xy_m=comp.tolist(),tcp_place_xy_m=place_xy.tolist())
        camera.stop();before=Pose(*robot.snapshot().tcp_pose);robot.connect_control();assert_pose_close(robot.snapshot().tcp_pose,before,.0002,math.radians(.1),'Control handover')
        grip_cfg=replace(s.grasp.gripper,open_position=PARTIAL_OPEN_POSITION);gripper=create_gripper(grip_cfg);gripper.connect();gripper.initialize(recalibrate=False)
        held=gripper.has_object();position=gripper.position();log('held_pretransfer',has_object=held,position=position)
        if held is not True or not 100<=position<=500:raise RuntimeError('Held feedback not credible before transfer')
        motion=Motion(s,robot,log);motion.expected=before
        def check():
            if gripper.has_object() is not True:raise RuntimeError('Object dropped during transfer')
        if abs(before.z-s.site.safe_high_z_m)>1e-6:
            motion.move(before.with_z(s.site.safe_high_z_m),'normalize_high_z',speed=.005)
        destination=Pose(*map(float,place_xy),s.site.safe_high_z_m,before.rx,before.ry,before.rz)
        motion.high(destination,'red_to_t0_high',check=check);motion.vertical(s.site.placement_contact_z_m,'red_t0_place',check=check)
        if not gripper.open():raise RuntimeError('Release command failed')
        time.sleep(max(3.,s.site.timing.release_settle_s))
        if gripper.has_object() is not False:raise RuntimeError('Release feedback missing')
        log('released',position=gripper.position(),has_object=False)
        motion.vertical(s.site.safe_high_z_m,'red_t0_retreat');motion.high(historic_home,'return_home')
        camera.start(serial=s.raw['camera_serial']);camera.configure_color(s.raw['camera_settings'])
        # Pipeline restart resets the device-local frame counter.
        obs=capture_observation(camera,robot,s.mapper.tcp_T_camera,s.calibration_sha256,session,robot.active_tcp_offset(),s.raw['camera_serial']);obs.save(a.output/'verification_frame')
        found=detect_batteries(obs.colour,obs.depth_m,tuple(t for t in s.vision.colors if t.name=='red'),obs.base_T_camera,camera.normalize_color_pixels,s.raw['paper_z_m'],roi=s.raw['target_roi'],max_height_mad_m=s.raw['max_height_mad_m'])
        if set(found)!={'red'}:raise ValueError('Released red battery not detected in T0 ROI')
        # The placed battery may occlude printed T0 details, so compare against
        # the live pre-transfer boundary rather than requiring re-detection.
        actual=np.asarray(found['red'].center_base_m[:2]);error=math.dist(actual,target_xy)
        if error>.010:raise ValueError(f'Released body center error too large: {error}')
        assert target_points is not None
        margins=[cv2.pointPolygonTest(np.asarray(target_points,np.float32),tuple(map(float,p)),True) for p in found['red'].corners_base_xy]
        if min(margins)<=s.raw['target_containment_margin_m']:raise ValueError('Released red body is not wholly inside T0')
        audit['completed']=True;audit['final_pose']=list(robot.snapshot().tcp_pose);audit['verification']={'red_body_xy_m':actual.tolist(),'target_xy_m':target_xy.tolist(),'center_error_m':error,'min_corner_margin_m':min(margins)};save()
        print(json.dumps({'TRANSFER_COMPLETE':True,'red_in_T0':True,'center_error_mm':error*1000,'home':audit['final_pose']},ensure_ascii=False,indent=2))
    except BaseException as exc:
        audit['error']=str(exc)
        if motion is not None and motion.expected.z<s.site.safe_high_z_m-.001:
            try:motion.vertical(s.site.safe_high_z_m,'error_vertical_retreat');audit['error_retreat_completed']=True
            except Exception as recovery:audit['error_retreat_error']=str(recovery)
        try:robot.stop_l(.2)
        except Exception as stop:audit['stop_error']=str(stop)
        save();raise
    finally:
        camera.stop();robot.disconnect()
        if gripper is not None:gripper.disconnect()

if __name__=='__main__':main()
