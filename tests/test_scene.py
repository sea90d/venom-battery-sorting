"""Synthetic tests for instance geometry; no hardware."""
import math
import numpy as np
import pytest
import cv2
from battery_sorting.scene import detect_batteries,SceneError
from battery_sorting.config import Pose
from .closed_loop_fakes import environment


def detect(rt):
    image,depth=rt.camera.capture()
    return detect_batteries(image,depth,rt.setup.vision.colors,rt.camera.transform,rt.camera.normalize_color_pixels,0.)


def test_dynamic_base_centers_survive_camera_motion_and_remaining_subset(tmp_path):
    rt,world,_=environment(tmp_path,handles=False)
    for pose in (Pose(0,0,.447856,math.pi,0,0),Pose(.04,.01,.50,3.0,.08,.03)):
        rt.robot._pose=pose
        found=detect(rt)
        assert set(found)==set(world)
        for color,obj in found.items():
            assert obj.center_base_m[:2]==pytest.approx(world[color]['xy'],abs=.002)
            assert obj.height_m==pytest.approx(.05,abs=.0001)
    world['blue']['held']=True
    assert set(detect(rt))=={'red','yellow','green'}


def test_printed_colors_do_not_count_as_batteries(tmp_path):
    rt,_,_=environment(tmp_path,handles=False)
    image,depth=rt.camera.capture()
    image[200:250,200:250]=(240,0,0)
    found=detect_batteries(image,depth,rt.setup.vision.colors,rt.camera.transform,rt.camera.normalize_color_pixels,0.)
    assert len(found)==4


def test_missing_edge_depth_rejected_not_shifted_center(tmp_path):
    rt,_,_=environment(tmp_path,handles=False)
    image,depth=rt.camera.capture()
    red=(image[:,:,2]>200)&(image[:,:,0]<20)&(image[:,:,1]<20)
    ys,xs=np.nonzero(red)
    missing=xs<np.quantile(xs,.25)
    depth[ys[missing],xs[missing]]=0
    with pytest.raises(SceneError,match='coverage'):
        detect_batteries(image,depth,rt.setup.vision.colors,rt.camera.transform,rt.camera.normalize_color_pixels,0.)


def test_same_color_duplicates_rejected(tmp_path):
    rt,_,_=environment(tmp_path,handles=False)
    image,depth=rt.camera.capture()
    blue=(image[:,:,0]>200)&(image[:,:,2]<20)
    image[blue]=(0,0,240)
    with pytest.raises(SceneError,match='Duplicate'):
        detect_batteries(image,depth,rt.setup.vision.colors,rt.camera.transform,rt.camera.normalize_color_pixels,0.)


def handled_scene(tmp_path, handle_size=(.075,.020), yaw=0.):
    rt,world,_=environment(tmp_path,handles=False)
    world['red']['yaw']=yaw
    for color in ('yellow','blue','green'):
        world[color]['held']=True
    image,depth=rt.camera.capture()
    camera=rt.camera
    xy=world['red']['xy']
    u=np.array([math.cos(yaw),math.sin(yaw)]);v=np.array([-u[1],u[0]])
    corners=np.array([xy+a*handle_size[0]/2*u+b*handle_size[1]/2*v for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))])
    mask=np.zeros(depth.shape,np.uint8)
    camera.polygon(mask,corners,.08,1,True)
    image[mask>0]=(0,0,240)
    yy,xx=np.nonzero(mask)
    t=camera.transform
    rays=np.column_stack((camera.normalize_color_pixels(np.column_stack((xx,yy))),np.ones(len(xx))))@t[:3,:3].T
    depth[yy,xx]=(.08-t[2,3])/rays[:,2]
    return rt,world,image,depth,mask


def run_image(rt,image,depth):
    return detect_batteries(image,depth,rt.setup.vision.colors,rt.camera.transform,rt.camera.normalize_color_pixels,0.)


@pytest.mark.parametrize('yaw',[0.,.35,1.2])
def test_two_level_body_and_handle_are_separate_planes(tmp_path,yaw):
    rt,world,image,depth,_=handled_scene(tmp_path,yaw=yaw)
    item=run_image(rt,image,depth)['red']
    assert item.height_m==pytest.approx(.05,abs=.001)
    assert item.center_base_m[:2]==pytest.approx(world['red']['xy'],abs=.003)
    assert item.handle['center_base_m'][:2]==pytest.approx(world['red']['xy'],abs=.003)
    assert item.handle['center_base_m'][2]==pytest.approx(.08,abs=.001)
    assert item.handle['height_above_body_m']==pytest.approx(.03,abs=.001)
    assert abs((item.handle['yaw_rad']-yaw+math.pi/2)%math.pi-math.pi/2)<.04


def test_flat_battery_does_not_invent_a_handle(tmp_path):
    rt,_,_=environment(tmp_path,handles=False)
    assert all(x.handle is None for x in detect(rt).values())


def test_wide_elevated_patch_is_not_a_handle(tmp_path):
    rt,_,image,depth,_=handled_scene(tmp_path,handle_size=(.045,.045))
    with pytest.raises(SceneError):
        run_image(rt,image,depth)


def test_continuous_ramp_is_not_a_two_level_battery(tmp_path):
    rt,_,image,depth,_=handled_scene(tmp_path)
    red=(image[:,:,2]>200)&(image[:,:,0]<20)
    yy,xx=np.nonzero(red)
    heights=.035+.06*(xx-xx.min())/np.ptp(xx)
    depth[yy,xx]=rt.camera.transform[2,3]-heights
    with pytest.raises(SceneError):
        run_image(rt,image,depth)


def test_split_upper_surface_is_rejected(tmp_path):
    rt,_,image,depth,mask=handled_scene(tmp_path)
    yy,xx=np.nonzero(mask)
    middle=abs(xx-np.median(xx))<9
    depth[yy[middle],xx[middle]]=rt.camera.transform[2,3]-.05
    with pytest.raises(SceneError):
        run_image(rt,image,depth)


def test_missing_handle_depth_is_not_a_valid_center(tmp_path):
    rt,_,image,depth,mask=handled_scene(tmp_path)
    depth[mask>0]=0
    with pytest.raises(SceneError,match='coverage|Duplicate'):
        run_image(rt,image,depth)


@pytest.mark.parametrize('name',['trial_grip_precheck','red_t0_trial_observation'])
def test_local_recorded_four_colors_with_handles(name):
    from pathlib import Path
    from battery_sorting.closed_loop import Setup,Observation,recorded_camera_model
    root=Path(__file__).resolve().parents[1]
    bundle=root/'output/site_20260915_initial'/name
    if not bundle.exists():
        pytest.skip('Optional local hardware recording, never collected by tests')
    s=Setup(root/'config/closed_loop.json');o=Observation.load(bundle);c=recorded_camera_model(o.metadata)
    found=detect_batteries(o.colour,o.depth_m,s.vision.colors,o.base_T_camera,c.normalize_color_pixels,
                          s.raw['paper_z_m'],roi=s.raw['pick_roi'])
    assert set(found)=={'red','yellow','blue','green'}
    # Labels checked against independently viewed RGB, not HSV-generated labels.
    expected={'red':(365,326),'yellow':(445,325),'blue':(365,422),'green':(445,419)}
    for color,point in expected.items():
        item=found[color]
        assert math.dist(item.centroid_px,point)<15
        assert .035<item.height_m<.055
        assert item.handle is not None
        assert .02<item.handle['height_above_body_m']<.04
        assert math.dist(item.center_base_m[:2],item.handle['center_base_m'][:2])<.010
