"""Synthetic physical world: no SDK device, socket or serial operations."""
import math
from dataclasses import replace
from pathlib import Path
import cv2
import numpy as np
from battery_sorting.closed_loop import Setup, Runtime
from battery_sorting.config import Pose
from battery_sorting.robot import FakeRobot
from battery_sorting.handeye import pose_vector_to_transform
from battery_sorting.handeye_runtime import HandEyeRuntimeMapper

ROOT=Path(__file__).resolve().parents[1]
PALETTE={'red':(0,0,240),'yellow':(0,240,240),'blue':(240,0,0),'green':(0,180,0)}


class Robot(FakeRobot):
    def active_tcp_offset(self):
        return [0.0]*6


class Camera:
    width,height,fx=1280,960,600.
    def __init__(self,robot,world,task='basic',draw_handles=True):
        self.robot,self.world,self.task=robot,world,task
        self.draw_handles=draw_handles
        self.number=0
        self.board_shift=0.
    def normalize_color_pixels(self,p):
        return (np.asarray(p)-[self.width/2,self.height/2])/self.fx
    def project_color_points(self,p):
        p=np.asarray(p)
        return p[:,:2]/p[:,2,None]*self.fx+[self.width/2,self.height/2]
    @property
    def transform(self):
        return pose_vector_to_transform(self.robot.snapshot().tcp_pose)
    def pixels(self,xy,z=0.):
        t=self.transform
        base=np.column_stack((np.asarray(xy),np.full(len(xy),z)))
        return np.rint(self.project_color_points((base-t[:3,3])@t[:3,:3])).astype(np.int32)
    def polygon(self,image,xy,z,color,filled=False):
        points=self.pixels(xy,z)
        if filled:
            cv2.fillConvexPoly(image,points,color)
        else:
            cv2.polylines(image,[points],True,color,2)
    def capture(self):
        self.number+=1
        t=self.transform
        yy,xx=np.mgrid[:self.height,:self.width]
        rays=np.column_stack((self.normalize_color_pixels(np.column_stack((xx.ravel(),yy.ravel()))),np.ones(xx.size)))@t[:3,:3].T
        depth=(-t[2,3]/rays[:,2]).reshape(xx.shape).astype(np.float32)
        image=np.full((self.height,self.width,3),255,np.uint8)
        if self.task=='basic':
            xy=np.array([[-.33,-.03],[-.13,-.03],[-.13,.07],[-.33,.07]])
            xy[:,0]+=self.board_shift
            self.polygon(image,xy,0,(0,0,0))
        else:
            for center in ((-.30,.15),(-.10,.15),(-.20,-.10)):
                for radius in (.075,.04):
                    angles=np.linspace(0,2*math.pi,180)
                    xy=np.column_stack((np.cos(angles),np.sin(angles)))*radius+center
                    xy[:,0]+=self.board_shift
                    self.polygon(image,xy,0,(0,0,0))
        for x in np.linspace(-.39,-.06,7):
            for y in (-.25,.27):
                cv2.circle(image,tuple(self.pixels([[x+self.board_shift,y]])[0]),3,(0,0,0),-1)
        for color,item in self.world.items():
            if item.get('held'):
                continue
            yaw=item['yaw']; u=np.array([math.cos(yaw),math.sin(yaw)])
            v=np.array([-u[1],u[0]])
            corners=np.array([item['xy']+a*.04*u+b*.035*v for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))])
            mask=np.zeros(xx.shape,np.uint8)
            self.polygon(mask,corners,.05,1,True)
            image[mask>0]=PALETTE[color]
            plane=((.05-t[2,3])/rays[:,2]).reshape(xx.shape)
            depth[mask>0]=plane[mask>0]
            if self.draw_handles:
                # A smaller raised rectangle gives the synthetic batteries the
                # same unambiguous handle axis used by the production runtime.
                handle=np.array([item['xy']+a*.026*u+b*.009*v for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))])
                handle_mask=np.zeros(xx.shape,np.uint8)
                self.polygon(handle_mask,handle,.08,1,True)
                image[handle_mask>0]=PALETTE[color]
                handle_plane=((.08-t[2,3])/rays[:,2]).reshape(xx.shape)
                depth[handle_mask>0]=handle_plane[handle_mask>0]
        return image,depth
    def observation_metadata(self):
        return dict(serial='234222302865',frame_number=self.number,timestamp_ms=self.number*33.,timestamp_domain='synthetic',
                    color_settings=dict(exposure=100,gain=16,white_balance=4500,enable_auto_exposure=0,enable_auto_white_balance=0),
                    intrinsics=dict(width=self.width,height=self.height,fx=self.fx,fy=self.fx,ppx=self.width/2,ppy=self.height/2,model='distortion.none',coeffs=[0.]*5))


class Gripper:
    def __init__(self,robot,world):
        self.robot,self.world=robot,world
        self.held=None
        self.actions=[]
        self.release_shift=0.
    def initialize(self,recalibrate=False):
        self.actions.append('initialize'); return True
    def open(self):
        self.actions.append('open')
        if self.held:
            item=self.world[self.held]
            item['held']=False
            item['xy']=np.array(self.robot.snapshot().tcp_pose[:2])+[self.release_shift,0]
            r=pose_vector_to_transform(self.robot.snapshot().tcp_pose)[:3,:3]
            item['yaw']=math.atan2(r[1,0],r[0,0])%math.pi
        self.held=None
        return True
    def close(self):
        self.actions.append('close')
        xy=self.robot.snapshot().tcp_pose[:2]
        closest=min(self.world,key=lambda c:math.dist(xy,self.world[c]['xy']))
        if math.dist(xy,self.world[closest]['xy'])<.005:
            self.held=closest; self.world[closest]['held']=True
        return True
    def has_object(self):
        return self.held is not None
    def position(self):
        return 239 if self.held else 1000
    def stop(self):
        self.actions.append('stop')


def environment(tmp_path,task='basic',handles=True,sideways=False):
    setup=Setup(ROOT/'config/closed_loop.json')
    home=Pose(0,0,.447856,math.pi,0,0)
    standby=Pose(0,0,.447856,math.pi/2,0,0) if sideways else home
    from battery_sorting.offsite import Workspace,Pose6
    setup.site=replace(setup.site,home_pose_m_rad=Pose6(*standby.as_list()),placement_contact_z_m=.069207,
                       workspace_bounds_m=Workspace(-.5,.5,-.4,.4,.0,.6),tool_rotation_vector_rad=(math.pi,0,0))
    setup.raw.update(active_tcp_offset_m_rad=[0.]*6,paper_z_m=0.,target_view_pose_m_rad=home.as_list(),
        sequence_target_view_pose_m_rad=home.as_list(),pick_roi=[690,0,590,960],
        target_roi=[0,0,630,960],sequence_target_roi=[0,0,630,960],grasp_reference_handle_yaw_rad=0.,
        grasp_center_offset_xy_m=[0.,0.],grip_position_range=[220,260],min_pick_separation_m=.10,
        camera_settings=dict(exposure=100,gain=16,white_balance=4500),
        geometry_and_gripper_verified_by_task=dict(basic=True,sequence=True))
    setup.mapper=HandEyeRuntimeMapper(np.eye(4))
    world={color:dict(xy=np.array(xy,float),yaw=0.) for color,xy in zip(PALETTE,((.12,.14),(.30,.14),(.12,-.14),(.30,-.14)))}
    robot=Robot(standby)
    camera=Camera(robot,world,task,handles)
    gripper=Gripper(robot,world)
    sleeps=[]
    class Clock:
        now=0.
        def time(self): return self.now
        def sleep(self,seconds): self.now+=seconds;sleeps.append(seconds)
    clock=Clock()
    runtime=Runtime(setup,camera,robot,gripper,tmp_path/'run',sleep=clock.sleep,clock=clock.time)
    return runtime,world,sleeps
