"""Metric table rectification and stationary-board checks, without hardware I/O."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass
class TableView:
    image: np.ndarray
    origin_xy: np.ndarray
    axes: np.ndarray
    scale: float

    def to_base(self, pixels):
        return self.origin_xy + np.asarray(pixels, float) * self.scale @ self.axes.T


def rectify_table(image, base_T_camera, normalize, project, paper_z, roi=None, reference=None):
    """Resample with the SDK's own projection model; circles stay circles.

    Axes follow the initial camera view, so P1/P2/P3 naming requires the
    operator to confirm the map is upright in the saved rectified image.
    """
    h,w=image.shape[:2]
    x,y,rw,rh=(0,0,w,h) if roi is None else roi
    if min(x,y)<0 or min(rw,rh)<32 or x+rw>w or y+rh>h:
        raise ValueError("Invalid target ROI")
    t=np.asarray(base_T_camera,float)
    if reference is None:
        pixels=np.array([[x,y],[x+rw-1,y],[x+rw-1,y+rh-1],[x,y+rh-1]],float)
        rays=np.column_stack((normalize(pixels),np.ones(4)))@t[:3,:3].T
        if np.any(abs(rays[:,2])<1e-8):
            raise ValueError("Table view contains horizon")
        distances=(paper_z-t[2,3])/rays[:,2]
        if np.any(distances<=0):
            raise ValueError("Table behind camera")
        corners=t[:3,3]+rays*distances[:,None]
        ex=t[:2,0].copy()
        if np.linalg.norm(ex)<0.2:
            raise ValueError("Camera too oblique for target view")
        ex/=np.linalg.norm(ex)
        ey=np.array([-ex[1],ex[0]])
        if ey@t[:2,1]<0:
            ey=-ey
        axes=np.column_stack((ex,ey))
        local=corners[:,:2]@axes
        lower=local.min(axis=0)
        scale=0.001
        width,height=np.ceil((local.max(axis=0)-lower)/scale).astype(int)+1
        if min(width,height)<64 or max(width,height)>1600 or width*height>1_600_000:
            raise ValueError("Target view too small/large; narrow target_roi")
        origin=lower@axes.T
    else:
        height,width=reference.image.shape[:2]
        origin,axes,scale=reference.origin_xy,reference.axes,reference.scale
    gy,gx=np.mgrid[:height,:width]
    xy=origin+np.column_stack((gx.ravel(),gy.ravel()))*scale@axes.T
    base=np.column_stack((xy,np.full(len(xy),paper_z)))
    camera=(base-t[:3,3])@t[:3,:3]
    if np.any(camera[:,2]<=0):
        raise ValueError("Rectified table behind camera")
    uv=project(camera).reshape(height,width,2).astype(np.float32)
    inside=(uv[:,:,0]>=x)&(uv[:,:,0]<x+rw)&(uv[:,:,1]>=y)&(uv[:,:,1]<y+rh)
    result=cv2.remap(image,uv[:,:,0],uv[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    result[~inside]=255
    return TableView(result,origin,axes,scale)


def require_stationary_board(reference, current, max_shift_m=0.003):
    """Track static printed features; occlusions are outliers, not movement.

    Requires distributed inliers. Texture-poor or obscured boards fail closed.
    This is a consistency check, not automatic re-registration of moved maps.
    """
    old=cv2.cvtColor(reference.image,cv2.COLOR_BGR2GRAY)
    new=cv2.cvtColor(current.image,cv2.COLOR_BGR2GRAY)
    # Only start at dark printed texture, excluding colored battery bodies.
    hsv=cv2.cvtColor(reference.image,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,1]<60)&(old<180)).astype(np.uint8)*255
    points=cv2.goodFeaturesToTrack(old,300,0.01,8,mask=mask)
    if points is None or len(points)<8:
        raise ValueError("Insufficient printed features to verify stationary board")
    moved,status,_=cv2.calcOpticalFlowPyrLK(old,new,points,None)
    back,back_status,_=cv2.calcOpticalFlowPyrLK(new,old,moved,None)
    valid=(status.ravel()==1)&(back_status.ravel()==1)&(np.linalg.norm(back-points,axis=2).ravel()<1)
    a,b=points.reshape(-1,2)[valid],moved.reshape(-1,2)[valid]
    if len(a)<8:
        raise ValueError("Printed board is occluded")
    affine,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=1.5)
    if affine is None or inliers.sum()<8:
        raise ValueError("Cannot verify stationary board")
    a=a[inliers.ravel()>0]
    if min(np.ptp(a,axis=0))*reference.scale<0.07:
        raise ValueError("Board verification features too concentrated")
    h,w=old.shape
    corners=np.array([[0,0],[w,0],[w,h],[0,h]],float)
    delta=np.column_stack((corners,np.ones(4)))@affine.T-corners
    if np.max(np.linalg.norm(delta,axis=1))*reference.scale>max_shift_m:
        raise ValueError("Printed board moved since target scan")
    return int(inliers.sum())
