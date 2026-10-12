"""Pure RGB-D instance geometry for the eye-in-hand competition runtime."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .geometry import validate_rigid_transform


class SceneError(ValueError):
    pass


@dataclass(frozen=True)
class BatteryInstance:
    color: str
    center_base_m: tuple[float, float, float]
    yaw_rad: float
    corners_base_xy: tuple
    size_m: tuple[float, float]
    height_m: float
    height_mad_m: float
    centroid_px: tuple[float, float]
    handle: dict | None = None


def _plane_geometry(pixels, z, transform, normalize_pixels):
    rays=np.asarray(normalize_pixels(np.asarray(pixels,float)),float)
    direction=np.column_stack((rays,np.ones(len(rays))))@transform[:3,:3].T
    if not np.isfinite(direction).all() or np.any(abs(direction[:,2])<1e-8):
        raise SceneError('Invalid surface rays')
    distance=(z-transform[2,3])/direction[:,2]
    if np.any(distance<=0):
        raise SceneError('Surface behind camera')
    points=transform[:3,3]+distance[:,None]*direction
    rect=cv2.minAreaRect(points[:,:2].astype(np.float32))
    corners=cv2.boxPoints(rect)
    edges=np.roll(corners,-1,axis=0)-corners
    edge=edges[np.argmax(np.linalg.norm(edges,axis=1))]
    return rect,corners,math.atan2(float(edge[1]),float(edge[0]))%math.pi


def _body_and_handle(z, pixels, shape, transform, normalize_pixels, tolerance, name):
    """Bounded two-level model, only for a rectangular body with one narrow handle.

    Mixed edge depths are not planar support. Each surface uses its own plane;
    never project the elevated handle silhouette onto the body plane.
    """
    def mode(values):
        edges=np.arange(float(values.min())-.002,float(values.max())+.004,.002)
        hist,_=np.histogram(values,edges)
        smooth=np.convolve(hist,np.ones(5),'full')[2:2+len(hist)]
        return float(edges[int(np.argmax(smooth))]+.001)
    body_mode=mode(z)
    upper=z[z>body_mode+.018]
    if len(upper)<max(150,.10*len(z)):
        raise SceneError(f'Nonplanar battery surface: {name}; no supported handle')
    handle_mode=mode(upper)
    bands=[abs(z-body_mode)<=.006,abs(z-handle_mode)<=.006]
    results=[]
    for index,band in enumerate(bands):
        mask=np.zeros(shape,np.uint8)
        mask[pixels[band,1],pixels[band,0]]=1
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        if index==1:
            ids=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=75]
            if len(ids)==0:
                raise SceneError(f'Ambiguous handle support: {name}')
            if len(ids)>1:
                raise SceneError(f'Ambiguous handle support: {name}')
            best=max(ids,key=lambda i:stats[i,cv2.CC_STAT_AREA])
            mask=(labels==best).astype(np.uint8)
        used=(mask[pixels[:,1],pixels[:,0]]>0)&band
        if used.sum()<150:
            raise SceneError(f'Insufficient planar support: {name}')
        median=float(np.median(z[used]));mad=float(np.median(abs(z[used]-median)))
        rect,corners,yaw=_plane_geometry(pixels[used],median,transform,normalize_pixels)
        pixel_rect=cv2.minAreaRect(pixels[used].astype(np.float32))
        occupancy=used.sum()/max(1,float(np.prod(pixel_rect[1])))
        results.append((mask,median,mad,rect,corners,yaw,occupancy))
    body,handle=results
    bs=sorted(body[3][1]);hs=sorted(handle[3][1])
    delta=handle[1]-body[1]
    # A handle is a narrow raised strip, not an arbitrary noisy upper patch.
    # This rejects broad elevated reflections and split/continuous depth
    # ramps before a grasp point is produced.
    if delta < .018 or hs[0] < .008 or hs[1] < .030 or hs[0]/max(hs[1],1e-9) > .75:
        raise SceneError(f'Nonplanar battery surface: {name}; weak two-plane support')
    if hs[1] > max(bs[1]*1.25, .045):
        raise SceneError(f'Nonplanar battery surface: {name}; weak two-plane support')
    descriptor=dict(center_base_m=[*map(float,handle[3][0]),handle[1]],
                    yaw_rad=handle[5],size_m=list(map(float,hs)),
                    height_above_body_m=delta,height_mad_m=handle[2],
                    centroid_px=list(map(float,np.mean(pixels[handle[0][pixels[:,1],pixels[:,0]]>0],axis=0))),
                    corners_base_xy=handle[4].tolist())
    return body,descriptor


def detect_batteries(colour, depth_m, thresholds, base_T_camera, normalize_pixels,
                     paper_z_m, *, roi=None, min_height_m=0.015,
                     max_height_m=0.12, max_height_mad_m=0.005, min_pixels=150):
    """Detect zero or more independent instances; reject ambiguous same colors.

    Use actual Base Z for foreground and dispersion. Plane intersections of
    each colour contour give metric body center/yaw, not distorted RGB moments.
    The calibrated handle offset belongs to the grasp planner, not this module.
    """
    if not isinstance(colour,np.ndarray) or colour.dtype!=np.uint8 or colour.ndim!=3 or colour.shape[2]!=3:
        raise SceneError("Expected uint8 BGR image")
    if not isinstance(depth_m,np.ndarray) or depth_m.shape!=colour.shape[:2] or not np.issubdtype(depth_m.dtype,np.floating):
        raise SceneError("Depth must be aligned floating point metres")
    values=(paper_z_m,min_height_m,max_height_m,max_height_mad_m,min_pixels)
    if not all(math.isfinite(v) for v in values) or not 0<min_height_m<max_height_m or max_height_mad_m<=0 or min_pixels<20:
        raise SceneError("Invalid depth/instance thresholds")
    transform=np.asarray(base_T_camera,float)
    validate_rigid_transform(transform,"base_T_camera")
    h,w=depth_m.shape
    x,y,rw,rh=(0,0,w,h) if roi is None else roi
    if any(type(v) is not int for v in (x,y,rw,rh)) or min(x,y)<0 or min(rw,rh)<=0 or x+rw>w or y+rh>h:
        raise SceneError("Invalid scene ROI")
    hsv=cv2.cvtColor(colour,cv2.COLOR_BGR2HSV)
    valid_depth=np.isfinite(depth_m)&(depth_m>=0.2)&(depth_m<=3.0)
    result={}
    all_masks=[]
    named=list(thresholds)
    if len({t.name for t in named})!=len(named):
        raise SceneError("Duplicate color thresholds")
    kernel=np.ones((3,3),np.uint8)
    for threshold in named:
        color_mask=np.zeros((h,w),np.uint8)
        for r in threshold.ranges:
            color_mask|=cv2.inRange(hsv,np.array([r.h_min,r.s_min,r.v_min],np.uint8),np.array([r.h_max,r.s_max,r.v_max],np.uint8))
        search=np.zeros((h,w),bool)
        search[y:y+rh,x:x+rw]=True
        color_mask[~search]=0
        if any(np.any((color_mask>0)&m) for m in all_masks):
            raise SceneError("Overlapping HSV thresholds")
        all_masks.append(color_mask>0)
        py,px=np.nonzero((color_mask>0)&valid_depth)
        visible_pixels=int(np.count_nonzero(color_mask))
        if len(px)<min_pixels:
            # A visible colour with no usable depth must not look like an
            # empty slot.  The caller may retry at the same observation pose,
            # but it must receive a deterministic diagnostic.
            if visible_pixels>=min_pixels:
                raise SceneError(f'Insufficient depth: {threshold.name}')
            continue
        pixels=np.column_stack((px,py)).astype(float)
        rays=np.asarray(normalize_pixels(pixels),float)
        if rays.shape!=pixels.shape or not np.isfinite(rays).all():
            raise SceneError("Invalid native camera rays")
        xyz=np.column_stack((rays,np.ones(len(rays))))*depth_m[py,px,None]
        base=xyz@transform[:3,:3].T+transform[:3,3]
        heights=base[:,2]-paper_z_m
        raised=(heights>=min_height_m)&(heights<=max_height_m)
        mask=np.zeros((h,w),np.uint8)
        mask[py[raised],px[raised]]=255
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        candidates=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=min_pixels]
        if len(candidates)>1:
            raise SceneError(f'Duplicate raised instances: {threshold.name}')
        if not candidates:
            if visible_pixels>=min_pixels:
                raise SceneError(f'Insufficient raised depth: {threshold.name}')
            continue
        label=candidates[0]
        lx,ly,lw,lh,_=stats[label]
        instance=labels==label
        # A component touching the configured ROI edge is incomplete.  Using
        # the selected raised component (rather than every RGB speckle in the
        # ROI) avoids treating printed graphics or background reflections as a
        # clipped battery.
        if lx<=x or ly<=y or lx+lw>=x+rw or ly+lh>=y+rh:
            raise SceneError(f'Clipped instance: {threshold.name}')
        # Check the original RGB component as well as the depth silhouette:
        # otherwise a missing strip of depth silently shifts the center.
        _,color_labels=cv2.connectedComponents((color_mask>0).astype(np.uint8))
        parent_ids=np.unique(color_labels[instance])
        parent=np.isin(color_labels,parent_ids[parent_ids!=0])
        parent_area=max(1,int(np.count_nonzero(parent)))
        if np.count_nonzero(parent&valid_depth)<0.80*parent_area:
            raise SceneError(f'Low original color depth coverage: {threshold.name}')
        selected=instance[py,px]
        z=base[selected,2]
        median=float(np.median(z))
        mad=float(np.median(abs(z-median)))
        handle=None
        body_geometry=None
        if mad>max_height_mad_m or np.quantile(abs(z-median),0.90)>3*max_height_mad_m:
            body_geometry,handle=_body_and_handle(z,np.column_stack((px[selected],py[selected])),
                (h,w),transform,normalize_pixels,max_height_mad_m,threshold.name)
            median,mad=body_geometry[1:3]
        # Count holes within the external silhouette to reject missing-depth
        # clipping that could otherwise move the inferred body center.
        contours,_=cv2.findContours(instance.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=max(contours,key=cv2.contourArea)
        filled=np.zeros((h,w),np.uint8)
        cv2.drawContours(filled,[contour],-1,1,-1)
        color_inside=(filled>0)&(color_mask>0)
        if np.count_nonzero(valid_depth&color_inside)<0.80*np.count_nonzero(color_inside):
            raise SceneError(f'Low component depth coverage: {threshold.name}')
        if body_geometry is not None:
            rect=body_geometry[3]
        else:
            rect,_,_=_plane_geometry(contour.reshape(-1,2),median,transform,normalize_pixels)
        corners=cv2.boxPoints(rect)
        edges=np.roll(corners,-1,axis=0)-corners
        lengths=np.linalg.norm(edges,axis=1)
        edge=edges[np.argmax(lengths)]
        short,long=sorted(rect[1])
        if short<0.02 or long>0.13 or long<0.03:
            raise SceneError(f'Unexpected battery dimensions: {threshold.name} ({short}, {long})')
        yaw=math.atan2(float(edge[1]),float(edge[0]))%math.pi
        moments=cv2.moments(contour)
        centroid=(moments['m10']/moments['m00'],moments['m01']/moments['m00'])
        result[threshold.name]=BatteryInstance(threshold.name,(*map(float,rect[0]),median),yaw,
            tuple(tuple(map(float,p)) for p in corners),(float(short),float(long)),
            median-paper_z_m,mad,centroid,handle)
    return result
