"""Shared competition standby rules; pure geometry, no hardware I/O."""
import json
import math
from pathlib import Path
import cv2
import numpy as np
from .config import Pose

ROOT=Path(__file__).resolve().parents[1]


def segment_zone_distance_xy(start,end,bounds):
    """Exact separation between a flange-to-TCP segment and an XY rectangle."""
    xmin,xmax,ymin,ymax=bounds
    p=np.asarray(start,float);q=np.asarray(end,float);delta=q-p
    lo,hi=0.,1.
    for axis,(lower,upper) in enumerate(((xmin,xmax),(ymin,ymax))):
        if abs(delta[axis])<1e-12:
            if not lower<=p[axis]<=upper:lo,hi=1.,0.;break
        else:
            a,b=sorted(((lower-p[axis])/delta[axis],(upper-p[axis])/delta[axis]))
            lo=max(lo,a);hi=min(hi,b)
    if lo<=hi:return 0.
    distances=[math.hypot(max(xmin-v[0],0.,v[0]-xmax),
                         max(ymin-v[1],0.,v[1]-ymax)) for v in (p,q)]
    length=float(delta@delta)
    for corner in ((xmin,ymin),(xmin,ymax),(xmax,ymin),(xmax,ymax)):
        v=np.asarray(corner,float)
        t=0. if length<1e-12 else float(np.clip((v-p)@delta/length,0.,1.))
        distances.append(float(np.linalg.norm(v-(p+t*delta))))
    return min(distances)


def require_competition_start(pose, rules=None):
    """Keep a sideways gripper's capsule footprint outside the printed zone."""
    if rules is None:
        rules=json.loads((ROOT/'config/t0_start_pose.json').read_text(encoding='utf-8'))
    xmin,xmax,ymin,ymax=rules['battery_zone_bounds_base_xy_m']
    radius=rules['tool_footprint_radius_m']
    margin=rules['outside_margin_m']
    tolerance=rules['max_parallel_tilt_deg']
    tool_length=rules['tool_axis_to_flange_m']
    if (not all(math.isfinite(v) for v in (xmin,xmax,ymin,ymax,radius,margin,tolerance,tool_length,*pose.as_list()))
            or xmin>=xmax or ymin>=ymax or radius<=0 or tool_length<=0 or margin<0 or not 0<tolerance<=1):
        raise ValueError('Invalid competition start constraints')
    normal=cv2.Rodrigues(np.asarray(pose.as_list()[3:],float))[0][:,2]
    flange_xy=np.array([pose.x,pose.y])-tool_length*normal[:2]
    distance=segment_zone_distance_xy([pose.x,pose.y],flange_xy,(xmin,xmax,ymin,ymax))
    if distance<radius+margin-1e-9:
        raise ValueError('Competition start tool footprint overlaps battery zone')
    tilt=math.degrees(math.asin(float(np.clip(abs(normal[2]),0.,1.))))
    if tilt>tolerance:
        raise ValueError(f'Competition start gripper long axis is not parallel to table: {tilt:.3f}deg')
    return {'zone_distance_m':distance,'tool_envelope_margin_m':distance-radius,
            'parallel_tilt_deg':tilt}


def competition_observation_pose(setup):
    pose=Pose(*setup.raw.get('pick_view_pose_m_rad', setup.raw['target_view_pose_m_rad']))
    normal=cv2.Rodrigues(np.asarray(pose.as_list()[3:],float))[0][:,2]
    if not np.allclose(normal,[0.,0.,-1.],atol=1e-6):
        raise ValueError('Competition observation camera/tool must point down')
    return pose


