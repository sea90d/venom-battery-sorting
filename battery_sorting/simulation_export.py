"""Serialize existing Python plans for the browser; never connect hardware.

Both adapters consume the planner's actual actions. They do not rebuild paths.
JSON contains data only: poses in UR base metres / axis-angle radians.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

COLORS = {"red", "yellow", "blue", "green"}
SCHEMA = "ur5-simulation"


def _vector(value, n=6):
    result = list(value)
    if len(result) != n or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in result):
        raise ValueError(f"Expected {n} finite numeric values")
    return result


def _quat(rv):
    angle = math.sqrt(sum(v*v for v in rv))
    return [0, 0, 0, 1] if angle < 1e-12 else [*(v * math.sin(angle/2)/angle for v in rv), math.cos(angle/2)]


def nominal_motion_seconds(start, end, speed, acceleration, angular_speed=0.5):
    """Stop-at-waypoint trapezoidal translation; orientation time is an estimate."""
    if any(not math.isfinite(v) or v <= 0 for v in (speed, acceleration, angular_speed)):
        raise ValueError("Motion speed and acceleration must be finite and positive")
    distance = math.dist(start[:3], end[:3])
    translation = (distance/speed + speed/acceleration if distance >= speed*speed/acceleration
                   else 2*math.sqrt(distance/acceleration))
    dot = abs(sum(a*b for a, b in zip(_quat(start[3:]), _quat(end[3:]))))
    rotation = 2*math.acos(min(1.0, dot))/angular_speed
    return max(translation, rotation)


def _bundle(actions, initial_pose, *, planner, label, data_kind, colors, targets,
            speeds, acceleration, object_centers=None, angular_speed=0.5):
    if data_kind not in {"configured", "demo"}:
        raise ValueError("data_kind must be configured or demo")
    initial = _vector(initial_pose)
    target_map = {k: _vector(v) for k, v in targets.items()}
    if len(colors) not in (1, 3) or len(set(colors)) != len(colors) or any(c not in COLORS for c in colors):
        raise ValueError("Task must contain one or three distinct colours")
    assigned = dict(zip(colors, ["T0"] if len(colors) == 1 else ["P1", "P2", "P3"])) if targets else {}
    if any(t not in target_map for t in assigned.values()):
        raise ValueError("Missing placement pose; cannot substitute demo coordinates")
    current = initial
    steps = []
    pick_contacts = {}
    for action in actions:
        kind, name = action["kind"], action["label"]
        if kind not in {"initialize", "move", "open", "close", "dwell", "check_object", "visual_accept"}:
            raise ValueError(f"Unsupported action: {kind}")
        color = name.split(":", 1)[0] if ":" in name else None
        color = color if color in COLORS else None
        start = _vector(action.get("start_pose") or current)
        if max(abs(a-b) for a,b in zip(start,current)) > 1e-9:
            raise ValueError(f"Discontinuous Python plan: {name}")
        end = _vector(action.get("end_pose") or current)
        profile = action.get("profile") if kind == "move" else None
        speed = action.get("speed_mps",speeds.get(profile)) if kind == "move" else None
        duration = nominal_motion_seconds(start, end, speed, acceleration, angular_speed) if kind == "move" else action.get("seconds", 0.0)
        if isinstance(duration, bool) or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"Invalid duration: {name}")
        if kind != "move" and end != start:
            raise ValueError(f"Non-motion action changed pose: {name}")
        if name.endswith(":pick_contact"):
            pick_contacts[color] = end
        object_id = action.get("object_id",color if kind in {"close", "check_object", "visual_accept", "open"} else None)
        if kind == "close" and not object_id:
            raise ValueError(f"Cannot identify held object for {name}")
        if kind == "close" and object_id not in pick_contacts:
            pick_contacts[object_id] = end
        steps.append(dict(label=name, kind=kind, start_pose=start, end_pose=end,
                          profile=profile, speed_mps=speed,
                          acceleration_mps2=acceleration if kind == "move" else None,
                          duration_s=duration, object_id=object_id,
                          target_id=assigned.get(color) if kind in {"open", "visual_accept"} else None))
        current = end
    if set(colors) - set(pick_contacts):
        raise ValueError("Missing pick_contact poses in Python plan")
    centers = {c: [p[0], p[1], .04] for c,p in pick_contacts.items()}
    if object_centers is not None:
        centers.update({c: _vector(p,3) for c,p in object_centers.items()})
    if any(c not in COLORS for c in centers):
        raise ValueError("Unknown object colour")
    source_file = Path(__file__).with_name(planner.split('.')[-1] + '.py')
    return dict(schema=SCHEMA, version=1, frame="ur_base", units=dict(position="m",rotation="rad"),
                source=dict(planner=planner, label=label, data_kind=data_kind,
                            planner_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
                            timing="nominal trapezoid; angular speed and IO latency are visualization assumptions",
                            angular_speed_rad_s=angular_speed,
                            geometry="CAD centered 40 mm above table Z=0 unless object_centers supplied; no physics or collision solver"),
                initial_pose=initial, initial_gripper_open=False,
                objects=[dict(id=c,color=c,center=p,quaternion=[0,0,0,1],size_m=[.07,.08,.08]) for c,p in centers.items()],
                targets=[dict(id=k,pose=p) for k,p in target_map.items()],steps=steps)


def export_mission_plan(plan, robot_config, task, *, data_kind="configured", label="MissionPlan", object_centers=None):
    """Call immediately after Mission.build_plan(), before any connect/execute."""
    from .task import BasicTask
    home = robot_config.taught_poses.get("home")
    if home is None:
        raise ValueError("home is not configured")
    colors = [task.target_color.value] if isinstance(task, BasicTask) else [c.value for c in task.color_order]
    names = ["T0"] if len(colors)==1 else ["P1","P2","P3"]
    if any(robot_config.taught_poses.get(n) is None for n in names):
        raise ValueError("Placement poses are not configured")
    actions = [dict(kind=a.kind.value,label=a.label,
                    end_pose=a.waypoint.pose.as_list() if a.waypoint else None,
                    profile=a.waypoint.profile if a.waypoint else None,seconds=a.seconds) for a in plan.actions]
    return _bundle(actions,home.as_list(),planner="battery_sorting.mission",label=label,data_kind=data_kind,
                   colors=colors,targets={n:robot_config.taught_poses[n].as_list() for n in names},
                   speeds=dict(transit=robot_config.transit_speed,approach=robot_config.approach_speed,contact=robot_config.contact_speed),
                   acceleration=robot_config.move_acceleration,object_centers=object_centers)


def export_offline_plan(plan, site, *, speeds, acceleration, data_kind="configured", label="OfflinePlan", object_centers=None):
    """OfflinePlan has no speed fields: caller must provide the timing profile."""
    from .offsite import validate_offline_plan
    validate_offline_plan(plan, site)
    names = ["T0"] if plan.task=="basic" else ["P1","P2","P3"]
    targets = {n:[*site.target_centers_base_xy_m[n],site.placement_contact_z_m,*site.tool_rotation_vector_rad] for n in names}
    return _bundle(plan.to_dict()["steps"],site.home_pose_m_rad.as_tuple(),planner="battery_sorting.offsite",label=label,
                   data_kind=data_kind,colors=list(plan.colors),targets=targets,speeds=speeds,acceleration=acceleration,
                   object_centers=object_centers)


def write_plan(bundle, path):
    """Output artifacts only. Never modifies robot configuration."""
    payload = json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(payload,encoding="utf-8")


def export_grasp_lift_plan(plan, config):
    """Trace the original successful execute_plan on recording fake devices.

    No user-supplied Python is executed. Only this repository's fixed execution
    function runs, with offline fakes and a recording (non-sleeping) clock.
    """
    from .robot import FakeRobot
    from .gripper import MockGripper
    from .grasp_lift import execute_plan, validate_plan
    validate_plan(plan,config)
    records=[]
    class RecordingRobot(FakeRobot):
        def move_l_without_tcp_force_guard(self,waypoint,speed,acceleration):
            records.append(dict(kind="move",label=waypoint.label,
                                start_pose=self.snapshot().tcp_pose,end_pose=waypoint.pose.as_list(),
                                profile=waypoint.profile,speed_mps=speed))
            return super().move_l_without_tcp_force_guard(waypoint,speed,acceleration)
    class RecordingGripper(MockGripper):
        def open(self):
            records.append(dict(kind="open",label="gripper_open_high"))
            return super().open()
        def close(self):
            records.append(dict(kind="close",label="gripper_close",object_id=plan.color))
            return super().close()
        def has_object(self):
            records.append(dict(kind="check_object",label="object_feedback",object_id=plan.color))
            return super().has_object()
    def wait(seconds):
        records.append(dict(kind="dwell",label="grip_settle",seconds=seconds))
    execute_plan(plan,config,RecordingRobot(plan.start_pose),RecordingGripper(object_after_close=True),sleep=wait)
    bundle=_bundle(records,plan.start_pose.as_list(),planner="battery_sorting.grasp_lift",
                   label=f"历史现场采点 · {plan.color} 抓取提升 · 成功分支",data_kind="configured",colors=[plan.color],
                   targets={},speeds={},acceleration=config.acceleration_mps2)
    bundle['source']['preconditions']="Start at validated pregrasp hover; gripper already initialized. Historical evidence, not a live camera capture."
    return bundle


def main(argv=None):
    parser=argparse.ArgumentParser(description="Export the existing offline Python plan for HTML simulation")
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--picks", required=True, type=Path, help="JSON mapping colour to actual Base [X,Y] metres")
    parser.add_argument("--task",choices=["basic","sequence"],default="basic")
    parser.add_argument("--colors",nargs="+",default=["green"])
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--transit-speed",type=float,required=True)
    parser.add_argument("--approach-speed",type=float,required=True)
    parser.add_argument("--contact-speed",type=float,required=True)
    parser.add_argument("--acceleration",type=float,required=True)
    args=parser.parse_args(argv)
    from .offsite import load_site, build_offline_plan
    site=load_site(args.site)
    picks=json.loads(args.picks.read_text(encoding="utf-8"))
    plan=build_offline_plan(site,args.task,args.colors,picks)
    bundle=export_offline_plan(plan,site,speeds=dict(transit=args.transit_speed,approach=args.approach_speed,contact=args.contact_speed),
                               acceleration=args.acceleration,label=args.site.name)
    write_plan(bundle,args.out)
    print(f"Exported {len(bundle['steps'])} Python actions to {args.out}; no hardware connected")


if __name__ == "__main__":
    main()
