import json
from dataclasses import replace

import pytest

from battery_sorting.simulation_export import export_mission_plan, export_offline_plan, nominal_motion_seconds, write_plan
from battery_sorting.offsite import build_offline_plan, load_site
from battery_sorting.mission import Mission
from battery_sorting.robot import FakeRobot
from battery_sorting.gripper import MockGripper
from battery_sorting.task import BasicTask, Color, SequenceTask
from battery_sorting.vision import ColorDetector
from tests.test_offsite import complete_site_dict, pick_xy


def test_mission_export_preserves_actual_actions_and_poses(valid_config, frame, tmp_path):
    task=SequenceTask((Color.BLUE,Color.RED,Color.YELLOW))
    mission=Mission(valid_config,FakeRobot(valid_config.robot.taught_poses['home']),MockGripper())
    detected=ColorDetector(valid_config.vision,valid_config.camera.width,valid_config.camera.height).detect(frame)
    plan=mission.build_plan(task,detected)
    bundle=export_mission_plan(plan,valid_config.robot,task)
    assert [s['label'] for s in bundle['steps']]==[a.label for a in plan.actions]
    for step,action in zip(bundle['steps'],plan.actions):
        if action.waypoint:
            assert step['end_pose']==action.waypoint.pose.as_list()
        if action.kind.value=='dwell':
            assert step['duration_s']==action.seconds
    altered=list(bundle['steps'])
    assert len(altered)==63
    path=tmp_path/'plan.json'
    write_plan(bundle,path)
    assert json.loads(path.read_text(encoding='utf-8'))==bundle


def test_offsite_preserves_home_rotation_and_placement_height():
    raw=complete_site_dict()
    raw['geometry']['home_pose_m_rad'][3:]=[-.041577688,-3.134181163,-.006394028]
    site=load_site(raw)
    plan=build_offline_plan(site,'basic',['green'],pick_xy())
    exported=export_offline_plan(plan,site,speeds=dict(transit=.05,approach=.02,contact=.005),acceleration=.1)
    assert [s['label'] for s in exported['steps']]==[s.label for s in plan.steps]
    assert any(s['label']=='orient_for_work' and s['duration_s']>0 for s in exported['steps'])
    assert exported['steps'][-1]['end_pose']==list(site.home_pose_m_rad.as_tuple())
    assert next(s for s in exported['steps'] if s['label']=='green:T0_contact')['end_pose'][2]==.070


def test_missing_geometry_cannot_export():
    from pathlib import Path
    site=load_site(Path(__file__).resolve().parents[1]/'config/competition_template.json')
    with pytest.raises(RuntimeError,match='阻塞'):
        build_offline_plan(site,'basic',['green'],pick_xy())


def test_timing_rejects_invalid_speed():
    with pytest.raises(ValueError):
        nominal_motion_seconds([0]*6,[0,0,.1,0,0,0],float('nan'),.1)
    assert nominal_motion_seconds([0]*6,[0,0,.1,0,0,0],.05,.1)==pytest.approx(2.5)


def test_export_cli_never_allows_execute():
    from main import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--export-simulation','tmp/result.json','--execute'])


def test_main_export_stops_before_any_hardware_connection(valid_config, frame, tmp_path, monkeypatch):
    import main
    def forbidden(*args, **kwargs):
        raise AssertionError('Export must not connect a device')
    monkeypatch.setattr(FakeRobot,'connect_readonly',forbidden)
    monkeypatch.setattr(FakeRobot,'connect_control',forbidden)
    monkeypatch.setattr(MockGripper,'connect',forbidden)
    # Old shared fixture rounds 0.1 degrees slightly above its exact limit.
    config_path=tmp_path/'valid.json'
    raw=json.loads(config_path.read_text(encoding='utf-8'))
    raw['robot']['segment_start_rotation_tolerance']=.0017
    config_path.write_text(json.dumps(raw),encoding='utf-8')
    detection=ColorDetector(valid_config.vision,valid_config.camera.width,valid_config.camera.height).detect(frame)
    monkeypatch.setattr(main,'detect',lambda *args,**kwargs:detection)
    image=tmp_path/'already_detected.png'
    out=tmp_path/'simulation.json'
    assert main.run(['--config',str(tmp_path/'valid.json'),'--image',str(image),'--export-simulation',str(out)])==0
    assert json.loads(out.read_text(encoding='utf-8'))['source']['planner']=='battery_sorting.mission'
