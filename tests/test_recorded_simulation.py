"""Recorded playback must preserve evidence, without constructing live devices."""
import json
from datetime import datetime
from pathlib import Path

import pytest

from battery_sorting.recorded_simulation import export_recorded_t0


ROOT = Path(__file__).resolve().parents[1]
RECORDING = ROOT / 'output/smooth_t0/20260915_212913'


def test_new_room_trial_uses_compliant_current_home_with_archived_scene(monkeypatch):
    import cv2
    import numpy as np
    from tools.export_smooth_t0_simulation import export_trial
    from tools.run_smooth_t0 import require_competition_start
    from battery_sorting.config import Pose
    from battery_sorting.robot import RealUR5
    from battery_sorting.gripper import AG105145Gripper
    recording=ROOT/'output/smooth_t0/new_room_20260929_111616'
    if not recording.exists():
        pytest.skip('Optional new-room recording')
    def no_hardware(*args,**kwargs):
        raise AssertionError('Offline start-pose verification attempted hardware I/O')
    monkeypatch.setattr(RealUR5,'__init__',no_hardware)
    monkeypatch.setattr(AG105145Gripper,'__init__',no_hardware)
    plan=export_trial(recording,ROOT/'config/closed_loop_scene_20260929.json')
    assert plan['initial_pose'][:3]==[-.25,.16,.447856]
    require_competition_start(Pose(*plan['initial_pose']))
    assert plan['source']['predicted_timing']['motion_to_home_s']>10
    turns=[s for s in plan['steps'] if s['label'] in ('开始信号后在区外转为朝下观察姿态','回到区外高位后恢复侧向 Home')]
    assert len(turns)==2
    for step in turns:
        assert step['start_pose'][:3]==step['end_pose'][:3]==plan['initial_pose'][:3]
    require_competition_start(Pose(*turns[-1]['end_pose']))
    down=cv2.Rodrigues(np.array(turns[0]['end_pose'][3:]))[0][:,2]
    assert down==pytest.approx([0.,0.,-1.],abs=1e-7)


def test_completed_recording_preserves_timestamps_and_commanded_endpoints(monkeypatch):
    def no_hardware(*args, **kwargs):
        raise AssertionError('Playback export attempted live hardware I/O')

    from battery_sorting.robot import RealUR5
    from battery_sorting.gripper import AG105145Gripper
    monkeypatch.setattr(RealUR5, '__init__', no_hardware)
    monkeypatch.setattr(AG105145Gripper, '__init__', no_hardware)
    plan = export_recorded_t0(RECORDING, ROOT / 'config/closed_loop.json')
    audit = json.loads((RECORDING / 'audit.json').read_text(encoding='utf-8'))
    origin = datetime.fromisoformat(audit['timestamp_utc'])
    elapsed = 0.
    for step in plan['steps']:
        elapsed += step['duration_s']
        if step.get('recorded_event_utc') and not step.get('subsegment_time_is_estimated'):
            assert elapsed == pytest.approx(
                (datetime.fromisoformat(step['recorded_event_utc']) - origin).total_seconds())
    assert elapsed == pytest.approx(plan['source']['captured_frame_offset_s'])
    expected = []
    for event in audit['events']:
        if event['event'] == 'move':
            expected.append(event['pose'])
        elif event['event'] == 'smooth_path':
            expected.extend(event['poses'])
    assert [s['end_pose'] for s in plan['steps'] if s['kind'] == 'move'] == expected
    assert plan['source']['verification'] == audit['verification']
    assert {o['id'] for o in plan['objects']} == {'red', 'green', 'blue', 'yellow'}
    assert len(plan['targets'][0]['outline_base_xy']) == 4


@pytest.mark.parametrize('field', ['executed', 'completed'])
def test_unexecuted_or_incomplete_recordings_are_rejected_before_loading_frames(tmp_path, field):
    audit = {'executed': True, 'completed': True}
    audit[field] = False
    (tmp_path / 'audit.json').write_text(json.dumps(audit), encoding='utf-8')
    with pytest.raises(ValueError, match='completed, executed'):
        export_recorded_t0(tmp_path, ROOT / 'config/closed_loop.json')


def test_new_trial_runs_current_cycle_with_nominal_timing_and_correct_handle_axes(monkeypatch):
    import math
    from tools.export_smooth_t0_simulation import export_trial
    from battery_sorting.robot import RealUR5

    def no_hardware(*args, **kwargs):
        raise AssertionError('Simulation attempted a live robot connection')

    monkeypatch.setattr(RealUR5, '__init__', no_hardware)
    plan = export_trial(RECORDING, ROOT / 'config/closed_loop.json')
    assert plan['source']['simulation_variant'] == 'timing_trial'
    assert 'verification' not in plan['source']
    steps = plan['steps']
    assert sum(s['duration_s'] for s in steps) == pytest.approx(
        plan['source']['predicted_timing']['motion_and_stability_s'])
    assert next(s for s in steps if '悬停' in s['label'])['duration_s'] == pytest.approx(2.1)
    transfer = next(s for s in steps if '搬运平面' in s['label'])
    assert transfer['end_pose'][2] == pytest.approx(.269207)
    assert (transfer['speed_mps'], transfer['acceleration_mps2']) == (.5, 1.)
    contact = [s for s in steps if '慢速接触' in s['label']]
    assert len(contact) == 2
    assert all((s['speed_mps'], s['acceleration_mps2']) == (.04, .1) for s in contact)
    opened = next(i for i, s in enumerate(steps) if s['kind'] == 'open')
    direct = steps[opened + 1]
    assert direct['kind'] == 'move'
    assert direct['label'] == '放置点 → home 直线返回（稳定计时继续）'
    assert direct['start_pose'] == steps[opened]['end_pose']
    assert direct['end_pose'] == plan['initial_pose']
    assert len([s for s in steps[opened + 1:] if s['kind'] == 'move']) == 1
    assert sum(s['duration_s'] for s in steps[opened + 1:]) == pytest.approx(3.1)
    for obj in plan['objects']:
        yaw = 2 * math.atan2(obj['quaternion'][2], obj['quaternion'][3])
        assert abs(yaw) < math.radians(6), 'Use the handle direction instead of the ambiguous body edge'
