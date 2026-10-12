import json
from pathlib import Path

import pytest

from battery_sorting.closed_loop import ROOT, Setup, main


@pytest.mark.parametrize('task,filename,y_max', [
    ('basic', 'closed_loop.json', .27),
    ('sequence', 'closed_loop_sequence.json', .37),
])
def test_task_defaults_use_actual_workspace(task, filename, y_max, capsys):
    assert main(['readiness', '--task', task]) == 0
    report = json.loads(capsys.readouterr().out)
    assert Path(report['config']).name == filename
    assert report['workspace_bounds_m']['y_max'] == y_max
    setup = Setup(report['config'])
    assert setup.robot_config().workspace_bounds.y_max == y_max


def test_explicit_configuration_is_preserved(capsys):
    assert main(['readiness', '--task', 'sequence', '--config', str(ROOT/'config/closed_loop.json')]) == 0
    report = json.loads(capsys.readouterr().out)
    assert Path(report['config']).name == 'closed_loop.json'
    assert report['workspace_bounds_m']['y_max'] == .27


def test_sequence_launchers_use_actual_configuration():
    for name in ('启动第二题三块放置.bat', '检查第二题就绪状态.bat'):
        assert '--config config\\closed_loop_sequence.json' in (ROOT/name).read_text(encoding='utf-8')


def test_site_test_is_ready_without_falsifying_acceptance(capsys):
    assert main(['readiness','--task','sequence','--site-test'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['ready'] is True and report['site_test'] is True
    assert report['site_verified'] is False
    setup=Setup(ROOT/'config/closed_loop_sequence.json')
    assert setup.blockers('sequence')==['geometry_and_gripper_verified_by_task.sequence=false']
    setup.require_ready('sequence',site_test=True)
    setup.raw['sequence_target_roi']=None
    with pytest.raises(ValueError,match='sequence_target_roi'):
        setup.require_ready('sequence',site_test=True)


def test_auto_start_still_requires_explicit_execution():
    with pytest.raises(ValueError,match='requires run --execute'):
        main(['run','--task','sequence','--auto-start','--site-test'])
    with pytest.raises(ValueError,match='only available for sequence'):
        main(['run','--task','basic','--execute','--site-test'])


def test_latest_recorded_scene_bounds_are_checked(tmp_path):
    setup = Setup(ROOT / 'config/closed_loop_sequence.json')
    report = tmp_path / 'analysis.json'
    report.write_text(json.dumps({'instances': {'yellow': {
        'center_base_m': [-0.55, -0.12, 0.03],
        'handle': {'center_base_m': [-0.55, -0.12, 0.05]},
    }}}), encoding='utf-8')
    setup.validate_recorded_scene_bounds(report)
    report.write_text(json.dumps({'instances': {'yellow': {
        'center_base_m': [-0.70, -0.12, 0.03],
    }}}), encoding='utf-8')
    with pytest.raises(ValueError, match='outside configured workspace'):
        setup.validate_recorded_scene_bounds(report)


def test_recorded_scene_controller_safety_is_checked_before_motion(tmp_path):
    setup = Setup(ROOT / 'config/closed_loop_sequence.json')
    report = tmp_path / 'analysis.json'
    report.write_text(json.dumps({'instances': {'yellow': {
        'center_base_m': [-0.55, -0.12, 0.03],
        'handle': {'center_base_m': [-0.55, -0.12, 0.05]},
    }}}), encoding='utf-8')

    class RejectingRobot:
        def pose_within_safety_limits(self, pose):
            return False

    with pytest.raises(ValueError, match='rejected by UR safety limits'):
        setup.validate_recorded_scene_bounds(report, robot=RejectingRobot())
