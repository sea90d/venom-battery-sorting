import json
import shutil
import numpy as np
import pytest
from battery_sorting.closed_loop import Setup
from tools import estimate_sequence_setup as estimator


def test_similarity_allows_reflected_image_to_base_axes():
    source = np.array([[0, 0], [300, 0], [300, 250], [0, 250]], float)
    expected = source@np.array([[0, .001], [.001, 0]])+[-.45, -.3]
    linear, offset = estimator.fit_similarity(source, expected)
    np.testing.assert_allclose(source@linear+offset, expected, atol=1e-12)


def test_map_estimate_recovers_labels_without_enabling_hardware(tmp_path, monkeypatch):
    original_root = estimator.ROOT
    paths = ['config/closed_loop.json', 'config/competition_template.json',
             'tmp/pdfs/board-1.png', 'tmp/pdfs/board-2.png',
             'task/《2026年挑战赛任务底图》.pdf',
             'output/new_room_high_20260929_111517_analysis/analysis.json']
    paths += ['output/new_room_high_20260929_111517/'+name
              for name in ('manifest.json', 'colour.png', 'depth_m.npy')]
    for name in paths:
        destination = tmp_path/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_root/name, destination)
    before = (tmp_path/'config/closed_loop.json').read_bytes()
    monkeypatch.setattr(estimator, 'ROOT', tmp_path)
    report = estimator.estimate()
    assert (tmp_path/'config/closed_loop.json').read_bytes() == before
    assert report['t0_independent_crosscheck_error_m'] < .003
    points = report['target_centers_base_xy_m']
    for a, b in [('P1', 'P2'), ('P2', 'P3')]:
        assert np.linalg.norm(np.array(points[a])-points[b]) == pytest.approx(.175, abs=.003)
    for label in points:
        np.testing.assert_allclose(report['projected_detector_centers_base_xy_m'][label], points[label], atol=.001)
    assert report['targets_outside_production_workspace'] == ['P2', 'P3']
    assert report['production_workspace_expanded'] is False
    estimator.estimate(apply=True)
    saved = json.loads((tmp_path/'config/closed_loop.json').read_text(encoding='utf-8'))
    assert saved['sequence_target_roi'] == report['sequence_target_roi']
    assert saved['geometry_and_gripper_verified_by_task']['sequence'] is False
    assert (tmp_path/'output/sequence_estimate/closed_loop_before_estimate.json').read_bytes()
    assert (tmp_path/'config/competition_template.json').read_bytes() == (original_root/'config/competition_template.json').read_bytes()


def test_candidate_configuration_keeps_live_verification_required():
    setup = Setup(estimator.ROOT/'config/closed_loop_sequence_estimate.json')
    assert setup.blockers('sequence') == ['geometry_and_gripper_verified_by_task.sequence=false']
    with pytest.raises(ValueError, match='geometry_and_gripper_verified_by_task.sequence'):
        setup.require_ready('sequence')
    # In-memory nominal check only; never save or run a verified flag.
    setup.raw['geometry_and_gripper_verified_by_task']['sequence'] = True
    setup.require_ready('sequence')
    from battery_sorting.config import Pose
    for xy in setup.site.target_centers_base_xy_m.values():
        if xy is not None:
            assert setup.in_workspace(Pose(*xy, setup.site.placement_contact_z_m, *setup.site.tool_rotation_vector_rad))
