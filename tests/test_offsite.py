from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from battery_sorting.offsite import (
    OffsiteError,
    OfflinePlan,
    PlanStep,
    Pose6,
    ReplayBundle,
    ReplayIntrinsics,
    StepKind,
    build_offline_plan,
    geometry_report,
    load_replay_bundle,
    load_site,
    readiness_report,
    save_replay_bundle,
    simulate_offline_plan,
    validate_offline_plan,
)


ROOT = Path(__file__).resolve().parents[1]
ROTATION = [-2.2448617460523574, -2.1864166279658708, 0.00325035854435066]


def complete_site_dict() -> dict:
    return {
        "schema_version": 1,
        "robot": {"host": "192.168.1.10", "dashboard_port": 29999, "rtde_port": 30004},
        "gripper": {"driver": "ag105145_modbus", "port": "COM11"},
        "calibration": {
            "handeye_result": str(
                ROOT / "output" / "handeye" / "session_20260910" / "handeye_result.json"
            )
        },
        "geometry": {
            "pick_contact_z_m": 0.069207,
            "safe_high_z_m": 0.447856,
            "final_slow_distance_m": 0.01,
            "tool_rotation_vector_rad": ROTATION,
            "home_pose_m_rad": [-0.05, -0.22, 0.447856, *ROTATION],
            "placement_contact_z_m": 0.070,
            "workspace_bounds_m": [-0.30, 0.30, -0.50, 0.10, 0.04, 0.50],
            "target_centers_base_xy_m": {
                "T0": [-0.114, -0.295],
                "P1": [-0.13, -0.24],
                "P2": [-0.04, -0.24],
                "P3": [-0.04, -0.34],
            },
        },
        "timing": {"grip_settle_s": 0.5, "lifted_hold_s": 2.1, "release_settle_s": 3.1},
        "verification": {
            "gripper_live_verified": True,
            "hover_verified_by_color": {
                "red": True,
                "yellow": True,
                "blue": True,
                "green": True,
            },
            "full_cycle_live_verified": True,
        },
    }


def pick_xy() -> dict[str, tuple[float, float]]:
    return {
        "red": (0.05, -0.20),
        "yellow": (0.13, -0.20),
        "blue": (0.05, -0.30),
        "green": (0.13, -0.30),
    }


def test_shipped_template_has_known_values_and_explicit_blockers() -> None:
    site = load_site(ROOT / "config" / "competition_template.json")
    report = readiness_report(site, "all")
    assert report["known"] == {
        "robot_host": "192.168.1.10",
        "gripper_port": "COM11",
        "pick_contact_z_m": 0.069207,
        "safe_high_z_m": 0.447856,
        "handeye_passed": True,
    }
    assert report["offline_plan_ready"] is False
    assert not any("home_pose" in item for item in report["blockers"])
    assert site.placement_contact_z_m == site.pick_contact_z_m == 0.069207
    assert not any("placement_contact_z" in item for item in report["blockers"])
    assert any("T0" in item for item in report["blockers"])


def test_strict_config_rejects_extra_key_and_wrong_site_identity() -> None:
    raw = complete_site_dict()
    raw["extra"] = 1
    with pytest.raises(OffsiteError, match="字段不匹配"):
        load_site(raw)
    raw = complete_site_dict()
    raw["robot"]["host"] = "192.168.0.10"
    with pytest.raises(OffsiteError, match="192.168.1.10"):
        load_site(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("pick_contact_z_m", 0.08, "pick_contact_z_m"),
        ("safe_high_z_m", 0.44, "safe_high_z_m"),
        ("final_slow_distance_m", 0.02, "final_slow_distance_m"),
        ("tool_rotation_vector_rad", [-2.2, -2.1, 0.0], "tool_rotation_vector_rad"),
    ),
)
def test_strict_config_rejects_changes_to_verified_geometry(
    field: str, value: object, message: str
) -> None:
    raw = complete_site_dict()
    raw["geometry"][field] = value
    with pytest.raises(OffsiteError, match=message):
        load_site(raw)


def test_basic_plan_and_state_simulation_cover_full_cycle() -> None:
    site = load_site(complete_site_dict())
    plan = build_offline_plan(site, "basic", ["green"], pick_xy())
    assert plan.steps[0].label == "initialize_gripper_at_home"
    assert plan.steps[1].label == "initial_gripper_open"
    labels = [step.label for step in plan.steps]
    assert labels.index("green:pick_contact") < labels.index("green:gripper_close")
    assert labels.index("green:pick_lift") < labels.index("green:T0_transit")
    assert labels.index("green:gripper_open") < labels.index("green:release_stability")
    assert labels[-1] == "return_home"
    assert simulate_offline_plan(plan, site)["passed"] is True


def test_unrotated_camera_home_rotates_only_in_place_at_safe_height() -> None:
    raw = complete_site_dict()
    raw["geometry"]["home_pose_m_rad"] = [
        -0.046166628952218576,
        -0.22000564307922596,
        0.447856,
        -0.04157768816703186,
        -3.1341811630889373,
        -0.00639402792769759,
    ]
    site = load_site(raw)
    plan = build_offline_plan(site, "basic", ["green"], pick_xy())
    moves = [step for step in plan.steps if step.kind is StepKind.MOVE]
    assert moves[0].label == "orient_for_work"
    assert moves[-2].label == "return_work_home"
    assert moves[-1].label == "return_home"
    for step in (moves[0], moves[-1]):
        assert step.start_pose is not None and step.end_pose is not None
        assert step.start_pose.as_tuple()[:3] == pytest.approx(step.end_pose.as_tuple()[:3])
        assert step.start_pose.z == pytest.approx(site.safe_high_z_m)


def test_sequence_maps_three_colors_to_p1_p2_p3() -> None:
    site = load_site(complete_site_dict())
    plan = build_offline_plan(site, "sequence", ["blue", "red", "yellow"], pick_xy())
    labels = [step.label for step in plan.steps]
    assert any("blue:P1_transit" == label for label in labels)
    assert any("red:P2_transit" == label for label in labels)
    assert any("yellow:P3_transit" == label for label in labels)
    assert sum(step.label.endswith(":release_stability") for step in plan.steps) == 3
    assert simulate_offline_plan(plan, site)["passed"] is True


def test_plan_geometry_is_safe_high_horizontal_and_pure_z_contact() -> None:
    site = load_site(complete_site_dict())
    plan = build_offline_plan(site, "sequence", ["red", "yellow", "blue"], pick_xy())
    for step in plan.steps:
        if step.kind is not StepKind.MOVE:
            continue
        assert step.start_pose is not None and step.end_pose is not None
        horizontal = np.hypot(
            step.end_pose.x - step.start_pose.x,
            step.end_pose.y - step.start_pose.y,
        )
        if horizontal > 1e-9:
            assert step.start_pose.z == pytest.approx(site.safe_high_z_m)
            assert step.end_pose.z == pytest.approx(site.safe_high_z_m)
        if step.profile == "contact":
            assert horizontal == pytest.approx(0.0)
            assert step.end_pose.z < step.start_pose.z


def test_tampered_low_horizontal_move_is_rejected() -> None:
    site = load_site(complete_site_dict())
    plan = build_offline_plan(site, "basic", ["green"], pick_xy())
    steps = list(plan.steps)
    index = next(i for i, step in enumerate(steps) if step.label == "green:T0_transit")
    original = steps[index]
    assert original.start_pose is not None and original.end_pose is not None
    start = replace(original.start_pose, z=0.2)
    end = replace(original.end_pose, z=0.2)
    steps[index] = replace(original, start_pose=start, end_pose=end)
    with pytest.raises(OffsiteError, match="路径不连续|安全高位"):
        validate_offline_plan(OfflinePlan(plan.task, plan.colors, tuple(steps)), site)


def test_tampered_upward_contact_move_is_rejected() -> None:
    site = load_site(complete_site_dict())
    plan = build_offline_plan(site, "basic", ["green"], pick_xy())
    steps = list(plan.steps)
    index = next(i for i, step in enumerate(steps) if step.label == "green:pick_contact")
    original = steps[index]
    assert original.start_pose is not None and original.end_pose is not None
    upward_end = replace(original.start_pose, z=original.start_pose.z + 0.01)
    steps[index] = replace(
        original,
        end_pose=upward_end,
    )
    if index + 1 < len(steps):
        for next_index in range(index + 1, len(steps)):
            next_step = steps[next_index]
            if next_step.kind is StepKind.MOVE:
                steps[next_index] = replace(next_step, start_pose=upward_end)
                break
    with pytest.raises(OffsiteError, match="向下"):
        validate_offline_plan(OfflinePlan(plan.task, plan.colors, tuple(steps)), site)


def test_missing_target_or_pick_xy_blocks_plan() -> None:
    raw = complete_site_dict()
    raw["geometry"]["target_centers_base_xy_m"]["T0"] = None
    site = load_site(raw)
    with pytest.raises(OffsiteError, match="T0"):
        build_offline_plan(site, "basic", ["green"], pick_xy())
    site = load_site(complete_site_dict())
    with pytest.raises(OffsiteError, match="缺少 green"):
        build_offline_plan(site, "basic", ["green"], {})


def replay_bundle(*, drift_m: float = 0.00005) -> ReplayBundle:
    colour = np.zeros((120, 160, 3), np.uint8)
    colour[20:100, 30:130] = (30, 120, 220)
    depth = np.full((120, 160), 0.62, np.float32)
    before = (0.1, -0.2, 0.4, -2.2, -2.1, 0.0)
    after = (before[0] + drift_m, *before[1:])
    return ReplayBundle(
        colour=colour,
        depth_m=depth,
        intrinsics=ReplayIntrinsics(642.2, 641.8, 80.0, 60.0),
        tcp_pose_before=before,
        tcp_pose_after=after,
        handeye_result_path="output/handeye/session_20260910/handeye_result.json",
        report={"valid": True, "source": "unit-test"},
    )


def test_replay_round_trip_uses_png_depth_npy_and_manifest(tmp_path: Path) -> None:
    path = save_replay_bundle(tmp_path / "bundle", replay_bundle())
    assert {item.name for item in path.iterdir()} == {"colour.png", "depth_m.npy", "manifest.json"}
    loaded = load_replay_bundle(path)
    assert np.array_equal(loaded.colour, replay_bundle().colour)
    assert np.array_equal(loaded.depth_m, replay_bundle().depth_m)
    assert loaded.tcp_pose_before == replay_bundle().tcp_pose_before
    with (path / "depth_m.npy").open("rb") as handle:
        assert np.array_equal(np.load(handle, allow_pickle=False), replay_bundle().depth_m)


def test_replay_rejects_tcp_drift_and_low_depth_coverage(tmp_path: Path) -> None:
    with pytest.raises(OffsiteError, match="TCP 漂移"):
        save_replay_bundle(tmp_path / "drift", replay_bundle(drift_m=0.001))
    bundle = replay_bundle()
    depth = np.zeros_like(bundle.depth_m)
    depth[:20] = 0.6
    with pytest.raises(OffsiteError, match="覆盖率"):
        save_replay_bundle(tmp_path / "sparse", replace(bundle, depth_m=depth))


def test_replay_integrity_hash_detects_tampering(tmp_path: Path) -> None:
    path = save_replay_bundle(tmp_path / "bundle", replay_bundle())
    depth = np.full((120, 160), 0.7, np.float32)
    with (path / "depth_m.npy").open("wb") as handle:
        np.save(handle, depth, allow_pickle=False)
    with pytest.raises(OffsiteError, match="完整性"):
        load_replay_bundle(path)


def test_replay_rejects_non_mapping_sha256_manifest(tmp_path: Path) -> None:
    path = save_replay_bundle(tmp_path / "bundle", replay_bundle())
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = "corrupted"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OffsiteError, match="manifest.sha256"):
        load_replay_bundle(path)


def test_geometry_report_uses_circle_corner_containment() -> None:
    report = geometry_report()
    assert report["nominal_per_side_margin_mm"]["T0"] == [15.0, 60.0]
    assert report['dimensions_mm']['P_inner_diameter']==80
    assert report['radial_corner_margin_mm']['P_inner']==pytest.approx(-13.150729)
    assert report['radial_corner_margin_mm']['P_outer']==pytest.approx(21.849271)
    assert "圆形" in report["risk"]
    assert "不要用人为偏置" in " ".join(report["strategy"])
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_release_stability_cannot_be_configured_below_three_seconds() -> None:
    raw = copy.deepcopy(complete_site_dict())
    raw["timing"]["release_settle_s"] = 2.9
    with pytest.raises(OffsiteError, match="至少 3 秒"):
        load_site(raw)
