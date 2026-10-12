from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from battery_sorting.config import BatterySortingConfig, ConfigError
from battery_sorting.gripper import create_gripper

from .conftest import parse_config


def test_example_is_valid_for_vision_only() -> None:
    config = BatterySortingConfig.from_json("config/example.json")
    assert config.validate_vision() == []
    assert any("null" in error for error in config.validate_planning())


def test_missing_gripper_defaults_to_not_configured(config_dict: dict, tmp_path: Path) -> None:
    config_dict.pop("gripper", None)
    config = parse_config(config_dict, tmp_path / "missing_gripper.json")
    assert config.gripper.driver == "not_configured"


def test_mock_gripper_rejects_real_fields(config_dict: dict, tmp_path: Path) -> None:
    config_dict["gripper"] = {"driver": "mock", "port": "COM9"}
    with pytest.raises(ConfigError, match="不得包含真实夹爪字段"):
        parse_config(config_dict, tmp_path / "mock_with_port.json")


def test_unknown_nested_key_rejected(config_dict: dict, tmp_path: Path) -> None:
    config_dict["vision"]["typo"] = 1
    with pytest.raises(ConfigError, match="未知字段"):
        parse_config(config_dict, tmp_path / "bad.json")


def test_string_false_is_not_boolean(config_dict: dict, tmp_path: Path) -> None:
    config_dict["dry_run"] = "false"
    with pytest.raises(ConfigError, match="true/false"):
        parse_config(config_dict, tmp_path / "bad_bool.json")


def test_slot_names_must_be_explicit_and_unique(config_dict: dict, tmp_path: Path) -> None:
    config_dict["vision"]["rois"][1]["slot_name"] = "slot1"
    config = parse_config(config_dict, tmp_path / "duplicate_slot.json")
    assert any("slot_name" in error for error in config.validate_vision())


def test_null_pose_fails_planning(config_dict: dict, tmp_path: Path) -> None:
    config_dict["robot"]["taught_poses"]["T0"] = None
    config = parse_config(config_dict, tmp_path / "null_pose.json")
    with pytest.raises(ConfigError, match="T0.*null"):
        config.require_planning_valid()


def test_near_duplicate_taught_points_rejected(config_dict: dict, tmp_path: Path) -> None:
    config_dict["robot"]["taught_poses"]["P2"] = list(
        config_dict["robot"]["taught_poses"]["P1"]
    )
    config = parse_config(config_dict, tmp_path / "duplicate_pose.json")
    assert any("P1/P2" in error for error in config.validate_planning())


def test_oversized_even_morphology_kernel_rejected(config_dict: dict, tmp_path: Path) -> None:
    config_dict["vision"]["morph_kernel_size"] = 32
    config = parse_config(config_dict, tmp_path / "kernel.json")
    assert any("1..31" in error for error in config.validate_vision())


def test_real_execution_is_refused_without_real_gripper(valid_config: BatterySortingConfig) -> None:
    gripper = create_gripper("mock")
    errors = valid_config.validate_execution(
        gripper_configured=gripper.is_configured,
        gripper_is_mock=gripper.is_mock,
        gripper_has_object_feedback=gripper.has_object_feedback,
    )
    assert any("真实夹爪" in error for error in errors)
    assert any("dry_run" in error for error in errors)


def test_pose_requires_six_finite_numbers(config_dict: dict, tmp_path: Path) -> None:
    config_dict["robot"]["taught_poses"]["home"] = [0, 0, 0]
    with pytest.raises(ConfigError, match="6 个数"):
        parse_config(config_dict, tmp_path / "pose.json")


def test_workspace_bounds_are_required(config_dict: dict, tmp_path: Path) -> None:
    config_dict["robot"]["workspace_bounds"] = None
    config = parse_config(config_dict, tmp_path / "workspace.json")
    assert any("workspace_bounds" in error for error in config.validate_planning())


def test_stop_distance_is_bounded(valid_config: BatterySortingConfig) -> None:
    unsafe_robot = replace(
        valid_config.robot,
        transit_speed=0.10,
        stop_deceleration=0.20,
        max_stop_distance_m=0.005,
    )
    unsafe = replace(valid_config, robot=unsafe_robot)
    assert any("制动距离" in error for error in unsafe.validate_planning())


def test_execution_requires_vision_calibrated(valid_config: BatterySortingConfig) -> None:
    errors = valid_config.validate_execution(
        gripper_configured=True,
        gripper_is_mock=False,
        gripper_has_object_feedback=True,
    )
    assert any("vision.calibrated" in error for error in errors)


def test_execution_requires_real_multiframe_majority(valid_config: BatterySortingConfig) -> None:
    one_frame = replace(
        valid_config,
        vision=replace(
            valid_config.vision,
            calibrated=True,
            sample_frames=1,
            required_consistent_frames=1,
        ),
    )
    errors = one_frame.validate_execution(
        gripper_configured=True,
        gripper_is_mock=False,
        gripper_has_object_feedback=True,
    )
    assert any("sample_frames >= 3" in error for error in errors)

    tied_vote = replace(
        valid_config,
        vision=replace(
            valid_config.vision,
            calibrated=True,
            sample_frames=4,
            required_consistent_frames=2,
        ),
    )
    errors = tied_vote.validate_execution(
        gripper_configured=True,
        gripper_is_mock=False,
        gripper_has_object_feedback=True,
    )
    assert any("严格多数" in error for error in errors)


def test_geometry_calibration_is_optional_outside_real_execution(
    valid_config: BatterySortingConfig,
) -> None:
    assert valid_config.validate_vision() == []
    assert all(
        roi.expected_component_area_px is None
        and roi.expected_long_axis_angle_deg is None
        for roi in valid_config.vision.rois
    )


def test_real_execution_requires_geometry_calibration(
    valid_config: BatterySortingConfig,
) -> None:
    errors = valid_config.validate_execution(
        gripper_configured=True,
        gripper_is_mock=False,
        gripper_has_object_feedback=True,
    )
    assert sum("expected_component_area_px" in error for error in errors) == 4
    assert sum("expected_long_axis_angle_deg" in error for error in errors) == 4


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("expected_component_area_px", -1.0, "expected_component_area_px"),
        ("expected_component_area_px", 999999.0, "超过 ROI 面积"),
        ("expected_long_axis_angle_deg", 90.0, "expected_long_axis_angle_deg"),
        ("area_tolerance_fraction", 0.0, "area_tolerance_fraction"),
        ("area_tolerance_fraction", 0.51, "area_tolerance_fraction"),
        ("max_angle_error_deg", 15.1, "max_angle_error_deg"),
    ),
)
def test_geometry_thresholds_are_strictly_validated(
    config_dict: dict,
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    config_dict["vision"]["rois"][0][field] = value
    config = parse_config(config_dict, tmp_path / f"bad_{field}.json")
    assert any(message in error for error in config.validate_vision())


def test_planning_requires_calibrated_homography(
    config_dict: dict, tmp_path: Path
) -> None:
    config_dict["vision"]["homography"]["calibrated"] = False
    config = parse_config(config_dict, tmp_path / "uncalibrated_mapping.json")
    assert any("homography.calibrated=false" in error for error in config.validate_planning())


def test_homography_config_rejects_duplicate_and_out_of_frame_points(
    config_dict: dict, tmp_path: Path
) -> None:
    config_dict["vision"]["homography"]["pixel_points"] = [
        [100.0, 100.0],
        [100.0, 100.0],
        [100.0, 380.0],
        [5000.0, 380.0],
    ]
    config = parse_config(config_dict, tmp_path / "bad_mapping_points.json")
    errors = config.validate_vision()
    assert any("重复点" in error for error in errors)
    assert any("超出相机画面" in error for error in errors)


def test_pick_xy_bounds_must_be_inside_workspace(
    config_dict: dict, tmp_path: Path
) -> None:
    config_dict["robot"]["pick_xy_bounds"]["x_max"] = 0.60
    config = parse_config(config_dict, tmp_path / "bad_pick_bounds.json")
    assert any("必须完全位于 workspace_bounds" in error for error in config.validate_planning())
