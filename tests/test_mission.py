from __future__ import annotations

import math
from dataclasses import replace

import pytest

from battery_sorting.config import BatterySortingConfig, Pose
from battery_sorting.gripper import MockGripper
from battery_sorting.mission import (
    ActionKind,
    Mission,
    MissionAction,
    MissionError,
    MissionPlan,
    compute_plan_metrics,
)
from battery_sorting.robot import FakeRobot, RobotSafetyError, RobotSnapshot, Waypoint, rotation_error
from battery_sorting.task import BasicTask, Color, SequenceTask, TaskError
from battery_sorting.vision import ColorDetector
from main import confirmation_phrase, parse_args

from .conftest import synthetic_frame


def detection(config: BatterySortingConfig):
    return ColorDetector(config.vision, config.camera.width, config.camera.height).detect(
        synthetic_frame(config)
    )


def make_mission(config: BatterySortingConfig, gripper: MockGripper | None = None):
    home = config.robot.taught_poses["home"]
    assert home is not None
    robot = FakeRobot(home)
    grip = gripper or MockGripper(object_after_close=True)
    mission = Mission(config, robot, grip, sleep_fn=lambda _seconds: None)
    return mission, robot, grip


def test_basic_maps_red_image_centroid_to_dynamic_xy_and_t0(
    valid_config: BatterySortingConfig,
) -> None:
    mission, _, _ = make_mission(valid_config)
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    moves = {action.label: action.waypoint.pose for action in plan.actions if action.waypoint}
    pick = moves["red:pick_contact"]
    assert pick.x == pytest.approx(0.09475, abs=1e-6)
    assert pick.y == pytest.approx(0.09250, abs=1e-6)
    assert pick.z == pytest.approx(valid_config.robot.pick_pose_template.z)
    assert plan.dynamic_pick_poses == (pick,)
    assert moves["red:place_contact"] == valid_config.robot.taught_poses["T0"]


def test_sequence_maps_order_to_p1_p2_p3(valid_config: BatterySortingConfig) -> None:
    mission, _, _ = make_mission(valid_config)
    task = SequenceTask((Color.BLUE, Color.RED, Color.YELLOW))
    plan = mission.build_plan(task, detection(valid_config))
    moves = {action.label: action.waypoint.pose for action in plan.actions if action.waypoint}
    assert moves["blue:place_contact"] == valid_config.robot.taught_poses["P1"]
    assert moves["red:place_contact"] == valid_config.robot.taught_poses["P2"]
    assert moves["yellow:place_contact"] == valid_config.robot.taught_poses["P3"]
    assert len(plan.dynamic_pick_poses) == 3
    assert all(pose.z == pytest.approx(0.10) for pose in plan.dynamic_pick_poses)


def test_sequence_requires_three_distinct_colors() -> None:
    with pytest.raises(TaskError):
        SequenceTask((Color.RED, Color.RED, Color.BLUE))


def test_external_base_xy_does_not_require_fixed_camera_homography(valid_config):
    from battery_sorting.config import HomographyConfig
    config=replace(valid_config,vision=replace(valid_config.vision,homography=HomographyConfig()))
    mission,_,_=make_mission(config)
    plan=mission.build_plan(BasicTask(Color.RED),detection(config),base_xy_by_color={'red':(.09,.09)})
    assert plan.dynamic_pick_poses[0].x==.09
    assert mission.mapping is None


def test_horizontal_motion_only_at_transit(valid_config: BatterySortingConfig) -> None:
    mission, _, _ = make_mission(valid_config)
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    home = valid_config.robot.taught_poses["home"]
    assert home is not None
    previous = home
    for waypoint in plan.waypoints:
        current = waypoint.pose
        if math.hypot(current.x - previous.x, current.y - previous.y) > 1e-9:
            assert previous.z == pytest.approx(valid_config.robot.safe_z_transit)
            assert current.z == pytest.approx(valid_config.robot.safe_z_transit)
        previous = current


def test_scoring_dwells_are_after_lift_and_after_retract(valid_config: BatterySortingConfig) -> None:
    mission, _, _ = make_mission(valid_config)
    actions = mission.build_plan(BasicTask(Color.RED), detection(valid_config)).actions
    labels = [action.label for action in actions]
    assert labels.index("red:lift_approach") < labels.index("red:lifted_hold")
    assert labels.index("red:release_retreat") < labels.index("red:release_stability")
    lifted = next(action for action in actions if action.label == "red:lifted_hold")
    released = next(action for action in actions if action.label == "red:release_stability")
    assert lifted.seconds >= 2.0
    assert released.seconds >= 3.0


def test_fake_execution_ends_at_home(valid_config: BatterySortingConfig) -> None:
    mission, robot, gripper = make_mission(valid_config)
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    mission.execute(plan)
    assert robot.snapshot().tcp_pose == tuple(valid_config.robot.taught_poses["home"].as_list())
    assert gripper.actions[:2] == ["open", "close"]
    assert robot.stop_count == 0


def test_failed_object_feedback_aborts_and_stops(valid_config: BatterySortingConfig) -> None:
    gripper = MockGripper(object_after_close=False)
    mission, robot, _ = make_mission(valid_config, gripper)
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    with pytest.raises(MissionError, match="未抓到"):
        mission.execute(plan)
    assert robot.stop_count == 1
    assert gripper.actions[-1] == "stop"


def test_dynamic_phrase_contains_host_and_task(valid_config: BatterySortingConfig) -> None:
    task = BasicTask(Color.GREEN)
    phrase = confirmation_phrase(valid_config, task)
    assert valid_config.robot.host in phrase
    assert task.description in phrase


def test_rotvec_sign_discontinuity_is_handled() -> None:
    first = [math.pi, 0.0, 0.0]
    second = [-math.pi, 0.0, 0.0]
    assert rotation_error(first, second) == pytest.approx(0.0, abs=1e-9)


def test_first_action_opens_gripper(valid_config: BatterySortingConfig) -> None:
    mission, _, _ = make_mission(valid_config)
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    assert plan.actions[0].kind is ActionKind.OPEN


def test_execute_revalidates_untrusted_plan(valid_config: BatterySortingConfig) -> None:
    mission, robot, _ = make_mission(valid_config)
    unsafe = Pose(0.50, 0.50, 0.10, math.pi, 0.0, 0.0)
    plan = MissionPlan(
        "unsafe",
        (MissionAction(ActionKind.MOVE, "diagonal_below_transit", Waypoint("bad", unsafe)),),
    )
    with pytest.raises(RobotSafetyError):
        mission.execute(plan)
    assert robot.records == []


def test_contact_pose_cannot_claim_transit_speed(valid_config: BatterySortingConfig) -> None:
    mission, robot, _ = make_mission(valid_config)
    contact = valid_config.robot.taught_poses["T0"]
    assert contact is not None
    plan = MissionPlan(
        "tampered profile",
        (MissionAction(ActionKind.MOVE, "bad_profile", Waypoint("bad_profile", contact, "transit")),),
    )
    with pytest.raises(RobotSafetyError, match="派生位姿"):
        mission.execute(plan)
    assert robot.records == []


def test_transit_cannot_skip_approach_and_move_directly_to_contact(
    valid_config: BatterySortingConfig,
) -> None:
    mission, robot, _ = make_mission(valid_config)
    contact = valid_config.robot.taught_poses["T0"]
    assert contact is not None
    plan = MissionPlan(
        "skip approach",
        (
            MissionAction(
                ActionKind.MOVE,
                "slot1_transit",
                mission.path.transit(contact, "slot1_transit"),
            ),
            MissionAction(
                ActionKind.MOVE,
                "slot1_contact",
                mission.path.contact(contact, "slot1_contact"),
            ),
        ),
    )
    with pytest.raises(RobotSafetyError, match="transit->contact"):
        mission.execute(plan)
    assert robot.records == []


def test_workspace_point_not_derived_from_teaching_is_rejected(
    valid_config: BatterySortingConfig,
) -> None:
    mission, robot, _ = make_mission(valid_config)
    arbitrary = Pose(0.30, -0.10, valid_config.robot.safe_z_transit, math.pi, 0.0, 0.0)
    plan = MissionPlan(
        "arbitrary",
        (MissionAction(ActionKind.MOVE, "arbitrary", Waypoint("arbitrary", arbitrary, "transit")),),
    )
    with pytest.raises(RobotSafetyError, match="不是已示教点"):
        mission.execute(plan)
    assert robot.records == []


def test_segment_start_drift_is_detected(valid_config: BatterySortingConfig) -> None:
    home = valid_config.robot.taught_poses["home"]
    assert home is not None

    class DriftingRobot(FakeRobot):
        def snapshot(self) -> RobotSnapshot:
            state = super().snapshot()
            if self.records:
                shifted = list(state.tcp_pose)
                shifted[0] += 0.01
                return RobotSnapshot(
                    tuple(shifted), state.robot_mode, state.safety_mode, state.runtime_state,
                    state.emergency_stopped, state.protective_stopped,
                )
            return state

    robot = DriftingRobot(home)
    mission = Mission(
        valid_config, robot, MockGripper(object_after_close=True), sleep_fn=lambda _seconds: None
    )
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    with pytest.raises(RobotSafetyError, match="起点检查"):
        mission.execute(plan)
    assert robot.stop_count == 1


def test_object_drop_during_transit_stops_before_descent(
    valid_config: BatterySortingConfig,
) -> None:
    class DroppingGripper(MockGripper):
        def __init__(self) -> None:
            super().__init__(object_after_close=True)
            self._feedback = iter((True, True, False))

        def has_object(self) -> bool:
            if not self._closed:
                return False
            return next(self._feedback)

    gripper = DroppingGripper()
    mission, robot, _ = make_mission(valid_config, gripper)
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    with pytest.raises(MissionError, match="check_before_place"):
        mission.execute(plan)
    labels = [record.label for record in robot.records]
    assert "red:place_transit" in labels
    assert "red:place_approach" not in labels
    assert gripper.actions.count("open") == 1


def test_execute_rejects_offline_image() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--execute", "--image", "old-frame.png"])


def test_vision_only_does_not_parse_irrelevant_sequence_task(monkeypatch: pytest.MonkeyPatch) -> None:
    import main as cli

    monkeypatch.setattr(cli, "detect", lambda *_args, **_kwargs: cli.DetectionResult((), True))
    assert cli.run([
        "--config", "config/example.json", "--vision-only", "--task", "sequence"
    ]) == 0


def test_plan_metrics_cover_motion_and_scoring_dwells(
    valid_config: BatterySortingConfig,
) -> None:
    mission, _, _ = make_mission(valid_config)
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    home = valid_config.robot.taught_poses["home"]
    assert home is not None
    metrics = compute_plan_metrics(plan, home, valid_config.robot)
    assert metrics.total_translation_m > 0.0
    assert metrics.nominal_motion_seconds > 0.0
    assert metrics.dwell_seconds == pytest.approx(
        sum(
            action.seconds
            for action in plan.actions
            if action.kind is ActionKind.DWELL
        )
    )
    assert metrics.nominal_total_seconds == pytest.approx(
        metrics.nominal_motion_seconds + metrics.dwell_seconds
    )


def test_plan_metrics_uses_trapezoidal_profile(valid_config: BatterySortingConfig) -> None:
    home = valid_config.robot.taught_poses["home"]
    assert home is not None
    target = Pose(home.x + 0.04, home.y, home.z, home.rx, home.ry, home.rz)
    plan = MissionPlan(
        "metric",
        (
            MissionAction(
                ActionKind.MOVE,
                "metric_move",
                Waypoint("metric_move", target, "transit"),
            ),
            MissionAction(ActionKind.DWELL, "metric_wait", seconds=2.0),
        ),
    )
    metrics = compute_plan_metrics(plan, home, valid_config.robot)
    # d=0.04, v=0.05, a=0.1 -> d/v + v/a = 1.3 seconds.
    assert metrics.total_translation_m == pytest.approx(0.04)
    assert metrics.nominal_motion_seconds == pytest.approx(1.3)
    assert metrics.nominal_total_seconds == pytest.approx(3.3)


def test_dynamic_pick_xy_outside_dedicated_bounds_is_rejected(
    valid_config: BatterySortingConfig,
) -> None:
    shifted_points = tuple(
        (x + 0.5, y) for x, y in valid_config.vision.homography.robot_xy_points_m
    )
    config = replace(
        valid_config,
        vision=replace(
            valid_config.vision,
            homography=replace(
                valid_config.vision.homography,
                robot_xy_points_m=shifted_points,
            ),
        ),
    )
    mission, _, _ = make_mission(config)
    with pytest.raises(MissionError, match="超出 pick_xy_bounds"):
        mission.build_plan(BasicTask(Color.RED), detection(config))


def test_execute_rejects_mismatched_dynamic_pick_declaration(
    valid_config: BatterySortingConfig,
) -> None:
    mission, robot, _ = make_mission(valid_config)
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    with pytest.raises(MissionError, match="声明.*不一致"):
        mission.execute(replace(plan, dynamic_pick_poses=()))
    assert robot.records == []


def test_execute_rejects_dynamic_pick_z_tampering(
    valid_config: BatterySortingConfig,
) -> None:
    mission, robot, _ = make_mission(valid_config)
    plan = mission.build_plan(BasicTask(Color.RED), detection(valid_config))
    actions = list(plan.actions)
    for index, action in enumerate(actions):
        if action.label == "red:pick_contact":
            assert action.waypoint is not None
            bad_pose = action.waypoint.pose.with_z(action.waypoint.pose.z + 0.01)
            actions[index] = replace(
                action,
                waypoint=replace(action.waypoint, pose=bad_pose),
            )
            break
    else:  # pragma: no cover - plan invariant
        raise AssertionError("missing red:pick_contact")
    tampered = replace(
        plan,
        actions=tuple(actions),
        dynamic_pick_poses=(bad_pose,),
    )
    with pytest.raises(MissionError, match="Z 或工具姿态"):
        mission.execute(tampered)
    assert robot.records == []
