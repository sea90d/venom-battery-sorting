#!/usr/bin/env python3
"""固定四槽电池分拣命令行入口；默认不连接真实机器人。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from battery_sorting.config import BatterySortingConfig, ConfigError, Pose
from battery_sorting.gripper import GripperBase, MockGripper, create_gripper
from battery_sorting.mission import ActionKind, Mission, MissionPlan, compute_plan_metrics
from battery_sorting.robot import (
    RUNTIME_STATE_STOPPED,
    RUNTIME_STATE_PLAYING,
    FakeRobot,
    RealUR5,
    RobotError,
    RobotBase,
    assert_pose_close,
    assert_pose_unchanged,
    require_snapshot_state,
)
from battery_sorting.task import BasicTask, Color, SequenceTask, Task, TaskError
from battery_sorting.vision import (
    Camera,
    ColorDetector,
    DetectionResult,
    HomographyModel,
    choose_consensus,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UR5 固定槽位彩色电池分拣")
    parser.add_argument("--config", type=Path, default=Path("config/example.json"))
    parser.add_argument("--task", choices=("basic", "sequence"), default="basic")
    parser.add_argument("--colors", nargs="+", default=["red"])
    parser.add_argument("--image", type=Path, help="使用离线图片，不打开相机")
    parser.add_argument("--vision-only", action="store_true", help="只识别四槽颜色，不要求示教位姿")
    parser.add_argument("--execute", action="store_true", help="申请真实机器人执行（多重门禁）")
    parser.add_argument("--export-simulation", type=Path, help="导出当前真实 Python 计划后退出，不连接机器人或夹爪")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.export_simulation and (args.execute or args.vision_only):
        parser.error("--export-simulation 不得与 --execute 或 --vision-only 同时使用")
    if args.vision_only and args.execute:
        parser.error("--vision-only 与 --execute 不能同时使用")
    if args.execute and args.image is not None:
        parser.error("--execute 禁止使用 --image；真机必须使用实时相机多帧结果")
    return args


def parse_task(task_name: str, raw_colors: list[str]) -> Task:
    try:
        colors = tuple(Color(value.lower()) for value in raw_colors)
    except ValueError as exc:
        raise TaskError("颜色只能是 red/blue/yellow/green") from exc
    if task_name == "basic":
        if len(colors) != 1:
            raise TaskError("basic 任务必须给 1 个 --colors")
        return BasicTask(colors[0])
    if len(colors) != 3:
        raise TaskError("sequence 任务必须给 3 个 --colors")
    return SequenceTask((colors[0], colors[1], colors[2]))


def detect(
    config: BatterySortingConfig, image_path: Path | None, *, strict_camera: bool = False
) -> DetectionResult:
    detector = ColorDetector(config.vision, config.camera.width, config.camera.height)
    if image_path is not None:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ConfigError(f"无法读取离线图片: {image_path}")
        return detector.detect(frame)
    with Camera(config.camera, strict_parameters=strict_camera) as camera:
        results = [detector.detect(camera.capture()) for _ in range(config.vision.sample_frames)]
    return choose_consensus(results, config.vision.required_consistent_frames)


def print_detection(result: DetectionResult) -> None:
    print("\n颜色识别（质心仅作槽位偏移诊断，不转换为机械臂坐标）:")
    for slot in result.slots:
        centroid = "None" if slot.centroid is None else f"({slot.centroid[0]:.1f},{slot.centroid[1]:.1f})"
        offset = "None" if slot.centroid_offset is None else f"({slot.centroid_offset[0]:+.1f},{slot.centroid_offset[1]:+.1f})"
        contour_area = (
            "None" if slot.contour_area_px is None else f"{slot.contour_area_px:.0f}px"
        )
        angle = (
            "None"
            if slot.long_axis_angle_deg is None
            else f"{slot.long_axis_angle_deg:.1f}deg"
        )
        status = "OK" if slot.valid else f"拒绝: {slot.reason}"
        print(
            f"  {slot.slot_name}: {slot.color or '-':6s} "
            f"coverage={slot.coverage:.3f} confidence={slot.confidence:.3f} "
            f"margin={slot.margin:.3f} centroid={centroid} offset={offset} "
            f"area={contour_area} long_axis={angle} [{status}]"
        )
    if not result.valid:
        print(f"  总体判定: 无效（{result.error}）")


def print_robot_xy(result: DetectionResult, mapping: HomographyModel) -> None:
    print(
        "\n图像生成的 UR 基坐标系抓取 XY "
        f"(标定RMSE={mapping.reprojection_rmse_m * 1000:.2f}mm, "
        f"最大误差={mapping.max_reprojection_error_m * 1000:.2f}mm):"
    )
    for color, centroid in sorted(result.color_to_centroid().items()):
        x, y = mapping.map_pixel_to_robot(*centroid)
        print(
            f"  {color:6s}: pixel=({centroid[0]:.1f},{centroid[1]:.1f}) "
            f"-> base_xy=({x:.6f},{y:.6f}) m"
        )


def print_plan(plan: MissionPlan, config: BatterySortingConfig) -> None:
    print(f"\n离线任务计划: {plan.task_description}")
    print(
        "  速度[m/s]: "
        f"transit={config.robot.transit_speed:.3f}, "
        f"approach={config.robot.approach_speed:.3f}, "
        f"contact={config.robot.contact_speed:.3f}; "
        f"accel={config.robot.move_acceleration:.3f}m/s², "
        f"stop={config.robot.stop_deceleration:.3f}m/s², "
        f"force_delta={config.robot.max_tcp_force_delta_n:.1f}N"
    )
    for index, action in enumerate(plan.actions, start=1):
        if action.kind is ActionKind.MOVE and action.waypoint is not None:
            values = ", ".join(f"{value:.6f}" for value in action.waypoint.pose.as_list())
            detail = f"profile={action.waypoint.profile} [{values}] m/rad"
        elif action.kind is ActionKind.DWELL:
            detail = f"{action.seconds:.2f}s"
        else:
            detail = ""
        print(f"  {index:02d}. {action.kind.value:12s} {action.label:30s} {detail}")
    home = config.robot.taught_poses.get("home")
    if home is not None:
        metrics = compute_plan_metrics(plan, home, config.robot)
        print(
            "\n名义估时（只含平移速度剖面与程序等待，不替代比赛视频计时）:\n"
            f"  平移距离 {metrics.total_translation_m:.3f} m; "
            f"运动 {metrics.nominal_motion_seconds:.1f} s; "
            f"等待 {metrics.dwell_seconds:.1f} s; "
            f"合计 {metrics.nominal_total_seconds:.1f} s"
        )


def confirmation_phrase(config: BatterySortingConfig, task: Task) -> str:
    return f"EXECUTE UR5 {config.robot.host} {task.description}"


def _home(config: BatterySortingConfig) -> Pose:
    home = config.robot.taught_poses.get("home")
    if home is None:
        raise ConfigError("robot.taught_poses.home 仍为 null")
    return home


def _safe_gripper_disconnect(gripper: GripperBase) -> Exception | None:
    """Best-effort cleanup that never hides the original mission error."""
    try:
        gripper.disconnect()
        return None
    except Exception as exc:  # pragma: no cover - hardware cleanup
        logging.exception("夹爪断开失败；请按夹爪说明书确认当前保持状态")
        return exc


def _safe_robot_disconnect(robot: RobotBase) -> Exception | None:
    """Best-effort cleanup that never hides the original mission error."""
    try:
        robot.disconnect()
        return None
    except Exception as exc:  # pragma: no cover - hardware cleanup
        logging.exception("机器人接口断开失败；请检查示教器状态")
        return exc


def _cleanup_resources(robot: RobotBase, gripper: GripperBase) -> None:
    errors = [
        error
        for error in (_safe_robot_disconnect(robot), _safe_gripper_disconnect(gripper))
        if error is not None
    ]
    # During exception unwinding, preserve the primary error. A cleanup failure on an
    # otherwise successful/early-return path must still make the command fail.
    if errors and sys.exc_info()[0] is None:
        raise RuntimeError(f"硬件清理失败: {errors[0]}") from errors[0]


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.execute:
        raise ConfigError("固定相机旧入口仅用于离线；末端 D455 请使用 python -m battery_sorting.closed_loop run")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = BatterySortingConfig.from_json(args.config)
    task: Task | None = None
    execution_gripper = None
    if args.vision_only:
        config.require_vision_valid()
    elif args.execute:
        task = parse_task(args.task, args.colors)
        execution_gripper = create_gripper(config.gripper)
        config.require_execution_valid(
            gripper_configured=execution_gripper.is_configured,
            gripper_is_mock=execution_gripper.is_mock,
            gripper_has_object_feedback=execution_gripper.has_object_feedback,
        )
    else:
        task = parse_task(args.task, args.colors)
        config.require_planning_valid()

    result = detect(config, args.image, strict_camera=args.execute)
    print_detection(result)
    if not result.valid:
        raise ConfigError("颜色识别不可靠，任务已停止")
    if args.vision_only:
        print("\nvision-only 完成；未规划路径、未连接机器人。")
        return 0

    assert task is not None
    home = _home(config)
    mapping = HomographyModel(config.vision.homography)
    print_robot_xy(result, mapping)
    if not args.execute:
        robot = FakeRobot(home)
        gripper = MockGripper(object_after_close=True)
        mission = Mission(
            config,
            robot,
            gripper,
            sleep_fn=lambda _seconds: None,
            mapping=mapping,
        )
        plan = mission.build_plan(task, result)
        print_plan(plan, config)
        if args.export_simulation:
            from battery_sorting.simulation_export import export_mission_plan, write_plan
            write_plan(export_mission_plan(plan, config.robot, task, label=args.config.name), args.export_simulation)
            print(f"仿真计划已导出: {args.export_simulation}；未连接机器人或夹爪")
            return 0
        try:
            robot.connect_readonly()
            robot.connect_control()
            gripper.connect()
            mission.execute(plan)
        finally:
            _cleanup_resources(robot, gripper)
        print(f"\ndry-run 完成：记录 {len(robot.records)} 个 moveL 路点；没有访问网络或真实硬件。")
        return 0

    assert execution_gripper is not None
    gripper = execution_gripper
    robot = RealUR5(config.robot)
    mission = Mission(config, robot, gripper, mapping=mapping)
    plan = mission.build_plan(task, result)
    print_plan(plan, config)
    try:
        gripper.connect()
        robot.connect_readonly()
        before = robot.snapshot()
        assert_pose_close(
            before.tcp_pose,
            home,
            config.robot.home_translation_tolerance,
            config.robot.home_rotation_tolerance,
            "人工确认前 home 检查",
        )
        phrase = confirmation_phrase(config, task)
        print("\n即将创建 RTDE 控制接口并执行上述路径。")
        try:
            answer = input(f"请输入完全一致的确认短语：\n{phrase}\n> ").strip()
        except EOFError:
            print("没有交互输入；未创建控制接口。")
            return 1
        if answer != phrase:
            print("确认短语不匹配；未创建控制接口。")
            return 1
        after = robot.snapshot()
        require_snapshot_state(after, {RUNTIME_STATE_STOPPED}, "人工确认后")
        assert_pose_unchanged(before.tcp_pose, after.tcp_pose)
        robot.connect_control()
        control_ready = robot.snapshot()
        require_snapshot_state(control_ready, {RUNTIME_STATE_PLAYING}, "控制接口创建后")
        assert_pose_unchanged(before.tcp_pose, control_ready.tcp_pose)
        assert_pose_close(
            control_ready.tcp_pose,
            home,
            config.robot.home_translation_tolerance,
            config.robot.home_rotation_tolerance,
            "控制接口创建后 home 检查",
        )
        if not gripper.initialize(recalibrate=False):
            raise RuntimeError("夹爪在 home 位姿初始化失败")
        after_gripper_init = robot.snapshot()
        require_snapshot_state(after_gripper_init, {RUNTIME_STATE_PLAYING}, "夹爪初始化后")
        assert_pose_close(
            after_gripper_init.tcp_pose,
            home,
            config.robot.home_translation_tolerance,
            config.robot.home_rotation_tolerance,
            "夹爪初始化后 home 检查",
        )
        mission.execute(plan)
        print("真实任务执行完成。")
        return 0
    finally:
        _cleanup_resources(robot, gripper)


def main() -> None:
    try:
        raise SystemExit(run())
    except (ConfigError, TaskError, RobotError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 1)


if __name__ == "__main__":
    main()
