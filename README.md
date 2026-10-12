# venom-battery-sorting

UR5 末端 RGB-D 感知与彩色电池搬运项目。包含第一题红色电池 T0 抓放、第二题黄→P1 / 绿→P2 / 蓝→P3 连续执行，以及本地图像控制台。

## 系统组成

- UR5 固定基座机械臂，通过 UR RTDE 控制与读取位姿。
- Intel RealSense D455 末端深度相机，用于颜色、把手和目标定位。
- AG105145 电动夹爪，通过 Modbus RTU 控制并读取夹持状态。
- Python 3.12 / Windows；OpenCV、NumPy、pyrealsense2、ur_rtde、pyserial。

## 环境与离线复现

在仓库根目录运行：
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-rgbd.txt -r requirements-gripper.txt
.\.venv\Scripts\python.exe -m battery_sorting.fixed_sequence plan
.\.venv\Scripts\python.exe -m pytest tests/test_fixed_sequence.py tests/test_competition_console.py tests/test_sequence_smooth_motion.py -q
```

`plan` 离线打印示教位置、执行顺序和时间预算。以上三个测试文件共 32 项测试通过，记录见 `evidence/pytest_32.txt` 与 `evidence/snapshot_pytest_32.txt`。其他测试使用的历史采集数据按各测试文件的路径配置。

## 现场入口

- 第一题：`启动第一题丝滑抓取.bat`，实现位于 `tools/run_smooth_t0.py`。
- 第二题：`启动第二题图像控制台.bat`，本地地址 `http://127.0.0.1:8768/?task=sequence`。
- 第二题固定执行：`启动第二题三块放置.bat` 或 `run_task2_fixed.bat`。
- 设备预检：`python -m battery_sorting.fixed_sequence preflight`。

第二题当前使用固定示教布局，执行顺序为黄→P1、绿→P2、蓝→P3。恢复基座、TCP、图纸和物体初始位置后运行。默认跳过软件全轨迹预计算及碰撞检查；`--check-trajectory` 可启用检查。启动核对 TCP 与空载状态，执行读取到位和夹持反馈，异常触发停止；UR 控制器原生安全限制生效。

现场配置使用机器人 `192.168.1.10`、夹爪 `COM11 / 115200 / slave 1`，TCP 偏移 246 mm。部署到其他设备时修改 `config/` 中的对应配置。

## 目录

| 路径 | 内容 |
|---|---|
| `battery_sorting/` | 感知、手眼坐标映射、运动、夹爪与任务执行 |
| `tools/` | 第一题入口、现场调试及仿真工具 |
| `config/` | TCP、手眼路径、示教位姿与运动参数 |
| `web_console/` | 图像控制台前端 |
| `web_sim/` | 仿真前端及示例计划 |
| `tests/` | 模拟测试与历史场景测试 |
| `evidence/` | 标定结果、实机日志、图像与验证记录 |
| `output/handeye/session_20260910/handeye_result.json` | 当前配置使用的手眼外参 |
| `output/competition_console/20261012_172952/events.json` | 固定示教点的来源记录 |

当前任务入口以以上脚本为准；根目录 `main.py` 为早期实现。

## 实测记录

手眼标定采用 27 组样本，变换链平移一致性 RMSE 为 2.974 mm。第一题完成红色电池 T0 抓放与图像复检，复检中心偏差 8.225 mm。
第二题五组完整流程记录平均 38.250 s，最好 37.844 s。计时覆盖展开、三块电池抓放与折回，运行中不进行视觉放置复检。离线模型在每次夹爪动作 0.6 s 的假设下给出约 28.700 s 预算。

## 第三方组件

使用 UR RTDE、Intel RealSense SDK、OpenCV、NumPy、pyserial 与 pytest 等组件。各组件许可与说明见其原项目。开发过程使用 Codex 与 OpenCode / DeepSeek 辅助流程。
