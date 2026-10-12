# Python 计划驱动的 UR5 仿真播放器

网页不再规划抓放路线。Python 导出动作和六维路点，网页按原顺序播放。JSON 文件只在本地浏览器读取；不会运行上传的 Python 或连接硬件。

2026-10-11 夹爪实体碰撞仿真：默认加载 `plans/tool-body-collision.json`，包含准备、计时夹取及复检，回放 Python 完整六维逆解的关节采样。法兰至 TCP 全长 246 mm 作为直径 150 mm 的实体圆柱，机器人连杆采用直径 100 mm 的胶囊包络，夹爪粗度为其 1.5 倍。全长参与检查，只允许共同法兰安装面正常连接。关节帧间使用保守运动包络递归细分，保证所导出的关节插值在该模型内至少有 10 mm 净空。旧 Home 被此模型判为碰撞，新路径的采样最小净空约 27 mm。

名义模型和历史场景不是实机碰撞验收；仿真计算不连接硬件。“新提速方案”按钮也同步了经过检查的关节轨迹，只显示比赛计时动作。其他未附带关节采样的历史及示例计划仍采用位置逆解 `ikPosition`，工具另行显示朝向，不能用来验证碰撞。新轨迹与尺寸说明见 [实体碰撞检查](../docs/TOOL_BODY_COLLISION_20261011.md)。

## 打开和导入

在项目根目录运行 `python -m http.server 8765 --bind 127.0.0.1`，打开 <http://127.0.0.1:8765/web_sim/>。用“导入计划 JSON”选择文件，或展开“粘贴 JSON 计划”。修改 Python 后重新导出、导入即可。

默认加载 `plans/accelerated-red-t0.json`：直接调用当前 `tools/run_smooth_t0.py` 的 `execute_cycle`，在假机器人、假夹爪和虚拟时钟上执行 `ten-second` 提速方案。使用 0.5 m/s、1.0 m/s²、抓取上方 15 cm 搬运；最后 10 mm 保持 0.04 m/s、0.1 m/s²。夹爪闭合反馈确认后立即提升，取消提升后的 2.1 秒原地等待；放置确认后从接触点沿一段直线直接回 home。到达 home 时立即返回夹取运行时间，3.1 秒释放稳定计时继续与返回动作重叠。

2026-10-10 起点修正：默认动画从电池区外侧向 Home 开始，先原地转为朝下观察，再夹取；返回区外高位后原地恢复侧向。工具长轴在等待开始时水平。该动作模型为 12.186 秒，尚未计入实机开始后的新鲜识别与通信；原先 9.049 秒采用朝下起点，已经不适用。历史场景读取冻结的 `config/closed_loop_scene_20260929.json`，动作读取当前 `config/closed_loop.json`。

```powershell
.\.venv\Scripts\python.exe tools/export_smooth_t0_simulation.py
.\.venv\Scripts\python.exe tools/simulate_tool_body_collision.py
```

电池 CAD 朝向采用窄把手轴，而不是接近方形的外轮廓最长边；后者会因尺寸估计交换长短边，使红色错误旋转 90°。把手轴采用模 180° 的统一表示，保留测量中的小角度差。

“载入历史红色 T0 运行”保留 `plans/recorded-red-t0.json`：2026-09-15 21:29 成功真机运行的红色电池 → T0 → home，16 个动作、10 个运动端点。48.24 秒为日志起点到复检帧采集的记录区间，包含约 10.13 秒准备及最终采集时间。原始日志是高位搬运、0.3 m/s、0.1 m/s²，不能作为新提速方案的实测结果。

四色初始 XY/方向和 T0 轮廓从同次 RGB-D 记录恢复，桌面采用实际纸面 Base Z。动作端点、阶段时长和验收结果来自 audit；没有连续 TCP 或关节采样，段内轨迹、连续下降的子段时间及释放时点为重建。电池高度/外形采用 CAD 示意，释放后动画位置按刚体变换推算，页面显示的 5.14 mm 中心误差是原日志视觉实测。

重新导出同次记录（仅离线读取文件，不连接设备）：

```powershell
.\.venv\Scripts\python.exe -m battery_sorting.recorded_simulation --recording output/smooth_t0/20260915_212913 --config config/closed_loop.json --out web_sim/plans/recorded-red-t0.json
```

页面“载入历史红色 T0 运行”可随时重新加载。另一个按钮仍保留 `plans/recorded-green-grasp.json`：来自历史报告链及原 `grasp_lift.build_plan/execute_plan`，在假设备上记录调用，含 14 段下降、夹持和 14 段提升，最后仍持物，不含放置区路线。

## 当前抓取程序导出

```powershell
.\.venv\Scripts\python.exe -m battery_sorting.grasp_lift plan --export-simulation web_sim/plans/recorded-green-grasp.json
```

随后点网页“载入现场绿色抓取计划”。这是原 `plan` 子命令，只读取报告、构造计划并在假设备上记录成功分支，未进入真机 `execute`。夹爪在开始前已初始化是该执行函数的前提。

## 完整 Mission 程序导出

在原离线命令添加 `--export-simulation`：

```powershell
.\.venv\Scripts\python.exe main.py --config config/your_complete_site.json --image output/your_scene.png --task sequence --colors blue red yellow --export-simulation output/simulation/mission.json
```

上例文件名是占位符，须换成完整现场配置和图像。图像检测、映射及 `Mission.build_plan` 照常运行；导出后立即返回，尚未连接机器人或夹爪。不加 `--image` 会调用原相机检测。不能同时使用 `--execute` 或 `--vision-only`。

手眼 RGB-D 入口也可直接接入，导出的就是刚生成的同一个计划：

```python
from battery_sorting.simulation_export import export_mission_plan, write_plan

plan = mission.build_plan(task, detection, base_xy_by_color=base_xy_by_color)
write_plan(export_mission_plan(plan, config.robot, task), "output/simulation/mission.json")
```

适配器直接遍历 `plan.actions`，保留顺序、路点和 DWELL，速度取自同一个 RobotConfig。MissionPlan 不包含 main.py 的连接及前置初始化流程。

## OfflinePlan 导出

```python
from battery_sorting.offsite import build_offline_plan
from battery_sorting.simulation_export import export_offline_plan, write_plan

plan = build_offline_plan(site, "sequence", ["blue", "red", "yellow"], actual_pick_xy)
bundle = export_offline_plan(plan, site,
    speeds={"transit": 0.05, "approach": 0.02, "contact": 0.005},
    acceleration=0.1)
write_plan(bundle, "output/simulation/offline.json")
```

OfflinePlan 本身没有速度字段，必须显式提供用于估时的速度，上例为演示值。CLI 等价入口：

```powershell
.\.venv\Scripts\python.exe -m battery_sorting.simulation_export --site config/competition_template.json --picks output/actual_pick_xy.json --task sequence --colors blue red yellow --transit-speed 0.05 --approach-speed 0.02 --contact-speed 0.005 --acceleration 0.1 --out output/simulation/offline.json
```

`actual_pick_xy.json` 是颜色到 Base 米制 `[X,Y]` 的映射。当前现场模板的放置 Z、目标中心和工作区仍为空，完整计划会拒绝生成，不会自动补入演示坐标。

## 示例和协议

`tools/generate_simulation_examples.py` 在内存中使用明确标识的演示参数，调用原 `offsite.build_offline_plan` 生成 4 份基础任务和 24 份序列任务，不写回现场配置。

```powershell
.\.venv\Scripts\python.exe tools/generate_simulation_examples.py
node web_sim/scene-geometry.test.mjs
node web_sim/plan-player.test.mjs
```

协议：`schema=ur5-simulation`、`version=1`、`frame=ur_base`、m/rad。每步有六维 `start_pose/end_pose`、`duration_s`；MOVE 有速度/加速度/速度档。动作支持 move、open、close、dwell、initialize、check_object、visual_accept。物块有初始中心、四元数 `[x,y,z,w]`、尺寸；夹持动作显式引用 object_id，不依赖中文或英文标签解析。

未知动作（包括未适配的 moveJ/moveC）、错误单位、缺失字段、路径不连续和错误持物顺序会在导入时拒绝，保留上一份有效计划。导入标签作为纯文本显示。规划器源码 SHA-256 用于追踪版本，不代表现场验证。

## 仿真范围

TCP 端点及等待时间原样来自 Python。平移采用逐段停止的梯形/三角速度曲线，姿态为最短路径四元数 SLERP。IO 无实测时长时按瞬时事件处理；纯旋转时长采用导出器标注的角速度估计，名义时长不是实机计时。

物块保持抓取时相对 TCP 的完整刚体变换；支持任意跳转和倒拖；释放后冻结当前位姿，不吸附目标中心。默认电池中心为桌面 Base Z=0 上方 40 mm，可由导出函数的 object_centers 覆盖。这是场景建模假设，不改变抓取 TCP。

UR5 连杆为简化 DH/位置 IK 示意，工具使用导入旋转向量。尚未接入真实法兰到 TCP 偏置、控制器关节解或碰撞模型；不模拟重力、夹空、滑落和急停。此版本用于流程及 TCP 路径检查，不能据此认定真机可达、无碰撞或抓取成功。
