# 验证证据索引

时间标识沿用项目输出目录；正文使用北京时间，JSON 内 utc 字段为 UTC。文件从原项目原样复制。

| 编号 | 包内文件 | 原始项目路径 |
|---|---|---|
| E01 | `E01/handeye_result.json` | `output/handeye/session_20260910/handeye_result.json` |
| E01 | `E01/session.json` | `output/handeye/session_20260910/session.json` |
| E01 | `E01/images/sample_026_corners.png` | `output/handeye/session_20260910/images/sample_026_corners.png` |
| E02 | `E02/audit.json` | `output/smooth_t0/20261011_221746/audit.json` |
| E02 | `E02/verification/colour.png` | `output/smooth_t0/20261011_221746/verification/colour.png` |
| E02 | `E02/verification/manifest.json` | `output/smooth_t0/20261011_221746/verification/manifest.json` |
| E03 | `E03/events.json` | `output/competition_console/20261012_172952/events.json` |
| E03 | `E03/summary.json` | `output/competition_console/20261012_172952/summary.json` |
| E03 | `E03/target_view_1.png` | `output/competition_console/20261012_172952/target_view_1.png` |
| E04 | `E04/summary.json` | `output/fixed_sequence/20261012_190249_0f229fcf/summary.json` |
| E04 | `E04/events.json` | `output/fixed_sequence/20261012_190249_0f229fcf/events.json` |
| E05 | `E05/summary.json` | `output/fixed_sequence/20261012_190530_23df6fc1/summary.json` |
| E05 | `E05/events.json` | `output/fixed_sequence/20261012_190530_23df6fc1/events.json` |
| E06 | `E06/summary.json` | `output/competition_console/20261012_190934/summary.json` |
| E06 | `E06/events.json` | `output/competition_console/20261012_190934/events.json` |
| E07 | `E07/summary.json` | `output/competition_console/20261012_191432/summary.json` |
| E07 | `E07/events.json` | `output/competition_console/20261012_191432/events.json` |
| E08 | `E08/summary.json` | `output/competition_console/20261012_191801/summary.json` |
| E08 | `E08/events.json` | `output/competition_console/20261012_191801/events.json` |
| E09 | `E09/events.json` | `output/competition_console/20261012_191407/events.json` |

E10：`pytest_32.txt` 为原项目的 32 项相关测试记录；`fixed_plan_offline.json` 为原项目离线计划输出。

`snapshot_pytest_32.txt` 与 `snapshot_fixed_plan.json` 为打包代码副本的复现验证；只运行离线计划与模拟测试，未连接硬件。

完成样本不代表整体成功率。E09 为夹持反馈异常中断；E04–E08 均未进行视觉摆放验收。

图片来源补充：`latex/figures/battery_view.png` 对应 `output/latest_analysis_20261012_v2/annotated.png`。其他图片对应上述文件。
