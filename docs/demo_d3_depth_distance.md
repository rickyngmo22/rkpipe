# Demo D3：检测 + 单目测距（Detect + Depth 数值化）

> 状态：已实现并板端验证（2026-08-29）。方案背景见 [compound_demo_ideas.md](compound_demo_ideas.md) D3。
> 形态：主任务 detect + 辅助任务 depth（Y5 多任务组合），每个检测框标注估计距离（米），
> 近距目标红框高亮并触发 `near_distance` 告警（webhook + 现场抓拍）。

## 演示效果

- 整帧深度 JET 伪彩叠加（原有能力）。
- 每个检测框左上角黑底黄字标注 `12.3m`；距离 ≤ `depth_dist_near_m` 的目标红框 + 红字。
- 触发近距时向 webhook POST `near_distance` 事件，并在 `alert_snapshot_dir` 保存带完整叠加的现场抓图。
- 告警抓拍/周期快照均取自叠加绘制之后，抓图自带框/伪彩/距离文字。

## 快速开始

```bash
# yaml 方式（已配好 D3 参数）
./build/console_detector -c configs/run_yolo26_detect_depth.yaml

# 命令行方式
./build/console_detector -m model/yolo26n.rknn --aux-model model/yolo26n_depth.rknn --aux-task depth \
    -i test_videos/baseline17.mp4 -t detect -p --depth-dist-text --depth-dist-near 5.0
```

## 配置项

| yaml 键 | CLI | 默认 | 说明 |
|---|---|---|---|
| `depth_dist_text` | `--depth-dist-text` | 0 | 检测框标注估计距离（米）。旧环境变量 `RK_PIPE_DEPTH_DIST_TEXT=1` 仍兼容 |
| `depth_dist_near_m` | `--depth-dist-near <m>` | 0 | 近距阈值（米），红框高亮 + near_distance 告警；0=只标注不告警 |
| `depth_dist_scale` | `--depth-dist-scale <f>` | 1.0 | 距离标定系数 = 实测米数 / 模型输出米数，按现场标定调整 |

依赖：`aux_model_path` + `aux_task: "depth"`。未配置 depth 辅助时启动自检会打印
`[warn-conflict][conflict-depth-dist]` 提示。近距告警复用 `alert_enabled / alert_webhook_url /
alert_interval_s / alert_snapshot_dir`，未配 webhook 时仅红框高亮、不发事件。

## near_distance 事件 JSON

```json
{"ts":1787972566643,"event":"near_distance","count":6,
 "targets":[{"class":0,"distance_m":8.24},{"class":2,"distance_m":13.74}, ...]}
```

- `class` 为类别 id（COCO 序），`distance_m` 为标定后估计距离。
- 去抖：与计数告警共用 `alert_interval_s` 窗口（两类事件合计每窗口最多一条）。
- 跨进程去重 key 为 `(task, "near_dist")`，多路并行同看一路时只报一次（需配 `alert_dedup_interval_s`）。

## 实现与调参

- 距离 = 框内深度**中值**（抽样 ≤4k 像素，抗个别噪点/遮挡），反算公式与归一化范围见
  `include/utils/depth_distance.h`（纯函数，`rk_pipe_unit_tests` 覆盖）。
- 深度图是米制输出（`yolov26_depth.cc`），`depth_lo/hi` 为每帧归一化范围；
  归一化上限默认 20m（`RK_PIPE_DEPTH_MAX` 可改），**更远的目标都钳位显示为 20.0m**——近距告警不受影响。
- `depth_dist_scale` 标定：拿卷尺/激光测距仪量一个静止目标的实际距离 D，画面读数 R，scale = D/R。
- 单目深度的绝对精度有限，演示口径写"估计距离"；横向对比（谁近谁远）与趋势变化是可靠的。

## 已知边界

- `mode: sequential` 路径未接距离叠加/近距告警（演示与生产用 pipeline 模式）。
- 近距判定基于单帧深度，闪烁目标可能瞬间越过阈值——由 `alert_interval_s` 去抖兜底。
- 距离文字按检测框左上角排布，密集小目标时文字可能互相重叠。
