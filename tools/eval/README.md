# rknn_eval —— RK3588 板端模型精度/性能一键评测

一个二进制完成：**配置校验 → 板端推理导出 → 精度评测 → 论文风格报告（MD + LaTeX + CSV）→ CI 断言**。
支持 detect / pose / seg / obb 四任务，INT8 vs FP16 自动对比与量化损失计算。

## 编译

```bash
cmake -S /userdata/rk_pipe -B /userdata/rk_pipe/build
cmake --build /userdata/rk_pipe/build --target rknn_eval -j 4
# 产物: build/rknn_eval（评测脚本自动定位到源码树 tools/eval/）
```

## 四任务标准命令（可直接复制）

```bash
# detect（COCO val2017，INT8 vs FP16 对比 + 线程扫描 + CI 断言）
build/rknn_eval --task detect \
  --model model/yolo26n.rknn --compare fp16=model/yolo26n_fp16.rknn \
  --images /userdata/rk_pipe/test_images \
  --ann coco_eval/annotations/instances_val2017.json \
  --label model/coco_80_labels_list.txt --obj-num 80 \
  --threads 4 --out-dir eval_runs/detect \
  --threads-sweep 1,6 --assert-ap50 40

# pose（必须用 person_keypoints 标注）
build/rknn_eval --task pose \
  --model model/yolo26n_pose.rknn --compare fp16=model/yolo26n_pose_fp16.rknn \
  --images /userdata/rk_pipe/test_images \
  --ann coco_eval/annotations/person_keypoints_val2017.json \
  --label model/coco_80_labels_list.txt --obj-num 1 \
  --out-dir eval_runs/pose

# seg（mask mAP）
build/rknn_eval --task seg \
  --model model/yolo26n_seg.rknn --compare fp16=model/yolo26n_seg_fp16.rknn \
  --images /userdata/rk_pipe/test_images \
  --ann coco_eval/annotations/instances_val2017.json \
  --label model/coco_80_labels_list.txt --obj-num 80 \
  --out-dir eval_runs/seg

# obb（DOTA v1.0 val：传原图+labelTxt 自动 1024/200 切片；或直接传切片目录 + --gt）
build/rknn_eval --task obb \
  --model model/yolo26n_obb.rknn --compare fp16=model/yolo26n_obb_fp16.rknn \
  --images coco_eval/dota/val_images --dota-labels coco_eval/dota/val_labelTxt \
  --label model/yolov8_obb_labels_list.txt --obj-num 15 \
  --out-dir eval_runs/obb
# 已切片时（跳过切片步骤）:
#   --images coco_eval/dota/patches --gt coco_eval/dota/patches_gt.jsonl
```

## 常用选项

| 选项 | 说明 |
|---|---|
| `--compare tag=FILE` | 对照模型，可多次；报告自动计算量化损失与速度比 |
| `--threads-sweep 1,6` | 线程扫描，只测 FPS 不重评精度（`/dev/null` dump，不占盘） |
| `--conf-sweep 0.001,0.25` | conf 扫描：同一 dump 离线重评，选部署阈值 |
| `--assert-ap X` / `--assert-ap50 X` | CI 断言：主指标 ×100 低于阈值 → 退出码 1 |
| `--reuse-dump FILE` | 跳过推理直接评测已有 dump（调参/复看最快路径） |
| `--vis-dir DIR` | 渲染主模型检测结果图（框/骨架/掩膜/旋转框 + 类别标签） |
| `--config FILE` | `key=value` 配置文件（`#` 注释），CLI 参数优先 |
| `--dry-run` | 校验配置与脚本后直接退出 |
| `--scripts-dir DIR` | 指定评测脚本目录（默认 exe 相对定位源码树 tools/eval，或 `$RK_EVAL_SCRIPTS`） |

### 单图开图出图

输入传单张图片路径即可，推理后 `--vis-dir` 直接输出检测结果图：

```bash
build/rknn_eval --task detect --model model/yolo26n.rknn \
  --images /path/to/one.jpg --conf 0.25 \
  --ann coco_eval/annotations/instances_val2017.json \
  --out-dir /tmp/vis --vis-dir /tmp/vis/out
# -> /tmp/vis/out/one_vis.jpg（带类别色框 + 标签）
```

四个任务的可视化内容：detect=类别色框+标签；pose=骨架+关键点；
seg=类色掩膜叠加+轮廓；obb=旋转四边形框。批量图片同样适用（每帧出一张 `*_vis.jpg`）。

## 产物（out-dir/）

| 文件 | 内容 |
|---|---|
| `report.md` | 论文风格报告：主表 + 量化损失表 +（obb）逐类表 + 协议说明 + LaTeX 表 |
| `summary_all.json` | 全部数值（含线程/conf 扫描），机器可读 |
| `dump_<tag>.jsonl` | 板端原始检测导出（每帧一行） |
| `summary_<tag>.json` | 单模型评测摘要（含逐类 AP） |
| `metrics_<tag>.json` | 流水线指标（FPS / 前处理 / 推理耗时） |
| `per_class.csv` | 逐类 AP / AP50（COCO 系任务） |

## 评测口径（论文式）

- 推理：conf=0.001 全量导出（评测端可再过滤），tracking/overlay/告警全关，pipeline 多线程。
- detect/seg：COCOeval bbox/segm，maxDets=100；pose：COCOeval keypoints，maxDets=20，仅 person。
- obb：DOTA v1.0 val 官方 1024/200 切片，旋转 IoU（凸多边形），IoU 0.5:0.95 十档 101 点插值 AP，
  difficult 忽略，GT 相交占比 ≥0.5 保留并裁剪到切片坐标。
- imgIds 限定为实际推理图集合；指标 ×100 保留 1 位小数。
- INT8 vs FP16 为**同口径差值**，直接衡量量化损失（与官方 PT 数值存在 letterbox/后处理口径差）。

## 数据与标注准备（一次性）

```bash
# COCO val2017 标注（官方 S3 在板上被限速，用 hf-mirror）
python3 -c "..."  # 见仓库 coco_eval/annotations/ 下已就绪文件
#   instances_val2017.json          ← detect/seg GT
#   person_keypoints_val2017.json   ← pose GT
# DOTA v1.0 val（Last-Bullet/DOTAv1.0 镜像，8 并发 ~25min）
#   coco_eval/dota/val_images + val_labelTxt
```

## 已知坑（工具已内置防护，人工直调时注意）

1. `--images` 路径必须含 `/`——输入源把纯文件名当单文件。
2. pose 必须用 `person_keypoints_*.json`（instances 文件不含 keypoints）。
3. seg 掩膜二值阈值默认 0.5（`RK_PIPE_SEG_MASK_THRESH=0`，rknn_eval 已自动设置? 未设时默认 logit -0.5）。
4. 进程中途被杀会产生 dump 尾行截断——评测脚本逐行容错跳过。
5. NPU governor 建议 performance 档，否则 FPS 波动大。
