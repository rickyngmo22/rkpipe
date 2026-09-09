# rknn_eval —— RK3588 板端模型精度/性能一键评测

一个二进制完成：**配置校验 → 板端推理导出 → 精度评测 → 论文风格报告（MD + LaTeX + CSV）→ CI 断言**。
支持 detect / pose / seg / obb 四任务，INT8 vs FP16 自动对比与量化损失计算。
配套：自定义数据集转换（convert_dataset.py）、远程评测驱动（run_eval.py，数据集在 PC）、
网页控制台（rknn_eval_web.py，零依赖）。框架设计见《远程评测与数据集框架设计.md》。

## 编译

```bash
cmake -B build
cmake --build build --target rknn_eval -j 4
# 产物: build/rknn_eval（评测脚本自动定位到源码树 tools/eval/）
```

## 四任务标准命令（可直接复制）

```bash
# detect（COCO val2017，INT8 vs FP16 对比 + 线程扫描 + CI 断言）
build/rknn_eval --task detect \
  --model model/yolo26n.rknn --compare fp16=model/yolo26n_fp16.rknn \
  --images test_images \\
  --ann coco_eval/annotations/instances_val2017.json \
  --label assets/labels/coco_80_labels_list.txt --obj-num 80 \
  --threads 4 --out-dir eval_runs/detect \
  --threads-sweep 1,6 --assert-ap50 40

# pose（必须用 person_keypoints 标注）
build/rknn_eval --task pose \
  --model model/yolo26n_pose.rknn --compare fp16=model/yolo26n_pose_fp16.rknn \
  --images test_images \\
  --ann coco_eval/annotations/person_keypoints_val2017.json \
  --label assets/labels/coco_80_labels_list.txt --obj-num 1 \
  --out-dir eval_runs/pose

# seg（mask mAP）
build/rknn_eval --task seg \
  --model model/yolo26n_seg.rknn --compare fp16=model/yolo26n_seg_fp16.rknn \
  --images test_images \\
  --ann coco_eval/annotations/instances_val2017.json \
  --label assets/labels/coco_80_labels_list.txt --obj-num 80 \
  --out-dir eval_runs/seg

# obb（DOTA v1.0 val：传原图+labelTxt 自动 1024/200 切片；或直接传切片目录 + --gt）
# PC 端数据整理/切片一键脚本: tools/eval/prepare_dota.py(.bat)；完整指南见 docs/benchmark_dota_obb.md
build/rknn_eval --task obb \
  --model model/yolo26n_obb.rknn --compare fp16=model/yolo26n_obb_fp16.rknn \
  --images coco_eval/dota/val_images --dota-labels coco_eval/dota/val_labelTxt \
  --label assets/labels/yolov8_obb_labels_list.txt --obj-num 15 \
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
| `--reuse-dump FILE` | 跳过推理直接评测已有 dump（调参/复看最快路径）；对照模型可用 `tag=dump:FILE` 评已拉回的 dump |
| `--vis-dir DIR` | 渲染主模型检测结果图（框/骨架/掩膜/旋转框 + 类别标签） |
| `--vis-sample N` | 可视化抽帧：每 N 帧渲 1 张（0=全渲） |
| `--preview` / `--preview-port N` | 推理时开板端 Web MJPEG 实时预览（dump 在叠加前写，不影响精度） |
| `--dump-only` | 只推理导出 dump+metrics，跳过评测/报告（远程模式用） |
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

四个任务的可视化内容：detect=类别色框+标签；pose=骨架+关键点
seg=类色掩膜叠加+轮廓；obb=旋转四边形框。批量图片同样适用（每帧出一张 `*_vis.jpg`）。
pose 任务按内容选帧：优先有人的图，单人/多人交替覆盖，无人帧仅兜底补齐。

## 自定义数据集（YOLO / VOC / COCO / labelme / yolo-obb）

```bash
# 1. 写数据集 spec（datasets/my.yaml，schema 见设计文档 §3.1）
#    detect: format=yolo|voc|coco；seg: format=labelme（类别表可省，自动汇总）；
#    obb: format=yolo-obb（小图直推）或 dota（大图切片）
# 2. 转换 + 校验 + GT 预览
python3 tools/eval/convert_dataset.py --spec datasets/my.yaml --vis 20
#    水平框/分割 -> my.coco.json；旋转框 -> my.gt.jsonl
#    + .check.json（漏标/非法框/类别分布）+ _preview/
# 3. 评测（ann/gt 传转换产物即可，类别映射全自动）
build/rknn_eval --task detect --model my.rknn --images /data/my/val/images \
  --ann datasets/my.coco.json --label model/my_labels_list.txt --obj-num N
```

eval_coco.py 类别映射自动判定：标注 ids 连续 1..N → 直映射；COCO 官方 80 类 → 内置空洞映射；
其他 → 报错（宁报错不错评）。

## 网页控制台（板端服务，PC 浏览器操作——推荐）

部署形态：**板端开启服务，PC 零环境**（Windows 直接用浏览器，不装任何东西）。

```bash
# 板端（一次性，或开机自启）
./tools/eval/eval_web_service.sh start        # 默认 :8081；PORT=9090 可改端口；stop/status/restart
# PC 浏览器打开（启动时也会打印）
#   http://<板IP>:8081
```

页面上直接完成一切：

1. **提交评测**（首页）：模型下拉（板上 `model/` 与已上传模型）或直接上传 `.rknn`；
   数据来源二选一：
   - **PC 文件夹同步（Windows 推荐）**：图片文件夹与标注文件**分开选**（目录随意，
     无需打包到一起）；**大数据集按「每批张数」分批增量上传**——提交时先传 2 批
     （1 批运行 + 1 批候选），板上跑完一批、页面自动补传下一批，**板上任意时刻
     最多 2 批**，不必一次性上传全部（如 COCO val 5000 张按 500/批只需常驻 ~1GB
     级别的两批）；标注选 COCO JSON（obb 选切片级 gt jsonl），类别数自动回填。
     小数据集设「每批张数=0」走一次性上传，任务结束即清缓存。
   - **板上路径**：数据已在板上时直填目录。
   模型下拉按任务过滤：对应关系登记在板上 `model/models.json`（文件名 → 任务，
   支持数组）；未登记的板上模型不出现在下拉里，PC 上传的模型不受限制。
   conf/threads/可视化/实时画面 → 提交后进队列（同板串行）。多模型对比：
   「对比模型」下拉选择（＋号可添加多个）即 **N 模型并行对比**——每实例线程数 =
   CPU 核数 ÷ 模型数，快照/结果图按模型同帧分列对比，报告含量化损失表。
   「每批张数」（默认 500）为分批消费模式：图片按批搬入工作目录、跑完即释放，
   板上任意时刻只装载一批（大数据集防涨内存/存储；注意分批会消费板上图片副本，
   重试需重新同步）。
2. **任务详情**（`/job/<id>`）：状态徽章 + 阶段 + 命令折叠；开「实时画面」的任务
   推理期间页面底部显示**抽帧快照流**（板端每 ~2s 抽 1 帧、页面每 3s 刷新一批，
   带检测叠加，不影响精度数字），**完成后保留为抽帧存档**继续可看；
   页面为 JS 局部刷新不整页闪刷，完成/失败自动展示报告；失败任务一键重试。
3. **队列与历史**（`/jobs`）：排队/运行中的任务可随时**终止**（整个进程组一起结束，
   状态标为 canceled，可重试）；历史任务可单个**删除**或一键**清空**（含产物目录，
   进行中的任务不受影响，均有确认框）；重启后自动还原全部历史——web 任务按 `web_job.json`
   恢复（中断标红可重试），CLI 产物（eval_runs/ 等）合成历史条目；
   `/compare` 跨任务精度/速度条形图对比。
4. **API**：`GET /api/job/<id>`（状态 JSON）、`POST /api/upload?name=`（zip/.rknn）、
   `POST /api/sync/manifest` + `POST /api/sync/file` + `GET /api/sync/done`
   （文件夹增量同步协议）。

模型也可以不传网页，直接拷到板上 `model/` 目录，下拉自动出现。

**存储安全（防爆板端存储）**：

- 同步/上传前做**空间预检**：所需空间超过板端剩余（保留 1GB 余量）直接拒绝并提示；
- 默认**用完即删**：任务结束（成功/失败/取消）自动清理该任务上传的数据缓存——失败重试需重传图片；
  提交页勾选「评测完成后保留上传的数据」可改为保留（二次评测增量只传差异，失败重试也建议勾选）；
- 评测标注为空/缺 `categories`（如 `{}` 占位文件）时给出明确报错，不再崩溃；
- 页面实时显示板端剩余空间；缓存根目录可用环境变量 `RK_EVAL_DATA` 指到大容量分区；
- 历史遗留缓存（无任务引用的 sync 残留）可手动清理：`rm -rf /userdata/rk_eval_data/*`。

自定义数据集图片名不限（`img_0012.jpg` 等非数字名均可）——评测器自动用标注内
`file_name` 反查 image_id；要求标注 JSON 覆盖同名文件。

## 远程评测 CLI（PC 端脚本化场景，备选）

无网页的自动化场景（CI、批量回归）才需要 PC 端环境：`run_eval.py` 从 PC 经
ssh/rsync 驱动板端（推图→板推理→拉 dump→PC 评测），链路细节见下；网页操作
用上面的板端服务即可，不涉及本节。

```bash
# 一次性：免密 ssh + 模型上板
ssh-copy-id user@<board-ip>
scp model/my.rknn user@<board-ip>:~/rkpipe/model/

# 一条命令：推图 → 板推理(dump-only) → 拉 dump → PC 评测 → 报告
python3 tools/eval/run_eval.py --spec datasets/my.yaml \
  --board user@<board-ip> \
  --model board:~/rkpipe/model/my.rknn \
  --compare fp16=board:~/rkpipe/model/my_fp16.rknn \
  --task detect --threads 4 --out-dir runs/my --preview
# spec.board_images 板上已有图片时可跳过推图；--keep-remote 保留板上工作目录

# 大数据集防涨满板端存储：分批推图（每批 500 张，批内推-跑-删，dump 自动合并；
# FPS 汇总含每批模型加载开销，口径略保守）
python3 tools/eval/run_eval.py --spec datasets/big.yaml ... --push-batch 500
```

链路健壮性（PC↔板，run_eval.py 内建）：

- ssh ControlMaster 连接复用：整个 job 一次握手，ssh/scp/rsync 全走同一条多路复用连接；
  保活探测（ServerAlive）+ 首连自动收 host key（`accept-new`）。
- rsync `--partial --timeout`：断点续传，瞬断后重试（自动 3 次）只补差异不重传。
- 整集推图只传图片扩展名（jpg/jpeg/png/bmp/tif/tiff），标注/labels 不占板端带宽与存储。
- `--remote-cwd`（默认 ~/rkpipe）：板端命令固定在该目录执行——ssh 非交互
  shell 的 cwd 是家目录，rknn_eval 的 label 等默认相对路径会失效。
- `--state-file PATH`：阶段状态机落盘（preparing→pushing(推图百分比/分批 i/n)→
  running(模型 i/n)→pulling→evaluating→done/error），供 Web 控制台轮询展示进度；
  失败时写入原因。Web 提交的远程任务自动启用。

多板脚本化评测：把 `boards.yaml`（模板见 tools/eval/）放工作目录供脚本参考，
用 `--board` 指定目标即可；**同一块板串行排队，不同板并行评测**（Web 端提交的
板端任务同板自动排队）。

## 历史对比

```bash
# CLI：跨目录汇总所有 summary 成对比表 + 量化损失
python3 tools/eval/compare_runs.py eval_runs/* web_runs/* --out compare.md
```

Web 控制台 `/compare` 页：按任务类型（bbox/segm/kpts/obb）分组，精度/速度 CSS 条形图，30s 自动刷新。

## 产物（out-dir/）

| 文件 | 内容 |
|---|---|
| `report.md` | 评测报告：主表 + 量化损失表 +（obb）逐类表 + 协议说明 |
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
# COCO val2017 标注（官方 S3，可用 hf-mirror 加速）放到 coco_eval/annotations/ 下：
#   instances_val2017.json          ← detect/seg GT
#   person_keypoints_val2017.json   ← pose GT
# DOTA v1.0 val（社区镜像）放到 coco_eval/dota/val_images + val_labelTxt
# 自定义数据集走上面的"自定义数据集"流程，不依赖 COCO/DOTA。
```

## 已知坑（工具已内置防护，人工直调时注意）

1. `--images` 路径必须含 `/`——输入源把纯文件名当单文件。
2. pose 标注必须含 keypoints 字段（`person_keypoints_*.json` 或任意同名内容的关键点
   COCO 标注，文件名不限；instances 标注会被内容校验拒绝）。
3. seg 掩膜二值阈值：rknn_eval 自动设为标准 mAP 口径 0.5（流水线默认 logit -0.5，可用 `RK_PIPE_SEG_MASK_THRESH` 覆盖）。
4. 进程中途被杀会产生 dump 尾行截断——评测脚本逐行容错跳过。
5. NPU governor 建议 performance 档，否则 FPS 波动大。
