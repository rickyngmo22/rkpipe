# OBB（旋转框）评测指南 — DOTA v1.0

在 RK3588 板端评测旋转框检测模型（YOLO26-OBB / YOLOv8-OBB）的精度与速度，
协议与 DOTA 官方对齐：原图 **1024×1024 / gap 200 切片** → 板端逐切片推理 →
**旋转 IoU 匹配**（IoU 0.5:0.95 十档，101 点插值 AP，COCO 风格）+ AP50。

支持两种数据形态：

| 形态 | 说明 | 适合 |
|---|---|---|
| 原图 + labelTxt | `rknn_eval` 自动切片（`--dota-labels`） | 首次评测 / 换切片参数 |
| 已切片 patches + gt jsonl | 跳过切片直接推理（`--gt`） | 反复调参 / 网页控制台 |

---

## 1. 准备 DOTA v1.0 val 数据（一次性）

DOTA v1.0 val 共 458 张航拍图 + 同名 labelTxt（每行：
`x1 y1 x2 y2 x3 y3 x4 y4 类别 difficult`，**顶点顺序任意但需闭合**，
类别 ∈ DOTA 15 类）。数据自行下载（官方 Google Drive 或社区镜像），
放在仓库根 `coco_eval/dota/`：

```bash
mkdir -p coco_eval/dota
# 解压后应得到:
#   coco_eval/dota/val_images/   458 张 *.png
#   coco_eval/dota/val_labelTxt/ 458 个 *.txt（与图片同名）
```

> 类别表固定 15 类（顺序与模型 label 文件一致）：
> plane, ship, storage-tank, baseball-diamond, tennis-court,
> basketball-court, ground-track-field, harbor, bridge,
> large-vehicle, small-vehicle, helicopter, roundabout,
> soccer-ball-field, swimming-pool
> 仓库已带对应 label 文件：`assets/labels/yolov8_obb_labels_list.txt`

### 1.5 PC 端一键预处理（Windows 推荐，网页评测前先做这步）

数据集压缩包不用整体上传——先在 PC 上切片转换，上传量能省一半以上
（458 张原图 png 约 3.3GB，切片后 jpg 仅约 1.5GB）：

```bat
:: 项目 tools/eval/prepare_dota.bat（Windows 双击/命令行均可）
:: 把 DOTA 压缩包(.zip)或已解压目录拖到窗口 → 自动:
::   1) 解压并整理出 val_images + val_labelTxt
::   2) 本机有 python+cv2 时自动 1024/200 切片 → patches + patches_gt.jsonl
::   3) 打印网页上传指引
prepare_dota.bat DOTA-val.zip
:: 只取前 50 张先试跑(小样验证):
prepare_dota.bat DOTA-val.zip --limit 50
```

产物 `dota_ready/` 目录结构即网页可直接使用的形式。PC 没装 python 也没关系：
脚本会整理好原图目录，改用网页「上传数据集 zip」（板端自动切片）。

---

## 2. 命令行评测（推荐先用它跑通）

### 2.1 原图自动切片（首次）

```bash
# 458 张原图 → 自动 1024/200 切片 → 推理 → 旋转 IoU 评测 → 报告
./build/rknn_eval --task obb \
  --model model/yolo26n_obb.rknn \
  --images coco_eval/dota/val_images \
  --dota-labels coco_eval/dota/val_labelTxt \
  --label assets/labels/yolov8_obb_labels_list.txt --obj-num 15 \
  --conf 0.001 --threads 4 --out-dir eval_runs/obb
```

切片产物在 `eval_runs/obb/patches/`（约 5300 张）与 `patches_gt.jsonl`，
一次切好后续可复用（见 2.2）。

### 2.2 已切片复用（调参/对比最快路径）

```bash
# 首次跑完 2.1 后, 切片结果已落盘; 之后直接对切片评测:
./build/rknn_eval --task obb \
  --model model/yolo26n_obb.rknn \
  --images eval_runs/obb/patches \
  --gt eval_runs/obb/patches_gt.jsonl \
  --label assets/labels/yolov8_obb_labels_list.txt --obj-num 15 \
  --conf 0.001 --threads 4 --out-dir eval_runs/obb_rerun
```

### 2.3 INT8 vs FP16 对比 / conf 扫描

```bash
./build/rknn_eval --task obb \
  --model model/yolo26n_obb.rknn --compare fp16=model/yolo26n_obb_fp16.rknn \
  --images eval_runs/obb/patches --gt eval_runs/obb/patches_gt.jsonl \
  --label assets/labels/yolov8_obb_labels_list.txt --obj-num 15 \
  --conf-sweep 0.001,0.25 --threads 4 --out-dir eval_runs/obb_cmp
# report.md: 主表 + 量化损失表 + 逐类 AP 表; summary_all.json 机器可读
```

---

## 3. 网页控制台评测

评测需要**切片后**的数据（`patches/` 切片图 + `patches_gt.jsonl` 单文件标注）。
两种方式二选一：

- **PC 文件夹同步（推荐）**：先用 1.5 的 `prepare_dota.bat/.py` 在 PC 生成
  `dota_ready/`，图片目录选 `patches/`，标注文件选 `patches_gt.jsonl`，
  类别数自动回填 15；
- **板上路径**：切片目录已拷到板上时直接填路径（最省流量）。

原始 DOTA（val_images + labelTxt）不要直接走网页——458 个 txt 标注无法经
网页单文件通道上传，请先在 PC 端预处理成切片形式。

## 4. 结果解读

- `report.md` Table 1：mAP（IoU 0.5:0.95 十档）、AP50、AP75、逐类 AP、FPS；
- 口径：GT 多边形与预测旋转框（`x,y,w,h,angle_rad`）做凸多边形相交 IoU；
  difficult 目标按忽略处理（命中不罚 FP）；切片时保留与切片相交 ≥50% 的目标并裁剪；
- AP50 与 DOTA 官方 VOC AP50 口径接近；mAP 十档比官方更严（官方只报 AP50），
  同口径对比请都以本工具同版本为准（INT8 vs FP16 为同口径差值，直接衡量量化损失）。

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| `--dota-labels` 指向的 txt 里有 8 个以上的数 | DOTA 允许任意顶点顺序，但类别后若还有顶点会被跳过；确保是标准 8 坐标 + 类别 + difficult |
| 类别对不上 / 大量漏检 | label 文件顺序必须与模型训练类别顺序一致（DOTA_CLASSES 顺序） |
| 切片慢 | 一次性切片后走 `--gt` 复用，别每次重切 |
| 想换切片尺寸 | `split_dota.py --subsize 640 --gap 0` 手动切后走 `--gt`（小目标场景常用 640） |
| 板上没 DOTA | 数据需自行下载；或从已准备 DOTA 的机器整目录拷贝 `val_images + val_labelTxt`（~2GB） |
