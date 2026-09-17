# rkpipe — RK3588 边缘端多任务视觉流水线

rkpipe 是一套面向 Rockchip RK3588 的实时视频理解流水线：多路视频输入 → RGA 零拷贝预处理 → NPU 推理 → 后处理/跟踪 → Web 预览 / 推流 / 结构化事件输出。单进程内支持检测、姿态、OBB、实例/语义分割、深度估计、OCR、人脸与多阶级联任务，并提供常驻守护进程与多路任务控制台。

仓库包含功能模块源码、稳定 C ABI 契约头、板端模型与评测工具。**流水线调度核心**（线程编排、任务队列、内存池、零拷贝推理引擎等）以预编译静态库 `prebuilt/aarch64/librkpipe_core.a` 随仓库直接分发，与契约头同版本成对发布，许可见 [LEGAL/RKPIPE-CORE-EULA.md](LEGAL/RKPIPE-CORE-EULA.md)；同目录 `sha256sums.txt` 可校验一致性。

## 任务能力

主任务一次跑一个（`--task` / YAML `task`），可选挂一路辅助任务（`aux_task`）；二级能力由专用配置键驱动。

| task | 说明 |
|---|---|
| `detect` | YOLO26 / YOLOv8 / YOLOv5 目标检测 |
| `pose` | YOLOv8-Pose / YOLO26-Pose 人体关键点检测（多人建议开启 tracking 锁定每人颜色） |
| `obb` | 旋转框检测 |
| `seg` | 实例分割 |
| `sem` | 语义分割（`model/yolo26n_sem.rknn`；19 类 label 需按数据集自备） |
| `depth` | 单目深度估计 |
| `ocr_det` | PPOCRv4 文本检测（别名 `text_det`） |
| `rtmpose` | 两阶段姿态：一级人体检测 → 框 crop → RTMPose 回归（17 关键点 + 姿态时序缓冲） |
| `composite_cls` | 两阶段级联：一级检测框 crop → 二级分类（MobileNetV2 top-1，别名 `detect_cls`） |
| `retinaface` | 人脸检测 + 5 点 landmark（别名 `face`） |

| 二级 / 组合能力 | 配置键 | 说明 |
|---|---|---|
| 文本识别 | `ocr_rec_model_path` | PPOCRv4 识别，接在 `ocr_det` 之后（CTC 解码，字典 `model/ppocr_keys_v1.txt`） |
| 动作识别 | `action_model_path` | 姿态时序动作识别，接在 `pose` 之后（`action_window_t` / `action_interval` / `action_norm`） |
| 同路组合（Y5） | `aux_model_path` / `aux_task` | 主任务 + `depth` / `pose` / `seg` / `sem` / `ocr_det` 辅助叠加 |
| 3D 线框（D3） | `detect3d_wireframe` | `detect` + `aux_task: depth` 时启用：框底接地带深度中值 → 几何反推 → 8 角点投影画 12 条棱；近距门限 `detect3d_min_depth_m`（默认 8.0） |

> 0.3.1 起旧的 `detect3d` 任务（KITTI 单目 3D 头）已整体下线，3D 线框改由上面的 `detect` + `aux_task: depth` 组合提供。

## 仓库结构

```
├── include/rkpipe/          契约层:稳定 C ABI(rkpipe.h)+ 跨层共享数据结构
├── include/{core,io,model,postprocess,detection,config,daemon,llm,utils}/
├── src/                     模块层实现(与头文件一一对应;daemon/ 常驻守护、llm/ 事件复检)
├── apps/console_detector/   板端控制台应用:仅用 C ABI 驱动完整流水线(兼作 ABI 验收)
├── examples/configs/        示例配置(最小可跑:detect/pose/obb/seg/depth,开箱即用)
├── configs/
│   ├── smoke/               各任务板端冒烟配置(rtmpose/ocr/composite_cls/detect+depth)
│   ├── streams/             daemon / 多路编排的 streams.json 样例
│   ├── run_yolo26_detect_depth3d.yaml   D3 3D 线框示例
│   └── llm.yaml.sample      LLM 复检配置样例(api_key 走环境变量)
├── prebuilt/aarch64/        预编译核心库 librkpipe_core.a + sha256sums.txt(随仓库分发)
├── model/                   板端 rknn 模型与 label(来源与许可见 NOTICE)
├── assets/labels/           评测/示例用 label
├── 3rdparty/                第三方头与库(rknn、curl;遵循各自许可)
├── tools/
│   ├── eval/                板端精度评测(rknn_eval + 网页评测控制台)
│   ├── convert/             ONNX→RKNN 转换与量化脚本(split6/split10/fp16/hybrid)
│   ├── conversion/          OCR 识别头转换脚本
│   └── diagnose_detect3d.js D3 线框质量诊断(框稳定性/几何自洽性/溢出 vs 距离)
├── tests/test_unit.cpp      单元测试(无硬件依赖;CI 模式剔除硬件用例)
├── docs/                    架构/构建/运行/评测/事件载荷/任务演示文档
├── scripts/multistream.py   多路部署编排器(一路一进程,NPU core 轮转/看护重启)
└── .github/workflows/ci.yml CI:纯逻辑单测(gcc + clang)+ 完整构建(缺库报错路径)
```

## 快速开始

### 1. 预编译核心库

核心库随仓库分发，clone 后无需额外下载，开箱即可构建：

```bash
cd prebuilt/aarch64 && sha256sum -c sha256sums.txt && cd ../..
```

### 2. 安装依赖（板端 RK3588，Debian/Ubuntu）

```bash
sudo apt install cmake g++ python3 ffmpeg \
    libopencv-dev libturbojpeg0-dev libcurl4-openssl-dev \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
```

RGA 与 rockchip-mpp 多数板厂镜像已自带，缺失时安装 `librga-dev`、`librockchip-mpp-dev`。自编 OpenCV 用 `cmake -B build -DOPENCV_ROOT=<prefix>`（或同名环境变量）替代系统包。完整环境说明见 [docs/build.md](docs/build.md)。

### 3. 构建

```bash
cmake -B build && cmake --build build -j $(nproc)

# 逐个目标构建（按需）
cmake --build build --target console_detector     # 板端控制台应用
cmake --build build --target rk_pipe_daemon       # 常驻守护 + 多路控制台
cmake --build build --target rknn_eval            # 精度/性能评测
cmake --build build --target rk_pipe_unit_tests   # 单元测试
```

无硬件环境（PC / CI）跑纯逻辑单测：

```bash
cmake -B build-ci -DRK_PIPE_CI_BUILD=ON && cmake --build build-ci && ctest --test-dir build-ci
```

### 4. 准备测试素材

测试视频不入库。两个来源二选一：

**来源 A：下载实测用视频（推荐）**

从网盘下载 `test_videos/` 压缩包，解压到仓库根的 `video/` 目录：

> 下载链接：https://pan.quark.cn/s/280ef6b0bc83?pwd=7P6U （提取码：7P6U）
>
> 内含 5 个测试视频（1280×720 / 1920×1080），README 精度与速度表格即基于这些视频实测。

```bash
mkdir -p video && cd video
unzip /path/to/test_videos.zip
```

**来源 B：ffmpeg 合成（无网盘/仅验证跑通）**

```bash
mkdir -p video
ffmpeg -y -f lavfi -i testsrc2=duration=60:size=1280x720:rate=30 video/demo.mp4
```

### 5. 运行

```bash
./build/console_detector examples/configs/detect_video.yaml
# 示例配置默认开启 Web 预览:浏览器打开 http://<板子IP>:8080 查看实时画面
```

从构建到多路部署的逐项验收清单（全部命令已在 RK3588 板端实测）见 [docs/run_guide.md](docs/run_guide.md)。

### 6. 常驻守护与多路控制台（rk_pipe_daemon）

`rk_pipe_daemon` 是本仓推荐的部署形态：fork 监督 N 个 `console_detector` 子进程，自带 REST API 与网页控制台。

```bash
./build/rk_pipe_daemon --streams-file configs/streams/daemon.sample.json \
  --api-port 8099 --watch 2 --restart
# 浏览器打开 http://<板子IP>:8099/ 进入任务控制台
```

- **监督与恢复**：子进程崩溃按指数退避（2s→60s，10 分钟窗口最多 5 次）自动拉起；`--watch <s>` 轮询配置热切换
- **控制台页**：任务卡片 + 各路内嵌 MJPEG 预览 + 板载负载（CPU/内存/温度/NPU 三核）+ 网页端新增/修改/重启/停止任务（表单 `extra_yaml` 可透传 rtmpose 两阶段等二级字段）
- **REST API**：`/api/health`、`/api/tasks`、`/api/summary`（聚合板载负载 + 全部任务）、`/api/tasks/<id>/{log,restart,status.json}`、`/api/reload`、`/api/events`(+SSE)、`/metrics`（Prometheus）、`/timeline`（事件时间线页）
- **事件闭环 + LLM 复检**：子进程 `alert_webhook_url` 指向 `http://127.0.0.1:<api-port>/internal/event` 即接入 → 事件 JSONL 落盘 → 可选 VLM 复检（OpenAI 兼容，`RK_PIPE_LLM_BASE_URL`/`RK_PIPE_LLM_MODEL`/`RK_PIPE_LLM_API_KEY` 走环境变量）→ `--forward-url` 转发业务 webhook
- **事件规则**：绊线穿越/入侵/滞留/离岗/聚集/遗留/跌倒，扁平 `event_*` 键或命名多规则 `event_rules` 列表
- **动态任务持久化**：REST 创建/修改的任务写入 `<仓库根>/daemon_runs/dynamic_tasks.json`，daemon 重启自动恢复（`--dump-dir` 或 `RK_PIPE_DUMP_DIR` 可改运行目录）

只要多路预览、不需要控制台时，也可用轻量编排器（一路一进程、NPU core 轮转）：

```bash
python3 scripts/multistream.py run --config cam_a.yaml cam_b.yaml cam_c.yaml --cores auto
```

## 模型精度评测(rknn_eval)

`rknn_eval` 用一个二进制完成板端评测:**配置校验 → 板端推理导出 → 精度评测(mAP) → 论文风格报告(MD + LaTeX + CSV) → CI 断言**,支持 detect / pose / seg / obb 四任务,可做 INT8 vs FP16 量化损失对比与线程扫描。

先做一次 dry-run(不需要数据集,只校验配置与脚本就位):

```bash
mkdir -p data/coco && echo '{}' > data/coco/annotations.json
./build/rknn_eval --task detect --model model/yolo26n.rknn \
  --images data/coco/val2017 --ann data/coco/annotations.json \
  --label assets/labels/coco_80_labels_list.txt --obj-num 80 --dry-run
```

真实评测(COCO val2017,数据集自行准备放在仓库根的 `data/coco/` 与 `coco_eval/` 下):

```bash
# INT8 主模型 + FP16 对照(自动计算量化损失),4 线程,AP50 低于 40 时退出码 1(CI 断言)
./build/rknn_eval --task detect \
  --model model/yolo26n.rknn --compare fp16=model/yolo26n_fp16.rknn \
  --images data/coco/val2017 \
  --ann coco_eval/annotations/instances_val2017.json \
  --label assets/labels/coco_80_labels_list.txt --obj-num 80 \
  --threads 4 --out-dir eval_runs/detect --assert-ap50 40
# 产物在 eval_runs/detect/:report.md(主表+量化损失表)、summary_all.json、per_class.csv 等
```

**网页控制台（亮点）**：板端起一个服务，PC 浏览器直接操作——无需任何 PC 端环境：

```bash
./tools/eval/eval_web_service.sh start     # 板端启动，默认 :8081
# PC 浏览器打开 http://<板IP>:8081
```

网页上完成一切：数据集上传（支持 PC 文件夹增量同步，二次评测只传差异图片）→ 模型下拉/上传 →
提交评测 → 实时日志与抽帧快照 → 自动出 mAP 报告（MD/CSV/逐类表）→ 历史任务对比。

pose / seg / obb 的标准命令、YOLO 四版本(yolo26n/v8n/11n/v5s)横向对比、自定义数据集转换、
DOTA 评测、`--conf-sweep`/`--vis-dir`(单图可视化)等进阶用法见 [tools/eval/README.md](tools/eval/README.md)。

## 板端性能与精度（RK3588 实测）

以下数据在 RK3588 板端实测（COCO val2017 5000 张，`thread_count: 9`，`conf=0.001`）。
2026-09-15 起模型全部换用新量化布局：**yolo26 detect/obb 为 split6 拆分布局（每尺度 box/cls 独立张量）、
seg 为 split10（box/cls/mask 拆分 + proto）、pose 为融合布局**。旧 sigmoid 融合版已删除，
转换脚本见 `tools/convert/`（`export_onnx_split6.py` → `convert_yolo26_split_int8.py` 等）。
拆分布局的动机与量化论证：one2one 检测头 box/cls 共享量化 scale 会被 cls 极值撑爆导致 AP 塌缩，
拆分后纯 INT8 即可正常工作（detect AP 33.0→35.8，obb mAP50 63.3→66.8，seg mask AP 27.0→28.5）。

**推理速度（pipeline 9 线程，视频流 1280×720@60fps）**

| 模型 | 任务 | 测试视频 | 分辨率 | FPS | 推理耗时/帧 |
|---|---|---|---|---|---|
| yolov8n | detect | test.mp4 | 1280×720@60 | 177.7 | 50.1ms |
| yolov5s | detect | test.mp4 | 1280×720@60 | 153.5 | 64.2ms |
| yolo11n | detect | test.mp4 | 1280×720@60 | 144.7 | 60.6ms |
| yolo26n | detect | test.mp4 | 1280×720@60 | 127.6 | 77.3ms |
| yolo26n_pose | pose | baseline9.mp4 | 1280×640@30 | 125.3 | 70.6ms |
| yolo11n_pose | pose | baseline9.mp4 | 1280×640@30 | 114.6 | 76.6ms |
| yolov8_obb | obb | baseline11.mp4 | 1280×720@24 | 173.1 | 44.8ms |
| yolo26n_seg | seg | baseline17_mid30s.mp4 | 1920×1080@30 | 39.0 | 107.3ms |

**INT8 vs FP16 推理速度对比**（yolo26 系列，9 线程）

| 模型 | 量化 | FPS | 推理耗时/帧 |
|---|---|---|---|
| yolo26n | INT8 | 127.6 | 77.3ms |
| yolo26n | FP16 | 58.6 | 218.4ms |
| yolo26n_obb | INT8 | 112.0 | 55.0ms |
| yolo26n_obb_fp16 | FP16 | 62.8 | 99.3ms |
| yolo26n_pose | INT8 | 125.2 | 71.6ms |
| yolo26n_pose_fp16 | FP16 | 58.1 | 111.5ms |
| yolo26n_seg | INT8 | 39.0 | 107.3ms |
| yolo26n_seg_fp16 | FP16 | 34.7 | 435.3ms |

> INT8 量化模型推理速度约为 FP16 的 2 倍以上；精度损失用 `--compare` 参数量化。

**detect 精度（COCO val2017, COCOeval bbox）**

| 模型 | AP | AP50 | AP75 | APs | APm | APl | AR | FPS |
|---|---|---|---|---|---|---|---|---|
| yolo26n | 35.8 | 50.8 | 38.8 | 15.4 | 39.6 | 53.9 | 53.6 | 129.3 |
| yolo26s | 41.1 | 57.1 | 44.9 | 22.0 | 45.7 | 58.4 | 60.5 | 73.5* |
| yolo26m | 43.8 | 59.7 | 47.8 | 27.0 | 49.4 | 60.1 | 65.1 | 31.4 |
| yolov8n | 34.0 | 48.5 | 37.0 | 14.3 | 37.7 | 50.1 | 44.5 | 170.8 |
| yolo11n | 35.9 | 50.8 | 38.9 | 15.5 | 39.3 | 54.5 | 45.9 | 141.7 |
| yolov5s | 31.2 | 48.4 | 33.8 | 14.3 | 35.9 | 42.5 | 39.5 | 147.5 |

\* y26s 的 sweep 口径存在 CPU 瓶颈，此处为端到端实测值；y26n/m 两口径一致。

**pose 精度（COCO val2017, COCOeval keypoints）**

| 模型 | AP | AP50 | AP75 | AR | FPS |
|---|---|---|---|---|---|
| yolo26n_pose | 49.5 | 77.5 | 53.1 | 46.8 | 123.5 |
| yolo26s_pose | 48.4 | 81.1 | 51.2 | 48.7 | 50.1 |
| yolo26m_pose | 55.7 | 85.4 | 61.5 | 56.2 | 30.3 |
| yolov8_pose | 49.3 | 78.4 | 52.8 | 47.9 | 119.3 |
| yolo11n_pose | 46.6 | 78.6 | 48.8 | 46.1 | 112.5 |

**seg 精度（COCO val2017, COCOeval segm）**

| 模型 | AP | AP50 | AP75 | FPS |
|---|---|---|---|---|
| yolo26n_seg | 28.5 | 48.8 | 29.0 | 39.0 |
| yolo26s_seg | 32.9 | 55.9 | 33.4 | 38.9 |
| yolo26m_seg | 36.6 | 61.9 | 37.3 | 38.0 |
| yolov8_seg | 12.7 | — | — | 58.5 |
| yolo11n_seg | 10.1 | 18.5 | 9.9 | 74.1 |

> yolov8/yolo11 的 seg 模型为早期导出链转换产物，数值异常偏低，仅作记录。

**obb 精度（DOTA v1.0 val, 旋转 IoU）**

| 模型 | mAP50 | mAP50:95 | FPS |
|---|---|---|---|
| yolo26n_obb | 66.8 | 35.4 | 112.0 |
| yolo26s_obb | 69.6 | 39.0 | 109.4 |
| yolo26m_obb | 72.1 | 42.5 | 105.2 |
| yolov8_obb | 65.4 | 32.1 | 104.9 |

**depth 推理速度**

| 模型 | FPS |
|---|---|
| yolo26n_depth | 63.6 |

> 以上均为 INT8 量化模型在 RK3588 NPU 上的板端实测值（非 PC 模拟）；
> 精度用 `rknn_eval` 全量评测（COCOeval / DOTA 旋转 IoU 标准口径，conf=0.001）。
> FPS 为 9 线程 pipeline 全量推理速度（非单帧推理耗时）。
> obb 精度在 DOTA v1.0 val（458 图 → 1024/200 切片 5297 张）上评测；obb/seg 的 640 固定输入
> 与官方模型页的 1024 原生输入口径不同，绝对值不可直接对标，家族内横向对比有效。

## C ABI 用法(配置即契约)

`rkpipe/rkpipe.h` 是对外的唯一稳定接口:传入与命令行语义一致的 YAML 配置即可驱动流水线,所有内部类型均不经过该头文件暴露。

```c
rkpipe_handle h = rkpipe_create("run.yaml");   // 加载配置(不启动)
rkpipe_set_event_cb(h, on_event, ctx);         // STARTED/RESULT/COMPLETED/ERROR
rkpipe_start(h);                               // 后台线程运行,非阻塞
rkpipe_wait(h, -1);                            // 等待自然结束(有限输入)
rkpipe_destroy(h);
```

**线程契约**:事件回调在库内部线程触发;回调内**不得**调用任何 `rkpipe_*` 接口(会死锁),如需投递请在回调内只做入队。`rkpipe_stop` 为协作式停止,实时流场景以终止输入/进程退出为主。

**逐帧结构化结果**:设置环境变量 `RK_PIPE_RESULT_JSONL=<文件路径>` 后,流水线每帧输出一行 schema v1 JSON(字段定义见 [docs/event_payload.md](docs/event_payload.md))。该格式与 `RK_PIPE_EVENT_RESULT` 事件 payload 完全一致;当前核心库即可用文件汇通道(仅 pipeline 模式),事件本身的下发与 YAML 配置键需配套核心库 Release。

## 持续集成

`.github/workflows/ci.yml` 在通用 ubuntu 上跑两个 job（矩阵展开为 gcc / clang / full-build 三项检查），全部不需要 RKNN/RGA/MPP 硬件：

- **unit-tests（gcc 与 clang 两档）**：`-DRK_PIPE_CI_BUILD=ON` 编译配置解析、事件规则、ROI 过滤、后处理纯函数、JSONL 序列化等无硬件依赖的源集并执行 `ctest`
- **full-build**：不带预编译核心库配置时必须给出明确错误（`预编译核心库不存在`）而非静默失败

板端全量单测（含硬件用例）在板上 `cmake --build build --target rk_pipe_unit_tests && ./build/rk_pipe_unit_tests` 运行。

## 许可

各部分许可不同，见 [LICENSE](LICENSE)（源码，Apache-2.0）、[NOTICE](NOTICE)（第三方组件与随仓库分发的模型权重）与 [LEGAL/RKPIPE-CORE-EULA.md](LEGAL/RKPIPE-CORE-EULA.md)（`prebuilt/` 下的预编译核心库属专有软件，不在 Apache-2.0 授权范围内）。

## 给贡献者

欢迎对模块层的改进:新模型后处理、新输入源、跟踪与事件规则、性能工具等。PR 前请运行 `ctest`。见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 文档

- [docs/build.md](docs/build.md) — 环境依赖、模型转换、交叉编译
- [docs/run_guide.md](docs/run_guide.md) — 运行指南:从构建到多路的逐项验收清单
- [docs/architecture.md](docs/architecture.md) — 数据流与模块职责(行为级描述)
- [docs/event_payload.md](docs/event_payload.md) — 逐帧结构化结果 JSONL / RESULT 事件 payload 的 schema
- [docs/demo_detect3d.md](docs/demo_detect3d.md) — D3 3D 线框实现与 P2 标定
- [docs/demo_d3_depth_distance.md](docs/demo_d3_depth_distance.md) — 深度与距离文字(3D 线框前置步骤)
- [docs/depth_accuracy.md](docs/depth_accuracy.md) — 深度/距离精度评估与已排除方案
- [docs/demo_m0_composite.md](docs/demo_m0_composite.md) — 检测 + 二级分类级联
- [docs/demo_m6_ocr.md](docs/demo_m6_ocr.md) — OCR 检测 + 识别全链路
- [tools/eval/README.md](tools/eval/README.md) — rknn_eval 评测:四任务标准命令、数据准备、进阶选项
- [docs/benchmark_dota_obb.md](docs/benchmark_dota_obb.md) — OBB 评测指南(DOTA v1.0:切片/命令行/网页控制台/口径)
- [docs/faq.md](docs/faq.md) — 常见问题(ABI 兼容、许可)
