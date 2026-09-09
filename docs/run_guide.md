# 运行指南(逐项验收)

按顺序执行以下步骤,每步都给出**命令**与**预期结果**,全部通过即代表仓库在本机处于健康状态。
步骤 1–2 不需要任何硬件(纯逻辑);步骤 3 起面向 RK3588 板端。

> 路径约定:本指南所有命令均在**仓库根目录**执行;示例一律使用仓库相对路径,
> `model/`、`video/`、`test_images/` 等本地素材目录已被 `.gitignore` 排除,不会入库。

## 验收清单总览

| # | 项目 | 命令入口 | 依赖 |
|---|---|---|---|
| 1 | 纯逻辑单元测试 | `ctest --test-dir build-ci` | cmake/g++/OpenCV |
| 2 | 板端完整构建 | `cmake --build build` | 预编译核心库 + RKNN/RGA/MPP/FFmpeg |
| 3 | 测试素材准备 | ffmpeg / 模型转换 | ffmpeg |
| 4 | 检测流水线 | `./build/console_detector examples/configs/detect_video.yaml` | 步骤 2/3 |
| 5 | 事件回调 + JSONL | `./build/console_detector`(JSONL 环境变量) | 步骤 4 |
| 6 | 评测工具 dry-run | `./build/rknn_eval --dry-run` | 步骤 2 |
| 7 | 多路编排器 | `python3 scripts/multistream.py run --help` | python3 |

## 0. 环境准备(板端 RK3588,Debian/Ubuntu)

```bash
sudo apt install cmake g++ python3 ffmpeg \
    libopencv-dev libturbojpeg0-dev \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev
```

RKNN Runtime / RGA / MPP 由板端系统提供或随仓库 `3rdparty/rknn/` 分发,详见 [build.md](build.md)。

## 1. 纯逻辑单元测试(无需硬件与核心库)

```bash
cmake -B build-ci -DRK_PIPE_CI_BUILD=ON
cmake --build build-ci -j $(nproc)
ctest --test-dir build-ci --output-on-failure
```

**预期**:`ctest` 汇总 `100% tests passed`。覆盖 YAML/命令行配置解析、事件规则引擎、ROI 过滤、后处理纯函数与性能统计。

## 2. 板端完整构建

```bash
cmake -B build
cmake --build build -j $(nproc)
```

**预期**:构建成功并产出 `build/console_detector`、`build/rknn_eval`。
若配置阶段报"预编译核心库不存在",先按 README 从 Releases 下载 `librkpipe_core.a` 放入 `prebuilt/aarch64/`。

## 3. 测试素材准备

```bash
mkdir -p video model
# 测试视频(ffmpeg 合成,流水线跑通用;真实检测效果请换自带素材或摄像头)。
# 用 60s 是因为流水线按算力满速处理(300 帧约 2.5s),太短的话 Web 预览来不及验证
ffmpeg -y -f lavfi -i testsrc2=duration=60:size=1280x720:rate=30 video/demo.mp4
# 模型:model/ 已随仓库附带(yolo26n / yolov8n / yolo11n / yolov5s 等),无需准备
```

**预期**:`video/demo.mp4` 可播放(`ffprobe video/demo.mp4` 有视频流),`model/` 下有所需 `.rknn`。
示例配置中的相对路径(`model/yolo26n.rknn` + `assets/labels/coco_80_labels_list.txt`)与之对应。

## 4. 检测流水线 + Web 预览

示例配置默认开启 Web 预览(`web_preview: 1`,端口 8080):

```bash
./build/console_detector examples/configs/detect_video.yaml &
# 等待模型加载完成出现 "listening on" 日志,然后:
curl -fsS -o /dev/null -w "root=%{http_code}\n" http://127.0.0.1:8080/          # 预期 200
curl -fsS -o /dev/null -w "healthz=%{http_code}\n" http://127.0.0.1:8080/healthz # 预期 200
# 无人观看时服务器跳过 JPEG 编码省 CPU;先连一次流再取快照:
( curl -s --max-time 4 -o /dev/null http://127.0.0.1:8080/stream.mjpg ) & sleep 2
curl -fsS -o /tmp/snap.jpg http://127.0.0.1:8080/snapshot.jpg                    # 预期 200,file 显示 JPEG
# 浏览器打开 http://<板子IP>:8080 可见实时检测画面
kill %1
```

**预期**:root/healthz 返回 200;`/tmp/snap.jpg` 为有效 JPEG;前台运行打印 `STARTED`,视频自然结束时打印 `COMPLETED`,退出码 0,无断言/段错误。pose / obb / seg / depth 示例配置用法相同。有限输入跑完后服务随流水线自然结束,常驻预览请接 RTSP/摄像头;不需要预览时在配置里设 `web_preview: 0`。

## 5. 事件回调与逐帧 JSONL

```bash
RK_PIPE_RESULT_JSONL=/tmp/results.jsonl ./build/console_detector examples/configs/detect_video.yaml | head -5
head -2 /tmp/results.jsonl
```

**预期**:`console_detector` 依次打印 `STARTED`/`RESULT`/`COMPLETED` 事件;`/tmp/results.jsonl` 每帧一行 JSON,首行含 `"schema_version":1` 等字段(字段定义见 [event_payload.md](event_payload.md))。

## 6. 评测工具 dry-run(不需要真实数据集)

```bash
# dry-run 只校验"配置合法 + 评测脚本就位",不执行推理;--ann 占位文件即可
mkdir -p data/coco && echo '{}' > data/coco/annotations.json
./build/rknn_eval --task detect --model model/yolo26n.rknn \
  --images data/coco/val2017 --ann data/coco/annotations.json \
  --label assets/labels/coco_80_labels_list.txt --obj-num 80 --dry-run
```

**预期**:打印 `[dry-run] task=detect scripts=... out=...` 并退出码 0。真实评测(COCO/DOTA 数据准备与完整命令)见 [tools/eval/README.md](../tools/eval/README.md)。

## 7. 多路部署编排器

```bash
# 预检:只打印 NPU core 分配,不启动
python3 scripts/multistream.py run --config cam_a.yaml cam_b.yaml --cores auto --dry-run
# 实际运行(每路一个进程;有限输入文件批量跑加 --no-restart,全部跑完自动退出):
python3 scripts/multistream.py run --config cam_a.yaml cam_b.yaml --cores auto --no-restart
```

**预期**:预检打印"预检通过: 2 路"与每路 `npu_core_start` 分配;实际运行时每路一个 `console_detector` 子进程、NPU core 轮转,子进程崩溃默认自动重启,`--no-restart` 下全部流退出后编排器以退出码 0 结束。

## 常见问题

| 现象 | 处理 |
|---|---|
| 配置阶段报缺 `librkpipe_core.a` | 从 Releases 下载放入 `prebuilt/aarch64/`,核对 `sha256sum -c prebuilt/aarch64/sha256sums.txt` |
| NPU 推理 FPS 波动大 | NPU governor 设为 performance:`echo performance | sudo tee /sys/class/devfreq/fdab0000.npu/governor` |
| 视频无法打开 | FFmpeg 编解码依赖未装齐,或文件路径不是相对仓库根目录执行 |
| Web 预览 8080 被占用 | 改配置 `web_preview_port` |
| 其余构建/转换问题 | 见 [build.md](build.md) 与 [faq.md](faq.md) |
