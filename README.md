# rkpipe — RK3588 边缘端多任务视觉流水线

rkpipe 是一套面向 Rockchip RK3588 的实时视频理解流水线:多路视频输入 → RGA 零拷贝预处理 → NPU 推理 → 后处理/跟踪 → Web 预览 / 推流 / 结构化事件输出。单进程内支持检测、姿态、OBB、实例分割、深度估计等任务。

仓库包含全部功能模块源码、稳定 C ABI 契约头、示例与工具链。**流水线调度核心**(线程编排、任务队列、内存池、零拷贝推理引擎等)以预编译静态库 `librkpipe_core.a` 随 Release 分发,不在本仓库(见 [边界说明](#开源边界))。

## 任务能力

| task | 说明 |
|---|---|
| `detect` | YOLO26 / YOLOv8 / YOLOv5 目标检测 |
| `pose` | YOLOv8-Pose / YOLO26-Pose 人体关键点检测(多人建议开启 tracking 锁定每人颜色) |
| `obb` | 旋转框检测 |
| `seg` | 实例分割 |
| `depth` | 单目深度估计 |

## 仓库结构

```
├── include/rkpipe/          契约层:稳定 C ABI(rkpipe.h)+ 跨层共享数据结构
├── include/core/            契约头子集(帧数据/任务结果/RKNN 上下文/事件规则…)
├── include/{io,model,postprocess,detection,config,utils}/
├── src/                     模块层实现(与头文件一一对应)
├── examples/
│   ├── rkpipe_cli/          CLI:仅用 C ABI 驱动完整流水线(兼作 ABI 验收)
│   ├── events_demo/         最小事件回调示例(纯 C)
│   └── configs/             示例配置
├── prebuilt/<arch>/         预编译闭源核心库(从 Releases 下载放入)
├── 3rdparty/rknn/           RKNN SDK 头与运行库(遵循 Rockchip 许可)
├── tools/                   标定工具、模型输出探针、板端精度评测(rknn_eval)
├── convert_*.py 等          ONNX→RKNN 模型转换辅助脚本
└── scripts/check_open_boundary.py   开源边界自检
```

## 快速开始

### 1. 获取预编译核心库

从本仓库 **Releases** 页下载对应版本的 `librkpipe_core.a`(目前提供 aarch64/RK3588),放入:

```
prebuilt/aarch64/librkpipe_core.a
```

### 2. 构建

```bash
# 板端(RK3588,Debian/Ubuntu):需要 opencv、turbojpeg、rga、rockchip-mpp、ffmpeg
cmake -B build && cmake --build build -j $(nproc)

# 无硬件环境:纯逻辑单元测试(配置解析/事件规则/ROI 过滤/后处理)
cmake -B build-ci -DRK_PIPE_CI_BUILD=ON && cmake --build build-ci && ctest --test-dir build-ci
```

### 3. 运行

```bash
./build/rkpipe_cli examples/configs/detect_video.yaml
# 浏览器打开 http://<板子IP>:8080 查看 Web 预览(detect_web_preview.yaml)
```

模型文件(.rknn)不随仓库分发:用 [tools/](tools/) 与根目录 `convert_*.py` 脚本从公开的 YOLO26 / YOLOv8 等权重自行转换,详见 [docs/build.md](docs/build.md)。

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

完整可运行示例见 [examples/events_demo](examples/events_demo)。

## 开源边界

| 层 | 形态 | 内容 |
|---|---|---|
| 契约层 | 源码(本仓库) | `rkpipe.h` C ABI、帧/结果数据结构、错误码、回调签名 |
| 模块层 | 源码(本仓库) | io / model / postprocess / detection / config / utils、事件规则引擎、模型转换工具 |
| 调度核心 | 预编译 `librkpipe_core.a` | 线程编排与任务队列、帧调度与保序、背压/跳帧策略、线程池与内存池、RKNN 零拷贝推理引擎、C ABI 门面实现 |

核心库不开源、不随源码分发;它通过且仅通过契约头与本仓库对话(链接期互相解析符号,见 CMake 中 `--start-group` 链接组)。`scripts/check_open_boundary.py` 在 CI 中持续自检"开源代码不引用闭源符号、无敏感内容"。

各部分许可不同,见 [LICENSE](LICENSE)(源码,Apache-2.0)、[NOTICE](NOTICE)(第三方组件)与 [LEGAL/RKPIPE-CORE-EULA.md](LEGAL/RKPIPE-CORE-EULA.md)(预编译核心)。

## 给贡献者

欢迎对模块层的改进:新模型后处理、新输入源、跟踪与事件规则、性能工具等。PR 前请运行 `ctest` 与边界自检;核心层不接受也无法接受源码 PR(它不在本仓库)。见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 文档

- [docs/build.md](docs/build.md) — 环境依赖、模型转换、交叉编译
- [docs/architecture.md](docs/architecture.md) — 数据流与模块职责(行为级描述)
- [docs/faq.md](docs/faq.md) — 常见问题(open-core 边界、ABI 兼容、许可)
