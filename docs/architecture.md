# 架构(行为级)

本文只描述**外部可观察的行为与模块职责**,不涉及调度核心的内部实现。

## 数据流

```
输入源(io)           模块层                    预编译核心                输出(io)
┌─────────────┐   ┌──────────────────┐   ┌(librkpipe_core.a)┐   ┌──────────────┐
│ 文件/目录/   │帧 │ preprocess(RGA   │→ │  线程编排/任务队列  │→ │ Web 预览(HTTP │
│ RTSP/v4l2 ──→│  │  零拷贝→NV12)    │   │  背压与自适应跳帧   │   │  MJPEG/WebRTC)│
└─────────────┘   │ model(RKNN 推理) │   │  保序输出/事件分发  │   │ 文件/RTSP 推流 │
                  │ postprocess 解码 │   └──────────────────┘   │ 结构化事件/告警│
                  │ detection 跟踪/  │         ↑ C ABI           └──────────────┘
                  │ 叠绘/事件规则     │─────────┘(rkpipe.h)
                  └──────────────────┘
```

- **io**:输入源抽象(`InputSource`:文件/图片目录/RTSP/v4l2)、输出路由(`OutputRouter`:文件、RTSP 推流、Web 预览)、帧的零拷贝释放。
- **preprocess**:BGR→NV12、letterbox,RGA 硬件加速路径与 CPU 回退路径。
- **model**:各任务的 RKNN 模型封装(加载/推理/输出后处理回调),INT8/FP16 统一接口。
- **postprocess**:各任务后处理实现(detect/pose/obb/seg/depth × yolo26/yolov8/yolov5)。
- **detection**:SimpleObjectTracker(ID + 平滑 + 遮挡恢复)、叠绘、事件规则引擎(EventEngine:绊线/入侵/滞留/离岗,纯逻辑可单测)。多人姿态场景开启 tracking 后,跟踪 ID 会为每个人锁定骨架/框颜色,避免人物进出画面时变色。
- **config**:YAML + 命令行配置(`AppConfig`),字段即契约。

## 两级并发形态

配置 `mode` 决定:

- `sequential`:单线程逐帧 读帧→推理→输出;
- `pipeline`:多 worker 流水线,输入/推理/输出三级并行,帧序保持一致;线程数 `thread_count`。

调度核心负责 worker 池、任务队列、在途帧背压(自适应跳帧反馈)、乱序结果的保序归队、事件回调分发——这些行为对模块层表现为:`PipelineFrame` 按 `index` 递增地流入、按 `index` 递增地流出。

## 路线

更多任务类型与多任务组合将在后续版本提供;契约头(`include/rkpipe/`、`include/core/` 开源子集)的演进见仓库 Release 说明。

## 契约层

- `rkpipe/rkpipe.h`:C ABI(create/start/stop/wait/status/事件回调),配置即契约。
- `rkpipe` 契约头与 `include/core/` 开源子集:帧数据(`pipeline_frame.h`)、任务结果变体(`task_result.h`)、RKNN 上下文(`rknn_context.h`)、任务类型、性能统计、内存池对外接口(实现闭源)。
- 闭源核心与开源模块在**链接期**互相解析符号(`--start-group`),运行期没有跨 .so 的 C++ ABI 依赖。
