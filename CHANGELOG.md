# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/),版本遵循语义化版本。

## [未发布]

### 新增
- M8 RetinaFace 人脸检测解码核心(`postprocess/retinaface_decode.h`):先验框生成(320×320 → 4200 anchors,strides 8/16/32,枚举顺序逐项对照 zoo `rknn_box_priors.h`;公式化生成,640×640 → 16800 亦可)、center-form 框解码(variances [0.1,0.2])、5 点 landmark 解码(variance 0.1)、conf 过滤 + 分数降序截断 + IoU NMS。纯函数无硬件依赖,进 CI 纯逻辑源集并新增 29 项单测(先验框基准值/零回归闭合/方差数学/NMS 行为/病态输入防护)

### 说明
- face 任务的流水线接线(任务类型/绘制/示例配置)依赖调度核心的 Face 结果分发支持,待配套新核心随 Release 发布后启用

## [0.2.0] — 首次公开发布

### 新增
- 基础任务:检测(detect)、姿态(pose)、旋转框(obb)、实例分割(seg)、单目深度(depth)
- 模型家族:YOLO26 / YOLOv8 / YOLOv5(detect 与 seg),YOLOv8-Pose / YOLO26-Pose,YOLOv8-OBB,YOLO26-Depth
- 稳定 C ABI(`rkpipe/rkpipe.h`):create / start / stop / wait / status / 事件回调,"配置即契约"
- 输入输出:本地文件 / 图片目录 / RTSP / v4l2 输入;文件 / RTSP 推流 / Web 预览(MJPEG/WebRTC)输出
- 跟踪与事件规则引擎:ID 跟踪、绊线 / 区域入侵 / 滞留 / 离岗(纯逻辑,CI 可测)
- 工具链:YOLO 系列 ONNX→RKNN 转换脚本、模型输出探针、相机标定
- RGA 零拷贝预处理路径(NV12 dma-buf)+ CPU 回退
- 预编译调度核心 `librkpipe_core.a`(aarch64/RK3588,随 Release 分发)

### 说明
- 采用 open-core 分发:功能模块开源(本仓库,Apache-2.0),调度核心为预编译库(专有许可)
- 无硬件环境可用 `RK_PIPE_CI_BUILD=ON` 运行纯逻辑单元测试
