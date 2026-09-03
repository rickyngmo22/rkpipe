#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo26n detect → RKNN 转换脚本（全 FP16）
=========================================
do_quantization=False → 全图 FP16，head 分数为真实分布（不再塌缩到 0.5）。
这是 2.3.2 下唯一能稳定拿到"有区分度分数"的路径（手动 hybrid 有 SiLU 融合 bug，
auto_hybrid 又只保护部分头）。代价：NPU 吞吐约为 INT8 的 1/2~1/3。

参考（板端实测）：yolo26n @640 INT8(layer) 4线程 ~117 FPS；全 FP16 预计 ~60-80 FPS，
对 25fps 视频源仍富余。板端 r_pipe 后处理已支持 fp16 输出（按张量类型解码），无需改代码。

前提（PC 端）:
  - rknn-toolkit2 == 2.3.2
  - ultralytics >= 8.4.110 导出 one2one 头的 ONNX（end2end=False）

用法:
  python3 convert_yolo26_fp16.py

输出: yolo26n_fp16.rknn
"""
import os

from rknn.api import RKNN

# ===== 按需修改 =====
ONNX_MODEL = "yolo26n.onnx"
RKNN_MODEL = "yolo26n_fp16.rknn"
TARGET_PLATFORM = "rk3588"
# ====================

assert os.path.isfile(ONNX_MODEL), f"ONNX not found: {ONNX_MODEL}"

rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform=TARGET_PLATFORM,
    optimization_level=0,
)
assert rknn.load_onnx(model=ONNX_MODEL) == 0, "load_onnx failed"

# do_quantization=False → 全图 FP16（无需校准集）
assert rknn.build(do_quantization=False) == 0, "build failed"
assert rknn.export_rknn(RKNN_MODEL) == 0, "export_rknn failed"
print(f"[OK] saved {RKNN_MODEL}")
print("[CHECK] list_outputs =", rknn.list_outputs())
rknn.release()
