#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo26n detect → RKNN 转换脚本（INT8 + 通道级量化）
=====================================================
对比默认 layer 量化：`quantized_method='channel'` 让 head 各通道独立 scale，
logits 量化步长变细，置信度不再塌缩到 0.5。速度仍是 INT8。

前提（PC 端）:
  - rknn-toolkit2 == 2.3.2（与板端 runtime 2.3.2 严格同版）
  - ultralytics >= 8.4.110 导出 one2one 头的 ONNX（end2end=False）
  - 校准图集 dataset.txt（每行一张图片绝对路径，建议 100~300 张，COCO 子集即可）

用法:
  python3 convert_yolo26_channel.py

输出: yolo26n_e2e_false_i8_channel.rknn
"""
import os

from rknn.api import RKNN

# ===== 按需修改 =====
ONNX_MODEL = "yolo26n.onnx"                 # ultralytics 导出的 ONNX（end2end=False）
RKNN_MODEL = "yolo26n_e2e_false_i8_channel.rknn"
DATASET = "dataset.txt"                     # 校准图集列表（绝对路径，每行一张）
TARGET_PLATFORM = "rk3588"
# ====================

assert os.path.isfile(ONNX_MODEL), f"ONNX not found: {ONNX_MODEL}"
assert os.path.isfile(DATASET), f"dataset not found: {DATASET}"

rknn = RKNN(verbose=True)

# 1) config：RGB 输入、0~255 归一化（与板端管线一致）；关键 quantized_method='channel'
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform=TARGET_PLATFORM,
    quantized_algorithm="kl_divergence",
    quantized_method="channel",          # 关键：通道级量化，head logits 保精度
    optimization_level=0,                # 保持 0，避免编译器改动输出头；跑通后可试 1~3
)

# 2) load ONNX（若导出时混入多余输出，用 outputs=[...] 指定 3 个 one2one 输出）
assert rknn.load_onnx(model=ONNX_MODEL) == 0, "load_onnx failed"

# 3) build（INT8 量化）
assert rknn.build(do_quantization=True, dataset=DATASET) == 0, "build failed"

# 4) export
assert rknn.export_rknn(RKNN_MODEL) == 0, "export_rknn failed"
print(f"[OK] saved {RKNN_MODEL}")

# 5) 自查：打印输出签名，必须 3 个 [1,84,H,W]，绝不能是 [1,300,6]
ret = rknn.list_outputs()
print("[CHECK] list_outputs =", ret)

rknn.release()
