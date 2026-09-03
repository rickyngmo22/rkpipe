#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo26n detect → RKNN 转换脚本（cls 已入图 sigmoid + INT8，方案2）
==================================================================
配套 add_cls_sigmoid_yolo26_onnx.py：先给 ONNX 的 cls 通道接 Sigmoid，再整模型 INT8。
cls 输出数学上必在 [0,1]，INT8 量化步长 ≈ 1/255，分数不再塌缩；box 直接距离(0~20)
值域温和，INT8 量化也够。整体为纯 INT8 计算，速度接近原 INT8 档（~100FPS 管线级）。

前提（PC 端）:
  - rknn-toolkit2 == 2.3.2（与板端 runtime 2.3.2 同版）
  - 输入 ONNX: yolo26n_cls_sigmoid.onnx（add_cls_sigmoid_yolo26_onnx.py 产物）
  - 校准图集 dataset.txt

用法:
  python3 convert_yolo26_cls_sigmoid_int8.py

输出: yolo26n.rknn（板端默认 detect 档位；裸名 = cls_sigmoid_i8，_fp16 后缀 = 全 FP16 档）
"""
import os

from rknn.api import RKNN

# ===== 按需修改 =====
ONNX_MODEL = "yolo26n_cls_sigmoid.onnx"
RKNN_MODEL = "yolo26n.rknn"
DATASET = "dataset.txt"
TARGET_PLATFORM = "rk3588"
# ====================

assert os.path.isfile(ONNX_MODEL), f"ONNX not found: {ONNX_MODEL}"
assert os.path.isfile(DATASET), f"dataset not found: {DATASET}"

rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform=TARGET_PLATFORM,
    quantized_algorithm="kl_divergence",
    quantized_method="channel",
    optimization_level=0,
)
assert rknn.load_onnx(model=ONNX_MODEL) == 0, "load_onnx failed"
assert rknn.build(do_quantization=True, dataset=DATASET) == 0, "build failed"
assert rknn.export_rknn(RKNN_MODEL) == 0, "export_rknn failed"
print(f"[OK] saved {RKNN_MODEL}")

# 自查：必须仍是 3 个 [1,84,H,W] 输出（绝不能是 [1,300,6]）
ret = rknn.list_outputs()
print("[CHECK] list_outputs =", ret)

# 可选：模拟器精度分析（对比原始 ONNX）
# rknn.accuracy_analysis(inputs=["test.npy"], output_dir="./acc_cls_sigmoid")

rknn.release()
