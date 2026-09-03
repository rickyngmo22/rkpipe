#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo26n OBB → RKNN 转换脚本（全 FP16）
========================================
do_quantization=False → 全图 FP16。INT8(layer) 会把 OBB head 的 cls logits 压平
（实测 yolo26n_obb INT8：所有格都报 plane 0.700，类/分无判别力，仅框回归可用），
FP16 是唯一稳定拿到真实分数/类别的路径。

前提（PC 端）:
  - rknn-toolkit2 == 2.3.2（与板端 runtime 2.3.2 严格同版）
  - ultralytics >= 8.4.110 导出 one2one 头的 ONNX（end2end=False）:
      from ultralytics import YOLO
      YOLO("yolo26n-obb.pt").export(format="onnx", opset=12, end2end=False)  # yolo26n-obb.onnx
  - 板端 yolo26n_obb 后处理已支持 fp16 输出（按张量类型解码），无需改代码

用法:
  python3 convert_yolo26_obb_fp16.py

输出: yolo26n_obb_fp16.rknn
"""
import os

from rknn.api import RKNN

# ===== 按需修改 =====
ONNX_MODEL = "yolo26n-obb.onnx"
RKNN_MODEL = "yolo26n_obb_fp16.rknn"
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
