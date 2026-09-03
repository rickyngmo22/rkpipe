#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo26n detect → RKNN 转换脚本（自动混合量化）
=============================================
【重要：2.3.2 手动 hybrid 有 bug】
  hybrid_quantization_step1/step2 手动把 head 标 FP16 的路径在 rknn-toolkit2 2.3.2 不可用：
  step2 的量化优化器 `_p_adjust_tanh_sigmoid` 会按名字查找已被融合进 SiLU 的
  `Sigmoid_output_0`，而 cfg 里只有融合后的 `act/Mul_output_0` → 必然 KeyError
  （YOLO26 head 全是 SiLU，必触发）。已实测：不改 cfg / custom_quantize_layers 标 42 层 /
  custom_hybrid 指定 head 区域 → step2 全部同样 KeyError；手动补 Sigmoid 条目 → 被判 invalid。
  因此只能走官方支持的 build(auto_hybrid=True)：按精度/溢出自动把部分层切 FP16，其余 INT8。

注意：auto_hybrid 的自动选择未必保护你想保护的头（本次实测只把 output2 即 20x20 尺度分支切了
FP16，output0/1 保持 INT8，检测分数仍塌缩）。若目的是拿到真实分数，优先用全 FP16 脚本
convert_yolo26_fp16.py。

前提（PC 端）:
  - rknn-toolkit2 == 2.3.2
  - ultralytics >= 8.4.110 导出 one2one 头的 ONNX（end2end=False）
  - 校准图集 dataset.txt（INT8 部分量化需要）

用法:
  python3 convert_yolo26_hybrid.py

输出: yolo26n_e2e_false_hybrid_auto.rknn
"""
import os

from rknn.api import RKNN

# ===== 按需修改 =====
ONNX_MODEL = "yolo26n.onnx"
RKNN_MODEL = "yolo26n_e2e_false_hybrid_auto.rknn"
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

# auto_hybrid=True：官方自动混合量化（FP16+INT8 混合，自动选择敏感层切 FP16）
assert rknn.build(do_quantization=True, dataset=DATASET, auto_hybrid=True) == 0, "build failed"
assert rknn.export_rknn(RKNN_MODEL) == 0, "export_rknn failed"
print(f"[OK] saved {RKNN_MODEL}")
print("[CHECK] list_outputs =", rknn.list_outputs())
rknn.release()
