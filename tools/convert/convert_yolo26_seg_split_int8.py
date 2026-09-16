#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo26-seg 拆分布局 ONNX（10 输出）→ RKNN 纯 INT8。

与 detect/obb split 完全同配置：kl_divergence / quantized_method=channel /
optimization_level=0，200 张 COCO 校准集（dataset/dataset_coco200.txt）。

输出布局（按导出顺序）：
  3 尺度 × (box [1,4,S,S] + cls [1,80,S,S] + mask [1,32,S,S]) + proto [1,32,160,160]

自查：
  1) 模拟器推理，输出数/shape 必须与源 ONNX 一致；
  2) 从模拟器输出估计每个张量的量化步长，cls 判别区 [-6.9,0]/[-12,0] 有效档位数；
  3) box/mask/proto 各自独立 scale（这正是拆分的目的）。

运行环境：WSL rknn232（rknn-toolkit2 2.3.2）
用法：
  python convert_yolo26_seg_split_int8.py --onnx onnx/yolo26n-seg_split10.onnx --output yolo26n-seg_split10_i8.rknn
"""
import argparse
import os

import numpy as np
import onnx

from rknn.api import RKNN

IMGSZ = 640


def estimate_step(vals: np.ndarray) -> float:
    """从反量化输出估计量化步长：排重后最小正差。"""
    v = np.unique(vals.astype(np.float64).ravel())
    if v.size < 2:
        return 0.0
    d = np.diff(v)
    d = d[d > 1e-9]
    return float(d.min()) if d.size else 0.0


def classify(dims, idx, total):
    """按导出顺序命名张量角色：3i=box, 3i+1=cls, 3i+2=mask, 最后=proto。"""
    if idx == total - 1:
        return "proto"
    kind = idx % 3
    return {0: "box", 1: "cls", 2: "mask"}[kind]


def main():
    parser = argparse.ArgumentParser(description="yolo26-seg 拆分布局 ONNX -> RKNN(INT8)")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="dataset/dataset_coco200.txt")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    assert os.path.isfile(args.onnx), f"ONNX not found: {args.onnx}"
    assert os.path.isfile(args.dataset), f"dataset not found: {args.dataset}"

    src = onnx.load(args.onnx)
    expected = []
    for o in src.graph.output:
        dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
        expected.append((o.name, dims))
    print("[EXPECT] 源 ONNX 输出:", expected)
    assert len(expected) == 10, f"seg 拆分布局应有 10 个输出，实际 {len(expected)}"
    for idx, (name, dims) in enumerate(expected):
        kind = classify(dims, idx, len(expected))
        ch = dims[1]
        if kind == "box":
            assert ch == 4, f"{name}: box 应 4 通道，实际 {ch}"
        elif kind == "mask":
            assert ch == 32, f"{name}: mask 应 32 通道，实际 {ch}"
        elif kind == "proto":
            assert ch == 32 and dims[2] == 160, f"{name}: proto 应 [32,160,160]，实际 {dims[1:]}"

    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.target,
        quantized_algorithm="kl_divergence",
        quantized_method="channel",
        optimization_level=0,
    )
    assert rknn.load_onnx(model=args.onnx) == 0, "load_onnx failed"
    assert rknn.build(do_quantization=True, dataset=args.dataset) == 0, "build failed"
    assert rknn.export_rknn(args.output) == 0, "export_rknn failed"
    print(f"[OK] saved {args.output}")

    # 自查：模拟器推理 + 步长/有效档位估计
    rknn.init_runtime()
    img = np.random.rand(1, args.imgsz, args.imgsz, 3).astype(np.float32)
    outs = rknn.inference(inputs=[img])
    assert len(outs) == len(expected), f"输出数异常: {len(outs)} != {len(expected)}"
    print("\n[CHECK] 输出 shape 与量化步长估计（输入为随机图，仅验结构）:")
    for i, (o, (name, dims)) in enumerate(zip(outs, expected)):
        a = np.asarray(o)
        assert list(a.shape) == dims, f"{name} shape 异常: {a.shape} != {dims}"
        step = estimate_step(a)
        kind = classify(dims, i, len(expected))
        msg = f"  {name}({kind}): shape={a.shape} step~{step:.6f}"
        if kind == "cls" and step > 0:
            lv69 = int(round(6.9 / step))
            lv12 = int(round(12.0 / step))
            msg += f" | conf0.001 判别区[-6.9,0]≈{lv69} 档, [-12,0]≈{lv12} 档"
        print(msg)
    print("[CHECK] done（步长仅估计值，最终以板端 rknn_tensor_attr.scale 为准）")

    rknn.release()


if __name__ == "__main__":
    main()
