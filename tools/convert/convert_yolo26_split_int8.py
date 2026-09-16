#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo26n 拆分布局 ONNX（6 输出）→ RKNN 纯 INT8。

与现网 i8 同配置：kl_divergence / quantized_method=channel / optimization_level=0，
200 张 COCO 校准集（dataset/dataset_coco200.txt）。

自查：
  1) 模拟器推理，输出数/shape 必须与源 ONNX 一致（6 个：box [1,4,S,S] × cls [1,80,S,S]）；
  2) 从模拟器输出估计每个张量的量化步长（取排重后最小正差），并给出 cls 判别区
     [-6.9, 0]（conf=0.001）与 [-12, 0] 的有效档位数 —— 直接验证"拆分+clamp"是否达标。

运行环境：WSL rknn232（rknn-toolkit2 2.3.2）
用法：
  python convert_yolo26_split_int8.py
  python convert_yolo26_split_int8.py --onnx onnx/yolo26n_split6_noclamp.onnx --output yolo26n_split6_noclamp_i8.rknn
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


def main():
    parser = argparse.ArgumentParser(description="yolo26 拆分布局 ONNX -> RKNN(INT8)")
    parser.add_argument("--onnx", default="onnx/yolo26n_split6.onnx")
    parser.add_argument("--output", default="yolo26n_split6_i8.rknn")
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
    assert len(expected) == 6, f"拆分布局应有 6 个输出，实际 {len(expected)}"

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
        kind = "box" if dims[1] == 4 else "cls"
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
