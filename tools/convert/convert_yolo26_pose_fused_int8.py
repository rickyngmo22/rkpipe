#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo26-pose 融合布局 ONNX（3 输出 [1,56,H,W]）→ RKNN 纯 INT8。

与 detect 同配置：kl_divergence / channel / optimization_level=0 / 200 张 COCO 校准。
pose 融合 INT8 实测无塌缩（y26n: AP 49.49 vs FP16 55.41，损失 5.9 AP 属正常量化损失），
s/m 沿用融合布局即可直接上板（yolov26_pose.cc 原生支持 3×56）。

运行环境：WSL rknn232
用法：
  python convert_yolo26_pose_fused_int8.py --onnx onnx/yolo26s-pose_e2e_false.onnx \
      --output yolo26s_pose_i8.rknn
"""
import argparse
import os

import numpy as np
import onnx

from rknn.api import RKNN

IMGSZ = 640


def main():
    parser = argparse.ArgumentParser(description="yolo26-pose 融合 ONNX -> RKNN(INT8)")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="dataset/dataset_coco200.txt")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    assert os.path.isfile(args.onnx), f"ONNX not found: {args.onnx}"
    assert os.path.isfile(args.dataset), f"dataset not found: {args.dataset}"

    src = onnx.load(args.onnx)
    expected = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
                for o in src.graph.output]
    print("[EXPECT] 源 ONNX 输出:", expected)
    assert len(expected) == 3, f"融合布局应有 3 个输出，实际 {len(expected)}"

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

    rknn.init_runtime()
    img = np.random.rand(1, args.imgsz, args.imgsz, 3).astype(np.float32)
    outs = rknn.inference(inputs=[img])
    assert len(outs) == 3, f"输出数异常: {len(outs)}"
    print("[CHECK] 输出 shape:")
    for o, (name, dims) in zip(outs, expected):
        a = np.asarray(o)
        assert list(a.shape) == dims, f"{name} shape 异常: {a.shape} != {dims}"
        print(f"  {name}: shape={a.shape} min={a.min():.4f} max={a.max():.4f}")
    print("[CHECK] done")

    rknn.release()


if __name__ == "__main__":
    main()
