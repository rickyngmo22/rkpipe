#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 YOLO26 detect「拆分布局」原始头 ONNX（one2one 头，box/cls 按尺度分成独立张量）。

输出签名（6 个输出，box_i / cls_i 交替）：
    box_i : [1, 4,  H, W]   直接距离回归 (l,t,r,b)，网格单位
    cls_i : [1, nc, H, W]   分类 logits（默认已 clamp 到 [--clamp-lo, --clamp-hi]）

为什么拆分：
  融合布局 [1, 4+nc, H, W] 的输出张量共享一个量化 scale，box 与 cls 互相撑爆量化步长
  （实测融合张量 scale 0.15~0.43，cls 判别区只有 ~28 档）。拆分后每张量独立 scale。

为什么默认加 clamp：
  cls logits 存在极端负值（FP32 全量集观测到 -174），就算单独成张量也会把 calib 值域撑到
  ±40 以上。clamp(-12, 8) 只截断 sigmoid(x) < 6e-6 的无意义负尾（conf 阈值 0.001 对应
  logit -6.9），且 clamp 保序 —— argmax 与阈值过滤结果不变，FP32 语义无损。
  box 不做 clamp：大目标距离可达数十网格单位，clamp 会截断真实框。

运行环境：WSL onnx_env（ultralytics）
用法：
  python export_onnx_split6.py                                  # 默认 clamp(-12, 8)
  python export_onnx_split6.py --no-clamp                       # 不加 clamp（对照）
  python export_onnx_split6.py --weights weights/yolo26n.pt --output onnx/yolo26n_split6.onnx
"""
import argparse
import os
import types

import torch

from ultralytics.nn.modules.head import Detect


def main():
    parser = argparse.ArgumentParser(description="YOLO26 box/cls 拆分导出")
    parser.add_argument("--weights", default="weights/yolo26n.pt")
    parser.add_argument("--output", default="onnx/yolo26n_split6.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--clamp-lo", type=float, default=-12.0)
    parser.add_argument("--clamp-hi", type=float, default=8.0)
    parser.add_argument("--no-clamp", action="store_true", help="不加 clamp（对照组）")
    args = parser.parse_args()

    from ultralytics import YOLO

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"[load] {args.weights}")
    model = YOLO(args.weights).model
    model.eval()
    try:
        model.fuse()
        print("[ok] 模型已 fuse")
    except Exception as e:
        print(f"[warn] fuse 失败，继续导出: {e}")

    head = model.model[-1]
    assert isinstance(head, Detect), f"不是 Detect 头: {type(head).__name__}"
    assert hasattr(head, "one2one_cv2"), "模型无 one2one 分支，请确认是 end2end 的 YOLO26 模型"
    assert head.reg_max == 1, f"reg_max 应为 1（直接距离回归），实际 {head.reg_max}"
    print(f"[head] {type(head).__name__} | nl={head.nl} nc={head.nc} reg_max={head.reg_max}")

    no_clamp = args.no_clamp
    clamp_lo, clamp_hi = args.clamp_lo, args.clamp_hi

    def raw(self, x):
        outs = []
        for i in range(self.nl):
            box = self.one2one_cv2[i](x[i])   # [1, 4, H, W]
            cls = self.one2one_cv3[i](x[i])   # [1, nc, H, W]
            if not no_clamp:
                cls = torch.clamp(cls, clamp_lo, clamp_hi)
            outs.append(box)
            outs.append(cls)
        return tuple(outs)

    head.forward = types.MethodType(raw, head)

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        out = model(dummy)
    shapes = [tuple(o.shape) for o in out]
    assert len(shapes) == 2 * head.nl, f"输出数异常: {len(shapes)} != {2 * head.nl}"
    for s in shapes:
        print(f"  -> {s}")

    output_names = []
    for i in range(head.nl):
        output_names += [f"box_{i}", f"cls_{i}"]

    print(f"[export] {args.output} (opset={args.opset}, imgsz={args.imgsz}, "
          f"clamp={'off' if no_clamp else f'[{clamp_lo}, {clamp_hi}]'})")
    torch.onnx.export(
        model,
        (dummy,),
        args.output,
        opset_version=args.opset,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,
        dynamo=False,  # 传统 TorchScript 导出器，避免依赖 onnxscript
    )
    print("[ok] 导出完成")


if __name__ == "__main__":
    main()
