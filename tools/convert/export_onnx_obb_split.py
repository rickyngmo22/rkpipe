#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 YOLO26-OBB「拆分布局」原始头 ONNX（one2one 头）。

输出签名（6 个输出，boxangle_i / cls_i 交替）：
    boxangle_i : [1, 5,  H, W]   4 box 距离回归(l,t,r,b) + 1 angle(弧度) —— 值域接近，共享 scale 无损
    cls_i      : [1, nc, H, W]   分类 logits（默认 clamp 到 [--clamp-lo, --clamp-hi]）

为什么拆分：
  融合布局 [1, 4+1+nc, H, W] 的输出张量共享量化 scale，15 类 DOTA 的 cls logits 极端负值
  把 box/angle 一起撑爆（历史实测：融合 INT8 全部格子报同一类同一分，无判别力）。
  拆分后 cls 独占 scale，box+angle 值域接近（都是 -1~数十网格单位）合在一起不互相撑。

运行环境：WSL onnx_env（ultralytics）
用法：
  python export_onnx_obb_split.py                                # 默认 weights 不存在 obb pt，需 --weights
  python export_onnx_obb_split.py --weights <yolo26n-obb.pt> --output onnx/yolo26n-obb_split6.onnx
"""
import argparse
import os
import types

import torch

from ultralytics.nn.modules.head import Detect, OBB26


def main():
    parser = argparse.ArgumentParser(description="YOLO26-OBB box+angle/cls 拆分导出")
    parser.add_argument("--weights", required=True, help="未 fuse 的 yolo26*-obb.pt checkpoint")
    parser.add_argument("--output", default="onnx/yolo26n-obb_split6.onnx")
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
    assert isinstance(head, OBB26), f"不是 OBB26 头: {type(head).__name__}"
    assert hasattr(head, "one2one_cv2"), "模型无 one2one 分支"
    assert head.reg_max == 1, f"reg_max 应为 1（直接距离回归），实际 {head.reg_max}"
    print(f"[head] {type(head).__name__} | nl={head.nl} nc={head.nc} reg_max={head.reg_max}")

    no_clamp = args.no_clamp
    clamp_lo, clamp_hi = args.clamp_lo, args.clamp_hi

    def raw(self, x):
        outs = []
        for i in range(self.nl):
            box = self.one2one_cv2[i](x[i])     # [1, 4, H, W]
            angle = self.one2one_cv4[i](x[i])   # [1, 1, H, W]
            cls = self.one2one_cv3[i](x[i])     # [1, nc, H, W]
            if not no_clamp:
                cls = torch.clamp(cls, clamp_lo, clamp_hi)
            ba = torch.cat([box, angle], dim=1)  # [1, 5, H, W]
            outs.append(ba)
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
        output_names += [f"boxangle_{i}", f"cls_{i}"]

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
        dynamo=False,
    )
    print("[ok] 导出完成")


if __name__ == "__main__":
    main()
