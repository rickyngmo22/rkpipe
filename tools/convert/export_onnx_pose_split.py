#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 YOLO26-Pose「拆分布局」原始头 ONNX（Pose26 one2one 头）。

背景：rk_pipe 板端 pose 后处理（yolov26_pose.cc）期望融合布局 3×[1,56,H,W]
（4 box + 1 cls + 51 kpt 共享 scale）——与 detect 塌缩前同构，cls/kpt-conf 的
极端负 logit 会撑爆共享 scale。本脚本按支路拆分输出张量，各张量独立 scale。

输出签名（--split4，默认，12 个输出）：
    box_i     : [1, 4,  H, W]   直接距离回归 (l,t,r,b)，网格单位，不 clamp
    cls_i     : [1, 1,  H, W]   分类 logit，clamp(-12, 8)（保序，FP32 语义无损）
    kpt_xy_i  : [1, 3K, H, W]   关键点 xy 原始值（RealNVP 流式头解码前），不 clamp
    kpt_cf_i  : [1, K,  H, W]   关键点可见性 logit，clamp(-12, 8)

--split3：每尺度 3 输出（box/cls/kpt 合并 51 通道）——kpt 内部 xy 与 conf 仍共享 scale，
仅用于对照「拆支路但 kpt 不细分」的效果。

用法（WSL onnx_env）：
  python export_onnx_pose_split.py --weights pose/yolo26n-pose.pt \
      --output onnx/yolo26n_pose_split12.onnx
"""
import argparse
import os
import types

import torch

from ultralytics.nn.modules.head import Pose


def main():
    parser = argparse.ArgumentParser(description="YOLO26-Pose 拆分导出")
    parser.add_argument("--weights", default="pose/yolo26n-pose.pt")
    parser.add_argument("--output", default="onnx/yolo26n_pose_split12.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--clamp-lo", type=float, default=-12.0)
    parser.add_argument("--clamp-hi", type=float, default=8.0)
    parser.add_argument("--split3", action="store_true",
                        help="每尺度 3 输出（kpt 51 通道不细分），默认 4 输出/尺度")
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
    assert isinstance(head, Pose), f"不是 Pose 头: {type(head).__name__}"
    assert hasattr(head, "one2one_cv2"), "模型无 one2one 分支，请确认是 end2end 的 YOLO26-pose"
    assert head.reg_max == 1, f"reg_max 应为 1，实际 {head.reg_max}"
    kpt_num = head.kpt_shape[0]
    print(f"[head] {type(head).__name__} | nl={head.nl} nc={head.nc} "
          f"kpt_shape={head.kpt_shape} reg_max={head.reg_max}")

    clamp_lo, clamp_hi = args.clamp_lo, args.clamp_hi
    split3 = args.split3

    def raw(self, x):
        outs = []
        for i in range(self.nl):
            box = self.one2one_cv2[i](x[i])   # [1, 4, H, W]
            cls = self.one2one_cv3[i](x[i])   # [1, 1, H, W]
            kpt = self.one2one_cv4[i](x[i])   # [1, 3K, H, W]（x,y,conf 交错）
            cls = torch.clamp(cls, clamp_lo, clamp_hi)
            if split3:
                outs += [box, cls, kpt]
            else:
                # kpt 通道序 [x,y,conf]*K：xy 占 ch 0..2K，conf 占 ch 2K..3K
                kpt_xy = kpt[:, : 2 * kpt_num]
                kpt_cf = torch.clamp(kpt[:, 2 * kpt_num:], clamp_lo, clamp_hi)
                outs += [box, cls, kpt_xy, kpt_cf]
        return tuple(outs)

    head.forward = types.MethodType(raw, head)

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        out = model(dummy)
    shapes = [tuple(o.shape) for o in out]
    per_scale = 3 if split3 else 4
    assert len(shapes) == per_scale * head.nl, f"输出数异常: {len(shapes)}"
    for s in shapes:
        print(f"  -> {s}")

    output_names = []
    for i in range(head.nl):
        if split3:
            output_names += [f"box_{i}", f"cls_{i}", f"kpt_{i}"]
        else:
            output_names += [f"box_{i}", f"cls_{i}", f"kpt_xy_{i}", f"kpt_cf_{i}"]

    print(f"[export] {args.output} (opset={args.opset}, imgsz={args.imgsz}, "
          f"clamp=[{clamp_lo}, {clamp_hi}], split={'3' if split3 else '4'})")
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
