#!/usr/bin/env python3
"""
导出 YOLO26 各任务的"原始头" ONNX（end2end=False，只走 one2one 头，无 decode/NMS/TopK）。

任务与输出签名（per-scale 为 one2one_cv2/3/4 原始卷积输出，cat 拼接）：
    detect: 3 × [1, 4+nc,   H, W]   4 box + nc cls(logits)
    pose  : 3 × [1, 4+1+51, H, W]   4 box + 1 score + 51 kpt(17×3)
    obb   : 3 × [1, 4+1+nc, H, W]   4 box + 1 angle + nc cls（按方案顺序 box,angle,cls）
    seg   : 3 × [1, 4+nc+32,H, W] + [1, 32, 160, 160] proto
            (4 box + nc cls + 32 mask)

运行环境：onnx_env
用法：
    python export_onnx_raw.py                          # 默认 weights/yolo26n.pt (detect)
    python export_onnx_raw.py --weights weights/yolo26n-pose.pt --output onnx/yolo26n-pose_e2e_false.onnx
"""
import argparse
import os
import types

import torch

from ultralytics.nn.modules.head import Detect, OBB26, Pose26, Segment26, SemanticSegment


def make_raw_forward(head):
    """按头类型返回 (raw_forward, n_outputs)。"""
    if isinstance(head, SemanticSegment):
        def raw(self, x):
            # 语义分割：单输出 [B, nc, H/8, W/8] 逐像素类别 logits（不上采样、不 argmax）
            return (self.classifier(x[0]),)

        return raw, 1

    if isinstance(head, Segment26):
        def raw(self, x):
            outs = []
            for i in range(self.nl):
                box = self.one2one_cv2[i](x[i])
                cls = self.one2one_cv3[i](x[i])
                msk = self.one2one_cv4[i](x[i])
                outs.append(torch.cat([box, cls, msk], dim=1))
            proto = self.proto(x)
            if isinstance(proto, tuple):
                proto = proto[0]
            return tuple(outs) + (proto,)

        return raw, 4

    if isinstance(head, Pose26):
        def raw(self, x):
            outs = []
            for i in range(self.nl):
                box = self.one2one_cv2[i](x[i])
                score = self.one2one_cv3[i](x[i])
                feat = self.one2one_cv4[i](x[i])
                kpts = self.one2one_cv4_kpts[i](feat)
                outs.append(torch.cat([box, score, kpts], dim=1))
            return tuple(outs)

        return raw, 3

    if isinstance(head, OBB26):
        def raw(self, x):
            outs = []
            for i in range(self.nl):
                box = self.one2one_cv2[i](x[i])
                angle = self.one2one_cv4[i](x[i])
                cls = self.one2one_cv3[i](x[i])
                outs.append(torch.cat([box, angle, cls], dim=1))
            return tuple(outs)

        return raw, 3

    # detect / 其余 Detect 派生头
    def raw(self, x):
        outs = []
        for i in range(self.nl):
            box = self.one2one_cv2[i](x[i])
            cls = self.one2one_cv3[i](x[i])
            outs.append(torch.cat([box, cls], dim=1))
        return tuple(outs)

    return raw, 3


def main():
    parser = argparse.ArgumentParser(description="导出 YOLO26 原始头 ONNX（全任务）")
    parser.add_argument("--weights", default="weights/yolo26n.pt")
    parser.add_argument("--output", default="onnx/yolo26n_e2e_false.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
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
    if isinstance(head, SemanticSegment):
        # 语义分割头：无 box/one2one/reg_max，单独分支
        print(f"[head] {type(head).__name__} | nl={head.nl} nc={head.nc}")
    else:
        assert isinstance(head, Detect), f"不是 Detect 头: {type(head).__name__}"
        assert hasattr(head, "one2one_cv2"), "模型无 one2one 分支，请确认是 end2end=True 的 YOLO26 模型"
        assert head.reg_max == 1, f"reg_max 应为 1（直接距离回归），实际 {head.reg_max}"
        print(f"[head] {type(head).__name__} | nl={head.nl} nc={head.nc} reg_max={head.reg_max}")

    raw_forward, n_out = make_raw_forward(head)
    head.forward = types.MethodType(raw_forward, head)

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        out = model(dummy)
    shapes = [tuple(o.shape) for o in out]
    assert len(shapes) == n_out, f"输出数异常: {len(shapes)} != {n_out}"
    for s in shapes:
        print(f"  -> {s}")

    output_names = [f"output{i}" for i in range(n_out)]
    print(f"[export] {args.output} (opset={args.opset}, imgsz={args.imgsz}, outputs={n_out})")
    torch.onnx.export(
        model,
        (dummy,),
        args.output,
        opset_version=args.opset,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,
        dynamo=False,  # 用传统 TorchScript 导出器，避免依赖 onnxscript
    )
    print("[ok] 导出完成")


if __name__ == "__main__":
    main()
