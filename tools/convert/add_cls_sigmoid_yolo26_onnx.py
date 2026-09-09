#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo26 各任务 ONNX 图手术：cls 通道入图 sigmoid（方案2）
=========================================================
目的：yolo26 输出原始 cls logits（可低至 -60~-80），INT8 量化 scale 被极端负值拉粗，
导致分数塌缩。v8/v11 免疫是因为它们的 cls 在图上已 sigmoid、输出 [0,1]。
本脚本把各任务输出的 cls 通道区间接上 Sigmoid，其余通道原样，Concat 后输出形状/名字不变。

通用三段式：box(原样) + sigmoid(cls) + tail(原样, 如有)
  detect: [1,84,H,W]   cls=[4:84]             (box=ch0:4)
  pose:   [1,56,H,W]   cls=[4:5]  tail=kpt     (box=ch0:4, kpt=ch5:56)
  obb:    [1,20,H,W]   cls=[5:20]             (box+angle=ch0:5)
  seg:    [1,116,H,W]  cls=[4:84] tail=mask    (box=ch0:4, mask=ch84:116)
          + proto[1,32,160,160] 自动跳过（通道数不足 cls 区间）
  depth:  单输出 [1,1,640,640]，无 cls，不适用

用法（PC 端，rknn-toolkit2 转换环境）:
  python3 add_cls_sigmoid_yolo26_onnx.py yolo26n_pose.onnx out.onnx --task pose
  python3 add_cls_sigmoid_yolo26_onnx.py yolo26n.onnx out.onnx --task detect

前提:
  - onnx >= 1.13
  - 输入 ONNX 为非 end2end（end2end=False）导出
"""
import argparse
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

INT_MAX = np.iinfo(np.int64).max

# 任务预设：cls 通道起始与类别数（tail 通道自动 = 总通道 - cls_start - class_count）
TASK_CFG = {
    "detect": dict(cls_start=4, class_count=80),
    "pose":   dict(cls_start=4, class_count=1),
    "obb":    dict(cls_start=5, class_count=15),
    "seg":    dict(cls_start=4, class_count=80),
}


def unique_name(model, base):
    names = set()
    for n in model.graph.node:
        names.update(list(n.input) + list(n.output))
    for vi in model.graph.value_info:
        names.add(vi.name)
    for i in model.graph.input:
        names.add(i.name)
    for o in model.graph.output:
        names.add(o.name)
    for ini in model.graph.initializer:
        names.add(ini.name)
    name, k = base, 0
    while name in names:
        k += 1
        name = f"{base}_{k}"
    return name


def output_channels(model, tensor_name):
    """从 value_info 取张量的通道数（NCHW dims[1]），取不到返回 0"""
    for vi in model.graph.value_info:
        if vi.name == tensor_name and vi.type.HasField("tensor_type"):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if len(dims) >= 2 and dims[1] > 0:
                return dims[1]
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_onnx")
    ap.add_argument("out_onnx")
    ap.add_argument("--task", default="detect", choices=list(TASK_CFG.keys()))
    ap.add_argument("--cls-start", type=int, default=None, help="覆盖任务预设的 cls 起始通道")
    ap.add_argument("--class-count", type=int, default=None, help="覆盖任务预设的类别数")
    args = ap.parse_args()

    cfg = TASK_CFG[args.task]
    cls_start = args.cls_start if args.cls_start is not None else cfg["cls_start"]
    class_count = args.class_count if args.class_count is not None else cfg["class_count"]
    print(f"[cfg] task={args.task} cls_start={cls_start} class_count={class_count}")

    model = onnx.load(args.in_onnx)
    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph

    outputs_processed = 0
    for out in list(g.output):
        oname = out.name

        # 定位产出该输出的节点
        producer = None
        for n in g.node:
            if oname in n.output:
                producer = n
                break
        if producer is None:
            print(f"[warn] {oname}: 找不到产出节点，跳过")
            continue
        # oname 若被其它节点消费，重命名产出会断链
        consumers = [n for n in g.node if oname in n.input]
        if consumers:
            print(f"[warn] {oname}: 还被 {len(consumers)} 个节点消费，跳过")
            continue

        channels = output_channels(model, oname)
        if channels < cls_start + class_count:
            print(f"[skip] {oname}: 通道数 {channels} < cls 区间 {cls_start}+{class_count}"
                  f"（seg 的 proto / depth 输出，无需处理）")
            continue
        tail_len = channels - cls_start - class_count
        print(f"[OK] {oname}: channels={channels} -> box[0:{cls_start}] + sigmoid(cls[{cls_start}:"
              f"{cls_start+class_count}]) + tail[{cls_start+class_count}:{channels}]")

        raw_name = unique_name(model, f"{oname}_raw")
        box_name = unique_name(model, f"{oname}_box")
        cls_name = unique_name(model, f"{oname}_cls")
        clssig_name = unique_name(model, f"{oname}_cls_sigmoid")

        # 1) 重命名产出节点输出为 raw（原输出名保留给 Concat）
        producer.output[producer.output.index(oname)] = raw_name

        # 2) Slice 常量（NCHW，axis=1 为通道）
        c = lambda base, arr: numpy_helper.from_array(
            np.array(arr, dtype=np.int64), name=unique_name(model, base))
        starts0 = c("s0", [0, 0, 0, 0])
        ends_box = c("eb", [INT_MAX, cls_start, INT_MAX, INT_MAX])
        ends_cls = c("ec", [INT_MAX, cls_start + class_count, INT_MAX, INT_MAX])
        ends_all = c("ea", [INT_MAX, INT_MAX, INT_MAX, INT_MAX])
        axes = c("ax", [0, 1, 2, 3])
        for ini in (starts0, ends_box, ends_cls, ends_all, axes):
            g.initializer.append(ini)

        g.node.append(helper.make_node(
            "Slice", inputs=[raw_name, starts0.name, ends_box.name, axes.name],
            outputs=[box_name], name=unique_name(model, f"{oname}_slice_box")))
        g.node.append(helper.make_node(
            "Slice", inputs=[raw_name, starts0.name, ends_cls.name, axes.name],
            outputs=[cls_name], name=unique_name(model, f"{oname}_slice_cls")))
        g.node.append(helper.make_node(
            "Sigmoid", inputs=[cls_name], outputs=[clssig_name],
            name=unique_name(model, f"{oname}_sigmoid")))

        concat_inputs = [box_name, clssig_name]
        if tail_len > 0:
            tail_name = unique_name(model, f"{oname}_tail")
            g.node.append(helper.make_node(
                "Slice", inputs=[raw_name, ends_cls.name, ends_all.name, axes.name],
                outputs=[tail_name], name=unique_name(model, f"{oname}_slice_tail")))
            concat_inputs.append(tail_name)

        g.node.append(helper.make_node(
            "Concat", inputs=concat_inputs, outputs=[oname],
            name=unique_name(model, f"{oname}_concat"), axis=1))
        outputs_processed += 1

    assert outputs_processed >= 1, "没有处理任何输出，请检查 ONNX 是否为 yolo26 非 end2end 导出"
    onnx.checker.check_model(model)
    onnx.save(model, args.out_onnx)
    print(f"[OK] saved {args.out_onnx}，共处理 {outputs_processed} 个输出")


if __name__ == "__main__":
    main()
