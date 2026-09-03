#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COCO val2017 评测：消费 console_detector --dump-detections 导出的 JSONL。

自动识别导出类型：
  detect（bbox mAP）  : 行含 "dets":[{"bbox":[x,y,w,h],"score","cls"}]
  pose  （keypoint AP): 行含 "poses":[{"box","score","kpts":[[x,y,v,(conf)]x17]}]
  seg   （segm AP）   : 行含 "segs":[{"box","score","cls","rle":[counts...]}]
                        rle 为检测框内裁剪掩膜的行级行程编码，此处重组全帧后评测

用法：
  python3 eval_coco.py --ann annotations/instances_val2017.json \
                       --dump dump_yolo26n_int8.jsonl [--label coco_80_labels_list.txt]
"""

import argparse
import json
import os
import sys

import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 标准 80 类（label 文件顺序）→ COCO 官方 category_id（1-90 跳 12 个空位）
CONTIG80_TO_COCO = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def detect_dump_kind(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if "poses" in d:
                    return "kpts"
                if "segs" in d:
                    return "segm"
                return "bbox"
    raise SystemExit("空 dump 文件: %s" % path)


def load_dump(path):
    frames, n_items = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
            key = "poses" if "poses" in frames[-1] else ("segs" if "segs" in frames[-1] else "dets")
            n_items += len(frames[-1][key])
    return frames, n_items


def image_id_from_file(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem)  # COCO val2017 文件名 000000xxxxxx.jpg = image_id


def check_label_order(label_path, coco):
    if not label_path or not os.path.exists(label_path):
        return "label 文件未提供，按标准 COCO-80 顺序映射"
    official_all = [c["name"] for c in sorted(coco.dataset["categories"], key=lambda c: c["id"])
                    if c["id"] in CONTIG80_TO_COCO]
    if len(official_all) < 80:
        return "标注类别 %d 个（如 person_keypoints 文件），跳过 label 顺序校验" % len(official_all)
    names = [l.strip() for l in open(label_path) if l.strip()]
    if names == official_all:
        return "label 文件与 COCO-80 官方顺序一致 (80 类)"
    return ("⚠ label 文件顺序与 COCO-80 官方不一致，映射可能错位！\n"
            "  label[:5]=%s official[:5]=%s" % (names[:5], official_all[:5]))


def build_results_bbox(frames, coco_gt, conf):
    results = []
    for fr in frames:
        img_id = image_id_from_file(fr["file"])
        if img_id not in coco_gt.imgs:
            continue
        for d in fr["dets"]:
            if d["score"] < conf or not (0 <= d["cls"] < 80):
                continue
            x, y, w, h = d["bbox"]
            if w <= 0 or h <= 0:
                continue
            results.append({"image_id": img_id, "category_id": CONTIG80_TO_COCO[d["cls"]],
                            "bbox": [float(x), float(y), float(w), float(h)],
                            "score": float(d["score"])})
    return results


def build_results_kpts(frames, coco_gt, conf):
    results = []
    for fr in frames:
        img_id = image_id_from_file(fr["file"])
        if img_id not in coco_gt.imgs:
            continue
        for p in fr["poses"]:
            if p["score"] < conf:
                continue
            kpts = [tuple(k[:3]) for k in p["kpts"]]  # 丢弃第 4 位原始 conf
            if len(kpts) != 17:
                continue
            flat = [v for kpt in kpts for v in kpt]
            results.append({"image_id": img_id, "category_id": 1,
                            "keypoints": flat, "score": float(p["score"])})
    return results


def decode_box_rle(counts, box, frame_w, frame_h):
    """框内行级 RLE → 全帧二值掩膜 (uint8 HxW)。"""
    x, y, w, h = box
    w = max(1, int(w)); h = max(1, int(h))
    crop = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val:
            crop[pos:pos + c] = 1
        pos += c
        val ^= 1
    if pos < h * w:  # 尾部补齐
        pass
    m = crop.reshape(h, w)
    full = np.zeros((frame_h, frame_w), dtype=np.uint8)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(frame_w, x0 + w), min(frame_h, y0 + h)
    if x1 > x0 and y1 > y0:
        full[y0:y1, x0:x1] = m[:y1 - y0, :x1 - x0]
    return full


def build_results_segm(frames, coco_gt, conf):
    results, skipped = [], 0
    for fr in frames:
        img_id = image_id_from_file(fr["file"])
        if img_id not in coco_gt.imgs:
            continue
        W, H = fr["width"], fr["height"]
        for s in fr["segs"]:
            if s["score"] < conf or not (0 <= s["cls"] < 80):
                continue
            if not s.get("rle"):
                skipped += 1
                continue
            m = decode_box_rle(s["rle"], s["bbox"], W, H)
            if not m.any():
                skipped += 1
                continue
            rle = maskUtils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")
            results.append({"image_id": img_id, "category_id": CONTIG80_TO_COCO[s["cls"]],
                            "segmentation": rle, "score": float(s["score"])})
    if skipped:
        print("(跳过空掩膜 %d 条)" % skipped)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="instances_val2017.json")
    ap.add_argument("--dump", required=True, help="--dump-detections 导出的 JSONL")
    ap.add_argument("--label", default=None, help="coco_80_labels_list.txt（顺序校验用）")
    ap.add_argument("--max-dets", type=int, default=None, help="默认 bbox/segm=100, kpts=20")
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--out", default=None, help="结果摘要 JSON 输出路径")
    args = ap.parse_args()

    kind = detect_dump_kind(args.dump)
    coco_gt = COCO(args.ann)
    if kind != "bbox":
        print(check_label_order(args.label, coco_gt))

    frames, n_items = load_dump(args.dump)
    builder = {"bbox": build_results_bbox, "kpts": build_results_kpts, "segm": build_results_segm}[kind]
    results = builder(frames, coco_gt, args.conf)

    dumped_ids = {image_id_from_file(fr["file"]) for fr in frames}
    print("类型: %s | 覆盖率: %d/%d 张 val 图 (%.1f%%) | 导出 %d -> 评测 %d 条" %
          (kind, len(dumped_ids), len(coco_gt.imgs),
           100.0 * len(dumped_ids) / len(coco_gt.imgs), n_items, len(results)))
    if not results:
        print("没有任何结果，退出")
        return 1

    coco_dt = coco_gt.loadRes(results)
    iou_type = {"bbox": "bbox", "kpts": "keypoints", "segm": "segm"}[kind]
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.params.imgIds = sorted(dumped_ids)  # 只评导出过的图
    max_dets = args.max_dets or (20 if kind == "kpts" else 100)
    ev.params.maxDets = [1, 10, max_dets]
    if kind == "kpts":
        ev.params.catIds = [1]  # 关键点评测只对 person
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    s = ev.stats
    # 逐类 AP：直接从已 accumulate 的 precision 张量提取（[T,R,K,A,M]），不重跑评测
    per_class = {}
    prec = ev.eval["precision"]
    for k, cid in enumerate(ev.params.catIds):
        name = coco_gt.cats[cid]["name"]
        p_all = prec[:, :, k, 0, -1]
        p_all = p_all[p_all > -1]
        p50 = prec[0, :, k, 0, -1]
        p50 = p50[p50 > -1]
        per_class[str(cid)] = {
            "name": name,
            "AP": float(np.mean(p_all)) if p_all.size else 0.0,
            "AP50": float(np.mean(p50)) if p50.size else 0.0,
        }

    summary = {"kind": kind, "dump": args.dump,
               "images_evaluated": len(dumped_ids), "images_total_val": len(coco_gt.imgs),
               "results": len(results), "max_dets": max_dets,
               "AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
               "AR_max": float(s[8]), "per_class": per_class}
    if len(s) >= 12:  # bbox/segm 有 small/medium/large 分档；keypoints 无
        summary.update({"AP_small": float(s[3]), "AP_medium": float(s[4]), "AP_large": float(s[5]),
                        "AR_1": float(s[6]), "AR_10": float(s[7]),
                        "AR_small": float(s[9]), "AR_medium": float(s[10]), "AR_large": float(s[11])})
    else:  # keypoints: AP, AP50, AP75, AP_M, AP_L, AR, AR@10, AR@20, AR_M, AR_L
        summary.update({"AP_medium": float(s[3]), "AP_large": float(s[4]),
                        "AR_10": float(s[6]), "AR_20": float(s[7]),
                        "AR_medium": float(s[8]), "AR_large": float(s[9])})
    print("\n===== 摘要 =====")
    print("mAP@[.5:.95] = %.4f   mAP50 = %.4f   mAP75 = %.4f" % (s[0], s[1], s[2]))
    if len(s) >= 12:
        print("AP_S=%.4f AP_M=%.4f AP_L=%.4f   AR@maxDets=%d: %.4f" %
              (s[3], s[4], s[5], max_dets, s[8]))
    else:  # keypoints 分档为 medium/large
        print("AP_M=%.4f AP_L=%.4f   AR@maxDets=%d: %.4f" %
              (s[3], s[4], max_dets, s[7]))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print("摘要已写入", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
