#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DOTA v1.0 val OBB 评测：旋转 IoU 匹配 + COCO 风格 AP。

GT：split_dota.py 产出的 patches_gt.jsonl（裁剪到切片的多边形）。
预测：console_detector --task obb --dump-detections 导出的 JSONL，
      box=[x,y,w,h,angle_rad]（左上角+宽高+弧度，原图=切片坐标）。

用法：
  python3 eval_dota.py --gt patches_gt.jsonl --dump dump_obb_int8.jsonl [--out summary.json]

口径：IoU 阈值 0.5:0.95:0.05 共 10 档，101 点插值 AP（COCO 风格）；difficult 目标按
忽略处理（命中 difficult 的预测不计 FP）。另报 AP50（与 DOTA 官方 VOC AP50 口径接近）。
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

DOTA_CLASSES = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
                "basketball-court", "ground-track-field", "harbor", "bridge",
                "large-vehicle", "small-vehicle", "helicopter", "roundabout",
                "soccer-ball-field", "swimming-pool"]


def box_to_poly(box):
    x, y, w, h, ang = box
    cx, cy = x + w / 2.0, y + h / 2.0
    c, s = math.cos(ang), math.sin(ang)
    dx, dy = w / 2.0, h / 2.0
    pts = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return np.array([[cx + px * c - py * s, cy + px * s + py * c] for px, py in pts], np.float32)


def poly_iou(p1, p2):
    inter_area, _ = cv2.intersectConvexConvex(p1.reshape(-1, 1, 2).astype(np.float32),
                                              p2.reshape(-1, 1, 2).astype(np.float32))
    if inter_area <= 0:
        return 0.0
    a1 = cv2.contourArea(p1.reshape(-1, 1, 2).astype(np.float32))
    a2 = cv2.contourArea(p2.reshape(-1, 1, 2).astype(np.float32))
    u = a1 + a2 - inter_area
    return inter_area / u if u > 0 else 0.0


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--conf", type=float, default=0.0, help="评测前按分数过滤预测")
    ap.add_argument("--iou-50-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gts = {g["patch"]: g for g in load_jsonl(args.gt)}
    dets = load_jsonl(args.dump)

    # 组装: cls -> list[(score, patch, poly)]；GT: cls -> patch -> {poly, difficult}
    preds = {c: [] for c in DOTA_CLASSES}
    gt_map = {c: {} for c in DOTA_CLASSES}
    n_gt = 0
    for patch, g in gts.items():
        for o in g["objects"]:
            poly = np.array(o["poly"], np.float32).reshape(-1, 2)
            if len(poly) < 3:
                continue
            gt_map[o["cls"]].setdefault(patch, []).append({"poly": poly, "difficult": o["difficult"]})
            if not o["difficult"]:
                n_gt += 1

    n_pred = 0
    for fr in dets:
        patch = os.path.basename(fr["file"])
        if patch not in gts:
            continue
        for d in fr.get("obbs", []):
            if d["score"] < args.conf:
                continue
            poly = box_to_poly(d["box"])
            if cv2.contourArea(poly.reshape(-1, 1, 2).astype(np.float32)) <= 1:
                continue
            preds[DOTA_CLASSES[d["cls"]]].append((float(d["score"]), patch, poly))
            n_pred += 1

    print("GT 目标(非difficult): %d | 预测: %d | 切片: %d" % (n_gt, n_pred, len(gts)))

    ious = [0.5] if args.iou_50_only else [0.5 + 0.05 * i for i in range(10)]
    ap50_per_cls = {}
    ap_all = []   # 各 IoU 档的各类 AP
    for thr_i, thr in enumerate(ious):
        aps = []
        for c in DOTA_CLASSES:
            gts_c = gt_map[c]
            # 标记匹配状态
            matched = {p: [False] * len(o) for p, o in gts_c.items()}
            difficult = {p: [o["difficult"] for o in gts_c[p]] for p in gts_c}
            ps = sorted(preds[c], key=lambda x: -x[0])
            tps, fps = [], []
            for score, patch, poly in ps:
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gts_c.get(patch, [])):
                    iou = poly_iou(poly, g["poly"])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_j >= 0 and best_iou >= thr:
                    if difficult.get(patch, [])[best_j]:
                        continue  # 忽略 difficult，不计 TP/FP
                    if not matched[patch][best_j]:
                        matched[patch][best_j] = True
                        tps.append(1)
                        fps.append(0)
                        continue
                tps.append(0)
                fps.append(1)
            # 全局按 score 已排序，逐类累计 PR
            npos = sum(1 for p, o in gts_c.items() for o in o if not o["difficult"])
            if npos == 0:
                aps.append(float("nan"))
                continue
            ctps, cfps = np.cumsum(tps), np.cumsum(fps)
            recall = ctps / npos
            precision = ctps / np.maximum(ctps + cfps, 1e-9)
            # 101 点插值
            q = np.zeros(101)
            for k, r in enumerate(np.linspace(0, 1, 101)):
                mask = recall >= r
                q[k] = precision[mask].max() if mask.any() else 0.0
            aps.append(q.mean())
            if thr == 0.5:
                ap50_per_cls[c] = aps[-1]
        ap_all.append(np.nanmean(aps))

    print("\n===== 每类 AP50 =====")
    for c in DOTA_CLASSES:
        print("  %-20s %.4f" % (c, ap50_per_cls.get(c, float("nan"))))
    print("\n===== 摘要 =====")
    print("mAP50      = %.4f" % ap_all[0])
    if len(ap_all) > 1:
        print("mAP50:0.95 = %.4f" % float(np.mean(ap_all)))
    if args.out:
        # kind 与 eval_coco.py 的 summary 对齐，历史对比页（compare_runs）
        # 靠它按任务类型分 tab，缺失会导致 obb 不出现在对比页
        out = {"kind": "obb",
               "mAP50": float(ap_all[0]),
               "mAP50_95": float(np.mean(ap_all)) if len(ap_all) > 1 else None,
               "AP50_per_class": ap50_per_cls, "n_gt": n_gt, "n_pred": n_pred}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print("摘要已写入", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

