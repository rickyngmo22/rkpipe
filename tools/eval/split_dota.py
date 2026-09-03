#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DOTA v1.0 val 切片（官方协议 1024×1024 / gap 200）+ GT 裁剪。

用法：
  python3 split_dota.py --images val_images --labels val_labelTxt \
                        --out patches --gt patches_gt.jsonl [--subsize 1024] [--gap 200] [--keep 0.5]

输出：
  patches/xxx_{x}_{y}.jpg         切片图（JPG q95）
  patches_gt.jsonl                每切片一行 {"patch","width","height",
                                   "objects":[{"poly":[x1,y1,...x4,y4],"cls","difficult"}]}
                                   多边形已裁剪到切片矩形；保留与切片相交面积 >= keep 的目标
"""
import argparse
import json
import os

import cv2
import numpy as np

DOTA_CLASSES = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
                "basketball-court", "ground-track-field", "harbor", "bridge",
                "large-vehicle", "small-vehicle", "helicopter", "roundabout",
                "soccer-ball-field", "swimming-pool"]


def parse_label(path):
    objs = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]
    for l in lines:
        parts = l.split()
        if len(parts) < 9:
            continue
        try:
            coords = list(map(float, parts[:8]))
        except ValueError:
            continue  # 头两行（images/sources）非数字
        name = parts[8]
        diff = int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else 0
        if name not in DOTA_CLASSES:
            continue
        objs.append({"poly": coords, "cls": name, "difficult": diff})
    return objs


def clip_poly(obj_poly, rect):
    """目标多边形与切片矩形求交（cv2 凸多边形相交），返回裁剪后多边形和交面积。"""
    rect_poly = np.array([[[rect[0], rect[1]]], [[rect[2], rect[1]]],
                          [[rect[2], rect[3]]], [[rect[0], rect[3]]]], np.float32)
    obj = np.array(obj_poly, np.float32).reshape(4, 1, 2)
    inter_area, inter_pts = cv2.intersectConvexConvex(rect_poly, obj)
    if inter_pts is None or len(inter_pts) < 3 or inter_area <= 0:
        return None, 0.0
    return inter_pts.reshape(-1, 2), float(inter_area)


def poly_area(pts):
    return cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.float32)) if len(pts) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--subsize", type=int, default=1024)
    ap.add_argument("--gap", type=int, default=200)
    ap.add_argument("--keep", type=float, default=0.5, help="目标相交面积占比 >= keep 才保留")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    step = args.subsize - args.gap
    gt_lines, n_patches = [], 0

    names = sorted(f for f in os.listdir(args.images) if f.lower().endswith((".png", ".jpg")))
    for idx, name in enumerate(names):
        stem = os.path.splitext(name)[0]
        img = cv2.imread(os.path.join(args.images, name))
        if img is None:
            print("skip unreadable", name)
            continue
        H, W = img.shape[:2]
        label = os.path.join(args.labels, stem + ".txt")
        objs = parse_label(label) if os.path.exists(label) else []

        # 与官方 ImgSplit 对齐：滑动窗口步长 step，边缘贴齐
        xs = list(range(0, max(W - args.subsize, 0) + 1, step))
        ys = list(range(0, max(H - args.subsize, 0) + 1, step))
        if xs[-1] + args.subsize < W:
            xs.append(W - args.subsize)
        if ys[-1] + args.subsize < H:
            ys.append(H - args.subsize)

        for x0 in xs:
            for y0 in ys:
                rect = (x0, y0, x0 + args.subsize, y0 + args.subsize)
                keep_objs = []
                for o in objs:
                    pts, inter = clip_poly(o["poly"], rect)
                    if pts is None:
                        continue
                    area_full = poly_area(np.array(o["poly"], np.float32).reshape(4, 1, 2))
                    if area_full > 0 and inter / area_full >= args.keep:
                        # 裁剪交点在原图坐标系，需平移到切片局部坐标系
                        local = pts - np.array([x0, y0], np.float32)
                        keep_objs.append({"poly": [float(v) for v in local.reshape(-1)],
                                          "cls": o["cls"], "difficult": o["difficult"]})
                patch_name = "%s_%d_%d.jpg" % (stem, x0, y0)
                cv2.imwrite(os.path.join(args.out, patch_name), img[y0:y0 + args.subsize, x0:x0 + args.subsize],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                gt_lines.append(json.dumps({"patch": patch_name, "width": args.subsize,
                                            "height": args.subsize, "objects": keep_objs}))
                n_patches += 1
        if (idx + 1) % 50 == 0:
            print("[%d/%d] patches=%d" % (idx + 1, len(names), n_patches))

    with open(args.gt, "w") as f:
        f.write("\n".join(gt_lines) + "\n")
    print("完成: %d 原图 -> %d 切片, GT -> %s" % (len(names), n_patches, args.gt))


if __name__ == "__main__":
    main()
