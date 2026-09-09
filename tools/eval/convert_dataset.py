#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert_dataset.py —— 自定义数据集注册/转换/校验（V1: yolo / voc / coco）。

统一中间格式 = COCO JSON（类别 id 连续 1..N，与模型 label 文件顺序一致），
评测器 eval_coco.py 自动按"连续直映射"评测，无需感知来源格式。

用法（推荐 spec 文件，与 Web 端同一 schema）：
  python3 convert_dataset.py --spec datasets/my_defect.yaml --vis 20

或直接参数：
  python3 convert_dataset.py --format yolo --images val/images --labels val/labels \
      --names data.yaml --out datasets/my_defect.coco.json --vis 20

spec schema：
  name: my_defect          # 数据集名（产物命名用）
  type: detect             # detect（obb/seg/pose 计划中）
  format: yolo             # yolo | voc | coco
  images: /data/x/val/images
  labels: /data/x/val/labels   # yolo: txt 目录；voc: xml 目录；coco: 留空
  names: /data/x/data.yaml     # 类别表（yaml 文件带 names: 或直接 list）；coco 可省略
  label_file: model/x_labels_list.txt  # 可选：模型 label 文件，做顺序一致性校验

输出：
  <out>                    COCO JSON（连续 1..N）
  <out>.check.json         校验报告（图片数/标注数/类别分布/漏标/非法框）
  <out>_preview/           随机 N 张 GT 叠加图（--vis N）
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

# 与 rknn_eval 可视化一致的 16 色调色板（BGR）
PALETTE = [(60, 76, 231), (28, 185, 69), (180, 51, 59), (163, 255, 10),
           (0, 160, 255), (255, 0, 121), (226, 43, 138), (255, 191, 0),
           (0, 215, 255), (128, 0, 128), (203, 192, 255), (50, 205, 50),
           (255, 165, 0), (255, 0, 255), (30, 105, 210), (222, 196, 176)]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))


def load_names(spec):
    """names 来源解析：yaml 文件（data.yaml 的 names:）/ 内联 list / label txt。"""
    names = spec.get("names")
    if names is None:
        return None
    if isinstance(names, list):
        return [str(x) for x in names]
    # 文件路径
    p = names
    if not os.path.exists(p):
        raise SystemExit("names 文件不存在: %s" % p)
    if p.endswith((".yaml", ".yml")):
        if yaml is None:
            raise SystemExit("需要 pyyaml 读取 %s（pip install pyyaml）" % p)
        d = yaml.safe_load(open(p, encoding="utf-8"))
        n = d.get("names")
        if isinstance(n, dict):  # ultralytics 新版: names: {0: person, ...}
            return [str(n[k]) for k in sorted(n)]
        if isinstance(n, list):
            return [str(x) for x in n]
        raise SystemExit("data.yaml 中未找到 names 列表")
    # 纯文本（每行一类，同模型 labels_list）
    return [l.strip() for l in open(p, encoding="utf-8", errors="ignore") if l.strip()]


def conv_yolo(spec, names):
    """YOLO txt（cls cx cy w h 归一化）→ COCO。"""
    img_dir, lbl_dir = spec["images"], spec["labels"]
    files = list_images(img_dir)
    images, annotations, bad = [], [], 0
    ann_id = 1
    for i, f in enumerate(files):
        stem = os.path.splitext(f)[0]
        path = os.path.join(img_dir, f)
        img = cv2.imread(path)
        if img is None:
            print("⚠ 无法读取，跳过:", f)
            continue
        H, W = img.shape[:2]
        images.append({"id": i + 1, "file_name": f, "width": W, "height": H})
        txt = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(txt):
            continue  # 漏标图：合法但计入 check
        for line in open(txt):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            except ValueError:
                bad += 1
                continue
            if not (0 <= c < len(names)):
                bad += 1
                continue
            bw, bh = w * W, h * H
            bx, by = cx * W - bw / 2.0, cy * H - bh / 2.0
            if bw <= 0 or bh <= 0:
                bad += 1
                continue
            annotations.append({"id": ann_id, "image_id": i + 1, "category_id": c + 1,
                                "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)],
                                "area": round(bw * bh, 2), "iscrowd": 0, "ignore": 0})
            ann_id += 1
    return images, annotations, bad


def conv_voc(spec, names):
    """VOC XML 目录 → COCO。"""
    import xml.etree.ElementTree as ET
    img_dir, lbl_dir = spec["images"], spec["labels"]
    files = list_images(img_dir)
    images, annotations, bad = [], [], 0
    ann_id = 1
    for i, f in enumerate(files):
        stem = os.path.splitext(f)[0]
        path = os.path.join(img_dir, f)
        img = cv2.imread(path)
        if img is None:
            print("⚠ 无法读取，跳过:", f)
            continue
        H, W = img.shape[:2]
        images.append({"id": i + 1, "file_name": f, "width": W, "height": H})
        xml = os.path.join(lbl_dir, stem + ".xml")
        if not os.path.exists(xml):
            continue
        root = ET.parse(xml).getroot()
        for obj in root.iter("object"):
            name = obj.findtext("name", "").strip()
            bb = obj.find("bndbox")
            if name not in names or bb is None:
                bad += 1
                continue
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                bad += 1
                continue
            annotations.append({"id": ann_id, "image_id": i + 1,
                                "category_id": names.index(name) + 1,
                                "bbox": [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                                "area": round(bw * bh, 2), "iscrowd": 0, "ignore": 0})
            ann_id += 1
    return images, annotations, bad


def check_coco(spec):
    """coco 直通校验：返回 (dataset, bad_count)。ids 必须连续 1..N 或 COCO-80 空洞。"""
    p = spec["labels"] or spec.get("ann")
    if not p or not os.path.exists(p):
        raise SystemExit("coco 格式需在 spec.labels（或 ann）提供标注 JSON 路径")
    d = json.load(open(p))
    ids = sorted(c["id"] for c in d.get("categories", []))
    n = len(ids)
    contiguous = ids == list(range(1, n + 1))
    from_coco80 = ids == [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    if not (contiguous or from_coco80):
        raise SystemExit("标注 category ids 非连续 1..%d 也非 COCO-80（%s...），"
                         "评测无法自动映射，请修正标注或重导出" % (n, ids[:6]))
    # 归一化成连续 id（COCO-80 空洞场景重映射，自定义连续场景原样）
    if from_coco80:
        remap = {old: new + 1 for new, old in enumerate(
            sorted(c["id"] for c in d["categories"]))}
        for c in d["categories"]:
            c["id"] = remap[c["id"]]
        for a in d.get("annotations", []):
            a["category_id"] = remap[a["category_id"]]
    # bbox 合法性
    bad = 0
    for a in d.get("annotations", []):
        x, y, w, h = a.get("bbox", [0, 0, 0, 0])
        if w <= 0 or h <= 0:
            bad += 1
    return d, bad


def conv_labelme(spec, names=None):
    """Labelme 目录（每图一个 .json：shapes[].polygon）→ COCO（segm）。

    类别表自动从标注汇总（按首次出现顺序 → 连续 1..N）；传入 names 时按其顺序。
    shape_type 只接受 polygon/rectangle（rectangle 转 bbox 语义的四点多边形）。
    """
    lbl_dir = spec["labels"]
    jsons = sorted(f for f in os.listdir(lbl_dir) if f.endswith(".json"))
    auto = names is None  # 未提供类别表 → 自动从标注汇总（按首次出现顺序）
    if names is None:
        names = []
    name_id = {n: i + 1 for i, n in enumerate(names)}
    images, annotations, bad = [], [], 0
    ann_id = 1
    for i, jf in enumerate(jsons):
        d = json.load(open(os.path.join(lbl_dir, jf), encoding="utf-8"))
        rel_img = d.get("imagePath") or ""
        img_name = os.path.basename(rel_img)
        img_path = os.path.join(spec["images"], img_name)
        img = cv2.imread(img_path)
        if img is None:
            print("⚠ 图片无法读取，跳过:", img_name)
            continue
        H = d.get("imageHeight") or img.shape[0]
        W = d.get("imageWidth") or img.shape[1]
        images.append({"id": i + 1, "file_name": img_name, "width": W, "height": H})
        for s in d.get("shapes", []):
            label = s.get("label", "")
            if not label:
                bad += 1
                continue
            if label not in name_id:
                if auto:
                    name_id[label] = len(name_id) + 1
                else:
                    bad += 1
                    continue
            pts = s.get("points", [])
            st = s.get("shape_type", "polygon")
            if st == "rectangle" and len(pts) == 2:
                (x1, y1), (x2, y2) = pts
                poly = [x1, y1, x2, y1, x2, y2, x1, y2]
            elif len(pts) >= 3:
                poly = [v for p in pts for v in p]
            else:
                bad += 1
                continue
            arr = np.array(poly, np.float32).reshape(-1, 2)
            area = float(cv2.contourArea(arr))
            if area <= 1:
                bad += 1
                continue
            annotations.append({"id": ann_id, "image_id": i + 1,
                                "category_id": name_id[label],
                                "segmentation": [[round(v, 1) for v in poly]],
                                "bbox": [round(float(arr[:, 0].min()), 2),
                                         round(float(arr[:, 1].min()), 2),
                                         round(float(arr[:, 0].max() - arr[:, 0].min()), 2),
                                         round(float(arr[:, 1].max() - arr[:, 1].min()), 2)],
                                "area": round(area, 2), "iscrowd": 0, "ignore": 0})
            ann_id += 1
    cats = sorted(name_id.items(), key=lambda kv: kv[1])
    return images, annotations, [n for n, _ in cats], bad


def conv_yolo_obb(spec, names):
    """YOLO-OBB txt（cls x1 y1 x2 y2 x3 y3 x4 y4 归一化四点）→ DOTA gt jsonl。

    输出与 split_dota.py 的 patches_gt.jsonl 同 schema（patch=文件名，不切片——
    yolo-obb 训练数据通常已是小图；DOTA 大图请用 split_dota.py 切片流程）。
    返回 (gt_lines, images_meta, bad)：gt_lines 为 jsonl 行列表。
    """
    img_dir, lbl_dir = spec["images"], spec["labels"]
    files = list_images(img_dir)
    gt_lines, images_meta, bad, n_obj = [], [], 0, 0
    for f in files:
        stem = os.path.splitext(f)[0]
        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            print("⚠ 无法读取，跳过:", f)
            continue
        H, W = img.shape[:2]
        images_meta.append({"id": len(images_meta) + 1, "file_name": f,
                            "width": W, "height": H})
        objs = []
        txt = os.path.join(lbl_dir, stem + ".txt")
        if os.path.exists(txt):
            for line in open(txt):
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    c = int(parts[0])
                    vals = list(map(float, parts[1:9]))
                except ValueError:
                    bad += 1
                    continue
                if not (0 <= c < len(names)):
                    bad += 1
                    continue
                poly = []
                ok = True
                for k in range(4):  # (x,y) 反归一化
                    px, py = vals[k * 2] * W, vals[k * 2 + 1] * H
                    poly += [round(px, 1), round(py, 1)]
                if cv2.contourArea(
                        np.array(poly, np.float32).reshape(-1, 1, 2)) <= 1:
                    ok = False
                if not ok:
                    bad += 1
                    continue
                objs.append({"poly": poly, "cls": names[c], "difficult": 0})
                n_obj += 1
        gt_lines.append(json.dumps({"patch": f, "width": W, "height": H,
                                    "objects": objs}))
    return gt_lines, images_meta, bad, n_obj


def make_check(images, annotations, names, bad, extra=""):
    from collections import Counter
    dist = Counter(a["category_id"] for a in annotations)
    labeled_imgs = {a["image_id"] for a in annotations}
    missing = [im["file_name"] for im in images if im["id"] not in labeled_imgs]
    check = {
        "images": len(images),
        "annotations": len(annotations),
        "categories": [{"id": i + 1, "name": n, "count": dist.get(i + 1, 0)}
                       for i, n in enumerate(names)],
        "unlabeled_images": len(missing),
        "unlabeled_examples": missing[:10],
        "invalid_boxes": bad,
        "notes": extra,
    }
    return check, dist


def render_preview_obb(images_meta, gt_lines, names, img_dir, out_dir, n):
    """旋转框 GT 预览：画四点多边形 + 类别名。"""
    os.makedirs(out_dir, exist_ok=True)
    random.seed(0)
    with_objs = []
    for line in gt_lines:
        d = json.loads(line)
        if d["objects"]:
            with_objs.append(d)
    if not with_objs:
        print("（无标注，跳过预览）")
        return 0
    picks = random.sample(with_objs, min(n, len(with_objs)))
    for d in picks:
        img = cv2.imread(os.path.join(img_dir, d["patch"]))
        if img is None:
            continue
        for o in d["objects"]:
            pts = np.array(o["poly"], np.float32).reshape(-1, 2)
            c = names.index(o["cls"]) if o["cls"] in names else 0
            color = PALETTE[c % len(PALETTE)]
            cv2.polylines(img, [pts.astype(int)], True, color, 2)
            cv2.putText(img, o["cls"], tuple(pts.min(0).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(d["patch"])[0] + "_gt.jpg"),
                    img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return len(picks)


def render_preview(images, annotations, names, img_dir, out_dir, n):
    os.makedirs(out_dir, exist_ok=True)
    by_img = {}
    for a in annotations:
        by_img.setdefault(a["image_id"], []).append(a)
    picks = [im for im in images if im["id"] in by_img]
    if not picks:
        print("（无标注，跳过预览）")
        return 0
    random.seed(0)
    picks = random.sample(picks, min(n, len(picks)))
    for im in picks:
        img = cv2.imread(os.path.join(img_dir, im["file_name"]))
        if img is None:
            continue
        for a in by_img.get(im["id"], []):
            x, y, w, h = [int(v) for v in a["bbox"]]
            c = a["category_id"] - 1
            color = PALETTE[c % len(PALETTE)]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = "%s(%d)" % (names[c], a["category_id"])
            cv2.putText(img, label, (max(0, x), max(14, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(im["file_name"])[0] + "_gt.jpg"),
                    img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return len(picks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", help="数据集 spec yaml")
    ap.add_argument("--format", choices=["yolo", "voc", "coco", "yolo-obb", "labelme"])
    ap.add_argument("--images")
    ap.add_argument("--labels")
    ap.add_argument("--names")
    ap.add_argument("--label-file", help="模型 label 文件（顺序一致性校验）")
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--out", help="输出 COCO JSON 路径")
    ap.add_argument("--vis", type=int, default=0, help="GT 叠加预览张数")
    args = ap.parse_args()

    if args.spec:
        if yaml is None:
            raise SystemExit("需要 pyyaml（pip install pyyaml）")
        spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    else:
        spec = {"name": args.name, "format": args.format, "images": args.images,
                "labels": args.labels, "names": args.names, "label_file": args.label_file}
    fmt = spec.get("format", "yolo")
    name = spec.get("name", "dataset")
    is_obb = fmt == "yolo-obb"
    out = args.out or os.path.join("datasets",
                                   name + (".gt.jsonl" if is_obb else ".coco.json"))

    names = load_names(spec)
    if fmt == "labelme":
        # seg 自定义数据集：类别表可省略（自动从标注汇总，按首次出现顺序 1..N）
        images, annotations, names, bad = conv_labelme(spec, names)
        if not names:
            raise SystemExit("labelme 标注中未发现任何类别")
        print("类别表 %d 类: %s" % (len(names), names))
        dataset = {"images": images, "annotations": annotations,
                   "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
        extra = "labelme → COCO segm（多边形）"
    elif fmt in ("yolo", "voc", "yolo-obb"):
        if not names:
            raise SystemExit("%s 格式需要类别表（spec.names: data.yaml 路径或内联 list）" % fmt)
        print("类别表 %d 类: %s%s" % (len(names), names[:6], "..." if len(names) > 6 else ""))
        if fmt == "yolo":
            images, annotations, bad = conv_yolo(spec, names)
        elif fmt == "yolo-obb":
            gt_lines, images, bad, n_obj = conv_yolo_obb(spec, names)
            annotations = []  # obb 走 jsonl，check 单独构造
        else:
            images, annotations, bad = conv_voc(spec, names)
        extra = ""
        lf = spec.get("label_file")
        if lf and os.path.exists(lf):
            lf_names = [l.strip() for l in open(lf, encoding="utf-8", errors="ignore") if l.strip()]
            extra = ("模型 label 文件顺序一致" if lf_names == names
                     else "⚠ 模型 label 文件与数据集类别顺序不一致！评测将按数据集顺序映射，"
                          "请确认模型训练时类别顺序 = 数据集类别顺序")

        if is_obb:
            from collections import Counter
            dist = Counter()
            for line in gt_lines:
                for o in json.loads(line)["objects"]:
                    dist[o["cls"]] += 1
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as f:
                f.write("\n".join(gt_lines) + "\n")
            check = {"images": len(images), "annotations": n_obj,
                     "categories": [{"name": n, "count": dist.get(n, 0)} for n in names],
                     "unlabeled_images": sum(1 for l in gt_lines
                                             if not json.loads(l)["objects"]),
                     "invalid_boxes": bad, "notes": extra or "yolo-obb → DOTA jsonl（未切片）"}
            json.dump(check, open(out + ".check.json", "w"), indent=2, ensure_ascii=False)
            print("转换完成: %d 图 / %d 旋转框 / %d 类 -> %s" %
                  (len(images), n_obj, len(names), out))
            print("校验: 空标注图 %d | 非法框 %d | 类别分布 %s%s" %
                  (check["unlabeled_images"], bad,
                   {k: v for k, v in sorted(dist.items())},
                   (" | " + extra) if extra else ""))
            if args.vis > 0:
                n = render_preview_obb(images, gt_lines, names, spec["images"],
                                       out + "_preview", args.vis)
                print("GT 预览 %d 张 -> %s" % (n, out + "_preview"))
            return
        dataset = {"images": images, "annotations": annotations,
                   "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    elif fmt == "coco":
        dataset, bad = check_coco(spec)
        names = [c["name"] for c in sorted(dataset["categories"], key=lambda c: c["id"])]
        images, annotations = dataset["images"], dataset["annotations"]
        extra = "coco 直通（ids 归一化为连续 1..%d）" % len(names)
    else:
        raise SystemExit("暂不支持 format=%s（detect: yolo/voc/coco；obb: yolo-obb/dota）" % fmt)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(dataset, open(out, "w"))
    print("转换完成: %d 图 / %d 标注 / %d 类 -> %s" %
          (len(images), len(annotations), len(names), out))

    check, dist = make_check(images, annotations, names, bad, extra)
    with open(out + ".check.json", "w") as f:
        json.dump(check, f, indent=2, ensure_ascii=False)
    print("校验: 漏标图 %d | 非法框 %d | 类别分布 %s%s" %
          (check["unlabeled_images"], bad,
           {names[k - 1]: v for k, v in sorted(dist.items())},
           (" | " + extra) if extra else ""))
    if args.vis > 0:
        n = render_preview(images, annotations, names, spec["images"],
                           out + "_preview", args.vis)
        print("GT 预览 %d 张 -> %s" % (n, out + "_preview"))


if __name__ == "__main__":
    main()
