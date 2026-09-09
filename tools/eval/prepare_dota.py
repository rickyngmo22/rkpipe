#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_dota.py —— PC 端 DOTA v1.0 数据集预处理（网页评测前一步）。

把官方/自行下载的 DOTA v1.0 压缩包（或已解压目录）一键整理成网页控制台
可直接使用的形式：

  输出目录结构（默认 ./dota_ready/）：
    val_images/                原图（png/jpg，已按官方布局配对整理）
    val_labelTxt/              同名 labelTxt
    patches/ + patches_gt.jsonl   若本机有 cv2+numpy：1024/200 切片产物
                                  （网页"PC 文件夹同步"直接选 patches 目录 +
                                   patches_gt.jsonl 单文件标注）

用法：
  # Windows（有 python 环境）: tools/eval/prepare_dota.bat <压缩包或目录>
  python3 tools/eval/prepare_dota.py --input DOTA-v1.0_val.zip [--out dota_ready]
  python3 tools/eval/prepare_dota.py --input ./已解压目录 [--limit 50] [--skip-slice]

参数：
  --input PATH   DOTA 压缩包(.zip) 或已解压目录（自动递归识别 images/labelTxt）
  --out DIR      输出目录（默认 ./dota_ready）
  --limit N      只取前 N 张原图（试跑/小样评测，正式评测请去掉）
  --skip-slice   只整理不切片（本机无 cv2 时自动跳过切片并提示）

随后两种用法二选一：
  A) 网页「上传数据集 zip」：把 dota_ready/val_images+val_labelTxt 重新打包上传，
     板上自动切片（适用于 PC 无 python 环境、且愿意传原图）
  B) 网页「PC 文件夹同步」：图片目录选 dota_ready/patches，标注选
     dota_ready/patches_gt.jsonl（推荐：切片 jpg 通常比原图 png 小, 且单文件标注）
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_pairs(root):
    """递归找 (图片目录, labelTxt 目录)：目录含图片、目录含 txt、同名配对>=3。"""
    imgdirs, lbldirs = {}, {}
    for dirpath, _, names in os.walk(root):
        imgs = [n for n in names if n.lower().endswith(IMG_EXT)]
        txts = [n for n in names if n.lower().endswith(".txt")]
        if imgs:
            imgdirs[dirpath] = {n.lower(): n for n in imgs}
        if txts:
            lbldirs[dirpath] = {n.lower(): n for n in txts}
    for ip, inames in imgdirs.items():
        for lp, lnames in lbldirs.items():
            hit = sum(1 for n in inames
                      if n.rsplit(".", 1)[0] + ".txt" in lnames)
            if hit >= 3 and len(inames) >= 3:
                return ip, lp
    return None, None


def unpack(input_path):
    """zip -> 临时目录; 目录 -> 原样。返回 (目录, 是否临时, 清理函数)。"""
    if os.path.isdir(input_path):
        return input_path, False, None
    tmp = tempfile.mkdtemp(prefix="dota_prep_")
    try:
        with zipfile.ZipFile(input_path) as z:
            # 防 zip slip
            base = os.path.abspath(tmp)
            for m in z.namelist():
                t = os.path.abspath(os.path.join(tmp, m))
                if not t.startswith(base + os.sep):
                    raise RuntimeError("zip 内路径非法: %s" % m)
            z.extractall(tmp)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return tmp, True, (lambda: shutil.rmtree(tmp, ignore_errors=True))


def collect(images_dir, labels_dir, limit):
    """按字母序收集 (原图路径, labelTxt路径) 配对列表。"""
    names = sorted(os.listdir(images_dir))
    picked = []
    for n in names:
        if not n.lower().endswith(IMG_EXT):
            continue
        stem = os.path.splitext(n)[0]
        lbl = os.path.join(labels_dir, stem + ".txt")
        if os.path.isfile(lbl):
            picked.append((os.path.join(images_dir, n), lbl))
        if limit and len(picked) >= limit:
            break
    return picked


def run_split(images_dir, labels_dir, out_root):
    """调 split_dota.py 切片; 返回 (patches_dir, gt_path) 或 (None, None)。"""
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "split_dota.py")
    patches = os.path.join(out_root, "patches")
    gt = os.path.join(out_root, "patches_gt.jsonl")
    if not os.path.exists(script):
        print("  ! 未找到 split_dota.py（%s），跳过切片" % script)
        return None, None
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        print("  ! 本机缺少 cv2/numpy，跳过切片（可用网页 zip 上传让板端切片）")
        return None, None
    cmd = ["python3", script, "--images", images_dir, "--labels", labels_dir,
           "--out", patches, "--gt", gt]
    print("  切片中（1024/200，%s 张原图）..." % len(os.listdir(images_dir)))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            print("  ! 切片失败: %s" % (r.stderr or r.stdout)[-500:])
            return None, None
    except Exception as e:
        print("  ! 切片异常: %s" % e)
        return None, None
    if os.path.exists(gt) and os.path.isdir(patches):
        return patches, gt
    print("  ! 切片产物缺失")
    return None, None


def main():
    ap = argparse.ArgumentParser(description="DOTA v1.0 预处理（网页评测用）")
    ap.add_argument("--input", required=True, help="DOTA zip 或已解压目录")
    ap.add_argument("--out", default="dota_ready")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 张(试跑)")
    ap.add_argument("--skip-slice", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print("输入不存在: %s" % args.input)
        sys.exit(2)

    root, is_tmp, cleanup = unpack(args.input)
    try:
        images_dir, labels_dir = find_pairs(root)
        if not images_dir:
            print("未在输入中识别到 DOTA 布局（图片 + 同名 labelTxt 配对）")
            sys.exit(2)
        print("识别到图片目录: %s" % images_dir)
        print("识别到标签目录: %s" % labels_dir)

        os.makedirs(args.out, exist_ok=True)
        oi = os.path.join(args.out, "val_images")
        ol = os.path.join(args.out, "val_labelTxt")
        os.makedirs(oi, exist_ok=True)
        os.makedirs(ol, exist_ok=True)

        pairs = collect(images_dir, labels_dir, args.limit)
        if not pairs:
            print("未找到任何 图片+labelTxt 配对")
            sys.exit(2)
        for img, lbl in pairs:
            shutil.copy2(img, os.path.join(oi, os.path.basename(img)))
            shutil.copy2(lbl, os.path.join(ol, os.path.basename(lbl)))
        print("已整理 %d 张原图 + 标签 → %s/val_images + val_labelTxt"
              % (len(pairs), args.out))

        patches, gt = (None, None)
        if not args.skip_slice:
            patches, gt = run_split(oi, ol, args.out)

        print()
        print("==== 下一步（二选一） ====")
        if patches and gt:
            print("A) 网页「PC 文件夹同步」：")
            print("     图片目录选 %s" % os.path.abspath(patches))
            print("     标注文件选 %s" % os.path.abspath(gt))
            print("     任务选 obb，类别数 15，label 选 assets/labels/yolov8_obb_labels_list.txt")
        print("B) 网页「上传数据集 zip」：把 %s/val_images 与 val_labelTxt "
              "打包上传（板上自动切片）" % os.path.abspath(args.out))
        print("    cd %s && zip -r dota.zip val_images val_labelTxt" % os.path.abspath(args.out))
    finally:
        if cleanup:
            cleanup()


if __name__ == "__main__":
    main()
