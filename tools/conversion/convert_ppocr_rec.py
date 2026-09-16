#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PP-OCRv4 Rec（文字识别）→ RKNN 转换脚本（M6，RKNN Model Zoo ppocr_rec 对接）
===========================================================================
输入 3x48x320，归一化 (x/255-0.5)/0.5 经 mean/std_values 入图（板端喂 UINT8 BGR 即可）。
输出 [1, T, 6625] CTC logits（6623 字典字符 + blank + 空格），板端 ctcGreedyDecode 解码。

量化档位建议（项目纪律：head logits INT8 易塌缩，见 yolo26量化方案.md）：
  - 默认 FP16（do_quantization=False）：分数可信，~6625 类输出无塌缩风险（推荐）
  - --int8：需要校准集（文字行图片 100 张，可用检测视频随机 crop），分数需板端探针复核；
    若出现全字符塌缩/分数聚簇，回退 FP16

前提（PC 端）:
  - rknn-toolkit2 == 2.3.2
  - PaddleOCR 官方 en_PP-OCRv4_rec_infer 导出 ONNX（paddle2onnx），
    或直接取 airockchip/rknn_model_zoo models/CV/ocr 提供的 onnx
  - 字典 ppocr_keys_v1.txt（6623 行，zoo仓库/ppocr 官方均有）→ 随模型一起拷板

用法:
  python3 convert_ppocr_rec.py                # FP16（推荐）
  python3 convert_ppocr_rec.py --int8         # INT8 + 校准集目录

输出: ppocrv4_rec_fp16.rknn（或 ppocrv4_rec_int8.rknn）
"""
import argparse
import glob
import os

from rknn.api import RKNN

ONNX_MODEL = "ppocrv4_rec.onnx"
TARGET_PLATFORM = "rk3588"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=ONNX_MODEL)
    ap.add_argument("--int8", action="store_true", help="INT8 量化（需校准集）")
    ap.add_argument("--dataset", default="./rec_calib_imgs", help="INT8 校准图片目录")
    args = ap.parse_args()

    assert os.path.isfile(args.onnx), f"ONNX not found: {args.onnx}"
    out = "ppocrv4_rec_int8.rknn" if args.int8 else "ppocrv4_rec_fp16.rknn"

    rknn = RKNN(verbose=True)
    # (x/255 - 0.5) / 0.5 == (x - 127.5) / 127.5
    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[127.5, 127.5, 127.5]],
        target_platform=TARGET_PLATFORM,
        optimization_level=3,
    )
    assert rknn.load_onnx(model=args.onnx) == 0, "load_onnx failed"

    if args.int8:
        imgs = sorted(
            glob.glob(os.path.join(args.dataset, "*.jpg"))
            + glob.glob(os.path.join(args.dataset, "*.png"))
        )
        assert imgs, f"no calibration images in {args.dataset}"
        with open("rec_calib_list.txt", "w") as f:
            f.write("\n".join(imgs))
        assert rknn.build(do_quantization=True, dataset="rec_calib_list.txt") == 0
    else:
        assert rknn.build(do_quantization=False) == 0, "build failed"

    assert rknn.export_rknn(out) == 0, "export_rknn failed"
    print(f"[OK] saved {out}")
    print("[CHECK] list_outputs =", rknn.list_outputs())
    print("[NEXT] 板端探针打印输出值域：若 INT8 分数聚簇/全同字，回退 FP16 档")
    rknn.release()


if __name__ == "__main__":
    main()
