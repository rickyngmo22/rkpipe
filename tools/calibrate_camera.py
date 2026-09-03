#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定工具：棋盘格照片 → 内参 K + 畸变系数 + detect3d_p2 配置行
==================================================================
用途：为 Detect3D 的 3D 线框投影提供真实 P2 矩阵（run_yolo26_detect3d.yaml 的
detect3d_p2 键）。内参必须用**实际演示用的那台相机**、**与视频相同的分辨率**标定。

步骤：
  1. 打印棋盘格（推荐 A4 横向，9x6 内角点，方格 20~25mm），贴硬纸板保持平整
  2. 固定相机分辨率/焦距（关自动对焦/变焦！），拍 15~25 张棋盘格照片：
     覆盖画面各区域（尤其四角/边缘），每张姿态不同（左右倾斜 15~30°）
  3. 运行本脚本 → 得到 detect3d_p2 配置行 + 重投影误差（RMS < 0.5px 为佳）
  4. 若镜头畸变明显（广角行车记录仪），用 --undistort 把演示视频先去畸变再喂模型
     （Detect3D/KITTI 训练数据是校正过的图像）

用法：
  python3 tools/calibrate_camera.py --images "calib/*.jpg" --cols 9 --rows 6
  python3 tools/calibrate_camera.py --images "calib/*.jpg" --cols 9 --rows 6 \
      --undistort demo.mp4 demo_rect.mp4

可在板端（opencv 4.x python）或 PC/WSL 运行；标定结果与运行设备无关，只与相机和分辨率有关。
"""
import argparse
import glob
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="棋盘格相机标定 → detect3d_p2")
    ap.add_argument("--images", required=True, help="标定照片通配符，如 \"calib/*.jpg\"")
    ap.add_argument("--cols", type=int, default=9, help="棋盘格内角点列数（默认 9）")
    ap.add_argument("--rows", type=int, default=6, help="棋盘格内角点行数（默认 6）")
    ap.add_argument("--square-mm", type=float, default=25.0, help="方格边长 mm（默认 25，只影响物理尺度）")
    ap.add_argument("--undistort", nargs=2, metavar=("IN_VIDEO", "OUT_VIDEO"), default=None,
                    help="用标定结果批量去畸变（广角镜头建议做）")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    if len(paths) < 5:
        print(f"[ERR] 标定照片太少：{len(paths)} 张（至少 5，建议 15~25），pattern={args.images}")
        sys.exit(1)

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm

    obj_points, img_points = [], []
    used = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] 读图失败: {path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
        if not found:
            found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if not found:
            print(f"[skip] 未找到角点: {path}")
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used += 1
        print(f"[ok] {path} ({img.shape[1]}x{img.shape[0]})")
    if used < 5:
        print(f"[ERR] 有效照片 {used} 张，不足（至少 5，建议 15~25）。检查棋盘格规格（--cols/--rows）与照片质量。")
        sys.exit(1)
    if used < 12:
        print(f"[warn] 有效照片仅 {used} 张，精度有限；建议补拍到 15~25 张。")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,
                                                     gray.shape[::-1], None, None)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"\n=== 标定结果（{used} 张）===")
    print(f"重投影误差 RMS = {rms:.3f} px（<0.5 良好，>1.0 建议重拍）")
    print(f"K = [[{fx:.2f}, 0, {cx:.2f}],\n     [0, {fy:.2f}, {cy:.2f}],\n     [0, 0, 1]]")
    print(f"dist = {np.round(dist.ravel(), 6).tolist()}")
    print(f"分辨率: {gray.shape[1]}x{gray.shape[0]}（内参与分辨率绑定，换分辨率需重标）")
    print("\n=== 写入 run_yolo26_detect3d.yaml ===")
    p2 = (f"{fx:.4f},0,{cx:.4f},0,0,{fy:.4f},{cy:.4f},0,0,0,1,0")
    print(f"detect3d_p2: \"{p2}\"")

    if args.undistort:
        src_path, dst_path = args.undistort
        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            print(f"[ERR] 打不开视频 {src_path}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 0)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, size, cv2.CV_16SC2)
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(cv2.remap(frame, map1, map2, cv2.INTER_LINEAR))
            n += 1
        cap.release()
        writer.release()
        print(f"[OK] 去畸变完成：{n} 帧 -> {dst_path}（把 demo yaml 的 input_path 指到它）")


if __name__ == "__main__":
    main()
