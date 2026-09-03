#ifndef _RKNN_DEMO_PREPROCESS_H_
#define _RKNN_DEMO_PREPROCESS_H_

#include <opencv2/core.hpp>

#include "common.h"
#include "utils.h"

int preprocess_image(image_buffer_t* src_image, image_buffer_t* dst_image, letterbox_t* letter_box, char color);

// 用 RGA 硬件将 NV12/NV21 帧转换为 BGR888 并写入 out_bgr。
// 可选 scale（0.1~1.0，1.0=原尺寸）：RGA 一次完成 NV12→BGR + 缩放，
// 供 Web 预览降采样场景使用，替代 CPU cvtColor + resize。
// 用于 Web 预览等需要 BGR 的场景；失败返回非 0（调用方回退 CPU 路径）。
// 要求 src 为 YUV420SP（NV12/NV21），且至少含 virt_addr 或 fd 之一。
int rga_nv12_to_bgr(const image_buffer_t& src, cv::Mat& out_bgr, double scale = 1.0);

// 用 RGA 硬件从 NV12/NV21 源帧裁剪 [src_x,src_y,src_w,src_h]（必须落在帧内）并
// 缩放到 out_rgb 的 [dst_x,dst_y,dst_w,dst_h] 区域（RGB888）。out_rgb 需为 CV_8UC3
// 连续 Mat，调用方负责预填背景色（未覆盖区域保留）。
// 用于 RTMPose 人体 ROI 直裁：一次 RGA 完成 裁剪+缩放+YCbCr→RGB，避免整帧 BGR 转换。
// 返回 0 成功；非 NV12/NV21、区域越界或 RGA 失败返回 -1（调用方回退 CPU 路径）。
int rga_nv12_crop_resize_rgb(const image_buffer_t& src,
                             int src_x, int src_y, int src_w, int src_h,
                             cv::Mat& out_rgb, int dst_x, int dst_y, int dst_w, int dst_h);

#endif
