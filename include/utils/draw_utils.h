#ifndef DRAW_UTILS_H
#define DRAW_UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "common.h"
#include "core/task_result.h"
#include "postprocess/postprocess.h"

// 绘制目标检测结果
void drawDetectionResults(image_buffer_t& frame, const object_detect_result_list& results);
bool drawDetectionResultsZeroCopy(image_buffer_t& frame, const object_detect_result_list& results);
// BGR(CPU)帧上的检测框绘制：opencv/ffmpeg_rkmpp 读取路径无零拷贝，需走这里
void drawDetectionResultsBGR(cv::Mat& frame, const object_detect_result_list& results);

// 绘制Pose结果
void drawPoseResults(image_buffer_t& frame, const pose_detect_result_list& results);
bool drawPoseResultsZeroCopy(image_buffer_t& frame, const pose_detect_result_list& results, bool draw_box = true);
// BGR(CPU)帧上的 Pose 绘制：opencv/ffmpeg_rkmpp 读取路径无零拷贝，需走这里。
// draw_box=false 时只画骨架/关键点，不画框（多任务叠加时框由主任务画）
void drawPoseResultsBGR(cv::Mat& frame, const pose_detect_result_list& results, bool draw_box = true);

bool drawOBBResultsZeroCopy(image_buffer_t& frame, const obb_detect_result_list& results);
// BGR(CPU)帧上的 OBB 绘制：opencv/ffmpeg_rkmpp 读取路径无零拷贝，需走这里
void drawOBBResultsBGR(cv::Mat& frame, const obb_detect_result_list& results);

bool drawSegResultsZeroCopy(image_buffer_t& frame, const seg_detect_result_list& results, bool draw_box = true);
// BGR(CPU)帧上的分割掩膜/框绘制：opencv/ffmpeg_rkmpp 读取路径无零拷贝，需走这里
// scale 用于把原始帧坐标的检测结果缩放到目标帧（如预览降采样后）。
// scale<1.0 时框/掩膜按比例缩小；掩膜会在内部 resize 到缩放后的框尺寸。
// draw_box=false 时只画掩膜，不画框（多任务叠加时框由主任务画）
void drawSegResultsBGR(cv::Mat& frame, const seg_detect_result_list& results, double scale = 1.0, bool draw_box = true);

// BGR(CPU)帧上的深度图绘制：depth 为 CV_8UC1（原帧分辨率，近=亮），JET 伪彩 + 混合到帧。
// RK_PIPE_DEPTH_REPLACE=1 时整帧 replace（alpha 强制 1.0，纯深度视图，不再与原始帧混合）。
void drawDepthResultsBGR(cv::Mat& frame, const cv::Mat& depth, double alpha = 0.7, const cv::Rect* roi = nullptr);


void drawOverlayText(image_buffer_t& frame, const std::string& text, int x, int y, const cv::Scalar& color, int scale = 1,
                     const cv::Scalar& bg_color = cv::Scalar(-1, -1, -1));

// BGR 帧标签:实底条 + 白字(Ultralytics 样式);供 tracking overlay 的 ID 文本复用
void putClassLabelBGR(cv::Mat& frame, const std::string& label, int x, int y1,
                     const cv::Scalar& bar_color, double scale, int thick);

// Ultralytics 官方 20 色调色板(BGR);同一类别/跟踪 ID 全局同色
cv::Scalar classColor(int cls_id);

// 显示阈值（默认 0.25，RK_PIPE_DISPLAY_THRESH 可覆盖）：worker 绘制与 tracking overlay 共用，
// 避免 tracking 路径用硬编码 0.5 把低分检测（如 yolo26 INT8）过滤到只剩 ID。
float displayThreshold();

#endif // DRAW_UTILS_H
