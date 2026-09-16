#ifndef DRAW_UTILS_H
#define DRAW_UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "utils/common.h"
#include "core/task_result.h"
#include "postprocess/postprocess.h"
#include "utils/depth_distance.h"

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
// D3 检测+单目测距叠加：每个检测框左上角标注"框内深度中值"估计距离（米）；
// near_m>0 时距离 ≤ near_m 的目标加红框、文字转红，并写入返回值供近距告警使用。
// draw_text=false 时只计算不绘制（overlay 关闭但告警开启的场景）。
// depth 为 CV_8UC1 反相深度图（近=亮），depth_lo/depth_hi 为其归一化范围（米，后处理输出），
// scale 为距离标定系数。
std::vector<DepthDistanceTarget> drawDepthDistanceOverlayBGR(
    cv::Mat& frame, const cv::Mat& depth, const cv::Rect* roi,
    float depth_lo, float depth_hi, const object_detect_result_list& dets,
    float near_m = 0.0f, float scale = 1.0f, bool draw_text = true);

// BGR(CPU) 帧上的单目 3D 检测绘制（Detect3D）：2D 框 + 类别/置信度 + 深度距离 + 3D 尺寸
// + 3D 投影中心十字标记；配置 P2（set_detect3d_p2）后追加 3D 线框投影。
void drawDetect3DResultsBGR(cv::Mat& frame, const Detect3DTaskResult& results);

// 设置 Detect3D 线框投影的 P2 矩阵（3x4 行主序 12 值，逗号分隔，如 KITTI cam2 标定）。
// 空串/解析失败 = 不绘制线框。运行时启动阶段调用一次。
void set_detect3d_p2(const std::string& p2_spec);
// Detect3D 深度全局缩放（场景尺度校准：单目深度在非训练场景上常有整体偏差，线框过大则调大）
void set_detect3d_depth_scale(float scale);
// 语义分割伪彩叠加：class_map 为 CV_8UC1 类别索引图（低分辨率有效区），放大到 roi 并按 alpha 混合。
// 前 19 类用 Cityscapes 标准色，其余类别用黄金角 HSV 伪彩兜底。
void drawSemResultsBGR(cv::Mat& frame, const cv::Mat& class_map, int class_num, const cv::Rect* roi = nullptr, double alpha = 0.6);

bool drawOCRResultsZeroCopy(image_buffer_t& frame, const OCRDetectTaskResult& results);
// M6 文字识别结果（多边形 + 识别文本）BGR 绘制；lines 为空时退化为多边形描边
void drawOCRTextBGR(cv::Mat& frame, const OCRDetectTaskResult& results);
// M0 两阶级联：检测框（原类别色）+ 每框二级 top-1 标签行
void drawCompositeClsBGR(cv::Mat& frame, const CompositeClsTaskResult& results);
// M8 人脸：绿色框 + 5 点 landmark（红点）+ 分数标签
void drawFaceResultsBGR(cv::Mat& frame, const FaceTaskResult& results);
// M13 动作识别：本帧新鲜动作（ActionTaskResult.actions）标注在对应 track 框上方第二行；
// labels 为动作名表（索引对齐 action_id），越界/缺表回退 "action N"
void drawActionLabelsBGR(cv::Mat& frame, const ActionTaskResult& results,
                         const std::vector<std::string>& labels);

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
