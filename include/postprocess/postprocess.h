#ifndef _RKNN_YOLOV8_DEMO_POSTPROCESS_H_
#define _RKNN_YOLOV8_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "core/rknn_context.h"
#include "utils/common.h"
#include "utils/utils.h"

namespace cv { class Mat; }

#define OBJ_NAME_MAX_SIZE 64
#define OBJ_NUMB_MAX_SIZE 128
#define MAX_CLASS_NUM 256
#define NMS_THRESH 0.45
#define BOX_THRESH 0.25
#define KEYPOINT_NUM 17 // COCO关键点数量
#define KEYPOINT_THRESH 0.2 // 关键点置信度阈值，降低到0.2以便显示更多关键点

// 关键点结构体
typedef struct {
    float x;       // 关键点x坐标
    float y;       // 关键点y坐标
    float conf;    // 关键点置信度
} pose_keypoint;

// 姿态估计结果结构体
typedef struct {
    image_rect_t box;          // 人体检测框
    float box_conf;           // 检测框置信度
    int cls_id;               // 类别ID
    int track_id;             // tracking 分配的稳定 ID（0=未跟踪）
    pose_keypoint keypoints[KEYPOINT_NUM]; // 17个关键点
} pose_detect_result;

// 姿态估计结果列表
typedef struct {
    int id;
    int count;
    pose_detect_result results[OBJ_NUMB_MAX_SIZE];
} pose_detect_result_list;

typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

typedef struct {
    image_obb_box_t box;
    float prop;
    int cls_id;
    int track_id;             // tracking 分配的稳定 ID（0=未跟踪）
} obb_detect_result;

typedef struct {
    int id;
    int count;
    obb_detect_result results[OBJ_NUMB_MAX_SIZE];
} obb_detect_result_list;

typedef struct {
    std::vector<cv::Mat> masks;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
} seg_detect_result_list;

int init_post_process(const char* label_file_path);

// ---- 标签表实例化（SDK 多 handle 异模型并发，专项#4-P2）----
// 标签表本体：init 时构造、deinit 时全局表清空；per-pipeline 表由
// shared_ptr 持有（hooks 捕获快照），线程经 bind 绑定后读取自己的表。
struct LabelTable {
    std::vector<std::string> names;
    int obj_class_num = 0;
};

// 捕获当前全局表快照（init 之后调用；供 PipelineRuntimeHooks 携带）
std::shared_ptr<const LabelTable> capture_label_table();
// 本线程绑定标签表（worker/output 线程开头调用一次；空指针=回退全局表）
void bind_thread_label_table(std::shared_ptr<const LabelTable> table);
void deinit_post_process();
int get_obj_class_num();
void set_obj_class_num(int class_num);
char *coco_cls_to_name(int cls_id);
int post_process_yolov8(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
int post_process_yolov5(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
// YOLO26 detect：无 DFL 直接距离回归 + cls logits + one2one 头免 NMS（topk 选择）
int post_process_yolov26(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, object_detect_result_list *od_results);
// YOLO26 pose：3 x [1,56,H,W]（4 box + 1 cls + 51 kpt），kpt 按 Pose26.kpts_decode 解码，免 NMS
int post_process_yolov26_pose(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, pose_detect_result_list *pd_results);
// YOLO26 OBB：3 x [1,4+nc+1,H,W]（4 box + nc cls + 1 angle），dist2rbox 解码，angle 原始弧度，免 NMS
int post_process_yolov26_obb(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, obb_detect_result_list *od_results);
// YOLO26 seg：3 x [1,4+nc+nm,H,W] + proto[1,nm,160,160]，mask = sigmoid(proto @ coeff)
int post_process_yolov26_seg(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, seg_detect_result_list *seg_results);
// YOLO26 depth：单输出 [1,1,H,W]，depth=exp(head)，反映射到原帧并归一化为 CV_8UC1（近=亮）。
// depth_lo/depth_hi 输出归一化范围（米），供距离文字把像素值还原为米制（可传 nullptr）。
int post_process_yolov26_depth(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, cv::Mat* depth_out, float* depth_lo = nullptr, float* depth_hi = nullptr);
// YOLO26 sem（语义分割）：单输出 [1,C,H,W]，逐像素 argmax 得 CV_8UC1 类别索引图（0..C-1）
int post_process_yolov26_sem(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, cv::Mat* class_map_out);
// YOLO26 Detect3D（单目 3D）：单输出 [1,300,14] 已解码免 NMS，
// 行布局见 detect3d_decode.h；坐标经 letterbox 逆映射回原帧，conf 阈值过滤
struct Detect3DTaskResult;  // 定义在 core/task_result.h（此处仅指针形参，前向声明避免循环包含）
int post_process_yolov26_detect3d(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, Detect3DTaskResult *d3_results);
int post_process_pose(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, pose_detect_result_list *pd_results);
// RTMPose SimCC 解码：simcc_x/simcc_y logits → argmax/split_ratio → 逆仿射映射回原图。
// results 为 rtmpose_decode_job_t*（定义在 model/rtmpose.h，经 run_rknn_inference 透传）
int post_process_rtmpose(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, void *results);
int post_process_obb(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, obb_detect_result_list *od_results);
// YOLOv8 分割后处理接口
int post_process_seg(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, seg_detect_result_list *seg_results);
void set_seg_mask_enabled(bool enabled);
bool get_seg_mask_enabled();
double get_last_obb_postprocess_time_ms();
double get_last_obb_postprocess_decode_time_ms();
double get_last_obb_postprocess_sort_time_ms();
double get_last_obb_postprocess_nms_time_ms();
double get_last_obb_postprocess_pack_time_ms();
int get_last_obb_candidate_count();
int get_last_obb_class_count();
long long get_last_obb_nms_comparisons();

double get_last_seg_postprocess_time_ms();
double get_last_seg_postprocess_decode_time_ms();
double get_last_seg_postprocess_sort_time_ms();
double get_last_seg_postprocess_nms_time_ms();
double get_last_seg_postprocess_mask_time_ms();
int get_last_seg_candidate_count();
int get_last_seg_class_count();

void deinitPostProcess();
#endif //_RKNN_YOLOV8_DEMO_POSTPROCESS_H_
