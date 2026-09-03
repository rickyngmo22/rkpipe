#pragma once

#include <unordered_map>
#include <vector>

#include "../utils/common.h"
#include "../postprocess/postprocess.h"

// 轻量级恒速 Kalman 滤波状态（SORT 风格，8 维）：
//   x = [cx, cy, w, h, vx, vy, vw, vh]
// P 为协方差矩阵，R 为测量噪声（按检测框尺寸初始化）。
struct KalmanFilter8 {
    double x[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    double P[8][8] = {};
    double R[4][4] = {};
    bool initialized = false;
};

struct TrackedDetection {
    object_detect_result det = {};
    int track_id = -1;
    bool is_new = false;
    bool is_confirmed = false;
    bool is_predicted = false;
};

// 可切换的跟踪算法档位（对应业界经典算法，能力逐级增强）：
//   Iou       - 单阶段 IoU 关联（最简，无遮挡恢复/两阶段）
//   Sort      - SORT：单阶段 IoU + missed 轨迹放宽 + 漏检期速度衰减
//   ByteTrack - ByteTrack：高分/低分两阶段关联
//   OcSort    - OC-SORT：两阶段 + 距离重连 + 观测速度再更新（默认，最强）
enum class TrackAlgorithm : int {
    Iou = 0,
    Sort = 1,
    ByteTrack = 2,
    OcSort = 3,
};

class SimpleObjectTracker {
public:
    SimpleObjectTracker(float iou_threshold,
                        int max_missed,
                        int min_confirm_hits,
                        float reid_iou_threshold,
                        float center_distance_threshold,
                        float box_smooth_alpha,
                        int render_max_missed,
                        float high_conf_threshold,
                        float low_conf_threshold,
                        float stage2_iou_threshold,
                        float size_ratio_threshold,
                        int max_tracks,
                        TrackAlgorithm algorithm = TrackAlgorithm::OcSort);

    // dt 为帧间隔（秒）归一化参数；默认 1.0 表示按帧为单位（与 SORT 一致）。
    std::vector<TrackedDetection> update(const object_detect_result_list& detections,
                                         std::unordered_map<int, int>* class_counter,
                                         float dt = 1.0f);

private:
    struct TrackState {
        int track_id = -1;
        int cls_id = -1;
        image_rect_t box = {};
        image_rect_t pred_box = {};
        image_rect_t smooth_box = {};
        image_rect_t last_obs_box = {};
        int missed = 0;
        int hits = 0;
        float last_prop = 0.0f;
        bool confirmed = false;
        bool counted = false;
        KalmanFilter8 kf;
    };

    // 几何工具
    static float computeIoU(const image_rect_t& a, const image_rect_t& b);
    static float computeCenterDistance(const image_rect_t& a, const image_rect_t& b);
    static float computeMaxCenterDistance(const image_rect_t& box, float scale);
    static bool sizeRatioOk(const image_rect_t& a, const image_rect_t& b, float min_ratio);
    static image_rect_t shiftBox(const image_rect_t& box, float dx, float dy, float dw, float dh);
    static image_rect_t blendBox(const image_rect_t& prev_box, const image_rect_t& cur_box, float alpha);
    static float computeAdaptiveSmoothAlpha(const image_rect_t& prev_box, const image_rect_t& new_box, float base_alpha);
    static bool isVelocityTooLarge(const TrackState& track);
    static image_rect_t applyMotionCompensation(const image_rect_t& box, const TrackState& track, float factor);

    // Kalman 滤波
    static void kfInit(KalmanFilter8& kf, const image_rect_t& box);
    static void kfPredict(KalmanFilter8& kf, float dt);
    static void kfUpdate(KalmanFilter8& kf, const image_rect_t& box);
    static image_rect_t boxFromKf(const KalmanFilter8& kf);
    static void invert4x4(const double in[4][4], double out[4][4]);

    // 关联（匈牙利算法，ByteTrack 两阶段）
    static void hungarianMinCost(int n, int m,
                                 const std::vector<std::vector<double>>& cost,
                                 std::vector<int>& row_assign,
                                 std::vector<int>& col_assign);
    void associate(const object_detect_result_list& detections,
                   const std::vector<int>& track_indices,
                   const std::vector<int>& det_indices,
                   float iou_gate,
                   bool relaxed_for_missed,
                   std::vector<int>& track_to_det,
                   std::vector<int>& det_to_track_id,
                   std::vector<char>& det_used,
                   std::vector<char>& track_used) const;

    // 算法能力开关（由 algorithm_ 决定）
    bool useTwoStage() const { return algorithm_ == TrackAlgorithm::ByteTrack || algorithm_ == TrackAlgorithm::OcSort; }
    bool useRelaxedMissed() const { return algorithm_ >= TrackAlgorithm::Sort; }
    bool useVelocityDecay() const { return algorithm_ >= TrackAlgorithm::Sort; }
    bool useRelink() const { return algorithm_ == TrackAlgorithm::OcSort; }
    bool useOru() const { return algorithm_ == TrackAlgorithm::OcSort; }

    float iou_threshold_ = 0.3f;
    int max_missed_ = 20;
    int min_confirm_hits_ = 3;
    float reid_iou_threshold_ = 0.15f;
    float center_distance_threshold_ = 1.8f;
    float box_smooth_alpha_ = 0.65f;
    float motion_compensation_ = 0.25f;
    int render_max_missed_ = 1;
    float high_conf_threshold_ = 0.5f;
    float low_conf_threshold_ = 0.25f;
    float stage2_iou_threshold_ = 0.5f;
    float size_ratio_threshold_ = 0.3f;
    // OC-SORT 风格遮挡恢复：近期漏检的轨迹允许仅凭中心距离重连
    float relink_center_scale_ = 0.9f;
    int relink_max_missed_ = 10;
    int max_tracks_ = 256;
    TrackAlgorithm algorithm_ = TrackAlgorithm::OcSort;
    int next_track_id_ = 1;
    std::vector<TrackState> tracks_;
};

object_detect_result_list buildTrackedResultList(const std::vector<TrackedDetection>& tracked);
