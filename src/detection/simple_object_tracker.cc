#include "detection/simple_object_tracker.h"
#include "utils/draw_utils.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr double kMatchInf = 1e9;   // 不可匹配/填充边代价（有限大，避免被当成"免费"）
constexpr double kMatchUnassigned = 1e8;  // 超过该代价视为未匹配
// 匈牙利算法内部的“无穷大”（minv/delta 初值）。必须显著大于 kMatchInf，
// 否则填充边(1e9)与算法 INF 相同会让严格比较失效，导致次优分配甚至死循环。
constexpr double kNoEdge = 1e18;
// 匈牙利矩阵规模上限（约 100×100）：超过则退回贪心分配。
// 遮挡恢复的距离重连会让候选变稠密，此时 O(N^3) 的匈牙利会成为瓶颈。
constexpr int kMaxHungarianCells = 10000;
// 输入级去重 IoU 门限：同帧同类、框重叠超过该值的低分候选视为同一目标的重复框
// （one2one/免 NMS 头多候选；真人对的框极少同帧重叠到 0.85 以上）
constexpr float kDupSuppressIoU = 0.75f;

}  // namespace

SimpleObjectTracker::SimpleObjectTracker(float iou_threshold,
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
                                         TrackAlgorithm algorithm)
    : iou_threshold_(std::max(0.0f, iou_threshold)),
      max_missed_(std::max(0, max_missed)),
      min_confirm_hits_(std::max(1, min_confirm_hits)),
      reid_iou_threshold_(std::max(0.0f, reid_iou_threshold)),
      center_distance_threshold_(std::max(0.1f, center_distance_threshold)),
      box_smooth_alpha_(std::min(0.95f, std::max(0.05f, box_smooth_alpha))),
      render_max_missed_(std::max(0, render_max_missed)),
      high_conf_threshold_(std::max(0.0f, high_conf_threshold)),
      low_conf_threshold_(std::max(0.0f, low_conf_threshold)),
      stage2_iou_threshold_(std::max(0.0f, stage2_iou_threshold)),
      size_ratio_threshold_(std::min(1.0f, std::max(0.05f, size_ratio_threshold))),
      max_tracks_(std::max(1, max_tracks)),
      algorithm_(algorithm) {}

std::vector<TrackedDetection> SimpleObjectTracker::update(const object_detect_result_list& detections,
                                                          std::unordered_map<int, int>* class_counter,
                                                          float dt) {
    const float t = std::max(1e-3f, dt);
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        kfPredict(tracks_[ti].kf, t);
        if (useVelocityDecay() && tracks_[ti].missed > 0) {
            // 遮挡/漏检期间对速度做指数衰减：目标更可能停下而非持续高速运动，
            // 避免预测框无限漂移导致重新出现时 IoU≈0 而无法重连（“跟丢”的主因）。
            for (int i = 4; i < 8; ++i) {
                tracks_[ti].kf.x[i] *= 0.9;
            }
        }
        tracks_[ti].pred_box = boxFromKf(tracks_[ti].kf);
    }

    int det_count = std::max(0, detections.count);
    std::vector<int> det_to_track_id(det_count, -1);
    std::vector<char> det_is_new(det_count, 0);
    std::vector<char> det_is_confirmed(det_count, 0);
    std::vector<image_rect_t> det_smooth_boxes(det_count);
    std::vector<char> det_has_smooth_box(det_count, 0);
    std::vector<int> track_to_det(tracks_.size(), -1);
    std::vector<char> track_used(tracks_.size(), 0);
    std::vector<char> det_used(det_count, 0);

    // 输入级去重：one2one/免 NMS 头（yolo26 系列）偶尔会从相邻网格/尺度对同一目标
    // 输出近乎重叠的多个候选。若都放行，未匹配的多余候选会直接建新轨迹 —— 同一人
    // 出现"影子轨迹"，与正主轮流抢匹配、ID 轮换，最终一方饿死再被识别成"新任务"。
    // 按置信度贪心保留高分候选，低分近重复框不再参与关联与建轨迹（输出行保留、无 ID）。
    std::vector<char> det_suppressed(det_count, 0);
    if (det_count > 1) {
        std::vector<int> order(det_count);
        for (int i = 0; i < det_count; ++i) {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&detections](int a, int b) {
            return detections.results[a].prop > detections.results[b].prop;
        });
        for (size_t oi = 0; oi < order.size(); ++oi) {
            const int i = order[oi];
            if (det_suppressed[i]) {
                continue;
            }
            for (size_t oj = oi + 1; oj < order.size(); ++oj) {
                const int j = order[oj];
                if (det_suppressed[j]) {
                    continue;
                }
                const object_detect_result& da = detections.results[i];
                const object_detect_result& db = detections.results[j];
                if (da.cls_id != db.cls_id) {
                    continue;
                }
                if (computeIoU(da.box, db.box) >= kDupSuppressIoU) {
                    det_suppressed[j] = 1;
                }
            }
        }
    }

    // 两阶段（ByteTrack/OcSort）：高分检测先与所有轨迹关联，未匹配轨迹再与低分检测关联。
    // 单阶段（Iou/Sort）：所有检测合成一组。
    std::vector<int> high_dets, low_dets;
    for (int di = 0; di < det_count; ++di) {
        if (det_suppressed[di]) {
            continue;
        }
        const float prop = detections.results[di].prop;
        if (useTwoStage()) {
            if (prop >= high_conf_threshold_) {
                high_dets.push_back(di);
            } else if (prop >= low_conf_threshold_) {
                low_dets.push_back(di);
            }
        } else {
            high_dets.push_back(di);
        }
    }

    // 阶段 A：全部轨迹 × 高分检测（Sort+ 档位对 missed>0 的轨迹放宽 IoU 门限以便遮挡恢复）
    std::vector<int> all_track_idx(tracks_.size());
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        all_track_idx[ti] = static_cast<int>(ti);
    }
    associate(detections, all_track_idx, high_dets, iou_threshold_, useRelaxedMissed(),
              track_to_det, det_to_track_id, det_used, track_used);

    // 阶段 B：未匹配轨迹 × 低分检测（要求更紧的 IoU；仅两阶段档位有低分检测）
    std::vector<int> unmatched_track_idx;
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        if (!track_used[ti]) {
            unmatched_track_idx.push_back(static_cast<int>(ti));
        }
    }
    std::vector<int> unused_low_dets;
    for (int di : low_dets) {
        if (!det_used[di]) {
            unused_low_dets.push_back(di);
        }
    }
    associate(detections, unmatched_track_idx, unused_low_dets, stage2_iou_threshold_, /*relaxed_for_missed=*/false,
              track_to_det, det_to_track_id, det_used, track_used);

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        int di = track_to_det[ti];
        if (di >= 0) {
            TrackState& track = tracks_[ti];
            const object_detect_result& det = detections.results[di];
            const image_rect_t& det_box = det.box;
            const image_rect_t prev_box = track.box;
            const image_rect_t prev_obs = track.last_obs_box;
            const int prev_missed = track.missed;
            kfUpdate(track.kf, det_box);
            if (useOru() && prev_missed > 0 && track.kf.initialized) {
                // ORU（观测速度再更新）：用跨遮挡期的观测位移重新估计速度，
                // 避免继续沿用衰减/滞后后的旧速度导致后续预测漂移；
                // 同时重置速度协方差，让滤波器重新学习。
                const double m = static_cast<double>(prev_missed) + 1.0;
                double vx = (det_box.left + det_box.right - prev_obs.left - prev_obs.right) / (2.0 * m);
                double vy = (det_box.top + det_box.bottom - prev_obs.top - prev_obs.bottom) / (2.0 * m);
                double vw = (static_cast<double>(det_box.right - det_box.left) -
                             static_cast<double>(prev_obs.right - prev_obs.left)) / m;
                double vh = (static_cast<double>(det_box.bottom - det_box.top) -
                             static_cast<double>(prev_obs.bottom - prev_obs.top)) / m;
                const double w = std::max(1.0, static_cast<double>(det_box.right - det_box.left));
                const double h = std::max(1.0, static_cast<double>(det_box.bottom - det_box.top));
                const double vmax = std::max(20.0, std::sqrt(w * w + h * h) * 1.5);
                track.kf.x[4] = std::max(-vmax, std::min(vmax, vx));
                track.kf.x[5] = std::max(-vmax, std::min(vmax, vy));
                track.kf.x[6] = std::max(-vmax, std::min(vmax, vw));
                track.kf.x[7] = std::max(-vmax, std::min(vmax, vh));
                for (int i = 4; i < 8; ++i) {
                    track.kf.P[i][i] = 100.0;
                }
            }
            track.last_obs_box = det_box;
            track.box = det_box;
            if (track.hits <= 1) {
                track.smooth_box = det_box;
            } else {
                float adaptive_alpha = computeAdaptiveSmoothAlpha(prev_box, det_box, box_smooth_alpha_);
                track.smooth_box = blendBox(track.smooth_box, det_box, adaptive_alpha);
            }
            track.smooth_box = applyMotionCompensation(track.smooth_box, track, motion_compensation_);
            track.last_prop = det.prop;
            tracks_[ti].missed = 0;
            tracks_[ti].hits++;
            if (!tracks_[ti].confirmed && tracks_[ti].hits >= min_confirm_hits_) {
                tracks_[ti].confirmed = true;
            }
            if (tracks_[ti].confirmed && !tracks_[ti].counted && class_counter) {
                (*class_counter)[tracks_[ti].cls_id] += 1;
                tracks_[ti].counted = true;
            }
            det_is_confirmed[di] = tracks_[ti].confirmed ? 1 : 0;
            det_smooth_boxes[di] = tracks_[ti].smooth_box;
            det_has_smooth_box[di] = 1;
        } else {
            tracks_[ti].missed++;
        }
    }

    // 丢弃统计（RK_PIPE_DEBUG_DROP=1）：诊断"跟丢重建"时区分未确认/超龄删除
    static const bool dbg_drop = []() {
        const char* v = getenv("RK_PIPE_DEBUG_DROP");
        return v && *v && strcmp(v, "0") != 0;
    }();
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [this, &dbg_drop](const TrackState& t) {
        const bool drop_unconfirmed = !t.confirmed && t.missed > 1;
        const bool drop_aged = t.missed > max_missed_;
        if (dbg_drop && (drop_unconfirmed || drop_aged)) {
            std::fprintf(stderr, "[track-drop] id=%d conf=%d hits=%d missed=%d prop=%.2f reason=%s\n",
                         t.track_id, t.confirmed ? 1 : 0, t.hits, t.missed, t.last_prop,
                         drop_unconfirmed ? "unconfirmed" : "aged");
        }
        return drop_unconfirmed || drop_aged;
    }), tracks_.end());

    // 仅未匹配的高分检测创建新轨迹；低分检测不建新轨迹（ByteTrack 策略，抑制噪声轨迹）
    for (int di : high_dets) {
        if (det_used[di]) {
            continue;
        }
        const object_detect_result& det = detections.results[di];
        TrackState track;
        track.track_id = next_track_id_++;
        track.cls_id = det.cls_id;
        track.box = det.box;
        track.pred_box = det.box;
        track.smooth_box = det.box;
        track.last_obs_box = det.box;
        track.missed = 0;
        track.hits = 1;
        track.last_prop = det.prop;
        track.confirmed = min_confirm_hits_ <= 1;
        track.counted = false;
        kfInit(track.kf, det.box);
        tracks_.push_back(track);
        det_to_track_id[di] = track.track_id;
        det_is_new[di] = 1;
        if (track.confirmed && class_counter) {
            (*class_counter)[det.cls_id] += 1;
            tracks_.back().counted = true;
        }
        det_is_confirmed[di] = track.confirmed ? 1 : 0;
        det_smooth_boxes[di] = det.box;
        det_has_smooth_box[di] = 1;
    }

    // 限制轨迹总量：优先淘汰未确认轨迹，再淘汰 missed 最多（最久未匹配）的。
    // 已确认并计数的轨迹即使被淘汰也不丢计数（计数记录在 class_counter 中）。
    if (static_cast<int>(tracks_.size()) > max_tracks_) {
        std::stable_sort(tracks_.begin(), tracks_.end(), [](const TrackState& a, const TrackState& b) {
            if (a.confirmed != b.confirmed) {
                return !a.confirmed;
            }
            return a.missed > b.missed;
        });
        tracks_.erase(tracks_.begin() + max_tracks_, tracks_.end());
    }

    std::vector<TrackedDetection> output;
    output.reserve(det_count);
    for (int di = 0; di < det_count; ++di) {
        TrackedDetection td;
        td.det = detections.results[di];
        if (det_has_smooth_box[di]) {
            td.det.box = det_smooth_boxes[di];
        }
        td.track_id = det_to_track_id[di];
        td.is_new = det_is_new[di] != 0;
        td.is_confirmed = det_is_confirmed[di] != 0;
        td.is_predicted = false;
        output.push_back(td);
    }

    for (const auto& track : tracks_) {
        if (!track.confirmed || track.missed <= 0 || track.missed > render_max_missed_) {
            continue;
        }
        // 已确认轨迹缺检时输出预测框（持续帧数由 render_max_missed_ 控制）。
        // last_prop 门槛只放行"原本就是高分"的目标，避免低分闪烁目标产生伪预测框。
        if (track.hits < min_confirm_hits_) {
            continue;
        }
        if (track.last_prop < 0.6f) {
            continue;
        }
        if (isVelocityTooLarge(track)) {
            continue;
        }
        TrackedDetection td;
        td.track_id = track.track_id;
        td.is_new = false;
        td.is_confirmed = true;
        td.is_predicted = true;
        td.det.cls_id = track.cls_id;
        td.det.prop = std::max(0.5f, track.last_prop * 0.9f);
        td.det.box = track.pred_box;
        output.push_back(td);
    }
    return output;
}

// ---------------- 几何工具 ----------------

float SimpleObjectTracker::computeIoU(const image_rect_t& a, const image_rect_t& b) {
    int left = std::max(a.left, b.left);
    int top = std::max(a.top, b.top);
    int right = std::min(a.right, b.right);
    int bottom = std::min(a.bottom, b.bottom);
    int w = std::max(0, right - left);
    int h = std::max(0, bottom - top);
    float inter = static_cast<float>(w) * static_cast<float>(h);
    float area_a = static_cast<float>(std::max(0, a.right - a.left)) * static_cast<float>(std::max(0, a.bottom - a.top));
    float area_b = static_cast<float>(std::max(0, b.right - b.left)) * static_cast<float>(std::max(0, b.bottom - b.top));
    float uni = area_a + area_b - inter;
    if (uni <= 1e-6f) {
        return 0.0f;
    }
    return inter / uni;
}

float SimpleObjectTracker::computeCenterDistance(const image_rect_t& a, const image_rect_t& b) {
    float acx = 0.5f * (static_cast<float>(a.left) + static_cast<float>(a.right));
    float acy = 0.5f * (static_cast<float>(a.top) + static_cast<float>(a.bottom));
    float bcx = 0.5f * (static_cast<float>(b.left) + static_cast<float>(b.right));
    float bcy = 0.5f * (static_cast<float>(b.top) + static_cast<float>(b.bottom));
    float dx = acx - bcx;
    float dy = acy - bcy;
    return std::sqrt(dx * dx + dy * dy);
}

float SimpleObjectTracker::computeMaxCenterDistance(const image_rect_t& box, float scale) {
    float w = static_cast<float>(std::max(1, box.right - box.left));
    float h = static_cast<float>(std::max(1, box.bottom - box.top));
    float diag = std::sqrt(w * w + h * h);
    return std::max(10.0f, diag * scale);
}

bool SimpleObjectTracker::sizeRatioOk(const image_rect_t& a, const image_rect_t& b, float min_ratio) {
    float area_a = static_cast<float>(std::max(1, a.right - a.left)) * static_cast<float>(std::max(1, a.bottom - a.top));
    float area_b = static_cast<float>(std::max(1, b.right - b.left)) * static_cast<float>(std::max(1, b.bottom - b.top));
    if (area_a <= 0.0f || area_b <= 0.0f) {
        return false;
    }
    float ratio = area_a < area_b ? area_a / area_b : area_b / area_a;
    return ratio >= min_ratio;
}

image_rect_t SimpleObjectTracker::shiftBox(const image_rect_t& box, float dx, float dy, float dw, float dh) {
    float left = static_cast<float>(box.left) + dx;
    float top = static_cast<float>(box.top) + dy;
    float right = static_cast<float>(box.right) + dx;
    float bottom = static_cast<float>(box.bottom) + dy;
    float width = std::max(2.0f, right - left + dw);
    float height = std::max(2.0f, bottom - top + dh);
    float cx = 0.5f * (left + right);
    float cy = 0.5f * (top + bottom);
    image_rect_t out;
    out.left = static_cast<int>(std::round(cx - 0.5f * width));
    out.top = static_cast<int>(std::round(cy - 0.5f * height));
    out.right = static_cast<int>(std::round(cx + 0.5f * width));
    out.bottom = static_cast<int>(std::round(cy + 0.5f * height));
    return out;
}

image_rect_t SimpleObjectTracker::blendBox(const image_rect_t& prev_box, const image_rect_t& cur_box, float alpha) {
    image_rect_t out;
    float beta = 1.0f - alpha;
    out.left = static_cast<int>(std::round(alpha * static_cast<float>(prev_box.left) + beta * static_cast<float>(cur_box.left)));
    out.top = static_cast<int>(std::round(alpha * static_cast<float>(prev_box.top) + beta * static_cast<float>(cur_box.top)));
    out.right = static_cast<int>(std::round(alpha * static_cast<float>(prev_box.right) + beta * static_cast<float>(cur_box.right)));
    out.bottom = static_cast<int>(std::round(alpha * static_cast<float>(prev_box.bottom) + beta * static_cast<float>(cur_box.bottom)));
    return out;
}

float SimpleObjectTracker::computeAdaptiveSmoothAlpha(const image_rect_t& prev_box, const image_rect_t& new_box, float base_alpha) {
    float move_dist = computeCenterDistance(prev_box, new_box);
    float max_dist = computeMaxCenterDistance(prev_box, 0.5f);
    float ratio = max_dist > 1e-6f ? move_dist / max_dist : 1.0f;
    float boosted_alpha = base_alpha + (1.0f - std::min(1.0f, ratio)) * 0.10f;
    return std::min(0.78f, std::max(0.30f, boosted_alpha));
}

bool SimpleObjectTracker::isVelocityTooLarge(const TrackState& track) {
    if (!track.kf.initialized) {
        return true;
    }
    double vx = track.kf.x[4];
    double vy = track.kf.x[5];
    float speed = static_cast<float>(std::sqrt(vx * vx + vy * vy));
    float max_dist = computeMaxCenterDistance(track.smooth_box, 0.35f);
    return speed > max_dist;
}

image_rect_t SimpleObjectTracker::applyMotionCompensation(const image_rect_t& box, const TrackState& track, float factor) {
    float dx = 0.0f;
    float dy = 0.0f;
    if (track.kf.initialized) {
        dx = static_cast<float>(track.kf.x[4]) * factor;
        dy = static_cast<float>(track.kf.x[5]) * factor;
    }
    return shiftBox(box, dx, dy, 0.0f, 0.0f);
}

// ---------------- Kalman 滤波 ----------------

void SimpleObjectTracker::kfInit(KalmanFilter8& kf, const image_rect_t& box) {
    double w = std::max(1.0, static_cast<double>(box.right - box.left));
    double h = std::max(1.0, static_cast<double>(box.bottom - box.top));
    kf.x[0] = 0.5 * (box.left + box.right);
    kf.x[1] = 0.5 * (box.top + box.bottom);
    kf.x[2] = w;
    kf.x[3] = h;
    kf.x[4] = 0;
    kf.x[5] = 0;
    kf.x[6] = 0;
    kf.x[7] = 0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            kf.P[i][j] = 0;
        }
    }
    kf.P[0][0] = std::max(9.0, 0.01 * w * w);
    kf.P[1][1] = std::max(9.0, 0.01 * h * h);
    kf.P[2][2] = std::max(9.0, 0.01 * w * w);
    kf.P[3][3] = std::max(9.0, 0.01 * h * h);
    kf.P[4][4] = kf.P[5][5] = kf.P[6][6] = kf.P[7][7] = 100.0;
    // 测量噪声：按框尺寸比例，避免大框/小框统一像素噪声导致的过拟合
    double rw = std::max(4.0, 0.04 * w);
    double rh = std::max(4.0, 0.04 * h);
    double rw2 = rw * rw;
    double rh2 = rh * rh;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            kf.R[i][j] = 0;
        }
    }
    kf.R[0][0] = rw2;
    kf.R[1][1] = rh2;
    kf.R[2][2] = rw2;
    kf.R[3][3] = rh2;
    kf.initialized = true;
}

void SimpleObjectTracker::kfPredict(KalmanFilter8& kf, float dt) {
    if (!kf.initialized) {
        return;
    }
    const double d = static_cast<double>(dt);
    // x = F * x  （F：x[i] += d * x[i+4] for i<4）
    for (int i = 0; i < 4; ++i) {
        kf.x[i] += kf.x[i + 4] * d;
    }
    // P = F * P * F^T + Q
    double FP[8][8];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double s = kf.P[i][j];
            if (i < 4) {
                s += d * kf.P[i + 4][j];
            }
            FP[i][j] = s;
        }
    }
    double Pn[8][8];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double s = FP[i][j];
            if (j < 4) {
                s += d * FP[i][j + 4];
            }
            Pn[i][j] = s;
        }
    }
    // 过程噪声（方差增量，随 dt 缩放）
    const double q_p = 1.0 * d;
    const double q_v = 64.0 * d;
    for (int i = 0; i < 8; ++i) {
        Pn[i][i] += (i < 4) ? q_p : q_v;
        for (int j = 0; j < 8; ++j) {
            kf.P[i][j] = Pn[i][j];
        }
    }
}

void SimpleObjectTracker::kfUpdate(KalmanFilter8& kf, const image_rect_t& box) {
    if (!kf.initialized) {
        return;
    }
    double z[4];
    z[0] = 0.5 * (box.left + box.right);
    z[1] = 0.5 * (box.top + box.bottom);
    z[2] = std::max(1.0, static_cast<double>(box.right - box.left));
    z[3] = std::max(1.0, static_cast<double>(box.bottom - box.top));

    double y[4];
    for (int i = 0; i < 4; ++i) {
        y[i] = z[i] - kf.x[i];
    }
    // S = H P H^T + R （H = [I4 | 0]，故 S[i][j] = P[i][j] + R[i][j]）
    double S[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            S[i][j] = kf.P[i][j] + kf.R[i][j];
        }
    }
    double Sinv[4][4];
    invert4x4(S, Sinv);
    double K[8][4];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            double s = 0;
            for (int k = 0; k < 4; ++k) {
                s += kf.P[i][k] * Sinv[k][j];
            }
            K[i][j] = s;
        }
    }
    for (int i = 0; i < 8; ++i) {
        double s = 0;
        for (int j = 0; j < 4; ++j) {
            s += K[i][j] * y[j];
        }
        kf.x[i] += s;
    }
    // P = (I - K H) P
    double Pn[8][8];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double s = kf.P[i][j];
            for (int k = 0; k < 4; ++k) {
                s -= K[i][k] * kf.P[k][j];
            }
            Pn[i][j] = s;
        }
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            kf.P[i][j] = Pn[i][j];
        }
    }
}

image_rect_t SimpleObjectTracker::boxFromKf(const KalmanFilter8& kf) {
    image_rect_t out;
    if (!kf.initialized) {
        out.left = out.top = out.right = out.bottom = 0;
        return out;
    }
    double cx = kf.x[0];
    double cy = kf.x[1];
    double w = std::max(1.0, kf.x[2]);
    double h = std::max(1.0, kf.x[3]);
    out.left = static_cast<int>(std::round(cx - 0.5 * w));
    out.top = static_cast<int>(std::round(cy - 0.5 * h));
    out.right = static_cast<int>(std::round(cx + 0.5 * w));
    out.bottom = static_cast<int>(std::round(cy + 0.5 * h));
    return out;
}

void SimpleObjectTracker::invert4x4(const double in[4][4], double out[4][4]) {
    double a[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            a[i][j] = in[i][j];
            a[i][j + 4] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int r = col + 1; r < 4; ++r) {
            if (std::fabs(a[r][col]) > std::fabs(a[piv][col])) {
                piv = r;
            }
        }
        if (std::fabs(a[piv][col]) < 1e-12) {
            continue;
        }
        if (piv != col) {
            for (int j = 0; j < 8; ++j) {
                std::swap(a[col][j], a[piv][j]);
            }
        }
        double inv = 1.0 / a[col][col];
        for (int j = 0; j < 8; ++j) {
            a[col][j] *= inv;
        }
        for (int r = 0; r < 4; ++r) {
            if (r == col) {
                continue;
            }
            double factor = a[r][col];
            if (std::fabs(factor) < 1e-12) {
                continue;
            }
            for (int j = 0; j < 8; ++j) {
                a[r][j] -= factor * a[col][j];
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[i][j] = a[i][j + 4];
        }
    }
}

// ---------------- 关联（匈牙利 + ByteTrack 两阶段） ----------------

void SimpleObjectTracker::hungarianMinCost(int n, int m,
                                           const std::vector<std::vector<double>>& cost,
                                           std::vector<int>& row_assign,
                                           std::vector<int>& col_assign) {
    row_assign.assign(n, -1);
    col_assign.assign(m, -1);
    if (n <= 0 || m <= 0) {
        return;
    }
    const int N = std::max(n, m);
    // 显式补成 N×N 方阵：越界的 padding 行/列用有限的 kMatchInf 填充。
    // 标准 e-maxx 要求完全方阵（每格有限代价），隐式 INF 会让 padding 行的
    // 增广路径依赖上一行残留的 way[]，可能成环死循环或产生次优分配。
    std::vector<std::vector<double>> a(N + 1, std::vector<double>(N + 1, kMatchInf));
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < m; ++c) {
            a[r + 1][c + 1] = cost[r][c];
        }
    }
    std::vector<double> u(N + 1, 0.0), v(N + 1, 0.0), minv(N + 1, 0.0);
    std::vector<int> p(N + 1, 0), way(N + 1, 0);
    std::vector<char> used(N + 1, 0);
    for (int i = 1; i <= N; ++i) {
        p[0] = i;
        int j0 = 0;
        std::fill(minv.begin(), minv.end(), kNoEdge);
        std::fill(used.begin(), used.end(), 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            double delta = kNoEdge;
            int j1 = 0;
            for (int j = 1; j <= N; ++j) {
                if (used[j]) {
                    continue;
                }
                const double cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= N; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    for (int j = 1; j <= N; ++j) {
        const int r = p[j];
        if (r >= 1 && r <= n && j <= m) {
            row_assign[r - 1] = j - 1;
            col_assign[j - 1] = r - 1;
        }
    }
}

void SimpleObjectTracker::associate(const object_detect_result_list& detections,
                                    const std::vector<int>& track_indices,
                                    const std::vector<int>& det_indices,
                                    float iou_gate,
                                    bool relaxed_for_missed,
                                    std::vector<int>& track_to_det,
                                    std::vector<int>& det_to_track_id,
                                    std::vector<char>& det_used,
                                    std::vector<char>& track_used) const {
    if (track_indices.empty() || det_indices.empty()) {
        return;
    }
    const int m = static_cast<int>(det_indices.size());
    // 只把“至少有一个有效候选”的轨迹纳入匈牙利矩阵：既降低 O(N^3) 开销，
    // 也能避免大量长期未匹配的 stale 轨迹挤占全局匹配。
    std::vector<std::vector<double>> cost;
    std::vector<int> row_track;
    cost.reserve(track_indices.size());
    for (int ri : track_indices) {
        const TrackState& track = tracks_[ri];
        std::vector<double> row(m, kMatchInf);
        bool has_candidate = false;
        for (int ci = 0; ci < m; ++ci) {
            const int di = det_indices[ci];
            const object_detect_result& det = detections.results[di];
            if (track.cls_id != det.cls_id) {
                continue;
            }
            const float center_dist = computeCenterDistance(track.pred_box, det.box);
            if (!sizeRatioOk(track.pred_box, det.box, size_ratio_threshold_)) {
                continue;
            }
            // OC-SORT 风格遮挡恢复：近期漏检的轨迹允许仅凭中心距离 + 尺寸比重连。
            // 预测漂移会导致 IoU≈0，几何上却仍是同一目标，此时不再要求框重叠。
            if (useRelink() && relaxed_for_missed && track.missed > 0 && track.missed <= relink_max_missed_) {
                const float max_dist = computeMaxCenterDistance(track.pred_box, relink_center_scale_);
                if (center_dist > max_dist) {
                    continue;
                }
                row[ci] = std::min(1.0, static_cast<double>(center_dist) / static_cast<double>(max_dist));
                has_candidate = true;
                continue;
            }
            const float iou = computeIoU(track.pred_box, det.box);
            float gate = iou_gate;
            if (relaxed_for_missed && track.missed > 0) {
                gate = reid_iou_threshold_;
            }
            if (iou < gate) {
                continue;
            }
            const float max_dist = computeMaxCenterDistance(track.pred_box, center_distance_threshold_);
            if (center_dist > max_dist) {
                continue;
            }
            row[ci] = 1.0 - static_cast<double>(iou);
            has_candidate = true;
        }
        if (!has_candidate) {
            continue;
        }
        cost.push_back(std::move(row));
        row_track.push_back(ri);
    }

    const int n = static_cast<int>(cost.size());
    if (n == 0) {
        return;
    }

    std::vector<int> assigned_row(n, -1);
    if (n * m <= kMaxHungarianCells) {
        std::vector<int> row_assign, col_assign;
        hungarianMinCost(n, m, cost, row_assign, col_assign);
        for (int r = 0; r < n; ++r) {
            const int ci = row_assign[r];
            if (ci < 0 || cost[r][ci] >= kMatchUnassigned) {
                continue;
            }
            assigned_row[r] = ci;
        }
    } else {
        // 稠密矩阵：按每行最优代价排序，逐行贪心选最优未用列（O(n*m)，避免匈牙利膨胀）
        std::vector<double> row_best(n, kMatchInf);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < m; ++c) {
                if (cost[r][c] < row_best[r]) {
                    row_best[r] = cost[r][c];
                }
            }
        }
        std::vector<int> order(n);
        for (int r = 0; r < n; ++r) {
            order[r] = r;
        }
        std::sort(order.begin(), order.end(), [&row_best](int a, int b) { return row_best[a] < row_best[b]; });
        std::vector<char> col_used(m, 0);
        for (int r : order) {
            int best_c = -1;
            double best_cost = kMatchInf;
            for (int c = 0; c < m; ++c) {
                if (col_used[c] || cost[r][c] >= best_cost) {
                    continue;
                }
                best_cost = cost[r][c];
                best_c = c;
            }
            if (best_c >= 0 && best_cost < kMatchUnassigned) {
                assigned_row[r] = best_c;
                col_used[best_c] = 1;
            }
        }
    }

    for (int r = 0; r < n; ++r) {
        const int ci = assigned_row[r];
        if (ci < 0) {
            continue;
        }
        const int ti = row_track[r];
        const int di = det_indices[ci];
        track_to_det[ti] = di;
        det_to_track_id[di] = tracks_[ti].track_id;
        track_used[ti] = 1;
        det_used[di] = 1;
    }
}

object_detect_result_list buildTrackedResultList(const std::vector<TrackedDetection>& tracked) {
    object_detect_result_list out = {};
    out.count = 0;
    for (const auto& td : tracked) {
        // 分数达标即画框（未确认目标也画，确认后才叠 ID——见 tracking_overlay）。
        // 原 0.6 门槛会把 0.3~0.6 的真实检测在零拷贝路径下过滤掉。
        if (td.det.prop < displayThreshold()) {
            continue;
        }
        if (out.count >= OBJ_NUMB_MAX_SIZE) {
            break;
        }
        out.results[out.count++] = td.det;
    }
    return out;
}
