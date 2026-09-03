#include "detection/tracking_runtime.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#include "detection/tracking_overlay.h"
#include "utils/draw_utils.h"

namespace {

TrackAlgorithm parseTrackAlgorithm(const std::string& name) {
    if (name == "iou") {
        return TrackAlgorithm::Iou;
    }
    if (name == "sort") {
        return TrackAlgorithm::Sort;
    }
    if (name == "bytetrack" || name == "byte") {
        return TrackAlgorithm::ByteTrack;
    }
    return TrackAlgorithm::OcSort;  // "ocsort" / "auto" / 未知值均回退到默认
}

// 判断 task_type 是否在逗号分隔的任务名白名单（track_tasks）中
bool taskTypeInList(TaskType task_type, const std::string& track_tasks) {
    std::string needle;
    switch (task_type) {
        case TaskType::Pose:
            needle = "pose";
            break;
        case TaskType::OBB:
            needle = "obb";
            break;
        case TaskType::Seg:
            needle = "seg";
            break;
        default:
            needle = "detect";
            break;
    }
    size_t start = 0;
    while (start <= track_tasks.size()) {
        size_t comma = track_tasks.find(',', start);
        std::string token = track_tasks.substr(
            start, comma == std::string::npos ? track_tasks.size() - start : comma - start);
        token.erase(std::remove_if(token.begin(), token.end(),
                                   [](unsigned char c) { return std::isspace(c) != 0; }),
                    token.end());
        std::transform(token.begin(), token.end(), token.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (token == needle) {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return false;
}

// 旋转框 → 轴对齐外接矩形（供 tracker 关联使用）
image_rect_t obbToAABB(const image_obb_box_t& box) {
    float cx = box.x + box.w * 0.5f;
    float cy = box.y + box.h * 0.5f;
    float c = std::cos(box.angle);
    float s = std::sin(box.angle);
    float hw = box.w * 0.5f;
    float hh = box.h * 0.5f;
    float dx = std::fabs(hw * c) + std::fabs(hh * s);
    float dy = std::fabs(hw * s) + std::fabs(hh * c);
    image_rect_t out;
    out.left = static_cast<int>(std::round(cx - dx));
    out.top = static_cast<int>(std::round(cy - dy));
    out.right = static_cast<int>(std::round(cx + dx));
    out.bottom = static_cast<int>(std::round(cy + dy));
    return out;
}

void renderTrackedOverlay(bool use_zero_copy,
                          cv::Mat& frame_mat,
                          image_buffer_t& frame_buffer,
                          const std::vector<TrackedDetection>& tracked,
                          const std::unordered_map<int, int>& class_counts,
                          bool show_count_overlay) {
    object_detect_result_list stable_results = buildTrackedResultList(tracked);
    if (use_zero_copy) {
        if (stable_results.count > 0) {
            drawDetectionResultsZeroCopy(frame_buffer, stable_results);
        }
        drawTrackingOverlay(frame_buffer, tracked, class_counts, show_count_overlay);
    } else if (!frame_mat.empty()) {
        // 与零拷贝路径对称:框/类别标签由统一绘制(Ultralytics 风格),overlay 只叠 ID 与计数
        if (stable_results.count > 0) {
            drawDetectionResultsBGR(frame_mat, stable_results);
        }
        drawTrackingOverlay(frame_mat, tracked, class_counts, show_count_overlay);
    }
}

}  // namespace

bool isTrackingActive(TaskType task_type, const AppConfig& options) {
    if (!options.enable_tracking) {
        return false;  // 总开关，默认关闭
    }
    if (!options.track_tasks.empty()) {
        return taskTypeInList(task_type, options.track_tasks);
    }
    switch (task_type) {
        case TaskType::Detect:
        case TaskType::Pose:
        case TaskType::OBB:
        case TaskType::Seg:
            return true;
        default:
            return false;  // OCR 等暂不支持
    }
}

SimpleObjectTracker createTracker(const AppConfig& options) {
    return SimpleObjectTracker(options.track_iou_threshold,
                               options.track_max_missed,
                               options.track_min_confirm_hits,
                               options.track_reid_iou_threshold,
                               options.track_center_distance_threshold,
                               options.track_box_smooth_alpha,
                               options.track_render_max_missed,
                               options.track_high_conf_threshold,
                               options.track_low_conf_threshold,
                               options.track_stage2_iou_threshold,
                               options.track_size_ratio_threshold,
                               options.track_max_tracks,
                               parseTrackAlgorithm(options.track_algorithm));
}

void renderPipelineTrackingOutput(const AppConfig& options,
                                  bool output_enabled,
                                  PipelineFrame& frame,
                                  SimpleObjectTracker& tracker,
                                  std::unordered_map<int, int>& class_counts,
                                  std::vector<TrackedDetection>* tracked_out) {
    if (!frame.hasResult) {
        if (tracked_out) {
            tracked_out->clear();
        }
        return;
    }

    const object_detect_result_list* detect_results = getDetectResultList(frame.result);
    if (!detect_results) {
        if (tracked_out) {
            tracked_out->clear();
        }
        return;
    }

    std::vector<TrackedDetection> tracked = tracker.update(*detect_results, &class_counts);
    if (tracked_out) {
        *tracked_out = tracked;  // 无输出叠加时也要回传（事件引擎按帧消费）
    }
    if (!output_enabled) {
        return;
    }

    // 调试：RK_PIPE_DEBUG_TRACK=1 打印 tracking 输出与过滤原因（排查"检测到但不显示"）
    static const bool dbg_track = []() {
        const char* v = getenv("RK_PIPE_DEBUG_TRACK");
        return v && *v && strcmp(v, "0") != 0;
    }();
    if (dbg_track) {
        fprintf(stderr, "[TRACK] dets=%d tracked=%zu:",
                detect_results->count, tracked.size());
        for (const auto& td : tracked) {
            fprintf(stderr, " [id=%d p=%.2f conf=%d pred=%d]",
                    td.track_id, td.det.prop, td.is_confirmed ? 1 : 0,
                    td.is_predicted ? 1 : 0);
        }
        fprintf(stderr, "\n");
    }

    renderTrackedOverlay(frame.isZeroCopy,
                         frame.matFrame,
                         frame.bufferFrame,
                         tracked,
                         class_counts,
                         options.show_count_overlay);
}

void updateAndRenderTracking(const AppConfig& options,
                             bool output_enabled,
                             bool overlay_enabled,
                             bool use_zero_copy,
                             cv::Mat& frame_mat,
                             image_buffer_t& frame_buffer,
                             const object_detect_result_list& detect_results,
                             SimpleObjectTracker& tracker,
                             std::unordered_map<int, int>& class_counts) {
    std::vector<TrackedDetection> tracked = tracker.update(detect_results, &class_counts);
    if (!output_enabled || !overlay_enabled) {
        return;
    }

    renderTrackedOverlay(use_zero_copy,
                         frame_mat,
                         frame_buffer,
                         tracked,
                         class_counts,
                         options.show_count_overlay);
}

// ---- pose / obb / seg 任务的 tracking ----

std::vector<TrackedDetection> trackPoseResults(SimpleObjectTracker& tracker,
                                               pose_detect_result_list& poses,
                                               std::unordered_map<int, int>* class_counter) {
    object_detect_result_list dets = {};
    for (int i = 0; i < poses.count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        object_detect_result& d = dets.results[dets.count++];
        d.box = poses.results[i].box;
        // 传真实模型置信度（不做抬底）：ByteTrack 两阶段里 prop < high_conf 的检测只用来
        // 救援既有轨迹、不会新建轨迹。抬到 0.6 会把低分闪烁候选全部变成高分 → 每次
        // 闪烁都新建一条轨迹，表现为人物"跟丢后被识别成新任务"。解码侧已按 conf 阈值
        // 过滤（默认 0.25），落在 [low_conf, high_conf) 的目标仍能靠低分段维持跟踪。
        d.prop = poses.results[i].box_conf;
        d.cls_id = poses.results[i].cls_id;
    }
    std::vector<TrackedDetection> tracked = tracker.update(dets, class_counter);
    for (int i = 0; i < poses.count && i < static_cast<int>(tracked.size()); ++i) {
        poses.results[i].track_id = tracked[i].track_id > 0 ? tracked[i].track_id : 0;
    }
    return tracked;
}

std::vector<TrackedDetection> trackOBBResults(SimpleObjectTracker& tracker,
                                              obb_detect_result_list& obbs,
                                              std::unordered_map<int, int>* class_counter) {
    object_detect_result_list dets = {};
    for (int i = 0; i < obbs.count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        object_detect_result& d = dets.results[dets.count++];
        d.box = obbToAABB(obbs.results[i].box);
        d.prop = obbs.results[i].prop;
        d.cls_id = obbs.results[i].cls_id;
    }
    std::vector<TrackedDetection> tracked = tracker.update(dets, class_counter);
    for (int i = 0; i < obbs.count && i < static_cast<int>(tracked.size()); ++i) {
        if (tracked[i].track_id > 0) {
            obbs.results[i].track_id = tracked[i].track_id;
            // 旋转框中心用跟踪平滑中心替换，保持 w/h/angle
            image_obb_box_t& ob = obbs.results[i].box;
            float cx = (tracked[i].det.box.left + tracked[i].det.box.right) * 0.5f;
            float cy = (tracked[i].det.box.top + tracked[i].det.box.bottom) * 0.5f;
            ob.x = cx - ob.w * 0.5f;
            ob.y = cy - ob.h * 0.5f;
        }
    }
    return tracked;
}

std::vector<TrackedDetection> trackSegResults(SimpleObjectTracker& tracker,
                                              SegTaskResult& segs,
                                              std::unordered_map<int, int>* class_counter) {
    const seg_detect_result_list& data = segs.data;
    object_detect_result_list dets = {};
    for (size_t i = 0; i < data.boxes.size() && dets.count < OBJ_NUMB_MAX_SIZE; ++i) {
        const cv::Rect& r = data.boxes[i];
        object_detect_result& d = dets.results[dets.count++];
        d.box.left = r.x;
        d.box.top = r.y;
        d.box.right = r.x + r.width;
        d.box.bottom = r.y + r.height;
        d.prop = i < data.scores.size() ? data.scores[i] : 0.6f;
        d.cls_id = i < data.class_ids.size() ? data.class_ids[i] : 0;
    }
    std::vector<TrackedDetection> tracked = tracker.update(dets, class_counter);
    segs.track_ids.assign(dets.count, 0);
    for (int i = 0; i < dets.count; ++i) {
        segs.track_ids[i] = tracked[i].track_id > 0 ? tracked[i].track_id : 0;
    }
    return tracked;
}

void renderPipelinePoseTrackingOutput(const AppConfig& options,
                                      bool output_enabled,
                                      PipelineFrame& frame,
                                      SimpleObjectTracker& tracker,
                                      std::unordered_map<int, int>& class_counts) {
    if (!frame.hasResult) {
        return;
    }
    PoseTaskResult* pose = std::get_if<PoseTaskResult>(&frame.result);
    if (!pose) {
        return;
    }
    std::vector<TrackedDetection> tracked = trackPoseResults(tracker, pose->data, &class_counts);
    if (std::getenv("RK_PIPE_TRACK_DEBUG") != nullptr) {
        std::fprintf(stderr, "[track][pose] dets=%d ids=", pose->data.count);
        for (int i = 0; i < pose->data.count; ++i) {
            std::fprintf(stderr, "%d,", pose->data.results[i].track_id);
        }
        std::fprintf(stderr, "\n");
    }
    if (!output_enabled) {
        return;
    }
    // 用跟踪平滑框替换检测框（骨架关键点保持原检测位置）
    for (int i = 0; i < pose->data.count && i < static_cast<int>(tracked.size()); ++i) {
        if (pose->data.results[i].track_id > 0) {
            pose->data.results[i].box = tracked[i].det.box;
        }
    }
    // 缺检帧：把 tracker 返回的"预测轨迹"（已确认目标、仅缺检≤1帧、高置信）也画出来，
    // 只画框不画骨架（keypoints 置 0）——消除"检出-漏检"闪烁，且不会产生伪框
    // （预测轨迹需连续命中+高置信才产生，单帧误检不满足条件）。
    for (size_t i = pose->data.count; i < tracked.size() && pose->data.count < OBJ_NUMB_MAX_SIZE; ++i) {
        if (!tracked[i].is_predicted) {
            continue;
        }
        pose_detect_result& pr = pose->data.results[pose->data.count++];
        pr.box = tracked[i].det.box;
        pr.box_conf = tracked[i].det.prop;
        pr.cls_id = tracked[i].det.cls_id;
        pr.track_id = tracked[i].track_id;
        memset(pr.keypoints, 0, sizeof(pr.keypoints));  // 无关键点 → 绘制侧只画框
    }
    if (frame.isZeroCopy) {
        drawPoseResultsZeroCopy(frame.bufferFrame, pose->data);
        drawTrackingOverlay(frame.bufferFrame, tracked, class_counts, options.show_count_overlay);
    } else if (!frame.matFrame.empty()) {
        drawPoseResultsBGR(frame.matFrame, pose->data);
        drawTrackedIdsBGR(frame.matFrame, tracked);
    }
}

void renderPipelineOBBTrackingOutput(const AppConfig& options,
                                     bool output_enabled,
                                     PipelineFrame& frame,
                                     SimpleObjectTracker& tracker,
                                     std::unordered_map<int, int>& class_counts) {
    if (!frame.hasResult) {
        return;
    }
    OBBTaskResult* obb = std::get_if<OBBTaskResult>(&frame.result);
    if (!obb) {
        return;
    }
    std::vector<TrackedDetection> tracked = trackOBBResults(tracker, obb->data, &class_counts);
    if (!output_enabled) {
        return;
    }
    if (frame.isZeroCopy) {
        drawOBBResultsZeroCopy(frame.bufferFrame, obb->data);
        drawTrackingOverlay(frame.bufferFrame, tracked, class_counts, options.show_count_overlay);
    } else if (!frame.matFrame.empty()) {
        drawOBBResultsBGR(frame.matFrame, obb->data);
        drawTrackedIdsBGR(frame.matFrame, tracked);
    }
}

void renderPipelineSegTrackingOutput(const AppConfig& options,
                                     bool output_enabled,
                                     PipelineFrame& frame,
                                     SimpleObjectTracker& tracker,
                                     std::unordered_map<int, int>& class_counts) {
    if (!frame.hasResult) {
        return;
    }
    SegTaskResult* seg = std::get_if<SegTaskResult>(&frame.result);
    if (!seg) {
        return;
    }
    std::vector<TrackedDetection> tracked = trackSegResults(tracker, *seg, &class_counts);
    if (!output_enabled) {
        return;
    }
    // 掩膜/框由 worker 照常绘制，这里只叠加 ID（分割掩膜不做框平滑，避免错位）
    if (frame.isZeroCopy) {
        drawTrackingOverlay(frame.bufferFrame, tracked, class_counts, options.show_count_overlay);
    } else if (!frame.matFrame.empty()) {
        drawTrackedIdsBGR(frame.matFrame, tracked);
    }
}
