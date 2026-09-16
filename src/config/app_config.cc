#include "config/app_config.h"

#include <getopt.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "utils/common.h"
#include "postprocess/postprocess.h"

namespace {

bool readStringFromAnyKey(const cv::FileStorage& fs, const std::vector<std::string>& keys, std::string& value) {
    for (const auto& key : keys) {
        const cv::FileNode node = fs[key];
        if (!node.empty()) {
            node >> value;
            return true;
        }
    }
    return false;
}

bool readIntFromAnyKey(const cv::FileStorage& fs, const std::vector<std::string>& keys, int& value) {
    for (const auto& key : keys) {
        const cv::FileNode node = fs[key];
        if (!node.empty()) {
            node >> value;
            return true;
        }
    }
    return false;
}

bool readFloatFromAnyKey(const cv::FileStorage& fs, const std::vector<std::string>& keys, float& value) {
    for (const auto& key : keys) {
        const cv::FileNode node = fs[key];
        if (!node.empty()) {
            node >> value;
            return true;
        }
    }
    return false;
}

bool readDoubleFromAnyKey(const cv::FileStorage& fs, const std::vector<std::string>& keys, double& value) {
    for (const auto& key : keys) {
        const cv::FileNode node = fs[key];
        if (!node.empty()) {
            node >> value;
            return true;
        }
    }
    return false;
}

bool readBoolFromAnyKey(const cv::FileStorage& fs, const std::vector<std::string>& keys, bool& value) {
    for (const auto& key : keys) {
        const cv::FileNode node = fs[key];
        if (!node.empty()) {
            int int_value = value ? 1 : 0;
            node >> int_value;
            value = int_value != 0;
            return true;
        }
    }
    return false;
}

void printValidationError(const std::string& message) {
    if (!message.empty()) {
        std::fprintf(stderr, "%s\n", message.c_str());
    }
}

}  // namespace

const char* AppConfig::localStreamOutput() {
    return "udp://127.0.0.1:1234?pkt_size=1316";
}

bool AppConfig::loadFromFile(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::printf("Failed to open config file: %s\n", filename.c_str());
        return false;
    }

    config_path = filename;

    readStringFromAnyKey(fs, {"model_path"}, model_path);
    readStringFromAnyKey(fs, {"aux_model_path", "aux_model"}, aux_model_path);
    readStringFromAnyKey(fs, {"aux_task"}, aux_task);
    readStringFromAnyKey(fs, {"person_model_path", "person_model"}, person_model_path);
    readStringFromAnyKey(fs, {"rtmpose_head_path", "head_model_path"}, rtmpose_head_path);
    readStringFromAnyKey(fs, {"input_path"}, input_path);
    // 非空时逐帧导出检测结果 JSONL(同 --dump-detections),供 C ABI 门面路径离线评测
    readStringFromAnyKey(fs, {"dump_detections_path", "dump_detections"}, dump_detections_path);
    readStringFromAnyKey(fs, {"output_video_path"}, output_video_path);
    readStringFromAnyKey(fs, {"output_backend"}, output_backend);
    readStringFromAnyKey(fs, {"task"}, task);
    readStringFromAnyKey(fs, {"mode"}, mode);
    readIntFromAnyKey(fs, {"thread_count"}, thread_count);
    readBoolFromAnyKey(fs, {"gui"}, gui);
    readStringFromAnyKey(fs, {"label_path"}, label_path);
    readIntFromAnyKey(fs, {"obj_class_num"}, obj_class_num);
    readFloatFromAnyKey(fs, {"conf_threshold", "conf_thresh"}, conf_threshold);
    readFloatFromAnyKey(fs, {"nms_threshold", "nms_thresh"}, nms_threshold);
    readDoubleFromAnyKey(fs, {"output_fps"}, output_fps);
    readDoubleFromAnyKey(fs, {"display_fps"}, display_fps);
    readDoubleFromAnyKey(fs, {"stream_fps", "write_fps"}, stream_fps);
    readStringFromAnyKey(fs, {"output_quality", "ffmpeg_quality"}, output_quality);
    readDoubleFromAnyKey(fs, {"output_bitrate_mbps", "ffmpeg_bitrate_mbps"}, output_bitrate_mbps);
    readIntFromAnyKey(fs, {"output_crf", "ffmpeg_crf"}, output_crf);
    readStringFromAnyKey(fs, {"output_preset", "ffmpeg_preset"}, output_preset);
    readBoolFromAnyKey(fs, {"output_low_latency", "ffmpeg_low_latency"}, output_low_latency);
    readBoolFromAnyKey(fs, {"web_preview", "preview_web"}, web_preview);
    readStringFromAnyKey(fs, {"web_preview_bind", "preview_web_bind"}, web_preview_bind);
    readIntFromAnyKey(fs, {"web_preview_port", "preview_web_port"}, web_preview_port);
    readIntFromAnyKey(fs, {"web_preview_quality", "preview_web_quality"}, web_preview_quality);
    readStringFromAnyKey(fs, {"web_preview_mode", "preview_web_mode"}, web_preview_mode);
    readIntFromAnyKey(fs, {"web_preview_eof_linger_ms", "preview_web_eof_linger_ms"}, web_preview_eof_linger_ms);
    readIntFromAnyKey(fs, {"web_preview_secondary_port", "preview_web_secondary_port"}, web_preview_secondary_port);
    readStringFromAnyKey(fs, {"web_preview_extra_ports", "preview_web_extra_ports"}, web_preview_extra_ports);
    readStringFromAnyKey(fs, {"web_preview_webrtc_urls", "preview_web_webrtc_urls"}, web_preview_webrtc_urls);
    readBoolFromAnyKey(fs, {"web_preview_replay_pace", "replay_pace"}, web_preview_replay_pace);
    readDoubleFromAnyKey(fs, {"web_preview_scale"}, web_preview_scale);
    readIntFromAnyKey(fs, {"npu_core_mask"}, npu_core_mask);
    readIntFromAnyKey(fs, {"npu_core_start"}, npu_core_start);
    readIntFromAnyKey(fs, {"rtmpose_det_core"}, rtmpose_det_core);
    readStringFromAnyKey(fs, {"rtmpose_pose_cores"}, rtmpose_pose_cores);
    readFloatFromAnyKey(fs, {"rtmpose_target_fps"}, rtmpose_target_fps);
    readIntFromAnyKey(fs, {"rtmpose_stage2_threads"}, rtmpose_stage2_threads);
    readFloatFromAnyKey(fs, {"rtmpose_box_pad"}, rtmpose_box_pad);
    readIntFromAnyKey(fs, {"drop_frames_on_overflow"}, drop_frames_on_overflow);
    readStringFromAnyKey(fs, {"roi"}, roi);
    readStringFromAnyKey(fs, {"mask_classes"}, mask_classes);
    readBoolFromAnyKey(fs, {"seg_mask"}, seg_mask);
    readBoolFromAnyKey(fs, {"enable_tracking", "tracking"}, enable_tracking);
    readStringFromAnyKey(fs, {"track_tasks", "track_tasks_whitelist"}, track_tasks);
    readBoolFromAnyKey(fs, {"show_count_overlay", "count_overlay"}, show_count_overlay);
    readFloatFromAnyKey(fs, {"track_iou_threshold", "track_iou"}, track_iou_threshold);
    readIntFromAnyKey(fs, {"track_max_missed", "track_miss"}, track_max_missed);
    readIntFromAnyKey(fs, {"track_min_confirm_hits", "track_confirm_hits"}, track_min_confirm_hits);
    readFloatFromAnyKey(fs, {"track_reid_iou_threshold", "track_reid_iou"}, track_reid_iou_threshold);
    readFloatFromAnyKey(fs, {"track_center_distance_threshold", "track_center_dist"}, track_center_distance_threshold);
    readFloatFromAnyKey(fs, {"track_box_smooth_alpha", "track_smooth_alpha"}, track_box_smooth_alpha);
    readIntFromAnyKey(fs, {"track_render_max_missed", "track_render_missed"}, track_render_max_missed);
    readFloatFromAnyKey(fs, {"track_high_conf_threshold", "track_high_conf"}, track_high_conf_threshold);
    readFloatFromAnyKey(fs, {"track_low_conf_threshold", "track_low_conf"}, track_low_conf_threshold);
    readFloatFromAnyKey(fs, {"track_stage2_iou_threshold", "track_stage2_iou"}, track_stage2_iou_threshold);
    readFloatFromAnyKey(fs, {"track_size_ratio_threshold", "track_size_ratio"}, track_size_ratio_threshold);
    readIntFromAnyKey(fs, {"track_max_tracks"}, track_max_tracks);
    readStringFromAnyKey(fs, {"track_algorithm", "track_algo"}, track_algorithm);
    readBoolFromAnyKey(fs, {"alert_enabled", "alert"}, alert_enabled);
    readStringFromAnyKey(fs, {"alert_webhook_url", "alert_webhook"}, alert_webhook_url);
    readIntFromAnyKey(fs, {"alert_min_count"}, alert_min_count);
    readDoubleFromAnyKey(fs, {"alert_interval_s", "alert_interval"}, alert_interval_s);
    readStringFromAnyKey(fs, {"alert_snapshot_dir"}, alert_snapshot_dir);
    readDoubleFromAnyKey(fs, {"alert_dedup_interval_s", "alert_dedup_interval"}, alert_dedup_interval_s);
    readStringFromAnyKey(fs, {"alert_dedup_state_dir"}, alert_dedup_state_dir);
    readBoolFromAnyKey(fs, {"alert_clip"}, alert_clip);
    readDoubleFromAnyKey(fs, {"alert_clip_pre_seconds"}, alert_clip_pre_seconds);
    readDoubleFromAnyKey(fs, {"alert_clip_max_fps"}, alert_clip_max_fps);
    readDoubleFromAnyKey(fs, {"snapshot_interval_s", "snapshot_interval"}, snapshot_interval_s);
    readStringFromAnyKey(fs, {"snapshot_dir"}, snapshot_dir);
    readStringFromAnyKey(fs, {"event_stats_path", "event_stats_file"}, event_stats_path);
    // 本地录像分段滚动
    readIntFromAnyKey(fs, {"record_segment_s", "record_segment"}, record_segment_s);
    readIntFromAnyKey(fs, {"record_keep"}, record_keep);
    // M0 两阶级联（task=composite_cls）
    readStringFromAnyKey(fs, {"cls_model_path", "cls_model"}, cls_model_path);
    readStringFromAnyKey(fs, {"cls_labels_path", "cls_labels"}, cls_labels_path);
    // M6 OCR 文字识别二级级联（task=ocr_det）
    readStringFromAnyKey(fs, {"ocr_rec_model_path", "ocr_rec_model"}, ocr_rec_model_path);
    readStringFromAnyKey(fs, {"ocr_dict_path", "ocr_dict"}, ocr_dict_path);
    readIntFromAnyKey(fs, {"ocr_rec_blank_index"}, ocr_rec_blank_index);
    readDoubleFromAnyKey(fs, {"ocr_rec_crop_pad"}, ocr_rec_crop_pad);
    // M13 动作识别级联（task=pose）
    readStringFromAnyKey(fs, {"action_model_path", "action_model"}, action_model_path);
    readStringFromAnyKey(fs, {"action_labels", "action_labels_path"}, action_labels);
    readIntFromAnyKey(fs, {"action_window_t", "action_window"}, action_window_t);
    readDoubleFromAnyKey(fs, {"action_interval"}, action_interval);
    readStringFromAnyKey(fs, {"action_norm"}, action_norm);
    // D3 检测+单目测距（需 aux_task=depth）
    readBoolFromAnyKey(fs, {"depth_dist_text"}, depth_dist_text);
    readFloatFromAnyKey(fs, {"depth_dist_near_m", "depth_dist_near"}, depth_dist_near_m);
    readFloatFromAnyKey(fs, {"depth_dist_scale"}, depth_dist_scale);
    readStringFromAnyKey(fs, {"detect3d_p2"}, detect3d_p2);
    readFloatFromAnyKey(fs, {"detect3d_depth_scale"}, detect3d_depth_scale);
    // 事件规则引擎（D1/D2）
    readStringFromAnyKey(fs, {"event_region"}, event_region);
    readStringFromAnyKey(fs, {"event_line"}, event_line);
    readStringFromAnyKey(fs, {"event_classes"}, event_classes);
    readBoolFromAnyKey(fs, {"event_line_cross"}, event_line_cross);
    readBoolFromAnyKey(fs, {"event_intrusion"}, event_intrusion);
    readBoolFromAnyKey(fs, {"event_dwell"}, event_dwell);
    readDoubleFromAnyKey(fs, {"event_dwell_seconds", "event_dwell"}, event_dwell_seconds);
    readBoolFromAnyKey(fs, {"event_absence"}, event_absence);
    readDoubleFromAnyKey(fs, {"event_absence_seconds", "event_absence"}, event_absence_seconds);
    readDoubleFromAnyKey(fs, {"event_stats_interval_s", "event_stats_interval"}, event_stats_interval_s);
    // 命名多规则（event_rules：YAML 序列，每项为 map）；非空时优先于扁平 event_* 键
    {
        const cv::FileNode rules_node = fs["event_rules"];
        if (rules_node.isSeq()) {
            for (const auto& n : rules_node) {
                EventRuleConfig r;
                const auto rdStr = [&](const char* key, std::string* v) {
                    const cv::FileNode f = n[key];
                    if (!f.empty()) {
                        f >> *v;
                    }
                };
                const auto rdNum = [&](const char* key, double* v) {
                    const cv::FileNode f = n[key];
                    if (!f.empty()) {
                        f >> *v;
                    }
                };
                const auto rdInt = [&](const char* key, int* v) {
                    const cv::FileNode f = n[key];
                    if (!f.empty()) {
                        f >> *v;
                    }
                };
                rdStr("id", &r.id);
                rdStr("type", &r.type);
                rdStr("line", &r.line);
                rdStr("region", &r.region);
                rdStr("classes", &r.classes);
                rdStr("direction", &r.direction);
                rdStr("ref_point", &r.ref_point);
                rdNum("dwell_seconds", &r.dwell_seconds);
                rdNum("absence_seconds", &r.absence_seconds);
                rdNum("seconds", &r.seconds);
                rdInt("min_count", &r.min_count);
                rdStr("llm_hint", &r.llm_hint);
                rdNum("fall_seconds", &r.fall_seconds);
                rdNum("fall_aspect", &r.fall_aspect);
                rdNum("speed_limit", &r.speed_limit);
                rdNum("px_per_meter", &r.px_per_meter);
                rdNum("speed_seconds", &r.speed_seconds);
                event_rules.push_back(std::move(r));
            }
        }
    }

    // push_local 默认关闭（无副作用）：只有 YAML 显式配置 push_local/local=1 时才允许
    // 本地 UDP 推流 + ffplay 拉起；此时若未指定输出路径，回退到默认本地流地址。
    bool read_push_local = readBoolFromAnyKey(fs, {"push_local"}, push_local);
    if (!read_push_local) {
        readBoolFromAnyKey(fs, {"local"}, push_local);
    }
    if (push_local && output_video_path.empty()) {
        output_video_path = localStreamOutput();
    }

    std::printf("Loaded configuration from %s\n", filename.c_str());
    return true;
}

bool AppConfig::validate(std::string* error_message) const {
    if (input_path.empty()) {
        if (error_message) {
            *error_message = "Error: --video is required argument (or via config file)";
        }
        return false;
    }
    if (model_path.empty()) {
        if (error_message) {
            *error_message = "Error: --model is required argument (or via config file)";
        }
        return false;
    }
    // rtmpose 两阶段：person_model_path 为第一阶段人体检测模型，必填
    {
        std::string lower_task = task;
        for (auto& ch : lower_task) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        if (lower_task == "rtmpose" && person_model_path.empty()) {
            if (error_message) {
                *error_message =
                    "Error: task=rtmpose requires person_model_path (stage-1 person detection model)";
            }
            return false;
        }
    }
    if (!aux_model_path.empty()) {
        std::string aux = aux_task;
        for (auto& ch : aux) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        const bool aux_ok = aux == "detect" || aux == "pose" || aux == "obb" ||
                            aux == "seg" || aux == "depth" || aux == "sem" || aux == "ocr_det";
        if (!aux_ok) {
            if (error_message) {
                *error_message =
                    "Error: --aux-task must be one of detect/pose/obb/seg/depth/sem/ocr_det";
            }
            return false;
        }
    }
    if (depth_dist_near_m < 0.0f) {
        if (error_message) {
            *error_message = "Error: depth_dist_near_m must be >= 0 (meters, 0=off)";
        }
        return false;
    }
    if (depth_dist_scale <= 0.0f) {
        if (error_message) {
            *error_message = "Error: depth_dist_scale must be > 0";
        }
        return false;
    }
    if (thread_count <= 0) {
        if (error_message) {
            *error_message = "Error: --threads must be positive";
        }
        return false;
    }
    if (output_fps < 0.0) {
        if (error_message) {
            *error_message = "Error: output_fps cannot be negative";
        }
        return false;
    }
    if (display_fps < 0.0) {
        if (error_message) {
            *error_message = "Error: display_fps cannot be negative";
        }
        return false;
    }
    if (stream_fps < 0.0) {
        if (error_message) {
            *error_message = "Error: stream_fps cannot be negative";
        }
        return false;
    }
    if (output_bitrate_mbps < 0.0) {
        if (error_message) {
            *error_message = "Error: output_bitrate_mbps cannot be negative";
        }
        return false;
    }
    if (output_crf < -1 || output_crf > 51) {
        if (error_message) {
            *error_message = "Error: output_crf must be -1 or between 0 and 51";
        }
        return false;
    }
    if (web_preview_port <= 0 || web_preview_port > 65535) {
        if (error_message) {
            *error_message = "Error: web_preview_port must be between 1 and 65535";
        }
        return false;
    }
    if (web_preview_quality < 1 || web_preview_quality > 100) {
        if (error_message) {
            *error_message = "Error: web_preview_quality must be between 1 and 100";
        }
        return false;
    }
    if (web_preview_scale < 0.1 || web_preview_scale > 1.0) {
        if (error_message) {
            *error_message = "Error: web_preview_scale must be between 0.1 and 1.0";
        }
        return false;
    }
    // npu_core_mask: -1 = RKNN_NPU_CORE_AUTO（驱动动态选核，线程数>核数时的均衡方案），0 轮转，>0 固定核
    if (npu_core_mask < -1) {
        if (error_message) {
            *error_message = "Error: npu_core_mask must be -1, 0 or a positive core mask";
        }
        return false;
    }
    if (npu_core_start < 0 || npu_core_start > 2) {
        if (error_message) {
            *error_message = "Error: npu_core_start must be between 0 and 2";
        }
        return false;
    }
    if (drop_frames_on_overflow < -1 || drop_frames_on_overflow > 1) {
        if (error_message) {
            *error_message = "Error: drop_frames_on_overflow must be -1, 0 or 1";
        }
        return false;
    }
    if (web_preview_mode != "live" && web_preview_mode != "replay") {
        if (error_message) {
            *error_message = "Error: web_preview_mode must be 'live' or 'replay'";
        }
        return false;
    }
    if (web_preview_eof_linger_ms < 0) {
        if (error_message) {
            *error_message = "Error: web_preview_eof_linger_ms cannot be negative";
        }
        return false;
    }
    if (web_preview_secondary_port < 0 || web_preview_secondary_port > 65535) {
        if (error_message) {
            *error_message = "Error: web_preview_secondary_port must be between 0 and 65535";
        }
        return false;
    }
    if (alert_interval_s < 0.0) {
        if (error_message) {
            *error_message = "Error: alert_interval_s cannot be negative";
        }
        return false;
    }
    if (alert_dedup_interval_s < 0.0) {
        if (error_message) {
            *error_message = "Error: alert_dedup_interval_s cannot be negative";
        }
        return false;
    }
    // 命名事件规则校验：类型白名单 + 几何必填 + 方向取值 + 阈值正数
    for (size_t i = 0; i < event_rules.size(); ++i) {
        const EventRuleConfig& r = event_rules[i];
        const std::string where = "event_rules[" + std::to_string(i) + "]";
        const bool is_line = r.type == "line_cross";
        const bool is_region = r.type == "intrusion" || r.type == "dwell" || r.type == "absence" ||
                               r.type == "crowd" || r.type == "abandoned";
        const bool is_fall = r.type == "fall";
        const bool is_speed = r.type == "speed";
        if (!is_line && !is_region && !is_fall && !is_speed) {
            if (error_message) {
                *error_message = "Error: " + where + ".type must be line_cross/intrusion/dwell/"
                                 "absence/crowd/abandoned/fall/speed";
            }
            return false;
        }
        if (is_line && r.line.empty()) {
            if (error_message) {
                *error_message = "Error: " + where + " (line_cross) requires line";
            }
            return false;
        }
        if (is_region && r.region.empty()) {
            if (error_message) {
                *error_message = "Error: " + where + " (" + r.type + ") requires region";
            }
            return false;
        }
        if (is_line && r.direction != "both" && r.direction != "A2B" && r.direction != "B2A") {
            if (error_message) {
                *error_message = "Error: " + where + ".direction must be both/A2B/B2A";
            }
            return false;
        }
        if ((is_line) && r.ref_point != "center" && r.ref_point != "bottom") {
            if (error_message) {
                *error_message = "Error: " + where + ".ref_point must be center/bottom";
            }
            return false;
        }
        if (is_speed && r.speed_limit <= 0.0) {
            if (error_message) {
                *error_message = "Error: " + where + " (speed) requires speed_limit > 0 (px/s)";
            }
            return false;
        }
        if (r.dwell_seconds < 0.0 || r.absence_seconds < 0.0 || r.seconds < 0.0 ||
            r.fall_seconds < 0.0 || r.fall_aspect <= 0.0 || r.px_per_meter < 0.0 ||
            r.speed_seconds < 0.0 || r.min_count < 1) {
            if (error_message) {
                *error_message = "Error: " + where +
                                 " thresholds must be non-negative (min_count >= 1, fall_aspect > 0)";
            }
            return false;
        }
    }
    if (snapshot_interval_s < 0.0) {
        if (error_message) {
            *error_message = "Error: snapshot_interval_s cannot be negative";
        }
        return false;
    }
    {
        // composite_cls 必须配二级分类模型（小写比较，与 CLI/yaml 任务名宽松匹配一致）
        std::string task_lower = task;
        std::transform(task_lower.begin(), task_lower.end(), task_lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if ((task_lower == "composite_cls" || task_lower == "detect_cls") &&
            cls_model_path.empty()) {
            if (error_message) {
                *error_message =
                    "Error: task=composite_cls requires cls_model_path (stage-2 classifier)";
            }
            return false;
        }
        // M13 动作级联：窗口帧数必须为正（且须与 ST-GCN 模型输入 T 一致，转换侧约束）
        if (!action_model_path.empty()) {
            if (action_window_t <= 0) {
                if (error_message) {
                    *error_message = "Error: action_window_t must be positive";
                }
                return false;
            }
            if (action_interval < 0.0) {
                if (error_message) {
                    *error_message = "Error: action_interval cannot be negative";
                }
                return false;
            }
            std::string norm_lower = action_norm;
            std::transform(norm_lower.begin(), norm_lower.end(), norm_lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (norm_lower != "box" && norm_lower != "image") {
                if (error_message) {
                    *error_message = "Error: action_norm must be \"box\" or \"image\"";
                }
                return false;
            }
        }
    }
    if (alert_clip && (alert_clip_pre_seconds < 0.5 || alert_clip_pre_seconds > 30.0 ||
                       alert_clip_max_fps < 0.5 || alert_clip_max_fps > 30.0)) {
        if (error_message) {
            *error_message =
                "Error: alert_clip_pre_seconds must be in [0.5, 30] and alert_clip_max_fps in [0.5, 30]";
        }
        return false;
    }
    if (!alert_webhook_url.empty() && alert_webhook_url.rfind("http://", 0) != 0 &&
        alert_webhook_url.rfind("https://", 0) != 0) {
        if (error_message) {
            *error_message = "Error: alert_webhook_url must start with http:// or https://";
        }
        return false;
    }
    if (!roi.empty()) {
        // ROI 格式："x1,y1,x2,y2,..."，数字个数为偶数且至少 6 个（≥3 点）
        int value_count = 0;
        bool malformed = false;
        std::stringstream ss(roi);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (token.empty()) {
                malformed = true;
                break;
            }
            for (char c : token) {
                if (!std::isdigit(static_cast<unsigned char>(c)) && c != '-' && c != ' ') {
                    malformed = true;
                    break;
                }
            }
            ++value_count;
        }
        if (malformed || value_count < 6 || (value_count % 2) != 0) {
            if (error_message) {
                *error_message = "Error: roi must be \"x1,y1,x2,y2,...\" with at least 3 points (even count)";
            }
            return false;
        }
    }
    return true;
}

void AppConfig::printSummary() const {
    std::printf("Final options snapshot:\n");
    std::printf("  model_path: %s\n", model_path.c_str());
    std::printf("  input_path: %s\n", input_path.c_str());
    std::printf("  output_backend: %s\n", output_backend.c_str());
    std::printf("  output_video_path: %s\n", output_video_path.empty() ? "<empty>" : output_video_path.c_str());
    std::printf("  task: %s\n", task.c_str());
    std::printf("  aux_task: %s\n", aux_model_path.empty() ? "<disabled>" : aux_task.c_str());
    std::printf("  aux_model_path: %s\n", aux_model_path.empty() ? "<empty>" : aux_model_path.c_str());
    std::printf("  person_model_path: %s\n", person_model_path.empty() ? "<empty>" : person_model_path.c_str());
    std::printf("  rtmpose_head_path: %s\n", rtmpose_head_path.empty() ? "<empty>" : rtmpose_head_path.c_str());
    std::printf("  depth_dist_text: %d\n", depth_dist_text ? 1 : 0);
    std::printf("  depth_dist_near_m: %.2f\n", depth_dist_near_m);
    std::printf("  depth_dist_scale: %.3f\n", depth_dist_scale);
    std::printf("  detect3d_p2: %s\n", detect3d_p2.empty() ? "<empty>" : "configured");
    std::printf("  detect3d_depth_scale: %.3f\n", detect3d_depth_scale);
    std::printf("  event_region: %s\n", event_region.empty() ? "<empty>" : event_region.c_str());
    std::printf("  event_line: %s\n", event_line.empty() ? "<empty>" : event_line.c_str());
    std::printf("  event_classes: %s\n", event_classes.empty() ? "<all>" : event_classes.c_str());
    std::printf("  event_line_cross: %d\n", event_line_cross ? 1 : 0);
    std::printf("  event_intrusion: %d\n", event_intrusion ? 1 : 0);
    std::printf("  event_dwell: %d (%.1fs)\n", event_dwell ? 1 : 0, event_dwell_seconds);
    std::printf("  event_absence: %d (%.1fs)\n", event_absence ? 1 : 0, event_absence_seconds);
    std::printf("  event_stats_interval_s: %.1f\n", event_stats_interval_s);
    if (!event_rules.empty()) {
        std::printf("  event_rules: %zu 条（优先于扁平 event_* 键）\n", event_rules.size());
        for (const auto& r : event_rules) {
            std::string geo;
            if (!r.line.empty()) {
                geo = "line=" + r.line;
            }
            if (!r.region.empty()) {
                geo = "region=" + r.region;
            }
            std::string extra;
            if (r.direction != "both") {
                extra = " dir=" + r.direction;
            }
            if (r.ref_point == "bottom") {
                extra += " ref=bottom";
            }
            if (r.type == "fall") {
                extra += " aspect=" + std::to_string(r.fall_aspect) +
                         " sustain=" + std::to_string(r.fall_seconds) + "s";
            }
            if (r.type == "speed") {
                extra += " limit=" + std::to_string(r.speed_limit) + "px/s" +
                         (r.px_per_meter > 0.0 ? " (calib " + std::to_string(r.px_per_meter) + "px/m)"
                                               : "") +
                         " sustain=" + std::to_string(r.speed_seconds) + "s";
            }
            std::printf("    - %s: %s %s%s\n", r.id.empty() ? "<auto>" : r.id.c_str(), r.type.c_str(),
                        geo.c_str(), extra.c_str());
        }
    }
    std::printf("  mode: %s\n", mode.c_str());
    std::printf("  thread_count: %d\n", thread_count);
    std::printf("  gui: %d\n", gui ? 1 : 0);
    std::printf("  push_local: %d\n", push_local ? 1 : 0);
    std::printf("  label_path: %s\n", label_path.empty() ? "<empty>" : label_path.c_str());
    std::printf("  obj_class_num: %d\n", obj_class_num);
    std::printf("  conf_threshold: %.4f\n", conf_threshold);
    std::printf("  nms_threshold: %.4f\n", nms_threshold);
    std::printf("  output_fps: %.2f\n", output_fps);
    std::printf("  display_fps: %.2f\n", display_fps);
    std::printf("  stream_fps: %.2f\n", stream_fps);
    std::printf("  output_quality: %s\n", output_quality.c_str());
    std::printf("  output_bitrate_mbps: %.2f\n", output_bitrate_mbps);
    std::printf("  output_crf: %d\n", output_crf);
    std::printf("  output_preset: %s\n", output_preset.empty() ? "<auto>" : output_preset.c_str());
    std::printf("  output_low_latency: %d\n", output_low_latency ? 1 : 0);
    std::printf("  web_preview: %d\n", web_preview ? 1 : 0);
    std::printf("  web_preview_bind: %s\n", web_preview_bind.c_str());
    std::printf("  web_preview_port: %d\n", web_preview_port);
    std::printf("  web_preview_quality: %d\n", web_preview_quality);
    std::printf("  web_preview_mode: %s\n", web_preview_mode.c_str());
    std::printf("  web_preview_eof_linger_ms: %d\n", web_preview_eof_linger_ms);
    std::printf("  web_preview_secondary_port: %d\n", web_preview_secondary_port);
    std::printf("  web_preview_extra_ports: %s\n",
                web_preview_extra_ports.empty() ? "<empty>" : web_preview_extra_ports.c_str());
    std::printf("  web_preview_webrtc_urls: %s\n",
                web_preview_webrtc_urls.empty() ? "<empty>" : web_preview_webrtc_urls.c_str());
    std::printf("  web_preview_replay_pace: %d\n", web_preview_replay_pace ? 1 : 0);
    std::printf("  web_preview_scale: %.2f\n", web_preview_scale);
    std::printf("  npu_core_mask: %d\n", npu_core_mask);
    std::printf("  npu_core_start: %d\n", npu_core_start);
    std::printf("  drop_frames_on_overflow: %d\n", drop_frames_on_overflow);
    std::printf("  roi: %s\n", roi.empty() ? "<empty>" : roi.c_str());
    std::printf("  mask_classes: %s\n", mask_classes.empty() ? "<empty>" : mask_classes.c_str());
    std::printf("  seg_mask: %d\n", seg_mask ? 1 : 0);
    std::printf("  enable_tracking: %d\n", enable_tracking ? 1 : 0);
    std::printf("  track_tasks: %s\n", track_tasks.empty() ? "<all>" : track_tasks.c_str());
    std::printf("  show_count_overlay: %d\n", show_count_overlay ? 1 : 0);
    std::printf("  track_iou_threshold: %.2f\n", track_iou_threshold);
    std::printf("  track_max_missed: %d\n", track_max_missed);
    std::printf("  track_min_confirm_hits: %d\n", track_min_confirm_hits);
    std::printf("  track_reid_iou_threshold: %.2f\n", track_reid_iou_threshold);
    std::printf("  track_center_distance_threshold: %.2f\n", track_center_distance_threshold);
    std::printf("  track_box_smooth_alpha: %.2f\n", track_box_smooth_alpha);
    std::printf("  track_render_max_missed: %d\n", track_render_max_missed);
    std::printf("  track_high_conf_threshold: %.2f\n", track_high_conf_threshold);
    std::printf("  track_low_conf_threshold: %.2f\n", track_low_conf_threshold);
    std::printf("  track_stage2_iou_threshold: %.2f\n", track_stage2_iou_threshold);
    std::printf("  track_size_ratio_threshold: %.2f\n", track_size_ratio_threshold);
    std::printf("  track_max_tracks: %d\n", track_max_tracks);
    std::printf("  track_algorithm: %s\n", track_algorithm.c_str());
    std::printf("  alert_enabled: %d\n", alert_enabled ? 1 : 0);
    std::printf("  alert_webhook_url: %s\n", alert_webhook_url.empty() ? "<empty>" : alert_webhook_url.c_str());
    std::printf("  alert_min_count: %d\n", alert_min_count);
    std::printf("  alert_interval_s: %.1f\n", alert_interval_s);
    std::printf("  alert_snapshot_dir: %s\n", alert_snapshot_dir.empty() ? "<empty>" : alert_snapshot_dir.c_str());
    std::printf("  alert_dedup_interval_s: %.1f\n", alert_dedup_interval_s);
    std::printf("  alert_dedup_state_dir: %s\n", alert_dedup_state_dir.empty() ? "<empty>" : alert_dedup_state_dir.c_str());
    std::printf("  alert_clip: %d", alert_clip ? 1 : 0);
    if (alert_clip) {
        std::printf(" (pre=%.1fs fps=%.1f, dir=alert_snapshot_dir)", alert_clip_pre_seconds,
                    alert_clip_max_fps);
    }
    std::printf("\n");
    std::printf("  snapshot_interval_s: %.1f\n", snapshot_interval_s);
    std::printf("  snapshot_dir: %s\n", snapshot_dir.empty() ? "<empty>" : snapshot_dir.c_str());
    if (!ocr_rec_model_path.empty()) {
        std::printf("  ocr_rec: %s (dict %s, blank=%d, crop_pad=%.2f)\n", ocr_rec_model_path.c_str(),
                    ocr_dict_path.c_str(), ocr_rec_blank_index, ocr_rec_crop_pad);
    }
    if (!cls_model_path.empty()) {
        std::printf("  composite_cls: %s (labels %s)\n", cls_model_path.c_str(),
                    cls_labels_path.empty() ? "<none>" : cls_labels_path.c_str());
    }
    if (!action_model_path.empty()) {
        std::printf("  action: %s (labels %s, window_t=%d, interval=%.1fs, norm=%s)\n",
                    action_model_path.c_str(), action_labels.empty() ? "<none>" : action_labels.c_str(),
                    action_window_t, action_interval, action_norm.c_str());
    }
}

namespace {

bool isStreamLikeInput(const std::string& uri) {
    std::string lower = uri;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lower.rfind("rtsp://", 0) == 0 || lower.rfind("rtsps://", 0) == 0 ||
           lower.rfind("rtmp://", 0) == 0 || lower.rfind("rtmps://", 0) == 0 ||
           lower.rfind("udp://", 0) == 0 || lower.rfind("srt://", 0) == 0 ||
           lower.rfind("http://", 0) == 0 || lower.rfind("https://", 0) == 0 ||
           lower.rfind("/dev/video", 0) == 0;
}

std::string lowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

}  // namespace

void AppConfig::warnConflicts() const {
    bool any_warning = false;
    auto warn = [&any_warning](const char* code, const char* msg) {
        std::printf("[rk_pipe][warn-conflict][%s] %s\n", code, msg);
        any_warning = true;
    };

    if (gui && web_preview) {
        warn("conflict-gui-web", "同时启用 gui(本地窗口) 与 web_preview(网页预览)，将产生双份渲染/编码开销；建议按需只开一种");
    }
    if (push_local && gui) {
        warn("conflict-push-gui", "push_local 与 gui 同时开启：本地推流由 GUI 显示替代，推流逻辑将被跳过");
    }
    if (push_local && web_preview) {
        warn("conflict-push-web", "push_local 与 web_preview 同时开启：本地 ffplay 与网页 MJPEG 会重复消费输出");
    }
    std::string backend = lowerAscii(output_backend);
    if (push_local && backend == "opencv") {
        warn("conflict-push-opencv", "output_backend=opencv 不支持 UDP 本地推流（udp:// 仅 FFmpeg 后端可用），本地推流可能失败");
    }
    if (web_preview && web_preview_mode == "replay" && isStreamLikeInput(input_path)) {
        warn("conflict-replay-stream", "web_preview_mode=replay 需要本地文件（可回放/有总帧数）；当前输入是实时流，将按 live 模式处理");
    }
    if (web_preview && web_preview_mode == "live" && !input_path.empty() && !isStreamLikeInput(input_path)) {
        warn("conflict-live-file", "web_preview_mode=live 用于实时流；当前输入像本地文件，replay 模式通常更合适");
    }
    if ((depth_dist_text || depth_dist_near_m > 0.0f) &&
        (aux_model_path.empty() || lowerAscii(aux_task) != "depth")) {
        warn("conflict-depth-dist",
             "depth_dist_text/depth_dist_near_m 需要 aux_model_path + aux_task=depth 的辅助模型；"
             "当前未启用 depth 辅助任务，距离标注与近距告警不会生效");
    }

    if (any_warning) {
        std::printf("--------------------------------------------------------------------------------\n");
    }
}

void AppConfig::printUsage(const char* program_name) {
    std::printf("Usage: %s --model <model_path> --video <video_path> [options]\n", program_name);
    std::printf("Options:\n");
    std::printf("  --model <path>       Path to model file\n");
    std::printf("  --input/--video <path> Path to input video\n");
    std::printf("  --output-video <path> Path to output video\n");
    std::printf("  --task <detect/pose/obb/seg/ocr_det/depth/sem/detect3d/rtmpose> Detection task type\n");
    std::printf("  --person-model <path> Stage-1 person detection model for task=rtmpose\n");
    std::printf("  --rtmpose-head <path> Split-deploy head model for task=rtmpose (model_path=feat)\n");
    std::printf("  --aux-model <path>  Auxiliary model for multi-task combo (detect+depth/pose etc., Y5)\n");
    std::printf("  --aux-task <task>   Auxiliary task type (default: detect)\n");
    std::printf("  --depth-dist-text   Annotate estimated distance (m) per box (requires --aux-task depth)\n");
    std::printf("  --depth-dist-near <m>   Near-distance threshold in meters: red box + near_distance alert (0=off)\n");
    std::printf("  --depth-dist-scale <f>  Distance calibration factor, applied to model output (default 1.0)\n");
    std::printf("  --dump-detections <file> Export per-frame detect results as JSONL (COCO mAP eval)\n");
    std::printf("  --action-model <path>  ST-GCN action model for task=pose cascade (M13)\n");
    std::printf("  --action-labels <path> Action label file, one name per line (aligned with model output)\n");
    std::printf("  --action-window-t <N>  Temporal window frames, must match model input T (default: 30)\n");
    std::printf("  --action-interval <s>  Min seconds between inferences per track (0 = once per window)\n");
    std::printf("  --action-norm <box|image> Skeleton normalization (image = mmaction2 PreNormalize2D, NTU weights)\n");
    std::printf("  --mode <sequential/pipeline> Processing mode\n");
    std::printf("  --threads <num>      Number of processing threads (default: 4)\n");
    std::printf("  --pipeline           Alias for --mode pipeline\n");
    std::printf("  --label <path>       Path to label file\n");
    std::printf("  --obj-num <num>      Number of classes in label file\n");
    std::printf("  --gui                Enable local display when using opencv backend\n");
    std::printf("  --config <path>      Path to config file\n");
    std::printf("  --conf <value>       Confidence threshold (default: %.2f)\n", BOX_THRESH);
    std::printf("  --nms <value>        NMS threshold (default: %.2f)\n", NMS_THRESH);
    std::printf("  --local              Enable local UDP streaming and launch ffplay\n");
    std::printf("  --disable-tracking   Disable ID tracking and class counting for detect task\n");
    std::printf("  --tracking           Enable ID tracking (default: off)\n");
    std::printf("  --track-tasks <list> Per-task tracking whitelist, comma separated: detect,pose,obb,seg (default: all when --tracking)\n");
    std::printf("  --track-iou <value>  IOU threshold for track association (default: 0.30)\n");
    std::printf("  --track-max-missed <num> Max missed frames before dropping track (default: 40)\n");
    std::printf("  --track-confirm-hits <num> Frames required before track is confirmed (default: 3)\n");
    std::printf("  --track-reid-iou <value> IoU threshold when re-linking missed tracks (default: 0.15)\n");
    std::printf("  --track-center-dist <value> Max center distance scale for association (default: 1.8)\n");
    std::printf("  --track-smooth-alpha <value> Temporal smoothing alpha for boxes (default: 0.65)\n");
    std::printf("  --track-render-missed <num> Continue rendering confirmed tracks when briefly missed (default: 1)\n");
    std::printf("  --track-high-conf <value> Score threshold for high-confidence detections (default: 0.50)\n");
    std::printf("  --track-low-conf <value> Score threshold for low-confidence detections (default: 0.25)\n");
    std::printf("  --track-stage2-iou <value> IoU threshold when matching low-conf detections (default: 0.50)\n");
    std::printf("  --track-size-ratio <value> Min area ratio for track-detection match (default: 0.30)\n");
    std::printf("  --track-max-tracks <num> Max concurrent tracks (evict stale ones beyond this, default: 256)\n");
    std::printf("  --track-algorithm <name> iou | sort | bytetrack | ocsort (default: ocsort)\n");
    std::printf("  --disable-count-overlay Disable on-screen class count overlay\n");
    std::printf("YAML output tuning:\n");
    std::printf("  output_quality: speed | balanced | source | fidelity\n");
    std::printf("  output_bitrate_mbps / output_crf / output_preset override the selected quality tier\n");
    std::printf("  web_preview: 1 enables browser MJPEG preview at http://<board-ip>:web_preview_port/\n");
}

namespace {

// CLI 数值参数解析：非法值（非数字/越界/带尾随垃圾）报友好错误并退出，
// 而不是让 std::stof/stoi 的未捕获异常直接 std::terminate
float parseFloatArg(const char* arg) {
    try {
        std::size_t pos = 0;
        const float v = std::stof(arg, &pos);
        if (pos != std::strlen(arg)) {
            throw std::invalid_argument("trailing characters");
        }
        return v;
    } catch (const std::exception&) {
        std::fprintf(stderr, "Error: invalid numeric argument '%s'\n", arg);
        std::exit(1);
    }
}

int parseIntArg(const char* arg) {
    try {
        std::size_t pos = 0;
        const long v = std::stol(arg, &pos);
        if (pos != std::strlen(arg) || v < INT_MIN || v > INT_MAX) {
            throw std::invalid_argument("out of range");
        }
        return static_cast<int>(v);
    } catch (const std::exception&) {
        std::fprintf(stderr, "Error: invalid integer argument '%s'\n", arg);
        std::exit(1);
    }
}

}  // namespace

AppConfig AppConfig::fromCommandLine(int argc, char** argv) {
    AppConfig config;
    config.conf_threshold = BOX_THRESH;
    config.nms_threshold = NMS_THRESH;

    for (int i = 1; i < argc; ++i) {
        if ((std::strcmp(argv[i], "-c") == 0 || std::strcmp(argv[i], "--config") == 0) && i + 1 < argc) {
            config.config_path = argv[i + 1];
            if (!config.loadFromFile(config.config_path)) {
                std::fprintf(stderr, "Error: cannot open config file: %s\n",
                             config.config_path.c_str());
                std::exit(1);
            }
            break;
        }
    }

    optind = 1;

    static struct option long_options[] = {
        {"model", required_argument, 0, 'm'},
        {"help", no_argument, 0, 'h'},
        {"input", required_argument, 0, 'i'},
        {"video", required_argument, 0, 'v'},
        {"output-video", required_argument, 0, 'o'},
        {"task", required_argument, 0, 't'},
        {"mode", required_argument, 0, 'd'},
        {"threads", required_argument, 0, 'j'},
        {"pipeline", no_argument, 0, 'p'},
        {"label", required_argument, 0, 'l'},
        {"obj-num", required_argument, 0, 'k'},
        {"gui", no_argument, 0, 'g'},
        {"config", required_argument, 0, 'c'},
        {"conf", required_argument, 0, 'C'},
        {"nms", required_argument, 0, 'N'},
        {"local", no_argument, 0, 'L'},
        {"disable-tracking", no_argument, 0, 'X'},
        {"tracking", no_argument, 0, 'F'},
        {"track-tasks", required_argument, 0, 'U'},
        {"track-iou", required_argument, 0, 'I'},
        {"track-max-missed", required_argument, 0, 'M'},
        {"track-confirm-hits", required_argument, 0, 'H'},
        {"track-reid-iou", required_argument, 0, 'R'},
        {"track-center-dist", required_argument, 0, 'D'},
        {"track-smooth-alpha", required_argument, 0, 'S'},
        {"track-render-missed", required_argument, 0, 'P'},
        {"track-high-conf", required_argument, 0, 'B'},
        {"track-low-conf", required_argument, 0, 'E'},
        {"track-stage2-iou", required_argument, 0, 'G'},
        {"track-size-ratio", required_argument, 0, 'Z'},
        {"track-max-tracks", required_argument, 0, 'T'},
        {"track-algorithm", required_argument, 0, 'A'},
        {"disable-count-overlay", no_argument, 0, 'Q'},
        {"aux-model", required_argument, 0, 1000},
        {"aux-task", required_argument, 0, 1001},
        {"person-model", required_argument, 0, 1005},
        {"rtmpose-head", required_argument, 0, 1006},
        {"depth-dist-text", no_argument, 0, 1002},
        {"depth-dist-near", required_argument, 0, 1003},
        {"depth-dist-scale", required_argument, 0, 1004},
        {"dump-detections", required_argument, 0, 1007},
        {"action-model", required_argument, 0, 1008},
        {"action-labels", required_argument, 0, 1009},
        {"action-window-t", required_argument, 0, 1010},
        {"action-interval", required_argument, 0, 1011},
        {"action-norm", required_argument, 0, 1012},
        {0, 0, 0, 0}
    };

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "m:i:v:o:t:d:j:pl:k:gc:C:N:LXI:M:H:R:D:S:P:QB:E:G:ZTAFU:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'h':
                printUsage(argv[0]);
                std::exit(0);
            case 'm':
                config.model_path = optarg;
                break;
            case 'i':
            case 'v':
                config.input_path = optarg;
                break;
            case 'o':
                config.output_video_path = optarg;
                break;
            case 't':
                config.task = optarg;
                break;
            case 1000:
                config.aux_model_path = optarg;
                break;
            case 1001:
                config.aux_task = optarg;
                break;
            case 1005:
                config.person_model_path = optarg;
                break;
            case 1006:
                config.rtmpose_head_path = optarg;
                break;
            case 1002:
                config.depth_dist_text = true;
                break;
            case 1003:
                config.depth_dist_near_m = parseFloatArg(optarg);
                break;
            case 1004:
                config.depth_dist_scale = parseFloatArg(optarg);
                break;
            case 1007:
                config.dump_detections_path = optarg;
                break;
            case 1008:
                config.action_model_path = optarg;
                break;
            case 1009:
                config.action_labels = optarg;
                break;
            case 1010:
                config.action_window_t = std::atoi(optarg);
                break;
            case 1011:
                config.action_interval = std::atof(optarg);
                break;
            case 1012:
                config.action_norm = optarg;
                break;
            case 'd':
                config.mode = optarg;
                break;
            case 'j':
                config.thread_count = parseIntArg(optarg);
                break;
            case 'p':
                config.mode = "pipeline";
                break;
            case 'l':
                config.label_path = optarg;
                break;
            case 'k':
                config.obj_class_num = parseIntArg(optarg);
                break;
            case 'g':
                config.gui = true;
                break;
            case 'c':
                config.config_path = optarg;
                break;
            case 'C':
                config.conf_threshold = parseFloatArg(optarg);
                break;
            case 'N':
                config.nms_threshold = parseFloatArg(optarg);
                break;
            case 'L':
                config.push_local = true;
                if (config.output_video_path.empty()) {
                    config.output_video_path = localStreamOutput();
                }
                break;
            case 'X':
                config.enable_tracking = false;
                break;
            case 'F':
                config.enable_tracking = true;
                break;
            case 'U':
                config.track_tasks = optarg;
                break;
            case 'I':
                config.track_iou_threshold = parseFloatArg(optarg);
                break;
            case 'M':
                config.track_max_missed = parseIntArg(optarg);
                break;
            case 'H':
                config.track_min_confirm_hits = parseIntArg(optarg);
                break;
            case 'R':
                config.track_reid_iou_threshold = parseFloatArg(optarg);
                break;
            case 'D':
                config.track_center_distance_threshold = parseFloatArg(optarg);
                break;
            case 'S':
                config.track_box_smooth_alpha = parseFloatArg(optarg);
                break;
            case 'P':
                config.track_render_max_missed = parseIntArg(optarg);
                break;
            case 'B':
                config.track_high_conf_threshold = parseFloatArg(optarg);
                break;
            case 'E':
                config.track_low_conf_threshold = parseFloatArg(optarg);
                break;
            case 'G':
                config.track_stage2_iou_threshold = parseFloatArg(optarg);
                break;
            case 'Z':
                config.track_size_ratio_threshold = parseFloatArg(optarg);
                break;
            case 'T':
                config.track_max_tracks = parseIntArg(optarg);
                break;
            case 'A':
                config.track_algorithm = optarg;
                break;
            case 'Q':
                config.show_count_overlay = false;
                break;
            default:
                printUsage(argv[0]);
                std::exit(1);
        }
    }

    std::string validation_error;
    if (!config.validate(&validation_error)) {
        printValidationError(validation_error);
        printUsage(argv[0]);
        std::exit(1);
    }

    config.warnConflicts();
    config.printSummary();
    return config;
}
