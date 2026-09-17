#pragma once

#include <unordered_map>

#include "config/app_config.h"
#include "core/task_type.h"
#include "core/pipeline_frame.h"
#include "detection/simple_object_tracker.h"

bool isTrackingActive(TaskType task_type, const AppConfig& options);

SimpleObjectTracker createTracker(const AppConfig& options);

// tracked_out 非空时回传本帧带 ID 的跟踪结果（事件规则引擎用），在 output_enabled 判定前回填
void renderPipelineTrackingOutput(const AppConfig& options,
                                  bool output_enabled,
                                  PipelineFrame& frame,
                                  SimpleObjectTracker& tracker,
                                  std::unordered_map<int, int>& class_counts,
                                  std::vector<TrackedDetection>* tracked_out = nullptr,
                                  // draw_boxes=false：只跑 tracking（事件引擎/统计照常），不画 2D 框与标签
                                  // （例如只要 3D 线框的干净画面）
                                  bool draw_boxes = true);

void updateAndRenderTracking(const AppConfig& options,
                             bool output_enabled,
                             bool overlay_enabled,
                             bool use_zero_copy,
                             cv::Mat& frame_mat,
                             image_buffer_t& frame_buffer,
                             const object_detect_result_list& detect_results,
                             SimpleObjectTracker& tracker,
                             std::unordered_map<int, int>& class_counts,
                             std::vector<TrackedDetection>* engine_tracked = nullptr);

// ---- pose / obb / seg 任务的 tracking（完整能力：ID + 平滑框 + 遮挡恢复） ----

// 从各任务结果提取轴对齐 bbox 跑 tracker，回填 track_id 与平滑框，返回 tracked（与输入索引对齐）。
std::vector<TrackedDetection> trackPoseResults(SimpleObjectTracker& tracker,
                                               pose_detect_result_list& poses,
                                               std::unordered_map<int, int>* class_counter);
std::vector<TrackedDetection> trackOBBResults(SimpleObjectTracker& tracker,
                                              obb_detect_result_list& obbs,
                                              std::unordered_map<int, int>* class_counter);
std::vector<TrackedDetection> trackSegResults(SimpleObjectTracker& tracker,
                                              SegTaskResult& segs,
                                              std::unordered_map<int, int>* class_counter);

// 输出线程统一渲染入口（与 detect 的 renderPipelineTrackingOutput 对应）。
// tracking 开启时 worker 不绘制（pose/obb 由这里重画平滑框+ID；seg 仅叠加 ID）。
void renderPipelinePoseTrackingOutput(const AppConfig& options,
                                      bool output_enabled,
                                      PipelineFrame& frame,
                                      SimpleObjectTracker& tracker,
                                      std::unordered_map<int, int>& class_counts,
                                    std::vector<TrackedDetection>* engine_tracked = nullptr);
void renderPipelineOBBTrackingOutput(const AppConfig& options,
                                     bool output_enabled,
                                     PipelineFrame& frame,
                                     SimpleObjectTracker& tracker,
                                     std::unordered_map<int, int>& class_counts,
                                     std::vector<TrackedDetection>* engine_tracked = nullptr);
void renderPipelineSegTrackingOutput(const AppConfig& options,
                                     bool output_enabled,
                                     PipelineFrame& frame,
                                     SimpleObjectTracker& tracker,
                                     std::unordered_map<int, int>& class_counts,
                                     std::vector<TrackedDetection>* engine_tracked = nullptr);
