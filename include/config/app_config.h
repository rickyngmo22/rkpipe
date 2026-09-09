#pragma once

#include <string>

class AppConfig {
public:
    std::string model_path;
    std::string input_path;
    std::string output_video_path;
    // 非空时：detect 任务逐帧导出检测结果 JSONL（--dump-detections），供 COCO mAP 离线评测
    std::string dump_detections_path;
    // 注意：AppConfig 与闭源核心按成员布局直接耦合（核心按构建期 sizeof 分配并直接访问
    // 成员），公开仓库不得增删字段——新增配置键必须随配套核心库 Release 协同发布。
    // 逐帧结果 JSONL 的过渡配置走环境变量 RK_PIPE_RESULT_JSONL（见 io/result_sink.h），
    // YAML 键 result_jsonl_path 待配套核心启用。
    std::string output_backend = "ffmpeg";
    std::string task = "detect";
    std::string mode = "sequential";
    int thread_count = 6;
    bool gui = false;
    std::string config_path;
    std::string label_path;
    int obj_class_num = 0;
    float conf_threshold = 0.0f;
    float nms_threshold = 0.0f;
    double output_fps = 0.0;
    double display_fps = 0.0;
    double stream_fps = 0.0;
    std::string output_quality = "balanced";
    double output_bitrate_mbps = 0.0;
    int output_crf = -1;
    std::string output_preset;
    bool output_low_latency = false;
    bool web_preview = false;
    std::string web_preview_bind = "0.0.0.0";
    int web_preview_port = 8080;
    int web_preview_quality = 80;
    std::string web_preview_mode = "live";
    int web_preview_eof_linger_ms = 1500;
    int web_preview_secondary_port = 0;
    std::string web_preview_extra_ports;
    std::string web_preview_webrtc_urls;  // 各窗格 WebRTC 播放 URL（'|' 分隔，对齐窗格顺序）；空=用 MJPEG
    bool web_preview_replay_pace = false;
    double web_preview_scale = 1.0;    // Web 预览降采样比例 (0.1~1.0，1.0=原尺寸)
    bool seg_mask = true;
    bool push_local = false;
    int npu_core_mask = 0;             // 0=按 NPU core 轮转分配；>0=固定 core mask（见 rknn_api.h）；<0=NPU_CORE_AUTO（驱动按负载动态选核，线程数>3 时的均衡方案）
    int npu_core_start = 0;            // NPU core 轮转起始偏移（0-2）；多路时由编排脚本按流分配，避免全部进程都从 core 0 起
    int drop_frames_on_overflow = -1;  // -1=自动(网络输出才丢帧)；0=队列满阻塞(背压)；1=队满丢帧
    // ROI 区域过滤（detect）："x1,y1,x2,y2,..." 多边形，框中心在外的不渲染/不计数/不告警（空=不启用）
    std::string roi;
    // 隐私遮蔽（detect）：需要打码的类别 id 列表，如 "0,1"（空=不启用）
    std::string mask_classes;
    bool enable_tracking = false;
    // 按任务开启 tracking 的白名单（逗号分隔：detect,pose,obb,seg）；空 = enable_tracking 时全部支持的任务开启
    std::string track_tasks;
    bool show_count_overlay = true;
    float track_iou_threshold = 0.3f;
    int track_max_missed = 20;
    int track_min_confirm_hits = 3;
    float track_reid_iou_threshold = 0.15f;
    float track_center_distance_threshold = 1.8f;
    float track_box_smooth_alpha = 0.65f;
    int track_render_max_missed = 1;
    // ByteTrack 两阶段关联参数
    float track_high_conf_threshold = 0.5f;
    float track_low_conf_threshold = 0.25f;
    float track_stage2_iou_threshold = 0.5f;
    float track_size_ratio_threshold = 0.3f;
    int track_max_tracks = 256;
    // 跟踪算法档位：iou | sort | bytetrack | ocsort（默认，能力最强）
    std::string track_algorithm = "ocsort";

    // 告警回调（detect 任务）：目标计数超阈值时 POST webhook + 可选现场抓图
    bool alert_enabled = false;
    std::string alert_webhook_url;
    int alert_min_count = 1;
    double alert_interval_s = 5.0;
    std::string alert_snapshot_dir;
    // 跨进程告警去重（多路并行时防止同一事件被多个进程重复上报）：
    // alert_dedup_interval_s>0 时启用，key=(task,class)，经共享状态文件 + flock 原子去重
    double alert_dedup_interval_s = 0.0;
    std::string alert_dedup_state_dir;
    // 周期快照：每 snapshot_interval_s 秒保存一帧到 snapshot_dir（0=关闭）
    double snapshot_interval_s = 0.0;
    std::string snapshot_dir;

    // 事件规则引擎（detect+tracking，D1/D2）：绊线穿越/区域入侵/滞留/离岗。
    // 规则命中产生 rule_events 告警（走 alert_* 链路）；enable_tracking 未开时自动为 detect 开启
    std::string event_region;        // 事件区域多边形 "x1,y1,x2,y2,..."（与 roi 同格式）
    std::string event_line;          // 绊线 "x1,y1,x2,y2"（两侧按有向直线 A/B 区分）
    std::string event_classes;       // 规则生效类别 id 列表，如 "0"（空=全部类别）
    bool event_line_cross = false;   // line_cross：轨迹穿越绊线（含 A->B/B->A 双向计数）
    bool event_intrusion = false;    // intrusion：目标进入事件区域
    bool event_dwell = false;        // dwell：在区域内停留超过 event_dwell_seconds
    double event_dwell_seconds = 10.0;
    bool event_absence = false;      // absence：区域出现过人后持续无人超过 event_absence_seconds（离岗）
    double event_absence_seconds = 30.0;

    static AppConfig fromCommandLine(int argc, char** argv);

    bool loadFromFile(const std::string& filename);
    bool validate(std::string* error_message = nullptr) const;
    void printSummary() const;
    // 启动自检：打印可能违背预期的参数组合告警（非致命，只提示）
    void warnConflicts() const;

    static void printUsage(const char* program_name);
    static const char* localStreamOutput();
};
