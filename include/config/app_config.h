#pragma once

#include <string>

class AppConfig {
public:
    std::string model_path;
    std::string input_path;
    // 多任务同路组合（Y5）：主任务 model_path/task 走完整链路，辅助任务只推理+叠加绘制。
    // aux_model_path 为空 = 不启用辅助任务。
    std::string aux_model_path;
    std::string aux_task = "detect";
    // task="rtmpose"（两阶段姿态）：model_path 为 RTMPose(SimCC) 模型，
    // person_model_path 为第一阶段人体检测模型。其他 task 忽略。
    // rtmpose 拆分部署（feat/head 两段）时：model_path=feat（主干），
    // rtmpose_head_path=head（SimCC 头部）。
    std::string person_model_path;
    std::string rtmpose_head_path;
    std::string output_video_path;
    // 非空时：detect 任务逐帧导出检测结果 JSONL（--dump-detections），供 COCO mAP 离线评测
    std::string dump_detections_path;
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
    int rtmpose_det_core = -1;         // rtmpose 任务：stage1(人体检测) 固定到的 NPU core（mask 值，如 0→RKNN_NPU_CORE_0=1）；-1=跟随 worker 级 core_mask
    std::string rtmpose_pose_cores;    // rtmpose 任务：stage2(pose) 使用的 NPU core 列表（逗号分隔，如 "1,2"），按 worker 轮转；空=跟随 worker 级 core_mask
    float rtmpose_target_fps = 0.0f;   // rtmpose 目标帧率（0=不限制/关闭预算）：单帧工作预算 = 1000*thread_count/fps，
                                       // 超预算只处理 top-K 人，把单帧耗时钳在帧间隔内 → 帧率稳定，密集场景降级不卡顿
    int rtmpose_stage2_threads = 0;    // rtmpose 二阶段全局工作池线程数（0=关闭/内联）：pool 线程并行做逐人 pose，
                                       // 一阶段(检测)与二阶段(pose)解耦并行，避免二阶段挤占一阶段线程
    float rtmpose_box_pad = 1.0f;      // rtmpose 仿射裁剪前框外扩系数：YOLO 框比 RTMDet 紧，精度不够可调 1.1~1.25
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

    // D3 检测+单目测距（aux_task=depth 时生效）：
    // depth_dist_text=1 在检测框上标注估计距离（米）；depth_dist_near_m>0 时近距目标
    // 红框高亮并触发 near_distance 告警；depth_dist_scale 为距离标定系数（实测/模型输出）
    bool depth_dist_text = false;
    float depth_dist_near_m = 0.0f;  // 近距阈值（米），0=关闭
    float depth_dist_scale = 1.0f;
    // Detect3D 线框投影的 P2 矩阵（3x4 行主序 12 值，逗号分隔，如 KITTI cam2 标定）。
    // 空 = 不绘制 3D 线框；内参需与实际相机一致，否则线框仅示意
    std::string detect3d_p2;
    // Detect3D 深度全局缩放（场景尺度校准：线框/距离整体偏大调大此值，默认 1.0）
    float detect3d_depth_scale = 1.0f;

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
