#pragma once

#include <string>
#include <vector>

// 命名事件规则（event_rules 列表项）：多绊线/多区域，各带类别与参数。
// 非空时优先生效，扁平 event_* 键不再合成默认规则。
struct EventRuleConfig {
    std::string id;        // 规则名（事件 payload 的 rule 字段与去重 key 用），空则自动编号
    std::string type;      // line_cross | intrusion | dwell | absence | crowd | abandoned | fall | speed
    std::string line;      // line_cross: "x1,y1,x2,y2"
    std::string region;    // 区域类: "x1,y1,x2,y2,..."（至少 3 点；fall/speed 可选，缺省全画面）
    std::string classes;   // 生效类别 id 列表（空=全部类别）
    std::string direction = "both";  // line_cross: both|A2B|B2A（单向即逆行/方向合规）
    std::string ref_point = "center";  // line_cross 判定点: "center"（默认）| "bottom"（脚底中心，行人惯例）
    double dwell_seconds = 10.0;
    double absence_seconds = 30.0;
    double seconds = 60.0;  // abandoned 静止时长阈值
    int min_count = 3;      // crowd 人数阈值
    double fall_seconds = 2.0;  // fall 躺倒持续时长阈值（持续该秒数才告警，去抖）
    double fall_aspect = 1.2;   // fall 宽高比阈值（框 w/h >= 该值视为躺倒形态）
    double speed_limit = 0.0;   // speed 阈值（px/s，>0 才生效）
    double px_per_meter = 0.0;  // speed 像素标定（>0 时 detail 换算 km/h；0=报 px/s）
    double speed_seconds = 1.0; // speed 超限持续去抖
    std::string llm_hint;       // LLM 复检提示：随事件 payload 上报，daemon 复检 prompt 拼入（可空）
};

class AppConfig {
    // 注意：AppConfig 与闭源核心按成员布局直接耦合（核心按构建期 sizeof 分配并直接访问
    // 成员），公开仓库不得增删字段——新增配置键必须随配套核心库 Release 协同发布。
    // 逐帧结果 JSONL 的过滤配置走环境变量 RK_PIPE_RESULT_JSONL（见 io/result_sink.h）。
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
    // 确认轨迹最多容忍的缺检帧数，超过即删除。40 ≈ 1.6s@25fps；"跟丢后被识别成
    // 新任务"的主要来源之一就是超龄删除（遮挡/漏检 >40 帧前应有机会重连）。代价：
    // 真正离开的目标会占号更久，同位置的新目标可能续用旧 ID，可按场景 --track-max-missed 调。
    int track_max_missed = 40;
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
    // 规则计数持久化文件（空=不持久化）：穿越计数跨重启累计（客流统计）。
    // 定期落盘 + 启动恢复（仅恢复与现存规则 id 匹配的计数）。
    std::string event_stats_path;
    // 本地录像分段滚动（仅本地文件；0=关闭）：每 record_segment_s 秒封段重开，
    // record_keep>0 时仅保留最近 N 段。事件片段(clip_*.avi)由 daemon GC 同目录治理。
    int record_segment_s = 0;
    int record_keep = 0;
    // 跨进程告警去重（多路并行时防止同一事件被多个进程重复上报）：
    // alert_dedup_interval_s>0 时启用，key=(task,class)，经共享状态文件 + flock 原子去重
    double alert_dedup_interval_s = 0.0;
    std::string alert_dedup_state_dir;
    // 事件视频片段（G4.1）：告警触发时把前 alert_clip_pre_seconds 的采样帧回溯拼成
    // MJPG AVI 片段（clip_<task>_<ts>.avi，存 alert_snapshot_dir），webhook body 携带 clip 路径。
    // 需 alert_enabled + alert_webhook_url + alert_snapshot_dir；内存预算 = pre × fps × 帧大小
    bool alert_clip = false;
    double alert_clip_pre_seconds = 5.0;  // 回溯时长（0.5~30s）
    double alert_clip_max_fps = 5.0;      // 采样帧率（0.5~30fps，降采样省内存）
    // 周期快照：每 snapshot_interval_s 秒保存一帧到 snapshot_dir（0=关闭）
    double snapshot_interval_s = 0.0;
    std::string snapshot_dir;

    // M0 两阶级联（task=composite_cls）：一级检测（model_path）+ 二级分类（cls_model_path，
    // MobileNetV2 等），每框 crop → top-1 叠加；cls_labels_path 可选（每行一个类名）
    std::string cls_model_path;
    std::string cls_labels_path;
    // M6 OCR 文字识别（task=ocr_det 时的二级级联）：检测多边形 → 透视矫正 →
    // ocr_rec_model_path（PP-OCRv4 Rec / LPRNet）→ CTC 解码（ocr_dict_path 字典，
    // 0=blank；ocr_rec_blank_index 供非标准布局覆盖）。不配 rec = 纯文字检测
    std::string ocr_rec_model_path;
    std::string ocr_dict_path;
    int ocr_rec_blank_index = 0;
    // 二级识别 crop 外扩系数（1.0=不外扩）：LPRNet 等整牌输入模型需要 >1
    // （det 检出的是文字笔画区，牌面大于文字区）；通用 OCR rec 保持 1.0
    double ocr_rec_crop_pad = 1.0;

    // M13 动作识别级联（task=pose 时生效）：pose 跟踪轨迹 → PoseSequenceBuffer 攒窗 →
    // ST-GCN（action_model_path）逐轨迹推理，结果附着 pose 帧（ActionTaskResult）。
    // action_labels 每行一个动作名（索引对齐模型输出）；action_window_t 为窗口帧数
    // （须与模型输入 T 一致，见 docs/action_rec.md §2）；action_interval 为同轨迹
    // 两次推理的最小间隔秒（0=每攒满一窗推一次，默认）
    std::string action_model_path;
    std::string action_labels;
    int action_window_t = 30;
    double action_interval = 0.0;
    // 骨架归一化口径（须与训练侧一致，docs/action_rec.md §2-3）：
    // box=逐帧框中心/尺度（自训）；image=原帧分辨率 (x-w/2)/(w/2)（mmaction2
    // PreNormalize2D 口径，NTU60 官方权重用此值）
    std::string action_norm = "box";

    // D3 检测+单目测距（aux_task=depth 时生效）：
    // depth_dist_text=1 在检测框上标注估计距离（米）；depth_dist_near_m>0 时近距目标
    // 红框高亮并触发 near_distance 告警；depth_dist_scale 为距离标定系数（实测/模型输出）
    bool depth_dist_text = false;
    float depth_dist_near_m = 0.0f;  // 近距阈值（米），0=关闭
    float depth_dist_scale = 1.0f;
    // depth 辅助结果伪彩叠加开关（默认开）。只想要检测框/距离/3D 线框的干净输出时关掉
    bool depth_pseudo = true;
    // 2D 检测框/标签绘制开关（默认开）。关掉后只画 3D 线框等其余叠加，画面更干净
    bool detect_box_draw = true;
    // 3D 线框投影的 P2 矩阵（3x4 行主序 12 值，逗号分隔，如 KITTI cam2 标定）。
    // 空 = 不绘制 3D 线框；内参需与实际相机一致，否则线框仅示意
    std::string detect3d_p2;
    // D3 3D 线框（路线 B，aux_task=depth 时生效）：2D 框 + depth 模型接地深度 + detect3d_p2
    // 几何反推 3D 包围盒（深度来自 aux depth 模型而非 3D 头回归，跨场景泛化好）。
    // 需同时配置 detect3d_p2 才会绘制。
    bool detect3d_wireframe = false;
    // 车辆相对朝向先验（度）：0=侧视（路侧车沿路停放/行驶，像素宽≈车长）；
    // 90=正视车头/车尾（像素宽≈车宽，长退化为先验比）
    float detect3d_alpha_deg = 0.0f;
    // 接地采样带高度占框高比例（0<band<1，默认 0.3 = 框底部 30%）
    float detect3d_ground_band = 0.3f;
    // 3D 线框最小绘制距离（米，默认 8.0；<=0 表示不限）。
    // ★为什么需要：单目 + 单个 2D 框**无法约束 3D 长方体沿车长方向的延伸**。
    //  一辆 4.5m 车长在 2~3m 距离下，其长方体投影必然横向摊开远超 2D 框（实测溢出画面，
    //  线框宽可达画面宽的 3 倍），这不是参数问题而是信息量极限。
    //  板端实测（baseline17_mid30s，901 帧，3355 个 car 框）「线框宽/2D框宽」比值：
    //    Z 0-5m   → 1.38（33% 溢出画面）
    //    Z 5-8m   → 1.37（ 2% 溢出）
    //    Z 8-12m  → 1.08（ 0% 溢出）  ← 8m 起线框与 2D 框基本吻合
    //    Z 12-18m → 1.04（ 0% 溢出）
    //  故默认 8.0：保留约 50% 的车辆线框，溢出率降为 0。近距车只画 2D 框+距离文字。
    float detect3d_min_depth_m = 8.0f;

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
    // 命名多规则（event_rules 列表）：非空时优先生效，上面的扁平键不再合成默认规则
    std::vector<EventRuleConfig> event_rules;
    // 周期统计报表（秒）：>0 时每该秒数经 webhook 上报 rule_stats（各规则穿越计数增量，
    // 上报后计数清零）并进 /status.json 缓存。0=关闭
    double event_stats_interval_s = 0.0;

    static AppConfig fromCommandLine(int argc, char** argv);

    bool loadFromFile(const std::string& filename);
    bool validate(std::string* error_message = nullptr) const;
    void printSummary() const;
    // 启动自检：打印可能违背预期的参数组合告警（非致命，只提示）
    void warnConflicts() const;

    static void printUsage(const char* program_name);
    static const char* localStreamOutput();
};
