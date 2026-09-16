// rk_pipe 最小单元测试集（无硬件依赖，仅覆盖配置解析 / Detector 工厂路由 / 后处理纯函数）
// 构建与运行：cmake --build build --target rk_pipe_unit_tests && ./build/rk_pipe_unit_tests
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <cstring>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unordered_map>
#include <vector>

#include "config/app_config.h"
#include "core/detection_filter.h"
#include "core/event_engine.h"
#include "core/performance.h"
#include "detection/crop_utils.h"
#include "io/output_tuning.h"
#include "io/http_request.h"
#include "postprocess/cls_top1.h"
#include "postprocess/retinaface_decode.h"
#include "postprocess/ctc_decode.h"
#include "postprocess/pose_sequence.h"
#include "daemon/events_store.h"
#include "daemon/daemon_internal.h"
#include "daemon/flywheel.h"
#include "core/thermal_governor.h"
#include "daemon/mini_json.h"
#include "postprocess/detect3d_decode.h"
#include "postprocess/postprocess_common.h"
#include "detection/simple_object_tracker.h"
#include "detection/tracking_runtime.h"
#include "utils/depth_distance.h"
#include "utils/http_post.h"
#ifndef RK_PIPE_CI
#include "detection/detector.h"
#endif

#include <turbojpeg.h>

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

static const char* kTestYaml = "/tmp/rk_pipe_unit_test_cfg.yaml";

static bool writeTestYaml(const std::string& content) {
    std::ofstream out(kTestYaml);
    if (!out.is_open()) {
        return false;
    }
    out << content;
    return true;
}

static void testConfigLoad() {
    std::printf("[test] AppConfig::loadFromFile\n");
    if (!writeTestYaml(
            "%YAML:1.0\n"
            "---\n"
            "model_path: \"/tmp/model.rknn\"\n"
            "input_path: \"/tmp/video.mp4\"\n"
            "task: \"pose\"\n"
            "mode: \"pipeline\"\n"
            "thread_count: 4\n"
            "conf_thresh: 0.35\n"
            "nms_thresh: 0.5\n"
            "web_preview: 1\n"
            "alert_enabled: 1\n"
            "alert_webhook_url: \"http://127.0.0.1:8888/hook\"\n"
            "depth_dist_text: 1\n"
            "depth_dist_near_m: 5.5\n"
            "depth_dist_scale: 1.25\n"
            "event_region: \"10,10,200,10,200,150,10,150\"\n"
            "event_line: \"0,80,400,80\"\n"
            "event_classes: \"0,16\"\n"
            "event_line_cross: 1\n"
            "event_intrusion: 1\n"
            "event_dwell: 1\n"
            "event_dwell_seconds: 12.5\n"
            "event_absence: 1\n"
            "event_absence_seconds: 45\n"
            "action_model_path: \"/tmp/stgcn_fp16.rknn\"\n"
            "action_labels: \"/tmp/action_labels.txt\"\n"
            "action_window_t: 100\n"
            "action_interval: 2.5\n")) {
        CHECK(false);
        return;
    }

    AppConfig cfg;
    CHECK(cfg.loadFromFile(kTestYaml));
    CHECK(cfg.model_path == "/tmp/model.rknn");
    CHECK(cfg.input_path == "/tmp/video.mp4");
    CHECK(cfg.task == "pose");
    CHECK(cfg.mode == "pipeline");
    CHECK(cfg.thread_count == 4);
    // 别名键 conf_thresh/nms_thresh 兼容
    CHECK(cfg.conf_threshold > 0.34f && cfg.conf_threshold < 0.36f);
    CHECK(cfg.nms_threshold > 0.49f && cfg.nms_threshold < 0.51f);
    CHECK(cfg.web_preview);
    CHECK(cfg.alert_enabled);
    CHECK(cfg.alert_webhook_url == "http://127.0.0.1:8888/hook");
    // D3 检测+单目测距配置
    CHECK(cfg.depth_dist_text);
    CHECK(cfg.depth_dist_near_m > 5.49f && cfg.depth_dist_near_m < 5.51f);
    CHECK(cfg.depth_dist_scale > 1.24f && cfg.depth_dist_scale < 1.26f);
    // D1/D2 事件规则配置
    CHECK(cfg.event_region == "10,10,200,10,200,150,10,150");
    CHECK(cfg.event_line == "0,80,400,80");
    CHECK(cfg.event_classes == "0,16");
    CHECK(cfg.event_line_cross && cfg.event_intrusion && cfg.event_dwell && cfg.event_absence);
    CHECK(cfg.event_dwell_seconds > 12.4 && cfg.event_dwell_seconds < 12.6);
    CHECK(cfg.event_absence_seconds > 44.9 && cfg.event_absence_seconds < 45.1);
    // M13 动作级联配置
    CHECK(cfg.action_model_path == "/tmp/stgcn_fp16.rknn");
    CHECK(cfg.action_labels == "/tmp/action_labels.txt");
    CHECK(cfg.action_window_t == 100);
    CHECK(cfg.action_interval > 2.4 && cfg.action_interval < 2.6);

    // CLI 形式
    std::string err;
    const char* argv2[] = {"console_detector", "--model", "/tmp/m.rknn", "--input", "/tmp/v.mp4",
                           "--task", "pose", "--action-model", "/tmp/a.rknn",
                           "--action-labels", "/tmp/l.txt", "--action-window-t", "300",
                           "--action-interval", "1.5", nullptr};
    AppConfig acfg = AppConfig::fromCommandLine(
        static_cast<int>(sizeof(argv2) / sizeof(argv2[0])) - 1, const_cast<char**>(argv2));
    CHECK(acfg.action_model_path == "/tmp/a.rknn");
    CHECK(acfg.action_labels == "/tmp/l.txt");
    CHECK(acfg.action_window_t == 300);
    CHECK(acfg.action_interval > 1.4 && acfg.action_interval < 1.6);
    CHECK(acfg.validate(&err));

    // 校验：window_t 非正拒绝；interval 负数拒绝；未配模型时不检查
    AppConfig bcfg;
    bcfg.input_path = "/tmp/x.mp4";
    bcfg.model_path = "/tmp/x.rknn";
    bcfg.action_model_path = "/tmp/a.rknn";
    bcfg.action_window_t = 0;
    CHECK(!bcfg.validate(&err));
    bcfg.action_window_t = 30;
    bcfg.action_interval = -1.0;
    CHECK(!bcfg.validate(&err));
    bcfg.action_interval = 0.0;
    bcfg.action_norm = "joint";
    CHECK(!bcfg.validate(&err));  // 非法归一化口径
    bcfg.action_norm = "image";
    CHECK(bcfg.validate(&err));
    AppConfig ncfg;
    ncfg.input_path = "/tmp/x.mp4";
    ncfg.model_path = "/tmp/x.rknn";
    ncfg.action_model_path.clear();  // 未配置模型：其余键不参与校验
    CHECK(ncfg.validate(&err));

    // ActionTaskResult 变体往返 + 双类型 pose 访问器
    ActionTaskResult at;
    at.data.count = 1;
    at.data.results[0].track_id = 7;
    at.actions.push_back(ActionItem{7, 3, 0.85f});
    TaskResult tr = at;
    const ActionTaskResult* back = std::get_if<ActionTaskResult>(&tr);
    CHECK(back && back->actions.size() == 1 && back->actions[0].track_id == 7 &&
          back->actions[0].action_id == 3);
    CHECK(getPoseResultList(tr) == &back->data);  // 动作类型内嵌 pose 数据可见
    TaskResult plain = PoseTaskResult{};
    CHECK(getPoseResultList(plain) != nullptr);   // 纯 pose 类型走原通道
}

static void testConfigValidate() {
    std::printf("[test] AppConfig::validate\n");
    AppConfig cfg;
    std::string err;
    CHECK(!cfg.validate(&err));  // 缺 input_path/model_path
    cfg.input_path = "/tmp/video.mp4";
    CHECK(!cfg.validate(&err));  // 缺 model_path
    cfg.model_path = "/tmp/model.rknn";
    CHECK(cfg.validate(&err));

    // 非法 web_preview_scale
    AppConfig bad;
    bad.input_path = "a.mp4";
    bad.model_path = "m.rknn";
    bad.web_preview_scale = 2.0;
    CHECK(!bad.validate(&err));
    // 非法 alert url
    AppConfig bad2;
    bad2.input_path = "a.mp4";
    bad2.model_path = "m.rknn";
    bad2.alert_webhook_url = "ftp://x";
    CHECK(!bad2.validate(&err));
}

static void testCommandLine() {
    std::printf("[test] AppConfig::fromCommandLine\n");
    const char* argv[] = {
        "console_detector", "--model", "/tmp/m.rknn", "--input", "/tmp/v.mp4",
        "--task", "detect", "--conf", "0.42", "--nms", "0.55",
        "--obj-num", "80", "--label", "/tmp/l.txt", "--pipeline", "--threads", "2",
        nullptr,
    };
    int argc = static_cast<int>(sizeof(argv) / sizeof(argv[0])) - 1;
    // fromCommandLine 内部会 validate（失败则 exit），参数需合法；并会打印 usage/summary
    AppConfig cfg = AppConfig::fromCommandLine(argc, const_cast<char**>(argv));
    CHECK(cfg.model_path == "/tmp/m.rknn");
    CHECK(cfg.input_path == "/tmp/v.mp4");
    CHECK(cfg.task == "detect");
    CHECK(cfg.conf_threshold > 0.41f && cfg.conf_threshold < 0.43f);
    CHECK(cfg.nms_threshold > 0.54f && cfg.nms_threshold < 0.56f);
    CHECK(cfg.obj_class_num == 80);
    CHECK(cfg.label_path == "/tmp/l.txt");
    CHECK(cfg.mode == "pipeline");
    CHECK(cfg.thread_count == 2);
}

#ifndef RK_PIPE_CI
static void testDetectorFactory() {
    std::printf("[test] Detector::createForModel\n");
    {
        auto d = Detector::createForModel("/tmp/yolov8n.rknn", "detect");
        CHECK(dynamic_cast<YOLOv8Detector*>(d.get()) != nullptr);
    }
    {
        auto d = Detector::createForModel("/tmp/yolov8_pose.rknn", "pose");
        CHECK(dynamic_cast<YOLOv8PoseDetector*>(d.get()) != nullptr);
    }
    {
        auto d = Detector::createForModel("/tmp/yolov8n_obb.rknn", "obb");
        CHECK(dynamic_cast<YOLOv8OBBDetector*>(d.get()) != nullptr);
    }
    {
        auto d = Detector::createForModel("/tmp/yolov8_seg.rknn", "seg");
        CHECK(dynamic_cast<YOLOv8SegDetector*>(d.get()) != nullptr);
    }
    {
        auto d = Detector::createForModel("/tmp/yolov5s.rknn", "detect");
        CHECK(dynamic_cast<YOLOv5Detector*>(d.get()) != nullptr);
    }
    {
        auto d = Detector::createForModel("/tmp/x.rknn", "ocr_det");
        CHECK(dynamic_cast<OCRDetectDetector*>(d.get()) != nullptr);
    }
    {
        auto d = Detector::createForModel("/tmp/yolo26n.rknn", "composite_cls");
        CHECK(dynamic_cast<CompositeClsDetector*>(d.get()) != nullptr);
    }
    {
        auto d = Detector::createForModel("/tmp/retinaface_m_fp16.rknn", "retinaface");
        CHECK(dynamic_cast<RetinaFaceDetector*>(d.get()) != nullptr);
    }
    // 未指定 task 且文件名无 yolo 系列/版本标记 → 默认 YOLOv8
    // （yolo11 文件名路由到 YOLO11Detector，其类型不在本测试的 include 面，不作 RTTI 断言）
    {
        auto d = Detector::createForModel("/tmp/x.rknn", "");
        CHECK(dynamic_cast<YOLOv8Detector*>(d.get()) != nullptr);
    }
}
#endif

static void testPostprocessMath() {
    std::printf("[test] postprocess 纯函数\n");
    // 两框部分重叠：交集 25，并集 175
    float overlap = CalculateOverlap(0.f, 0.f, 10.f, 10.f, 5.f, 5.f, 15.f, 15.f);
    CHECK(overlap > 0.14f && overlap < 0.15f);
    // 不重叠 → 0
    CHECK(CalculateOverlap(0.f, 0.f, 10.f, 10.f, 100.f, 100.f, 110.f, 110.f) == 0.f);

    // 反量化：(qnt - zp) * scale
    CHECK(deqnt_affine_to_f32(10, 5, 0.5f) == 2.5f);
    CHECK(deqnt_affine_to_f32(0, 128, 0.01f) == -1.28f);

    CHECK(sigmoidf(0.f) == 0.5f);
    CHECK(sigmoidf(0.f) == 0.5f);
    CHECK(clamp(15, 0, 10) == 10);
    CHECK(clamp(-3, 0, 10) == 0);

    // fp16 边界：1.0 <-> 0x3C00
    CHECK(fp16_to_float(0x3C00) == 1.0f);
}

// D3 检测+单目测距：像素→米换算 / 框→深度图映射 / 框内中值统计（纯函数）
static void testDepthDistance() {
    std::printf("[test] depth distance math\n");
    // 反相图端点：v=255 → lo，v=0 → hi；scale 乘法生效
    CHECK(std::fabs(depthPixelToMeters(255, 1.0f, 21.0f) - 1.0f) < 1e-4f);
    CHECK(std::fabs(depthPixelToMeters(0, 1.0f, 21.0f) - 21.0f) < 1e-4f);
    CHECK(std::fabs(depthPixelToMeters(128, 0.0f, 10.0f) - 4.98f) < 0.02f);
    CHECK(std::fabs(depthPixelToMeters(255, 1.0f, 21.0f, 2.0f) - 2.0f) < 1e-4f);
    CHECK(depthPixelToMeters(0, 5.0f, 5.0f) == 0.0f);  // lo==hi 无效范围

    // 框→深度子图映射：roi 平移 + 缩放 + 越界裁剪
    const cv::Rect roi(10, 10, 30, 20);
    const cv::Size dsize(60, 40);  // 深度图是 roi 的 2 倍分辨率
    cv::Rect mapped;
    image_rect_t box{10, 10, 25, 20};  // roi 左上角 → (0,0)，尺寸 ×2
    CHECK(mapBoxToDepthRect(box, roi, dsize, &mapped));
    CHECK(mapped.x == 0 && mapped.y == 0 && mapped.width == 31 && mapped.height == 21);
    box = image_rect_t{-50, -50, 25, 20};  // 左上越界 → 裁剪到 0
    CHECK(mapBoxToDepthRect(box, roi, dsize, &mapped));
    CHECK(mapped.x == 0 && mapped.y == 0);
    box = image_rect_t{100, 100, 120, 130};  // 完全在 roi 外 → 无效
    CHECK(!mapBoxToDepthRect(box, roi, dsize, &mapped));

    // 框内中值统计：单个离群块不拉偏结果（均值会被拉偏）
    cv::Mat depth(40, 60, CV_8UC1, cv::Scalar(120));
    cv::rectangle(depth, cv::Rect(0, 0, 8, 8), cv::Scalar(10), cv::FILLED);  // 近距噪点块
    box = image_rect_t{0, 0, 59, 39};
    float meters = 0.0f;
    CHECK(boxDepthMeters(depth, cv::Rect(0, 0, 60, 40), box, 0.0f, 20.0f, 1.0f, &meters));
    // v=120 → meters = (255-120)*20/255 ≈ 10.59
    CHECK(std::fabs(meters - 10.59f) < 0.05f);
    CHECK(!boxDepthMeters(depth, cv::Rect(0, 0, 60, 40), image_rect_t{100, 100, 120, 130},
                          0.0f, 20.0f, 1.0f, &meters));
}

// D1/D2 事件规则引擎：绊线/入侵/滞留/离岗状态机（纯逻辑）
static void testEventEngine() {
    std::printf("[test] EventEngine rules\n");
    const auto t0 = std::chrono::steady_clock::now();
    auto mk = [](int id, int cls, int l, int t, int r, int b, bool confirmed = true) {
        TrackedDetection td;
        td.track_id = id;
        td.det.cls_id = cls;
        td.is_confirmed = confirmed;
        td.det.box = {l, t, r, b};
        return td;
    };
    const auto ms = [](long long v) { return std::chrono::milliseconds(v); };

    // 绊线：水平线 y=100（cfg.event_line "0,100,200,100"），只对 person（cls 0）生效
    AppConfig cfg;
    cfg.event_line = "0,100,200,100";
    cfg.event_line_cross = true;
    cfg.event_classes = "0";
    EventEngine eng(cfg);
    CHECK(eng.active() && eng.lineCrossEnabled());

    // 下方(+侧) → 上方(-侧)：B->A；回到下方：A->B；同侧继续移动不触发
    eng.update({mk(1, 0, 90, 140, 110, 160)}, t0);
    auto evs = eng.update({mk(1, 0, 90, 40, 110, 60)}, t0 + ms(33));
    CHECK(evs.size() == 1 && evs[0].type == "line_cross" && evs[0].detail == "B->A");
    evs = eng.update({mk(1, 0, 90, 140, 110, 160)}, t0 + ms(66));
    CHECK(evs.size() == 1 && evs[0].type == "line_cross" && evs[0].detail == "A->B");
    evs = eng.update({mk(1, 0, 90, 240, 110, 260)}, t0 + ms(99));
    CHECK(evs.empty());
    CHECK(eng.lineCrossAB() == 1 && eng.lineCrossBA() == 1);

    // 类别过滤：car（cls 2）穿线不触发；未确认轨迹不触发
    eng.update({mk(2, 2, 90, 140, 110, 160)}, t0 + ms(132));
    evs = eng.update({mk(2, 2, 90, 40, 110, 60)}, t0 + ms(165));
    CHECK(evs.empty());
    eng.update({mk(3, 0, 90, 140, 110, 160, false)}, t0 + ms(198));
    evs = eng.update({mk(3, 0, 90, 40, 110, 60, false)}, t0 + ms(231));
    CHECK(evs.empty());

    // 区域：入侵 + 滞留 + 离岗（2s 阈值便于测试）
    AppConfig rcfg;
    rcfg.event_region = "0,0,100,0,100,100,0,100";
    rcfg.event_intrusion = true;
    rcfg.event_dwell = true;
    rcfg.event_dwell_seconds = 2.0;
    rcfg.event_absence = true;
    rcfg.event_absence_seconds = 2.0;
    EventEngine reng(rcfg);
    CHECK(reng.active() && reng.intrusionEnabled() && reng.dwellEnabled() && reng.absenceEnabled());

    // 区域外 → 区域内：intrusion
    reng.update({mk(1, 0, 190, 40, 210, 60)}, t0);
    evs = reng.update({mk(1, 0, 40, 40, 60, 60)}, t0 + ms(100));
    CHECK(evs.size() == 1 && evs[0].type == "intrusion" && evs[0].track_id == 1);
    // 持续滞留：>2s 触发一次 dwell，之后不重复
    evs = reng.update({mk(1, 0, 40, 40, 60, 60)}, t0 + ms(1200));
    CHECK(evs.empty());
    evs = reng.update({mk(1, 0, 40, 40, 60, 60)}, t0 + ms(2200));
    CHECK(evs.size() == 1 && evs[0].type == "dwell");
    evs = reng.update({mk(1, 0, 40, 40, 60, 60)}, t0 + ms(3000));
    CHECK(evs.empty());
    // 人离开（区域空）：空满 2s 后触发一次 absence；之后不再重复
    evs = reng.update({}, t0 + ms(4000));  // 区域空计时开始
    CHECK(evs.empty());
    evs = reng.update({}, t0 + ms(5200));  // 空 1.2s，不足阈值
    CHECK(evs.empty());
    evs = reng.update({}, t0 + ms(6100));  // 空 2.1s → absence
    CHECK(evs.size() == 1 && evs[0].type == "absence" && evs[0].track_id == 0);
    evs = reng.update({}, t0 + ms(6300));
    CHECK(evs.empty());
    // 重新进入：absence 后区域状态已复位，再次触发 intrusion
    evs = reng.update({mk(1, 0, 40, 40, 60, 60)}, t0 + ms(6400));
    CHECK(evs.size() == 1 && evs[0].type == "intrusion");

    // 非法配置：规则开了但没给几何 → 规则被剔除，引擎不激活
    AppConfig bad;
    bad.event_line_cross = true;
    EventEngine bad_eng(bad);
    CHECK(!bad_eng.active());
}

// 命名多规则：方向过滤（逆行）+ 多规则 rule_id + crowd + abandoned
static void testEventEngineMultiRule() {
    std::printf("[test] EventEngine 多规则\n");
    const auto t0 = std::chrono::steady_clock::now();
    auto mk = [](int id, int cls, int l, int t, int r, int b, bool confirmed = true) {
        TrackedDetection td;
        td.track_id = id;
        td.det.cls_id = cls;
        td.is_confirmed = confirmed;
        td.det.box = {l, t, r, b};
        return td;
    };
    const auto ms = [](long long v) { return std::chrono::milliseconds(v); };

    // YAML 解析：event_rules 列表（优先于扁平键）
    AppConfig cfg;
    cfg.event_line_cross = true;  // 扁平键应被 event_rules 忽略（无 line 几何，也不该合成规则）
    EventRuleConfig line_rule;
    line_rule.id = "gate";
    line_rule.type = "line_cross";
    line_rule.line = "0,100,200,100";
    line_rule.direction = "A2B";  // 只上报 A->B（B->A 即逆行，不报但计数）
    EventRuleConfig crowd_rule;
    crowd_rule.id = "plaza";
    crowd_rule.type = "crowd";
    crowd_rule.region = "0,0,100,0,100,100,0,100";
    crowd_rule.min_count = 2;
    EventRuleConfig left_rule;
    left_rule.id = "lobby";
    left_rule.type = "abandoned";
    left_rule.region = "200,0,300,0,300,100,200,100";
    left_rule.seconds = 1.0;
    cfg.event_rules = {line_rule, crowd_rule, left_rule};
    EventEngine eng(cfg);
    CHECK(eng.active() && eng.lineCrossEnabled());

    // 方向过滤：B->A（下方→上方）穿越不产生事件（逆行方向），计数仍累计
    eng.update({mk(1, 0, 90, 140, 110, 160)}, t0);
    auto evs = eng.update({mk(1, 0, 90, 40, 110, 60)}, t0 + ms(33));
    CHECK(evs.empty());
    CHECK(eng.lineCrossAB() == 0 && eng.lineCrossBA() == 1);
    // A->B（上方→下方）：合法方向，事件带 rule_id
    evs = eng.update({mk(1, 0, 90, 140, 110, 160)}, t0 + ms(66));
    CHECK(evs.size() == 1 && evs[0].type == "line_cross" && evs[0].rule_id == "gate" &&
          evs[0].detail == "A->B");
    CHECK(eng.lineCrossAB() == 1 && eng.lineCrossBA() == 1);

    // crowd：1 人不触发；2 人越阈值触发一次（带人数），持续不重复；回落重新武装
    eng.update({mk(10, 0, 40, 40, 60, 60)}, t0 + ms(100));
    evs = eng.update({mk(10, 0, 40, 40, 60, 60), mk(11, 0, 70, 70, 90, 90)}, t0 + ms(133));
    CHECK(evs.size() == 1 && evs[0].type == "crowd" && evs[0].rule_id == "plaza" &&
          evs[0].detail == "2人");
    evs = eng.update({mk(10, 0, 40, 40, 60, 60), mk(11, 0, 70, 70, 90, 90)}, t0 + ms(166));
    CHECK(evs.empty());
    // 回落到 1 人 → 再到 2 人：重新触发
    eng.update({mk(10, 0, 40, 40, 60, 60)}, t0 + ms(200));
    evs = eng.update({mk(10, 0, 40, 40, 60, 60), mk(11, 0, 70, 70, 90, 90)}, t0 + ms(233));
    CHECK(evs.size() == 1 && evs[0].type == "crowd");

    // abandoned：静止 1s 触发一次；持续静止不重复；位移后重新计时
    eng.update({mk(20, 5, 210, 40, 240, 80)}, t0 + ms(300));   // 进入区域，锚定
    evs = eng.update({mk(20, 5, 211, 41, 241, 81)}, t0 + ms(600));  // 微动（阈值内）不触发
    CHECK(evs.empty());
    evs = eng.update({mk(20, 5, 212, 42, 242, 82)}, t0 + ms(1400));  // 静止 >1s → abandoned
    CHECK(evs.size() == 1 && evs[0].type == "abandoned" && evs[0].rule_id == "lobby" &&
          evs[0].track_id == 20);
    evs = eng.update({mk(20, 5, 212, 42, 242, 82)}, t0 + ms(1800));  // 仍静止：不重复
    CHECK(evs.empty());
    // 大幅移动 → 重新锚定；再静止 1s 再次触发
    eng.update({mk(20, 5, 280, 40, 300, 80)}, t0 + ms(2200));
    evs = eng.update({mk(20, 5, 281, 41, 301, 81)}, t0 + ms(3300));
    CHECK(evs.size() == 1 && evs[0].type == "abandoned");
}

// 跌倒规则（M1）：宽高比躺倒形态 + 持续时长去抖状态机
static void testEventEngineFall() {
    std::printf("[test] EventEngine fall\n");
    const auto t0 = std::chrono::steady_clock::now();
    auto mk = [](int id, int cls, int l, int t, int r, int b, bool confirmed = true) {
        TrackedDetection td;
        td.track_id = id;
        td.det.cls_id = cls;
        td.is_confirmed = confirmed;
        td.det.box = {l, t, r, b};
        return td;
    };
    const auto ms = [](long long v) { return std::chrono::milliseconds(v); };

    AppConfig cfg;
    EventRuleConfig fall_rule;
    fall_rule.id = "ward";
    fall_rule.type = "fall";
    fall_rule.classes = "0";       // 只对人形生效
    fall_rule.fall_aspect = 1.2;   // w/h >= 1.2 视为躺倒
    fall_rule.fall_seconds = 1.0;  // 躺倒持续 1s 才告警
    cfg.event_rules = {fall_rule};
    EventEngine eng(cfg);
    CHECK(eng.active());

    // 站立（瘦高框）不触发
    auto evs = eng.update({mk(1, 0, 40, 20, 80, 100)}, t0);   // aspect 0.5
    CHECK(evs.empty());
    evs = eng.update({mk(1, 0, 40, 20, 80, 100)}, t0 + ms(2000));
    CHECK(evs.empty());

    // 躺倒（宽扁框）：持续不足阈值不触发，满 1s 触发一次
    evs = eng.update({mk(1, 0, 10, 60, 130, 100)}, t0 + ms(2300));  // aspect 3.0，锚定
    CHECK(evs.empty());
    evs = eng.update({mk(1, 0, 10, 60, 130, 100)}, t0 + ms(2900));  // 躺倒 0.6s
    CHECK(evs.empty());
    evs = eng.update({mk(1, 0, 10, 60, 130, 100)}, t0 + ms(3400));  // 躺倒 1.1s → fall
    CHECK(evs.size() == 1 && evs[0].type == "fall" && evs[0].rule_id == "ward" &&
          evs[0].track_id == 1 && evs[0].cls_id == 0 && evs[0].detail == "1s");
    evs = eng.update({mk(1, 0, 10, 60, 130, 100)}, t0 + ms(4000));  // 仍躺倒：不重复
    CHECK(evs.empty());

    // 起身（宽高比回落）复位 → 再次躺倒可重复触发
    evs = eng.update({mk(1, 0, 40, 20, 80, 100)}, t0 + ms(4300));
    CHECK(evs.empty());
    evs = eng.update({mk(1, 0, 10, 60, 130, 100)}, t0 + ms(4600));  // 锚定
    CHECK(evs.empty());
    evs = eng.update({mk(1, 0, 10, 60, 130, 100)}, t0 + ms(5700));  // 再躺 1.1s
    CHECK(evs.size() == 1 && evs[0].type == "fall");

    // 类别过滤：car（cls 2）天然宽扁也不触发；未确认轨迹不触发
    eng.update({mk(2, 2, 10, 60, 130, 100)}, t0 + ms(6100));
    evs = eng.update({mk(2, 2, 10, 60, 130, 100)}, t0 + ms(7300));
    CHECK(evs.empty());
    eng.update({mk(3, 0, 10, 60, 130, 100, false)}, t0 + ms(7600));
    evs = eng.update({mk(3, 0, 10, 60, 130, 100, false)}, t0 + ms(8800));
    CHECK(evs.empty());

    // region 限定：区域内躺倒触发，区域外躺倒不触发
    AppConfig zcfg;
    EventRuleConfig zfall;
    zfall.id = "yard";
    zfall.type = "fall";
    zfall.region = "0,0,100,0,100,100,0,100";
    zfall.fall_seconds = 0.5;
    zcfg.event_rules = {zfall};
    EventEngine zeng(zcfg);
    zeng.update({mk(5, 0, 10, 60, 130, 100)}, t0 + ms(100));    // 区域内躺倒锚定
    evs = zeng.update({mk(5, 0, 10, 60, 130, 100)}, t0 + ms(700));
    CHECK(evs.size() == 1 && evs[0].type == "fall" && evs[0].rule_id == "yard");
    zeng.update({mk(6, 0, 210, 60, 330, 100)}, t0 + ms(1000));  // 区域外躺倒锚定
    evs = zeng.update({mk(6, 0, 210, 60, 330, 100)}, t0 + ms(1700));
    CHECK(evs.empty());

    // 配置校验：fall 无 region 合法；fall_aspect <= 0 拒绝
    AppConfig vcfg;
    vcfg.input_path = "/tmp/x.mp4";   // validate() 要求 input/model 必填
    vcfg.model_path = "/tmp/x.rknn";
    EventRuleConfig vfall;
    vfall.type = "fall";
    vcfg.event_rules = {vfall};
    std::string err;
    CHECK(vcfg.validate(&err));
    vfall.fall_aspect = 0.0;
    vcfg.event_rules = {vfall};
    CHECK(!vcfg.validate(&err));
    // 类型白名单：fall 进入合法类型
    EventRuleConfig badtype;
    badtype.type = "slip";
    vcfg.event_rules = {badtype};
    CHECK(!vcfg.validate(&err));
}

// G3.2 绊线参考点（ref_point=center|bottom）+ G3.1 测速规则（speed）
static void testEventEngineSpeedRef() {
    std::printf("[test] EventEngine speed/ref_point\n");
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    auto mk = [](int id, int cls, int l, int t, int r, int b, bool confirmed = true) {
        TrackedDetection td;
        td.track_id = id;
        td.det.cls_id = cls;
        td.is_confirmed = confirmed;
        td.det.box = {l, t, r, b};
        return td;
    };
    const auto ms = [](long long v) { return std::chrono::milliseconds(v); };

    // ---- ref_point：同一移动（中心不穿线，脚底穿线），bottom 触发 / center 不触发 ----
    // 绊线 y=110。t0 框 (90,60,110,108)：中心 y=84 在上，脚底 y=108 在上（线更下方）
    // t1 框 (90,80,110,128)：中心 y=104 仍在上（未穿），脚底 y=128 已在下（穿过）
    AppConfig bcfg;
    EventRuleConfig bline;
    bline.id = "gate";
    bline.type = "line_cross";
    bline.line = "0,110,200,110";
    bline.ref_point = "bottom";
    bcfg.event_rules = {bline};
    EventEngine beng(bcfg);
    beng.update({mk(1, 0, 90, 60, 110, 108)}, t0);
    auto evs = beng.update({mk(1, 0, 90, 80, 110, 128)}, t0 + ms(33));
    CHECK(evs.size() == 1 && evs[0].type == "line_cross" && evs[0].detail == "A->B");

    AppConfig ccfg;
    EventRuleConfig cline = bline;
    cline.ref_point = "center";  // 默认语义
    ccfg.event_rules = {cline};
    EventEngine ceng(ccfg);
    ceng.update({mk(1, 0, 90, 60, 110, 108)}, t0);
    evs = ceng.update({mk(1, 0, 90, 80, 110, 128)}, t0 + ms(33));
    CHECK(evs.empty());  // 中心参考：84→104 未穿线
    // 反向区分：中心穿线、脚底不穿线
    ceng.update({mk(2, 0, 90, 80, 110, 130)}, t0 + ms(66));    // 中心105上，脚底130下
    evs = ceng.update({mk(2, 0, 90, 100, 110, 150)}, t0 + ms(99));  // 中心105→125 穿线
    CHECK(evs.size() == 1 && evs[0].type == "line_cross");
    beng.update({mk(2, 0, 90, 80, 110, 130)}, t0 + ms(66));
    evs = beng.update({mk(2, 0, 90, 100, 110, 150)}, t0 + ms(99));  // 脚底130→150 都在下
    CHECK(evs.empty());

    // ---- speed：EMA 速度 + 持续去抖 + km/h 换算 ----
    AppConfig scfg;
    scfg.input_path = "/tmp/x.mp4";
    scfg.model_path = "/tmp/x.rknn";
    EventRuleConfig spd;
    spd.id = "road";
    spd.type = "speed";
    spd.speed_limit = 100;  // px/s
    spd.speed_seconds = 0.3;
    scfg.event_rules = {spd};
    CHECK(scfg.validate(&err));
    EventEngine seng(scfg);

    seng.update({mk(1, 0, 90, 90, 110, 110)}, t0);  // 锚定（首帧无速度）
    // 等步长 +150ms、中心位移 30px（瞬时恒 200px/s）：超限自 f1 起算，f3 持续满 0.3s 触发
    evs = seng.update({mk(1, 0, 120, 90, 140, 110)}, t0 + ms(150));  // EMA=200 超限锚定
    CHECK(evs.empty());  // 持续 0
    evs = seng.update({mk(1, 0, 150, 90, 170, 110)}, t0 + ms(300));  // EMA=200
    CHECK(evs.empty());  // 持续 0.15s < 0.3
    evs = seng.update({mk(1, 0, 180, 90, 200, 110)}, t0 + ms(450));  // 持续 0.30s ≥ 0.3 → 触发
    CHECK(evs.size() == 1 && evs[0].type == "speed" && evs[0].rule_id == "road" &&
          evs[0].detail == "200px/s");
    // 持续超限不重复
    evs = seng.update({mk(1, 0, 210, 90, 230, 110)}, t0 + ms(600));
    CHECK(evs.empty());
    // 两个慢帧把 EMA 拖回限内（0.6^n 衰减：121 → 77）→ 复位
    evs = seng.update({mk(1, 0, 215, 90, 235, 110)}, t0 + ms(2600));  // 5px/2s → EMA=121 仍超限
    CHECK(evs.empty());
    evs = seng.update({mk(1, 0, 216, 90, 236, 110)}, t0 + ms(2700));  // EMA≈77 < 100 → 复位
    CHECK(evs.empty());
    // 复位后同目标再次连续超限：f7 锚定，f10 持续满 0.3s 再触发
    seng.update({mk(1, 0, 246, 90, 266, 110)}, t0 + ms(2800));  // 30px/0.1s=300 → EMA=166 锚定
    evs = seng.update({mk(1, 0, 276, 90, 296, 110)}, t0 + ms(2900));  // EMA=220，持续 0.1s
    CHECK(evs.empty());
    evs = seng.update({mk(1, 0, 306, 90, 326, 110)}, t0 + ms(3000));  // 持续 0.2s
    CHECK(evs.empty());
    evs = seng.update({mk(1, 0, 336, 90, 356, 110)}, t0 + ms(3100));  // 持续 0.30s → 再触发
    CHECK(evs.size() == 1 && evs[0].type == "speed");

    // km/h 换算：px_per_meter=10 → 200px/s / 10 * 3.6 = 72km/h
    AppConfig kcfg;
    EventRuleConfig kspd;
    kspd.id = "hw";
    kspd.type = "speed";
    kspd.speed_limit = 100;
    kspd.speed_seconds = 0.1;
    kspd.px_per_meter = 10;
    kcfg.event_rules = {kspd};
    EventEngine keng(kcfg);
    keng.update({mk(3, 2, 90, 90, 110, 110)}, t0);
    keng.update({mk(3, 2, 110, 90, 130, 110)}, t0 + ms(100));  // EMA=200
    evs = keng.update({mk(3, 2, 130, 90, 150, 110)}, t0 + ms(200));
    CHECK(evs.size() == 1 && evs[0].detail == "72.0km/h");

    // 校验：speed 缺 speed_limit 拒绝；ref_point 非法值拒绝
    AppConfig vcfg;
    vcfg.input_path = "/tmp/x.mp4";
    vcfg.model_path = "/tmp/x.rknn";
    EventRuleConfig vspd;
    vspd.type = "speed";
    vcfg.event_rules = {vspd};
    CHECK(!vcfg.validate(&err));  // 缺 speed_limit
    vspd.speed_limit = 100;
    vcfg.event_rules = {vspd};
    CHECK(vcfg.validate(&err));
    EventRuleConfig badref;
    badref.type = "line_cross";
    badref.line = "0,0,10,10";
    badref.ref_point = "left";
    vcfg.event_rules = {badref};
    CHECK(!vcfg.validate(&err));
    badref.ref_point = "bottom";
    vcfg.event_rules = {badref};
    CHECK(vcfg.validate(&err));
}

// Detect3D（单目 3D）：14 列行解码 + letterbox 逆映射（纯函数）
static void testDetect3DDecode() {
    std::printf("[test] detect3d decode\n");
    // 行布局: [x1,y1,x2,y2, conf, cls, cx3d,cy3d, depth, sin,cos, h3,w3,l3]
    const float row[14] = {100.f, 50.f, 300.f, 200.f, 0.87f, 0.f,
                           200.f, 190.f, 12.5f, 0.6f, 0.8f, 1.5f, 1.8f, 4.1f};
    Detect3DItem it = detect3dDecodeRow(row);
    CHECK(it.box.left == 100 && it.box.top == 50 && it.box.right == 300 && it.box.bottom == 200);
    CHECK(std::fabs(it.conf - 0.87f) < 1e-6f);
    CHECK(it.cls_id == 0);
    CHECK(std::fabs(it.depth_m - 12.5f) < 1e-5f);
    CHECK(std::fabs(it.h3 - 1.5f) < 1e-6f && std::fabs(it.w3 - 1.8f) < 1e-6f && std::fabs(it.l3 - 4.1f) < 1e-6f);

    // letterbox 逆映射：模型 416x1280，原帧 832x1280（scale=0.5, 无 pad, crop=0）
    letterbox_t lb{};
    lb.scale = 0.5f;
    lb.x_pad = 0.f;
    lb.y_pad = 0.f;
    lb.crop_x = 0;
    lb.crop_y = 0;
    lb.crop_w = 1280;
    lb.crop_h = 832;
    detect3dMapToFrame(it, &lb, 1280, 416, 1280, 832);
    CHECK(it.box.left == 200 && it.box.top == 100 && it.box.right == 600 && it.box.bottom == 400);
    CHECK(it.center_u == 400 && it.center_v == 380);

    // 带 pad + crop：模型坐标先减 pad 再除 scale 加 crop，并裁剪到原帧
    Detect3DItem it2 = detect3dDecodeRow(row);
    letterbox_t lb2{};
    lb2.scale = 1.0f;
    lb2.x_pad = 10.f;
    lb2.y_pad = 20.f;
    lb2.crop_x = 5;
    lb2.crop_y = 6;
    lb2.crop_w = 100;
    lb2.crop_h = 100;
    detect3dMapToFrame(it2, &lb2, 1280, 416, 640, 480);
    CHECK(it2.box.left == 95 && it2.box.top == 36 && it2.box.right == 295 && it2.box.bottom == 186);
    CHECK(it2.center_u == 195 && it2.center_v == 176);

    // 无 letterbox（scale<=0）：坐标视为已归一化，仅裁剪
    Detect3DItem it3 = detect3dDecodeRow(row);
    detect3dMapToFrame(it3, nullptr, 1280, 416, 150, 100);
    CHECK(it3.box.right == 149 && it3.box.bottom == 99);

    // 板端布局（38 列）：原始量，解码全在 C++
    float row38[38] = {0};
    row38[0] = 10.f; row38[1] = 20.f; row38[2] = 60.f; row38[3] = 80.f;  // 框：中心(35,50) 尺寸(50,60)
    row38[4] = 0.10f; row38[5] = 0.90f; row38[6] = 0.30f;               // cls=1 (Pedestrian)
    row38[7] = 0.1f; row38[8] = -0.5f;                                   // 中心偏移
    row38[9] = std::log(9.5f);                                           // log 深度
    row38[10] = 0.0f; row38[11] = 0.0f; row38[12] = 0.0f;                // 尺寸残差 0 → 先验原值
    row38[13] = -3.0f;                                                   // q3d（忽略）
    row38[14 + 3] = 5.0f;                                                // bin3 胜出
    row38[26 + 3] = 1.0f;                                                // 残差 tanh(1.0)
    Detect3DItem b = detect3dDecodeRowCols(row38, 38);
    CHECK(b.box.left == 10 && b.box.bottom == 80);
    CHECK(std::fabs(b.conf - 0.90f) < 1e-6f);
    CHECK(b.cls_id == 1);
    // 中心 = 框中心 + 偏移×框宽高：(35+0.1*50, 50-0.5*60) = (40, 20)
    CHECK(std::fabs(b.center_u - 40.0f) < 1e-4f);
    CHECK(std::fabs(b.center_v - 20.0f) < 1e-4f);
    CHECK(std::fabs(b.depth_m - 9.5f) < 1e-4f);
    // 尺寸 = Pedestrian 先验 × exp(0)
    CHECK(std::fabs(b.h3 - 1.76f) < 1e-4f);
    CHECK(std::fabs(b.w3 - 0.66f) < 1e-4f);
    CHECK(std::fabs(b.l3 - 0.84f) < 1e-4f);
    // multibin：alpha = 3×(2π/12) + tanh(1)×(π/12)
    const float bin_size = 2.0f * static_cast<float>(M_PI) / 12.0f;
    const float alpha = 3.0f * bin_size + std::tanh(1.0f) * (bin_size * 0.5f);
    CHECK(std::fabs(b.sin_alpha - std::sin(alpha)) < 1e-5f);
    CHECK(std::fabs(b.cos_alpha - std::cos(alpha)) < 1e-5f);

    // 3D 线框投影：简单内参 P2（fx=fy=100, cx=cy=50）手算对照
    // 目标：3D 底面中心 (0,0,10)；sin=0,cos=1 → ry=atan2(0,1)=0（车长沿相机 x 轴，
    // 2026-09-02 约定修正：模型 sin/cos 即最终 ry，不再叠加 theta）
    Detect3DItem w{};
    w.center_u = 50.f; w.center_v = 50.f; w.depth_m = 10.f;
    w.cos_alpha = 1.f; w.sin_alpha = 0.f;
    w.h3 = 1.5f; w.w3 = 2.0f; w.l3 = 4.0f;
    const float p2[12] = {100.f, 0, 50.f, 0, 0, 100.f, 50.f, 0, 0, 0, 1.f, 0};
    float corners[8][2];
    CHECK(computeDetect3DCorners2D(w, p2, corners));
    // 中心语义（y 角点 ±h/2）；ry=0 → c=1,s=0：X3=xc, Z3=zc+10
    // 角点 0（xc=+2, zc=+1, 底面 y=+0.75）：X3=2, Z3=11 → px=(200+550)/11≈68.182
    CHECK(std::fabs(corners[0][0] - 68.1818f) < 1e-3f && std::fabs(corners[0][1] - 56.8182f) < 1e-3f);
    // 角点 2（xc=-2, zc=-1, 底面）：X3=-2, Z3=9 → px=(-200+450)/9≈27.778, py=(75+450)/9≈58.333
    CHECK(std::fabs(corners[2][0] - 27.7778f) < 1e-2f && std::fabs(corners[2][1] - 58.3333f) < 1e-3f);
    // 角点 4（xc=+2, zc=+1, 顶面 y=-0.75）：X3=2, Z3=11 → py=(-75+550)/11≈43.182
    CHECK(std::fabs(corners[4][0] - 68.1818f) < 1e-3f && std::fabs(corners[4][1] - 43.1818f) < 1e-3f);
    // 非法输入：深度非正 → false
    Detect3DItem bad = w;
    bad.depth_m = 0.0f;
    CHECK(!computeDetect3DCorners2D(bad, p2, corners));
}

static void testPerformanceAverage() {
    std::printf("[test] PerformanceMetrics 平均与计数\n");
    PerformanceMetrics m;
    // 无测量时平均为 0
    CHECK(m.getAverageInferenceTime() == 0.0);
    CHECK(m.getAverageTotalTime() == 0.0);
    // 一次完整测量后计数 +1，平均非负
    m.startPreprocessing();
    m.stopPreprocessing();
    m.startInference();
    m.stopInference();
    m.startPostprocessing();
    m.stopPostprocessing();
    m.startTotal();
    m.stopTotal();
    m.addMeasurement();
    CHECK(m.getTotalMeasurements() == 1);
    CHECK(m.getAverageInferenceTime() >= 0.0);
    CHECK(m.getAveragePostprocessingTime() >= 0.0);
    // reset 后归零
    m.reset();
    CHECK(m.getTotalMeasurements() == 0);
}

static void testDetectionFilter() {
    std::printf("[test] DetectionFilter ROI 过滤\n");
    AppConfig cfg;
    cfg.roi = "0,0,100,0,100,100,0,100";  // 100x100 正方形
    DetectionFilter filter(cfg);
    CHECK(filter.active());

    object_detect_result_list list{};
    // 框1 中心 (50,50) 在 ROI 内 → 保留
    list.results[0].box = {40, 40, 60, 60};
    list.results[0].cls_id = 0;
    list.results[0].prop = 0.9f;
    // 框2 中心 (150,50) 在 ROI 外 → 移除
    list.results[1].box = {140, 40, 160, 60};
    list.results[1].cls_id = 0;
    list.results[1].prop = 0.9f;
    list.count = 2;

    filter.apply(nullptr, &list);
    CHECK(list.count == 1);
    CHECK(list.results[0].box.left == 40);

    // 未配置 ROI → 不启用
    AppConfig empty_cfg;
    DetectionFilter empty_filter(empty_cfg);
    CHECK(!empty_filter.active());
}

static void testTurbojpeg() {
    std::printf("[test] turbojpeg BGR→JPEG 冒烟\n");
    // 纯 CPU 冒烟：32x32 BGR 图，验证 tjCompress2 可用且输出非空
    cv::Mat bgr(32, 32, CV_8UC3, cv::Scalar(0, 128, 255));
    tjhandle handle = tjInitCompress();
    CHECK(handle != nullptr);
    if (!handle) {
        return;
    }
    unsigned char* jpeg_buf = nullptr;
    unsigned long jpeg_size = 0;
    CHECK(tjCompress2(handle, bgr.data, bgr.cols, bgr.step, bgr.rows,
                      TJPF_BGR, &jpeg_buf, &jpeg_size, TJSAMP_420, 80, TJFLAG_FASTDCT) == 0);
    CHECK(jpeg_size > 0);
    tjFree(jpeg_buf);
    tjDestroy(handle);
}


// daemon mini_json（G2 补测）：解析/转义/\u 转义（UTF-8+代理对）/dump 往返/非法输入
static void testMiniJson() {
    std::printf("[test] mini_json\n");
    using namespace mini_json;
    bool ok = false;
    JsonValue v = JsonParser("{\"a\":1,\"b\":[1,2,3],\"c\":\"x\",\"d\":true,\"e\":null}").parse(ok);
    CHECK(ok && v.type == JsonValue::Type::Object);
    CHECK(v.get("a") && v.get("a")->type == JsonValue::Type::Number && v.get("a")->num == 1.0);
    CHECK(v.get("d") && v.get("d")->b);
    CHECK(v.get("e") && v.get("e")->type == JsonValue::Type::Null);
    CHECK(v.get("b") && v.get("b")->arr.size() == 3 && v.get("b")->arr[1].num == 2.0);
    CHECK(v.get("c") && v.get("c")->str == "x");
    CHECK(v.get("missing") == nullptr);

    // 转义字符还原
    JsonValue esc = JsonParser("\"a\\nb\\\"c\\\\d\\t\"").parse(ok);
    CHECK(ok && esc.type == JsonValue::Type::String && esc.str == "a\nb\"c\\d\t");

    // \uXXXX → UTF-8：中文（3 字节）与代理对组合（emoji 4 字节）
    JsonValue uni = JsonParser("\"\\u4E2D\\u6587\"").parse(ok);
    CHECK(ok && uni.str == "\xE4\xB8\xAD\xE6\x96\x87");
    JsonValue emoji = JsonParser("\"\\ud83d\\ude00\"").parse(ok);
    CHECK(ok && emoji.str == "\xF0\x9F\x98\x80");

    // 大整数精度（原 %g 截断 bug）：毫秒时间戳
    JsonValue big = JsonParser("{\"ts\":1788366121382}").parse(ok);
    CHECK(ok && big.get("ts") && big.get("ts")->num == 1788366121382.0);

    // dump 往返 + 流式转义
    JsonValue rt = JsonParser("{\"k\":\"a\\\"b\",\"n\":-2.5,\"arr\":[true,null,\"\\u4e2d\"]}").parse(ok);
    CHECK(ok);
    JsonValue rt2 = JsonParser(dump(rt)).parse(ok);
    CHECK(ok && rt2.get("k")->str == "a\"b" && rt2.get("n")->num == -2.5 &&
          rt2.get("arr")->arr.size() == 3 && rt2.get("arr")->arr[2].str == "\xE4\xB8\xAD");
    std::ostringstream os;
    appendEscaped(os, "x\"y\n");
    CHECK(os.str() == "x\\\"y\\n");

    // 非法输入：置 ok=false 而非崩溃/静默成功
    JsonParser("{").parse(ok);
    CHECK(!ok);
    JsonParser("[1,,").parse(ok);
    CHECK(!ok);
    JsonParser("").parse(ok);
    CHECK(!ok);
    JsonParser("\"unterminated").parse(ok);
    CHECK(!ok);
}

// daemon EventStore（G2 补测）：环形缓冲/查询过滤/复核挂接/落盘回放/轮转
static void testEventStore() {
    std::printf("[test] EventStore\n");
    // 纯内存环形缓冲 + id 自增 + 非法 body 拒绝
    EventStore mem("", 4, 0);
    const long id1 = mem.append("{\"task\":\"detect\",\"event\":\"rule_events\",\"rule_type\":\"fall\"}");
    const long id2 = mem.append("{\"task\":\"gate\",\"event\":\"rule_events\",\"rule_type\":\"line_cross\"}");
    const long id3 = mem.append("{\"task\":\"detect\",\"event\":\"count\",\"total\":3}");
    CHECK(id1 == 1 && id2 == 2 && id3 == 3);
    CHECK(mem.append("not json") == -1);
    CHECK(mem.append("[1,2]") == -1);  // 非对象
    CHECK(mem.size() == 3);

    // 查询：task 过滤 / type（rule_events 细分看 rule_type）/ limit 保最近
    auto fall = mem.query("", "fall", 0, 0);
    CHECK(fall.size() == 1 && fall[0].id == id1);
    auto gate = mem.query("gate", "", 0, 0);
    CHECK(gate.size() == 1 && gate[0].id == id2);
    auto count_ev = mem.query("", "count", 0, 0);
    CHECK(count_ev.size() == 1 && count_ev[0].id == id3);
    auto rule_ev = mem.query("", "rule_events", 0, 0);
    CHECK(rule_ev.size() == 2);  // type=rule_events 按 body.event 命中（fall + line_cross 两条）
    auto recent = mem.query("", "", 0, 2);
    CHECK(recent.size() == 2 && recent.front().id == 2 && recent.back().id == 3);

    // 环形容量淘汰：cap=4，喂到 7 条后只留 id 4..7
    for (int i = 0; i < 4; ++i) {
        mem.append("{\"task\":\"x\"}");
    }
    CHECK(mem.size() == 4);
    EventStore::Item it;
    CHECK(!mem.get(3, &it) && mem.get(4, &it) && it.id == 4);
    auto since3 = mem.query("", "", 3, 0);
    CHECK(since3.size() == 4 && since3.front().id == 4);  // since=仅返回 id 更大者

    // 复核挂接 + 落盘回放 + id 续号 + 轮转
    const std::string path = "/tmp/rkpipe_test_events.jsonl";
    std::filesystem::remove(path);
    std::filesystem::remove(path + ".1");
    EventStore disk(path, 16, 4096);  // file_max_bytes=4KB 便于触发轮转
    const long a = disk.append("{\"task\":\"detect\",\"event\":\"rule_events\",\"rule_type\":\"intrusion\"}");
    EventStore::Review rv;
    rv.verdict = "false";
    rv.reason = "tree not person";
    rv.confidence = 0.9;
    rv.source = "llm";
    rv.ts_ms = 123456;
    CHECK(disk.attachReview(a, rv));
    CHECK(!disk.attachReview(999, rv));  // 事件不在内存：仅落盘，返回 false
    EventStore::Item got;
    CHECK(disk.get(a, &got) && got.has_review && got.review.verdict == "false");
    // 重启回放：新 store 从文件恢复事件与复核结论，id 续号不重发
    {
        EventStore disk2(path, 16, 0);
        CHECK(disk2.size() == 1);
        EventStore::Item replay;
        CHECK(disk2.get(a, &replay) && replay.has_review &&
              replay.review.source == "llm" && replay.review.reason == "tree not person");
        CHECK(disk2.append("{\"task\":\"y\"}") == a + 1);
    }
    // 超限轮转：events.jsonl → events.jsonl.1
    for (int i = 0; i < 300; ++i) {
        disk.append("{\"task\":\"bulk\",\"pad\":\"0123456789012345678901234567890123456789\"}");
    }
    CHECK(std::filesystem::exists(path + ".1"));
    std::filesystem::remove(path);
    std::filesystem::remove(path + ".1");
}

// M0 底座：两阶级联共享的仿射裁剪几何（正/逆系数往返精度 + 宽高比正形扩展）
struct TestAffine {
    float m[6];
    float im[6];
};
static void testCropAffine() {
    std::printf("[test] crop_utils affine\n");
    using crop_utils::buildSquareAffine;
    using crop_utils::mapPointBack;
    const auto nearEq = [](float a, float b) { return std::fabs(a - b) < 1e-3f; };

    // 瘦高人框（60x160）裁到 256x192（横宽模型）：h 撑到 w/aspect 正形
    TestAffine aff{};
    buildSquareAffine({100, 50, 160, 210}, 256, 192, 1.0f, &aff);
    const float aspect = 256.0f / 192.0f;
    const float w0 = 60.0f, h0 = 160.0f;              // w < h*aspect → 宽撑大到 h*aspect 正形
    const float exp_w = w0 > h0 * aspect ? w0 : h0 * aspect;
    const float exp_h = w0 > h0 * aspect ? w0 / aspect : h0;
    CHECK(nearEq(aff.m[0], 256.0f / exp_w));          // sx = model_w / 正形宽
    CHECK(nearEq(aff.m[4], 192.0f / exp_h));          // sy = model_h / 正形高
    CHECK(nearEq(aff.m[0], aff.m[4]));                // 正形 → x/y 等比
    // 框中心应映射到模型中心
    const float cx = (100 + 160) * 0.5f, cy = (50 + 210) * 0.5f;
    CHECK(nearEq(aff.m[0] * cx + aff.m[2], 128.0f));
    CHECK(nearEq(aff.m[4] * cy + aff.m[5], 96.0f));
    // 逆映射是正映射的逆：crop 中心 → 原图框中心
    float ox = 0.f, oy = 0.f;
    mapPointBack(aff, 128.0f, 96.0f, &ox, &oy);
    CHECK(nearEq(ox, cx) && nearEq(oy, cy));

    // 宽扁框（200x80）+ pad=1.25：正形按 w 主导扩展，逆映射仍闭合
    buildSquareAffine({0, 0, 200, 80}, 256, 192, 1.25f, &aff);
    const float wp = 200.0f * 1.25f, hp = 80.0f * 1.25f;
    const float exph = wp > hp * aspect ? wp / aspect : hp * aspect;
    CHECK(nearEq(aff.m[0], 256.0f / wp) && nearEq(aff.m[4], 192.0f / exph));
    mapPointBack(aff, 0.0f, 0.0f, &ox, &oy);
    // crop 左上角对应原图：x = cx - 正形宽/2，y = cy - 正形高/2
    CHECK(nearEq(ox, 100.0f - wp * 0.5f) && nearEq(oy, 40.0f - exph * 0.5f));

    // 极小框：尺寸钳到 >=1px，不产生除零/负缩放
    buildSquareAffine({10, 10, 10, 11}, 256, 192, 1.0f, &aff);
    CHECK(std::isfinite(aff.m[0]) && aff.m[0] > 0.0f && std::isfinite(aff.im[0]) && aff.im[0] > 0.0f);
    mapPointBack(aff, 128.0f, 96.0f, &ox, &oy);
    CHECK(nearEq(ox, 10.0f) && nearEq(oy, 10.5f));
}

// M13 动作识别：PoseSequenceBuffer 时序骨架窗口缓冲（入缓冲/攒满/缺帧补零/gap 作废/
// stale 老化/归一化数值/track_id 过滤/节流重置）
static void fillPose(pose_detect_result_list& poses, int idx, int track_id, int l, int t, int r,
                     int b, float kx, float ky, float conf) {
    pose_detect_result& pr = poses.results[idx];
    pr.box = {l, t, r, b};
    pr.box_conf = 0.9f;
    pr.cls_id = 0;
    pr.track_id = track_id;
    for (int k = 0; k < KEYPOINT_NUM; ++k) {
        pr.keypoints[k] = {kx, ky, conf};
    }
}

static void testPoseSequence() {
    std::printf("[test] PoseSequenceBuffer\n");
    const auto nearEq = [](float a, float b) { return std::fabs(a - b) < 1e-5f; };
    const int T = 3;
    const size_t V = KEYPOINT_NUM;
    const size_t plane = T * V;

    // track_id=0（未跟踪）不入缓冲
    PoseSequenceBuffer buf(T);
    CHECK(buf.windowT() == T && buf.numJoints() == static_cast<int>(V));
    pose_detect_result_list poses = {};
    fillPose(poses, 0, 0, 0, 0, 100, 100, 50.f, 50.f, 0.8f);
    poses.count = 1;
    buf.update(poses, 0);
    CHECK(buf.trackCount() == 0);

    // 连续帧攒窗：未满不 ready，攒满 ready
    fillPose(poses, 0, 7, 0, 0, 100, 100, 30.f, 70.f, 0.8f);
    buf.update(poses, 0);
    CHECK(buf.trackCount() == 1 && !buf.ready(7));
    buf.update(poses, 1);
    CHECK(!buf.ready(7));
    buf.update(poses, 2);
    CHECK(buf.ready(7));

    // 归一化数值：box(0,0,100,100) 中心(50,50) 尺度100；kpt(30,70)→(-0.2,0.2)
    PoseWindowTensor win;
    CHECK(buf.buildWindow(7, &win));
    CHECK(win.T == T && win.V == static_cast<int>(V) && win.C == 3);
    CHECK(win.rknn_dims.size() == 4 && win.rknn_dims[0] == 1 && win.rknn_dims[1] == 3 &&
          win.rknn_dims[2] == static_cast<uint32_t>(T) &&
          win.rknn_dims[3] == static_cast<uint32_t>(V));
    CHECK(win.data.size() == 3 * plane);
    CHECK(nearEq(win.data[0 * plane + 2 * V + 0], -0.2f));   // x 通道，最后一帧
    CHECK(nearEq(win.data[1 * plane + 2 * V + 0], 0.2f));    // y 通道
    CHECK(nearEq(win.data[2 * plane + 2 * V + 0], 0.8f));    // visibility = conf 原值
    CHECK(!buf.buildWindow(999, &win));                      // 不存在轨迹
    CHECK(!buf.buildWindow(7, nullptr));                     // 非法出参

    // 缺帧补零：f1 轨迹缺席（空帧）→ 零槽补位（visibility=0），f2 重现攒满
    PoseSequenceBuffer buf2(T);
    fillPose(poses, 0, 7, 0, 0, 100, 100, 30.f, 70.f, 0.8f);
    buf2.update(poses, 0);
    poses.count = 0;  // f1 空帧
    buf2.update(poses, 1);
    poses.count = 1;
    buf2.update(poses, 2);
    CHECK(buf2.ready(7));
    PoseWindowTensor win2;
    CHECK(buf2.buildWindow(7, &win2));
    const size_t zero_slot_off = 1 * V;  // t=1 是补零槽
    bool zero_ok = true;
    for (size_t c = 0; c < 3; ++c) {
        for (size_t v = 0; v < V; ++v) {
            if (std::fabs(win2.data[c * plane + zero_slot_off + v]) > 1e-6f) {
                zero_ok = false;
            }
        }
    }
    CHECK(zero_ok);
    CHECK(nearEq(win2.data[2 * plane + 2 * V + 0], 0.8f));  // t=2 真实帧恢复

    // gap >= window_t：旧窗作废清空重来（全零窗无动作证据）
    buf.update(poses, 100);  // buf 上次 f2，gap=97 ≥ 3
    CHECK(!buf.ready(7));
    CHECK(buf.buildWindow(7, &win) == false);

    // stale 老化：f100 后连续缺席帧（补零槽不续命），f104（距最后真实检测差 4 > stale=3）整条清除
    poses.count = 0;
    buf.update(poses, 101);
    buf.update(poses, 102);
    CHECK(buf.trackCount() == 1);
    buf.update(poses, 104);
    CHECK(buf.trackCount() == 0);
    poses.count = 1;

    // 窗口滚动：T=3 攒 5 帧 → 只留最后 3 帧（conf 0.3/0.4/0.5，f0/f1 被推出）
    PoseSequenceBuffer buf3(T);
    for (int f = 0; f < 5; ++f) {
        fillPose(poses, 0, 7, 0, 0, 100, 100, 30.f, 70.f, 0.1f + 0.1f * f);
        buf3.update(poses, f);
    }
    CHECK(buf3.ready(7));
    PoseWindowTensor win3;
    CHECK(buf3.buildWindow(7, &win3));
    CHECK(nearEq(win3.data[2 * plane + 0 * V], 0.3f));
    CHECK(nearEq(win3.data[2 * plane + 1 * V], 0.4f));
    CHECK(nearEq(win3.data[2 * plane + 2 * V], 0.5f));

    // 同帧同轨迹重复结果取第一条
    PoseSequenceBuffer buf4(1);
    pose_detect_result_list dup = {};
    fillPose(dup, 0, 9, 0, 0, 100, 100, 30.f, 70.f, 0.9f);
    fillPose(dup, 1, 9, 0, 0, 100, 100, 30.f, 70.f, 0.1f);
    dup.count = 2;
    buf4.update(dup, 0);
    CHECK(buf4.ready(9));
    PoseWindowTensor win4;
    CHECK(buf4.buildWindow(9, &win4));
    CHECK(nearEq(win4.data[2 * V + 0], 0.9f));  // conf 通道取第一条结果

    // 推理节流：resetTrack 清空重攒（每窗口仅推一次）
    PoseSequenceBuffer buf5(T);
    fillPose(poses, 0, 7, 0, 0, 100, 100, 30.f, 70.f, 0.8f);
    buf5.update(poses, 0);
    buf5.update(poses, 1);
    buf5.update(poses, 2);
    CHECK(buf5.ready(7));
    buf5.resetTrack(7);
    CHECK(!buf5.ready(7) && buf5.trackCount() == 0);

    // reset：全清
    buf5.update(poses, 0);
    buf5.update(poses, 1);
    fillPose(poses, 1, 8, 0, 0, 100, 100, 30.f, 70.f, 0.8f);
    poses.count = 2;
    buf5.update(poses, 2);
    CHECK(buf5.trackCount() == 2);
    buf5.reset();
    CHECK(buf5.trackCount() == 0);

    // image 归一化口径（mmaction2 PreNormalize2D）：200x100 帧，kpt(150,25)
    // → x_n=(150-100)/100=0.5, y_n=(25-50)/50=-0.5
    PoseSequenceBuffer bufi(1, KEYPOINT_NUM, -1, "image");
    fillPose(poses, 0, 7, 100, 40, 200, 60, 150.f, 25.f, 0.8f);
    poses.count = 1;
    bufi.update(poses, 0, 200, 100);
    CHECK(bufi.ready(7));
    PoseWindowTensor wini;
    CHECK(bufi.buildWindow(7, &wini));
    CHECK(nearEq(wini.data[0 * V + 0], 0.5f));
    CHECK(nearEq(wini.data[1 * V + 0], -0.5f));
    CHECK(nearEq(wini.data[2 * V + 0], 0.8f));
    // 帧尺寸未传时沿用最近一次有效值（流式多帧同一视频）
    bufi.update(poses, 1, 0, 0);
    bufi.update(poses, 2, 0, 0);
    CHECK(bufi.buildWindow(7, &wini));
    CHECK(nearEq(wini.data[0 * V + 0], 0.5f) && nearEq(wini.data[1 * V + 0], -0.5f));
}

static void testCtcDecode() {
    std::printf("[test] ctc decode\n");
    // 字典：类别 1='a', 2='b', 3='c'；类别 0=blank；类别 4（=C-1，>dict.size()）=空格
    const std::vector<std::string> dict = {"a", "b", "c"};
    const int C = 5;

    // 全 blank → 空文本、0 分
    {
        std::vector<float> logits;
        for (int t = 0; t < 4; ++t) {
            logits.insert(logits.end(), {5.0f, 0.1f, 0.1f, 0.1f, 0.1f});  // blank=idx0 最强
        }
        const CtcLine r = ctcGreedyDecode(logits.data(), 4, C, 0, dict);
        CHECK(r.text.empty() && r.score == 0.0f);
    }
    // "ab"：t0='a', t1=blank, t2='b', t3='b'（重复折叠）→ "ab"
    {
        const float rows[4][C] = {
            {0.1f, 2.0f, 0.1f, 0.1f, 0.1f},  // a
            {3.0f, 0.1f, 0.1f, 0.1f, 0.1f},  // blank
            {0.1f, 0.1f, 2.5f, 0.1f, 0.1f},  // b
            {0.1f, 0.1f, 2.5f, 0.1f, 0.1f},  // b（重复，折叠）
        };
        std::vector<float> logits;
        for (const auto& row : rows) {
            logits.insert(logits.end(), row, row + C);
        }
        const CtcLine r = ctcGreedyDecode(logits.data(), 4, C, 0, dict);
        CHECK(r.text == "ab");
        CHECK(r.score > 0.5f && r.score <= 1.0f);
        // 分数 = 两个采纳字符的 softmax 概率均值（a 与 b 的两步）≈ (p_a + p_b)/2
        const float p_a = ctcSoftmaxProb(logits.data(), C, 0, 1);
        const float p_b = ctcSoftmaxProb(logits.data(), C, 2, 2);
        CHECK(std::fabs(r.score - (p_a + p_b) / 2.0f) < 1e-4f);
    }
    // 同字符中间隔 blank 不折叠："aab" 语义 = a,a,b → "aab"
    {
        const float rows[5][C] = {
            {0.1f, 2.0f, 0.1f, 0.1f, 0.1f},  // a
            {0.1f, 2.0f, 0.1f, 0.1f, 0.1f},  // a（连续重复 → 折叠）
            {3.0f, 0.1f, 0.1f, 0.1f, 0.1f},  // blank（分隔）
            {0.1f, 2.0f, 0.1f, 0.1f, 0.1f},  // a（blank 后不折叠）
            {0.1f, 0.1f, 2.5f, 0.1f, 0.1f},  // b
        };
        std::vector<float> logits;
        for (const auto& row : rows) {
            logits.insert(logits.end(), row, row + C);
        }
        const CtcLine r = ctcGreedyDecode(logits.data(), 5, C, 0, dict);
        CHECK(r.text == "aab");
    }
    // 空格类别（C-1 越过字典）与非 blank 默认值
    {
        const float rows[2][C] = {
            {0.1f, 0.1f, 0.1f, 0.1f, 2.0f},  // 空格（idx 4 = C-1 > dict.size()）
            {0.1f, 2.0f, 0.1f, 0.1f, 0.1f},  // a
        };
        std::vector<float> logits;
        for (const auto& row : rows) {
            logits.insert(logits.end(), row, row + C);
        }
        const CtcLine r = ctcGreedyDecode(logits.data(), 2, C, 0, dict);
        CHECK(r.text == " a");
    }
    // 非法输入保护
    CHECK(ctcGreedyDecode(nullptr, 4, C, 0, dict).text.empty());
    std::vector<float> one(1, 0.0f);
    CHECK(ctcGreedyDecode(one.data(), 1, 1, 0, dict).text.empty());

    // logits_are_probs=true：分数取原值（不做二次 softmax）
    {
        const float rows[2][C] = {
            {0.05f, 0.90f, 0.02f, 0.02f, 0.01f},  // a（概率 0.90）
            {0.05f, 0.02f, 0.88f, 0.02f, 0.03f},  // b（概率 0.88）
        };
        std::vector<float> logits;
        for (const auto& row : rows) {
            logits.insert(logits.end(), row, row + C);
        }
        const CtcLine r = ctcGreedyDecode(logits.data(), 2, C, 0, dict, true);
        CHECK(r.text == "ab");
        CHECK(std::fabs(r.score - 0.89f) < 1e-4f);  // (0.90+0.88)/2

        // 同数据按 raw logits 解码（默认）：分数是二次 softmax，明显低于概率均值
        // （5 类 softmax(0.90) ≈ 0.37；类数越多塌得越狠——这正是 rec 必须走 prob 模式的原因）
        const CtcLine r2 = ctcGreedyDecode(logits.data(), 2, C, 0, dict, false);
        CHECK(r2.text == "ab");
        CHECK(r2.score > 0.2f && r2.score < 0.6f);
    }

    // M0 二级分类解码：argmax + softmax 分数（raw logits / 已 softmax 两模式）
    {
        const float logits[4] = {0.5f, 2.0f, 1.0f, -1.0f};
        const ClsTop1 t = clsTop1(logits, 4, false);
        CHECK(t.cls_id == 1);
        // softmax（max 基准）：1/[1+exp(-1.5)+exp(-1)+exp(-3)] = 1/1.6408 ≈ 0.6095
        CHECK(std::fabs(t.score - 0.6095f) < 2e-3f);
        const float probs[3] = {0.10f, 0.70f, 0.20f};
        const ClsTop1 p = clsTop1(probs, 3, true);
        CHECK(p.cls_id == 1 && std::fabs(p.score - 0.70f) < 1e-6f);
        CHECK(clsTop1(nullptr, 4, false).cls_id == -1);
    }

    // 布局识别（交付说明：rec=[1,40,6625] TC；LPRNet=[1,68,18] CT）
    const RecLayout rec = resolveRecLayout(40, 6625, 6623);
    CHECK(rec.time_steps == 40 && rec.num_classes == 6625 && !rec.channel_major);
    const RecLayout lpr = resolveRecLayout(68, 18, 68);
    CHECK(lpr.time_steps == 18 && lpr.num_classes == 68 && lpr.channel_major);
    // 字典信息缺失 → 默认 TC
    const RecLayout nodict = resolveRecLayout(40, 6625, 0);
    CHECK(nodict.time_steps == 40 && nodict.num_classes == 6625 && !nodict.channel_major);
    // dict_size+2（blank+空格在字典外）也落在类数区间
    const RecLayout rec2 = resolveRecLayout(25, 70, 68);
    CHECK(rec2.time_steps == 25 && rec2.num_classes == 70 && !rec2.channel_major);
}

// M8 RetinaFace 解码纯函数：先验框数量/锚点值/中心形式解码闭合/NMS
static void testRetinafaceDecode() {
    std::printf("[test] retinaface decode\n");
    const std::vector<float> priors = retinafacePriorBoxes(320);
    CHECK(priors.size() == 4200 * 4);
    // 首 anchor：网格(0,0) min_size 16 → c=(0.5*8/320)=0.0125, w=16/320=0.05
    CHECK(std::fabs(priors[0] - 0.0125f) < 1e-4f && std::fabs(priors[1] - 0.0125f) < 1e-4f &&
          std::fabs(priors[2] - 0.05f) < 1e-4f && std::fabs(priors[3] - 0.05f) < 1e-4f);
    // 末 anchor：末层(9,9) min_size 512 → c=0.95, w=1.6
    const float* last = &priors[4199 * 4];
    CHECK(std::fabs(last[0] - 0.95f) < 1e-3f && std::fabs(last[3] - 1.6f) < 1e-3f);
    // 第 40*40*2 个 = 第二层第一个：(0.5*16/320)=0.025, w=64/320=0.2
    const float* l2 = &priors[40 * 40 * 2 * 4];
    CHECK(std::fabs(l2[0] - 0.025f) < 1e-4f && std::fabs(l2[2] - 0.2f) < 1e-4f);

    // 解码：构造单 anchor 命中——priors[0] 中心零回归 → 框 = 中心±w/2（像素）
    std::vector<float> box(4200 * 4, 0.0f);
    std::vector<float> cls(4200 * 2, 0.0f);
    std::vector<float> landm(4200 * 10, 0.0f);
    cls[0 * 2 + 1] = 0.9f;   // anchor 0：face 0.9
    landm[0] = 0.0f;         // 零回归 → 点都在 anchor 中心
    cls[5 * 2 + 1] = 0.7f;   // anchor 5：face 0.7（与 0 位置不同，不互抑）
    cls[3 * 2 + 1] = 0.6f;   // anchor 3：与 anchor 0 同网格邻近（间距 0.0125*320=4px，重 IoU 应被抑制）
    RetinaFaceRawHeads heads{box.data(), cls.data(), landm.data(), 4200};
    auto items = retinafaceDecode(heads, 0.5f, 0.4f, 320);
    // anchor0 与 anchor3 框完全重叠（零回归同中心）→ NMS 抑制 anchor3
    CHECK(items.size() == 2);
    CHECK(std::fabs(items[0].score - 0.9f) < 1e-5f);
    // anchor0 框：cx=cy=0.0125*320=4px, w=h=0.05*320=16px → (4-8)=clamp 0 起
    CHECK(items[0].x1 <= 0.0f && std::fabs(items[0].x2 - 12.0f) < 1e-3f);
    // 零回归 landmark 全部落在 anchor 中心 (4,4)
    CHECK(std::fabs(items[0].landmarks[0] - 4.0f) < 1e-3f &&
          std::fabs(items[0].landmarks[1] - 4.0f) < 1e-3f);
    // 空头保护
    RetinaFaceRawHeads empty{};
    CHECK(retinafaceDecode(empty, 0.5f, 0.4f, 320).empty());
}

// G2 第二坡：输出链路纯逻辑（URL 分类/封装选择/SRT 规范化/rkmpp 开关/软编码参数推算）
static void testOutputTuning() {
    std::printf("[test] output tuning\n");
    CHECK(isNetworkUrl("rtmp://a/live") && isNetworkUrl("SRT://host:1000") &&
          isNetworkUrl("rtsps://x") && isNetworkUrl("udp://239.1.1.1:1234"));
    CHECK(!isNetworkUrl("/tmp/out.mp4") && !isNetworkUrl("http://x/y"));
    CHECK(isRtmpUrl("rtmps://x") && !isRtmpUrl("rtsp://x"));

    CHECK(std::string(getOutputFormatName("rtmp://a/b")) == "flv");
    CHECK(std::string(getOutputFormatName("udp://a")) == "mpegts");
    CHECK(std::string(getOutputFormatName("rtsp://a")) == "rtsp");
    CHECK(getOutputFormatName("/tmp/a.mp4") == nullptr);

    // SRT 规范化：直观形式 → MediaMTX streamid；带 query / 无路径 / 非 srt 原样
    CHECK(normalizeSrtPublishUrl("srt://1.2.3.4:9000/live/key") ==
          "srt://1.2.3.4:9000?mode=caller&streamid=#!::r=live/key,m=publish");
    CHECK(normalizeSrtPublishUrl("srt://h:1?x=y") == "srt://h:1?x=y");
    CHECK(normalizeSrtPublishUrl("srt://h:1") == "srt://h:1");
    CHECK(normalizeSrtPublishUrl("/tmp/a.mp4") == "/tmp/a.mp4");

    // rkmpp 开关优先级：DISABLE > FORCE > 网络 URL 默认
    setenv("RK_PIPE_DISABLE_RKMPP", "1", 1);
    setenv("RK_PIPE_FORCE_RKMPP", "1", 1);
    CHECK(!shouldUseRkMppForOutput("rtmp://x/y"));  // DISABLE 压过 FORCE
    unsetenv("RK_PIPE_DISABLE_RKMPP");
    CHECK(shouldUseRkMppForOutput("/tmp/a.mp4"));   // FORCE 强制本地也开
    unsetenv("RK_PIPE_FORCE_RKMPP");
    CHECK(!shouldUseRkMppForOutput("/tmp/a.mp4"));  // 本地默认关
    CHECK(shouldUseRkMppForOutput("rtmp://x/y"));   // 网络 URL 默认开

    // 画质档位别名归一
    CHECK(normalizeQualityMode("SPEED") == "speed");
    CHECK(normalizeQualityMode("low-latency") == "speed");
    CHECK(normalizeQualityMode("visually_lossless") == "fidelity");
    CHECK(normalizeQualityMode("nonsense") == "balanced");
    CHECK(normalizeQualityMode("") == "balanced");

    // 软编码参数推算：balanced 本地 1080p30 → bpp 0.18 推算
    EncodeTuningParams p;
    auto t = buildSoftwareEncodeTuning(p, false, 1920, 1080, 30.0);
    CHECK(t.bit_rate == 11197440);  // 1920*1080*30*0.18（未触钳位）
    CHECK(t.rc_buffer_size == t.bit_rate * 2);
    CHECK(t.gop_size == 60 && t.crf == 18 && t.preset == "fast" && !t.low_latency);
    // speed + 网络：低延时 + 档位钳位
    p.qualityMode = "speed";
    t = buildSoftwareEncodeTuning(p, true, 1920, 1080, 30.0);
    CHECK(t.low_latency && t.preset == "veryfast" && t.crf == 22 && t.gop_size == 30);
    CHECK(t.bit_rate >= 4000000 && t.bit_rate <= 12000000);
    CHECK(t.rc_buffer_size == t.bit_rate);  // low_latency 缓冲 1x
    // 显式覆盖：crf/preset/bitrate/lowLatency(medium→fast 联动)
    p = EncodeTuningParams{};
    p.crf = 12;
    p.preset = "medium";
    p.lowLatency = true;
    t = buildSoftwareEncodeTuning(p, false, 640, 480, 25.0);
    CHECK(t.crf == 12 && t.preset == "fast" && t.low_latency && t.gop_size == 25);
    p = EncodeTuningParams{};
    p.bitrateMbps = 10;
    t = buildSoftwareEncodeTuning(p, false, 640, 480, 25.0);
    CHECK(t.bit_rate == 10000000);
}

// tracker 测试依赖跟踪实现（非 CI 纯逻辑源集），CI 构建下整体剔除
#ifndef RK_PIPE_CI
static SimpleObjectTracker makeTestTracker(TrackAlgorithm mode = TrackAlgorithm::OcSort) {
    // 与 run_detect_web.yaml 相同的默认参数
    return SimpleObjectTracker(0.30f, 35, 4, 0.12f, 2.4f, 0.62f, 1, 0.5f, 0.25f, 0.5f, 0.3f, 256, mode);
}

static object_detect_result_list makeDets(const std::vector<std::pair<int, int>>& centers,
                                          float prop) {
    object_detect_result_list list = {};
    list.count = static_cast<int>(centers.size());
    for (int i = 0; i < list.count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        int x = centers[i].first;
        int y = centers[i].second;
        list.results[i].box.left = x;
        list.results[i].box.top = y;
        list.results[i].box.right = x + 30;
        list.results[i].box.bottom = y + 50;
        list.results[i].prop = prop;
        list.results[i].cls_id = 0;
    }
    return list;
}

static void testTrackerIds() {
    std::fprintf(stderr, "[test] SimpleObjectTracker ID 稳定性\n");
    SimpleObjectTracker tracker = makeTestTracker();
    std::unordered_map<int, int> counts;
    std::vector<int> stable_ids(3, -1);
    for (int f = 0; f < 200; ++f) {
        std::vector<std::pair<int, int>> centers;
        for (int o = 0; o < 3; ++o) {
            centers.push_back({20 + o * 100 + f * 2, 30 + o * 20});
        }
        std::vector<TrackedDetection> tracked = tracker.update(makeDets(centers, 0.9f), &counts);
        if (f < 10) {
            continue;
        }
        std::vector<int> ids;
        for (const auto& td : tracked) {
            if (td.is_confirmed && !td.is_predicted) {
                ids.push_back(td.track_id);
            }
        }
        CHECK(ids.size() == 3);
        if (ids.size() == 3) {
            CHECK(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2]);
            for (int o = 0; o < 3; ++o) {
                if (stable_ids[o] == -1) {
                    stable_ids[o] = ids[o];
                } else {
                    CHECK(stable_ids[o] == ids[o]);  // 同一目标 ID 全程稳定
                }
            }
        }
    }
    std::fprintf(stderr, "  IDs: %d/%d/%d\n", stable_ids[0], stable_ids[1], stable_ids[2]);
    CHECK(counts[0] == 3);  // 3 个独立目标各计数一次
}

static void testTrackerOcclusion() {
    std::fprintf(stderr, "[test] SimpleObjectTracker 遮挡恢复\n");
    SimpleObjectTracker tracker = makeTestTracker();
    std::unordered_map<int, int> counts;
    int seen_id = -1;
    for (int f = 0; f < 30; ++f) {
        std::vector<std::pair<int, int>> centers = {{50 + f * 2, 40}};
        std::vector<TrackedDetection> tracked = tracker.update(makeDets(centers, 0.9f), &counts);
        for (const auto& td : tracked) {
            if (td.is_confirmed && !td.is_predicted) {
                seen_id = td.track_id;
            }
        }
    }
    CHECK(seen_id > 0);
    // 遮挡 20 帧
    for (int f = 0; f < 20; ++f) {
        tracker.update(makeDets({}, 0.9f), &counts);
    }
    // 重新出现：位置按原速度(2px/帧)继续
    int recovered_id = -1;
    for (int f = 0; f < 30; ++f) {
        std::vector<std::pair<int, int>> centers = {{50 + (30 + f) * 2, 40}};
        std::vector<TrackedDetection> tracked = tracker.update(makeDets(centers, 0.9f), &counts);
        for (const auto& td : tracked) {
            if (td.is_confirmed && !td.is_predicted) {
                recovered_id = td.track_id;
            }
        }
    }
    CHECK(recovered_id == seen_id);  // 遮挡后恢复为原 ID
}

static void testTrackerPerf() {
    std::fprintf(stderr, "[test] SimpleObjectTracker 大量失配轨迹性能\n");
    SimpleObjectTracker tracker = makeTestTracker();
    std::unordered_map<int, int> counts;
    auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < 300; ++f) {
        std::vector<std::pair<int, int>> centers;
        centers.reserve(60);
        for (int o = 0; o < 60; ++o) {
            // 每帧跳到伪随机位置 → 永远匹配不上 → 轨迹持续堆积，考验轨迹上限与匈牙利复杂度
            centers.push_back({(o * 37 + f * 131) % 1280, (o * 53 + f * 67) % 720});
        }
        tracker.update(makeDets(centers, 0.8f), &counts);
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  300 帧 × 60 失配目标: %.1f ms 总耗时 (%.2f ms/帧)\n", ms, ms / 300.0);
    CHECK(ms < 3000.0);  // 平均 <10ms/帧，证明匈牙利未因轨迹累积而膨胀
}

static void testTrackerRelink() {
    std::fprintf(stderr, "[test] SimpleObjectTracker 距离重连（遮挡期位移较大）\n");
    SimpleObjectTracker tracker = makeTestTracker();
    std::unordered_map<int, int> counts;
    int expected_id = -1;
    // 15 帧静止目标，确认其 ID
    for (int f = 0; f < 15; ++f) {
        std::vector<TrackedDetection> tracked = tracker.update(makeDets({{100, 50}}, 0.9f), &counts);
        for (const auto& td : tracked) {
            if (td.is_confirmed && !td.is_predicted) {
                expected_id = td.track_id;
            }
        }
    }
    CHECK(expected_id > 0);
    // 遮挡 5 帧（期间目标实际移动到 x=140，位置跳变远超框宽）
    for (int f = 0; f < 5; ++f) {
        tracker.update(makeDets({}, 0.9f), &counts);
    }
    // 重新出现：从 x=140 以 8px/帧 继续运动，全程应保持原 ID
    bool failed = false;
    int recovered_id = -1;
    int x = 140;
    for (int f = 0; f < 30; ++f) {
        std::vector<TrackedDetection> tracked = tracker.update(makeDets({{x, 50}}, 0.9f), &counts);
        x += 8;
        for (const auto& td : tracked) {
            if (td.is_confirmed && !td.is_predicted) {
                if (recovered_id == -1) {
                    recovered_id = td.track_id;
                } else if (td.track_id != recovered_id) {
                    failed = true;
                }
            }
        }
    }
    CHECK(recovered_id == expected_id);  // 距离重连保住原 ID
    CHECK(!failed);                      // 后续连续运动不丢不换
}

static void testTrackerModes() {
    std::fprintf(stderr, "[test] SimpleObjectTracker 算法档位切换\n");
    struct { TrackAlgorithm mode; const char* name; } modes[] = {
        {TrackAlgorithm::Iou, "iou"},
        {TrackAlgorithm::Sort, "sort"},
        {TrackAlgorithm::ByteTrack, "bytetrack"},
        {TrackAlgorithm::OcSort, "ocsort"},
    };
    for (const auto& m : modes) {
        SimpleObjectTracker tracker = makeTestTracker(m.mode);
        std::unordered_map<int, int> counts;
        std::vector<int> stable_ids(3, -1);
        bool ok = true;
        for (int f = 0; f < 200 && ok; ++f) {
            std::vector<std::pair<int, int>> centers;
            for (int o = 0; o < 3; ++o) {
                centers.push_back({20 + o * 100 + f * 2, 30 + o * 20});
            }
            std::vector<TrackedDetection> tracked = tracker.update(makeDets(centers, 0.9f), &counts);
            if (f < 10) {
                continue;
            }
            std::vector<int> ids;
            for (const auto& td : tracked) {
                if (td.is_confirmed && !td.is_predicted) {
                    ids.push_back(td.track_id);
                }
            }
            if (ids.size() != 3) {
                ok = false;
                continue;
            }
            for (int o = 0; o < 3; ++o) {
                if (stable_ids[o] == -1) {
                    stable_ids[o] = ids[o];
                } else if (stable_ids[o] != ids[o]) {
                    ok = false;
                }
            }
        }
        std::fprintf(stderr, "  %s: %s\n", m.name, ok ? "ok" : "FAILED");
        CHECK(ok);
    }
}

// 合成检测流：目标平滑移动，prop 0.50~0.89（覆盖高分/低分），可选周期遮挡
static std::vector<object_detect_result_list> makeTrackerBenchStream(int frames,
                                                                     int objs,
                                                                     bool with_occlusion) {
    std::vector<object_detect_result_list> stream(frames);
    for (int f = 0; f < frames; ++f) {
        object_detect_result_list& list = stream[f];
        list.count = 0;
        for (int o = 0; o < objs; ++o) {
            if (with_occlusion && (f % 30) >= 20 && (o % 2) == 0) {
                continue;  // 一半目标周期性消失 10 帧
            }
            int x = 60 + ((o * 137 + f * 4) % 1160);
            int y = 60 + ((o * 71 + f * 3) % 600);
            object_detect_result& r = list.results[list.count++];
            r.box.left = x;
            r.box.top = y;
            r.box.right = x + 40;
            r.box.bottom = y + 70;
            r.prop = 0.50f + static_cast<float>((o * 13 + f * 7) % 40) / 100.0f;
            r.cls_id = 0;
        }
    }
    return stream;
}

static void testTrackerBench() {
    std::fprintf(stderr, "[test] SimpleObjectTracker 耗时基准\n");
    struct { TrackAlgorithm mode; const char* name; } modes[] = {
        {TrackAlgorithm::Iou, "iou"},
        {TrackAlgorithm::Sort, "sort"},
        {TrackAlgorithm::ByteTrack, "bytetrack"},
        {TrackAlgorithm::OcSort, "ocsort"},
    };
    const int kFrames = 3000;
    struct { int objs; bool occlusion; const char* tag; } cases[] = {
        {20, false, "20目标·普通"},
        {60, false, "60目标·普通"},
        {60, true, "60目标·周期遮挡"},
    };
    for (const auto& c : cases) {
        std::vector<object_detect_result_list> stream = makeTrackerBenchStream(kFrames, c.objs, c.occlusion);
        {
            volatile int sink = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (const auto& list : stream) {
                sink += list.count;
            }
            auto t1 = std::chrono::steady_clock::now();
            (void)sink;
            std::fprintf(stderr, "  [%s] 基线(空循环): %.2f us/帧\n", c.tag,
                         std::chrono::duration<double, std::micro>(t1 - t0).count() / kFrames);
        }
        for (const auto& m : modes) {
            SimpleObjectTracker tracker = makeTestTracker(m.mode);
            std::unordered_map<int, int> counts;
            auto t0 = std::chrono::steady_clock::now();
            for (const auto& list : stream) {
                tracker.update(list, &counts);
            }
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / kFrames;
            std::fprintf(stderr, "  [%s] %-10s: %.2f us/帧\n", c.tag, m.name, us);
        }
    }
}

static void testTrackerOtherTasks() {
    std::fprintf(stderr, "[test] SimpleObjectTracker pose/obb/seg 任务\n");

    // pose：3 个人匀速移动，ID 应稳定且互异
    SimpleObjectTracker pose_tracker = makeTestTracker();
    std::unordered_map<int, int> counts;
    std::vector<int> stable_pose(3, -1);
    bool pose_ok = true;
    for (int f = 0; f < 200 && pose_ok; ++f) {
        pose_detect_result_list poses = {};
        poses.count = 3;
        for (int o = 0; o < 3; ++o) {
            int x = 20 + o * 100 + f * 2;
            poses.results[o].box = {x, 30 + o * 20, x + 30, 80 + o * 20};
            poses.results[o].box_conf = 0.9f;
            poses.results[o].cls_id = 0;
        }
        trackPoseResults(pose_tracker, poses, &counts);
        if (f < 10) {
            continue;
        }
        for (int o = 0; o < 3; ++o) {
            if (stable_pose[o] == -1) {
                stable_pose[o] = poses.results[o].track_id;
            } else if (stable_pose[o] != poses.results[o].track_id) {
                pose_ok = false;
            }
        }
    }
    std::fprintf(stderr, "  pose ids: %d/%d/%d\n", stable_pose[0], stable_pose[1], stable_pose[2]);
    CHECK(pose_ok);

    // obb：3 个旋转目标匀速移动，ID 稳定
    SimpleObjectTracker obb_tracker = makeTestTracker();
    std::unordered_map<int, int> counts2;
    std::vector<int> stable_obb(3, -1);
    bool obb_ok = true;
    for (int f = 0; f < 200 && obb_ok; ++f) {
        obb_detect_result_list obbs = {};
        obbs.count = 3;
        for (int o = 0; o < 3; ++o) {
            int cx = 40 + o * 120 + f * 2;
            obbs.results[o].box.x = cx - 20;
            obbs.results[o].box.y = 40 + o * 30;
            obbs.results[o].box.w = 40;
            obbs.results[o].box.h = 20;
            obbs.results[o].box.angle = 0.3f;
            obbs.results[o].prop = 0.9f;
            obbs.results[o].cls_id = 0;
        }
        trackOBBResults(obb_tracker, obbs, &counts2);
        if (f < 10) {
            continue;
        }
        for (int o = 0; o < 3; ++o) {
            if (stable_obb[o] == -1) {
                stable_obb[o] = obbs.results[o].track_id;
            } else if (stable_obb[o] != obbs.results[o].track_id) {
                obb_ok = false;
            }
        }
    }
    std::fprintf(stderr, "  obb ids: %d/%d/%d\n", stable_obb[0], stable_obb[1], stable_obb[2]);
    CHECK(obb_ok);

    // seg：3 个实例匀速移动，ID 稳定
    SimpleObjectTracker seg_tracker = makeTestTracker();
    std::unordered_map<int, int> counts3;
    SegTaskResult segs;
    std::vector<int> stable_seg(3, -1);
    bool seg_ok = true;
    for (int f = 0; f < 200 && seg_ok; ++f) {
        segs.data.boxes.clear();
        segs.data.scores.clear();
        segs.data.class_ids.clear();
        for (int o = 0; o < 3; ++o) {
            int x = 30 + o * 110 + f * 2;
            segs.data.boxes.emplace_back(x, 40 + o * 25, 40, 60);
            segs.data.scores.push_back(0.9f);
            segs.data.class_ids.push_back(0);
        }
        trackSegResults(seg_tracker, segs, &counts3);
        if (f < 10) {
            continue;
        }
        for (int o = 0; o < 3; ++o) {
            if (stable_seg[o] == -1) {
                stable_seg[o] = segs.track_ids[o];
            } else if (stable_seg[o] != segs.track_ids[o]) {
                seg_ok = false;
            }
        }
    }
    std::fprintf(stderr, "  seg ids: %d/%d/%d\n", stable_seg[0], stable_seg[1], stable_seg[2]);
    CHECK(seg_ok);
}
#endif  // !RK_PIPE_CI


// ---- LLM 模块（纯逻辑，CI 可测）----------------------------------------------

#include "llm/llm_analyzer.h"
#include "llm/llm_config.h"
#include "llm/detection_json.h"
#include "core/task_result_json.h"

static void testLlmConfig() {
    std::printf("[test] LlmConfig::loadFromFile\n");
    if (!writeTestYaml(
            "%YAML:1.0\n"
            "---\n"
            "llm_enabled: 1\n"
            "llm_provider: \"cloud\"\n"
            "llm_min_conf: 0.4\n"
            "llm_interval_s: 2.0\n"
            "llm_queue_size: 8\n"
            "llm_results_path: \"/tmp/llm_results.jsonl\"\n"
            "llm_model: \"glm-4.5v\"\n"
            "llm_endpoint: \"http://example.invalid/v1/chat/completions\"\n"
            "llm_api_key_env: \"LLM_API_KEY\"\n"
            "llm_send_image: 1\n")) {
        CHECK(false);
        return;
    }
    LlmConfig cfg;
    CHECK(cfg.loadFromFile(kTestYaml));
    CHECK(cfg.enabled);
    CHECK(cfg.provider == "cloud");
    CHECK(cfg.model == "glm-4.5v");
    CHECK(cfg.min_conf > 0.39f && cfg.min_conf < 0.41f);
    CHECK(cfg.queue_size == 8);
    CHECK(cfg.send_image);
    // 非法值钳位
    CHECK(writeTestYaml("%YAML:1.0\n---\nllm_enabled: 1\nllm_queue_size: 0\nllm_interval_s: -1.0\n"));
    LlmConfig clamped;
    CHECK(clamped.loadFromFile(kTestYaml));
    CHECK(clamped.queue_size == 1);
    CHECK(clamped.interval_s == 0.0);
    // 默认发现：文件不存在 → enabled=false（静默禁用）
    setenv("RKPIPE_LLM_CONFIG", "/nonexistent/rk_pipe_llm.yaml", 1);
    const LlmConfig missing = LlmConfig::fromDefaultLocation();
    unsetenv("RKPIPE_LLM_CONFIG");
    CHECK(!missing.enabled);
}

static void testLlmAnalyzer() {
    std::printf("[test] LlmAnalyzer (mock provider)\n");

    // 1) 检测结果序列化（模块自带 detection_json）
    TaskResult result = DetectTaskResult{};
    DetectTaskResult* det = std::get_if<DetectTaskResult>(&result);
    det->data.count = 2;
    det->data.results[0].box = {1, 2, 11, 22};
    det->data.results[0].prop = 0.42f;  // borderline → 触发 min_conf 门控
    det->data.results[0].cls_id = 5;
    det->data.results[1].box = {5, 6, 15, 26};
    det->data.results[1].prop = 0.91f;
    det->data.results[1].cls_id = 0;
    const std::string result_json = llm::taskResultToJson(result, {});
    CHECK(result_json.find("\"type\":\"detect\"") != std::string::npos);
    CHECK(result_json.find("0.42") != std::string::npos);

    // 2) mock provider 端到端：提交 → worker 完成 → 落盘/缓存/对话
    const std::string results_path =
        "/tmp/rk_pipe_llm_test_results_" + std::to_string((int)getpid()) + ".jsonl";
    std::remove(results_path.c_str());
    CHECK(writeTestYaml(
        "%YAML:1.0\n"
        "---\n"
        "llm_enabled: 1\n"
        "llm_provider: \"mock\"\n"
        "llm_min_conf: 0.5\n"
        "llm_interval_s: 0.0\n"
        "llm_queue_size: 4\n"
        "llm_send_image: 1\n"
        "llm_results_path: \"" + results_path + "\"\n"));
    LlmConfig cfg;
    CHECK(cfg.loadFromFile(kTestYaml));

    PipelineFrame frame;
    frame.index = 7;
    frame.sourceName = "cam0";
    frame.hasResult = true;
    frame.matFrame = cv::Mat(32, 32, CV_8UC3, cv::Scalar(8, 16, 32));
    frame.result = result;
    det = std::get_if<DetectTaskResult>(&frame.result);  // 后续改写必须作用于 frame 内副本

    llm::LlmAnalyzer analyzer(cfg);
    analyzer.start();
    analyzer.observe(frame.result);
    analyzer.submit(frame);
    bool done = false;
    std::string status;
    for (int i = 0; i < 500; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        status = analyzer.statusJson();
        if (status.find("\"completed\":1") != std::string::npos) {
            done = true;
            break;
        }
    }
    CHECK(done);
    CHECK(status.find("\"submitted\":1") != std::string::npos);
    CHECK(status.find("real_defect") != std::string::npos);

    // 3) 全部高置信度 → 门控拦截，submitted 不增长
    det->data.results[0].prop = 0.95f;
    analyzer.submit(frame);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    status = analyzer.statusJson();
    CHECK(status.find("\"submitted\":1") != std::string::npos);

    // 4) chat 同步问答
    const llm::LlmResponse answer = analyzer.chat("现在哪类缺陷最多？", "{\"car\":2}");
    CHECK(answer.ok);
    CHECK(answer.text.find("模拟回答") != std::string::npos);
    analyzer.stop();

    // 5) 结论 JSONL 落盘
    std::ifstream in(results_path);
    CHECK(in.is_open());
    std::string line;
    int lines = 0;
    bool has_verdict = false;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            ++lines;
            has_verdict = has_verdict || line.find("real_defect") != std::string::npos;
        }
    }
    CHECK(lines == 1);
    CHECK(has_verdict);
    in.close();
    std::remove(results_path.c_str());
}



// ---------------- daemon supervision（daemon_supervision.cc 纯逻辑，CI 可跑） ----------------

static void testDaemonUrlDecode() {
    CHECK(urlDecode("") == "");
    CHECK(urlDecode("abc") == "abc");
    CHECK(urlDecode("a%20b+c") == "a b c");          // %20 与 + 都解为空格
    CHECK(urlDecode("%2Ftmp%2Fa.mp4") == "/tmp/a.mp4");
    CHECK(urlDecode("100%") == "100%");              // 非法转义原样保留
    CHECK(urlDecode("%zz") == "%zz");
}

static void testDaemonParseQuery() {
    const auto kv = parseQuery("task=detect&input=%2Ftmp%2Fa.mp4&threads=2");
    CHECK(kv.at("task") == "detect");
    CHECK(kv.at("input") == "/tmp/a.mp4");
    CHECK(kv.at("threads") == "2");
    CHECK(kv.find("flag") == kv.end());              // 无 '=' 的段忽略
    CHECK(parseQuery("").empty());
}

static void testDaemonYamlScalar() {
    CHECK(yamlScalar("3") == "3");                   // 数字不加引号
    CHECK(yamlScalar("0.5") == "0.5");
    CHECK(yamlScalar("-2") == "-2");
    CHECK(yamlScalar("abc") == "\"abc\"");
    CHECK(yamlScalar("") == "\"\"");
    CHECK(yamlScalar("/tmp/x.yaml") == "\"/tmp/x.yaml\"");
    CHECK(yamlQuoted(std::string("a\"b\\c")) == "\"a\\\"b\\\\c\"");  // 转义双引号与反斜杠
}

static void testDaemonPassthroughKeys() {
    CHECK(isPassthroughKey("event_line"));
    CHECK(isPassthroughKey("event_dwell_seconds"));
    CHECK(isPassthroughKey("alert_webhook_url"));
    CHECK(isPassthroughKey("extra_yaml"));
    CHECK(!isPassthroughKey("model"));
    CHECK(!isPassthroughKey("events"));              // 前缀须为 event_/alert_
    CHECK(!isPassthroughKey("event-line"));          // 含非法字符
    CHECK(isEventRuleKey("type"));
    CHECK(isEventRuleKey("llm_hint"));
    CHECK(isEventRuleKey("fall_seconds"));
    CHECK(!isEventRuleKey("foo"));
}

static void testDaemonSpecCompare() {
    TaskSpec a, b;
    a.input = "/a.mp4";
    a.model = "/m.rknn";
    a.threads = "2";
    b = a;
    CHECK(sameSpec(a, b));
    CHECK(taskSpecSameIdentity(a, b));
    b.threads = "4";
    CHECK(!sameSpec(a, b));
    CHECK(taskSpecSameIdentity(a, b));               // 身份不含 threads
    b.passthrough = {{"event_line", "1"}};
    CHECK(!sameSpec(a, b));

    TaskSpec p, q;
    p.id = "x";
    q.id = "x";
    CHECK(taskSpecSameIdentity(p, q));
    q.id = "y";
    CHECK(!taskSpecSameIdentity(p, q));
    q.id.clear();
    CHECK(!taskSpecSameIdentity(p, q));              // 一方有 id 另一方空 → 非同身份
    TaskSpec r, s;                                   // 双方无 id → 按 input/task/model
    r.input = "/a.mp4";
    s.input = "/a.mp4";
    CHECK(taskSpecSameIdentity(r, s));
    s.task = "pose";
    CHECK(!taskSpecSameIdentity(r, s));
}

static void testDaemonNextTaskId() {
    const int base = nextTaskId("");
    CHECK(nextTaskId("") == base + 1);
    CHECK(nextTaskId("50") == 50);                   // 数字 spec_id 直接采用并推高计数器
    CHECK(nextTaskId("") >= 51);
    CHECK(nextTaskId("abc") >= 52);                  // 非数字 → 计数器自增
}

static void testDaemonTaskSpecJson() {
    TaskSpec s;
    s.task = "detect";
    s.input = "/x.mp4";
    s.model = "/m.rknn";
    s.threads = "2";
    bool ok = false;
    const mini_json::JsonValue v = mini_json::JsonParser(taskSpecJson(s)).parse(ok);
    CHECK(ok);
    CHECK(v.type == mini_json::JsonValue::Type::Object);
    CHECK(v.getString("task") == "detect");
    CHECK(v.getString("input") == "/x.mp4");
    CHECK(v.getString("threads") == "2");
    CHECK(v.getString("model") == "/m.rknn");
}

static void testDaemonWriteYaml() {
    Task t;
    t.id = 1;                                        // id==1 时额外写出 extra_ports
    t.port = 8123;
    t.spec.task = "detect";
    t.spec.input = "/tmp/in.mp4";
    t.spec.model = "/tmp/m.rknn";
    t.spec.threads = "2";
    t.spec.tracking = "1";
    t.yaml_path = "/tmp/rk_test_task.yaml";
    CHECK(writeYaml(t, {8124}));
    std::ifstream in(t.yaml_path);
    CHECK(in.is_open());
    std::ostringstream buf;
    buf << in.rdbuf();
    const std::string yaml = buf.str();
    CHECK(yaml.find("model_path: \"/tmp/m.rknn\"") != std::string::npos);
    CHECK(yaml.find("input_path: \"/tmp/in.mp4\"") != std::string::npos);
    CHECK(yaml.find("task: \"detect\"") != std::string::npos);
    CHECK(yaml.find("thread_count: 2") != std::string::npos);
    CHECK(yaml.find("web_preview_port: 8123") != std::string::npos);
    CHECK(yaml.find("web_preview_extra_ports: \"8124\"") != std::string::npos);
    CHECK(yaml.find("enable_tracking: 1") != std::string::npos);
    std::remove(t.yaml_path.c_str());
}

static void testDaemonLoadConfig() {
    const char* path = "/tmp/rk_test_streams.json";
    {
        std::ofstream f(path);
        f << R"({
            "streams": [{
                "id": "42", "task": "detect", "input": "/x.mp4", "model": "/m.rknn",
                "threads": 2, "port": 8091, "tracking": 1,
                "events": {"line": "0,0,10,10", "line_cross": 1},
                "alert": {"enabled": 1, "webhook_url": "http://h/hook"},
                "rules": [{"id": "r1", "type": "line_cross", "line": "1,1,2,2"}]
            }]
        })";
    }
    const auto specs = loadConfig(path);
    CHECK(specs.size() == 1);
    const TaskSpec& s = specs[0];
    CHECK(s.id == "42");
    CHECK(s.task == "detect");
    CHECK(s.input == "/x.mp4");
    CHECK(s.model == "/m.rknn");
    CHECK(s.threads == "2");                          // 数字键统一字符串化
    CHECK(s.port == "8091");
    CHECK(s.tracking == "1");
    bool has_line = false, has_cross = false, has_hook = false;
    for (const auto& kv : s.passthrough) {            // 嵌套 events{}/alert{} 展开为扁平透传键
        if (kv.first == "event_line") has_line = true;
        if (kv.first == "event_line_cross") has_cross = true;
        if (kv.first == "alert_webhook_url") has_hook = true;
    }
    CHECK(has_line);
    CHECK(has_cross);
    CHECK(has_hook);
    CHECK(s.rules.size() == 1);                       // 命名规则透传
    bool rule_id = false, rule_type = false;
    for (const auto& kv : s.rules[0]) {
        if (kv.first == "id" && kv.second == "r1") rule_id = true;
        if (kv.first == "type" && kv.second == "line_cross") rule_type = true;
    }
    CHECK(rule_id);
    CHECK(rule_type);
    std::remove(path);
    CHECK(loadConfig("/tmp/rk_nonexistent_streams.json").empty());  // 缺文件 → 空表不崩溃
}

static void testDaemonMemBudget() {
    TaskSpec s;
    s.task = "detect";
    s.threads = "2";
    s.input = "/x.mp4";
    CHECK(estimateTaskMemMb(s) > 0);
    int est = -1;
    CHECK(resourceBudgetOk(s, &est));
    CHECK(est >= 0);
}


static void testHttpRequest() {
    CHECK(httpRequestMethod("POST /x HTTP/1.1\r\n") == "POST");
    CHECK(httpRequestMethod("nonsense") == "GET");
    CHECK(httpRequestPath("GET /stream.mjpg?t=123 HTTP/1.1\r\nHost:h\r\n\r\n") == "/stream.mjpg");
    CHECK(httpRequestPath("GET / HTTP/1.1\r\n\r\n") == "/");
    CHECK(httpRequestPath("garbage") == "/");
    CHECK(httpRequestBody("POST /a HTTP/1.1\r\n\r\n{\"k\":1}") == "{\"k\":1}");
    CHECK(httpRequestBody("GET / HTTP/1.1\r\n\r\n").empty());
    CHECK(httpContentLength("POST /a HTTP/1.1\r\nContent-Length: 7\r\n\r\n1234567") == 7);
    CHECK(httpContentLength("POST /a HTTP/1.1\r\ncontent-length:12\r\n\r\n") == 12);  // 头名大小写不敏感
    CHECK(httpContentLength("GET / HTTP/1.1\r\n\r\n") == -1);
    const auto kv = parseQuery("a=1&b=%2Fx");
    CHECK(kv.at("a") == "1");
    CHECK(kv.at("b") == "/x");
}


static void testDaemonPrometheus() {
    const std::string board =
        R"({"cpu_usage":12.5,"temp_c":41.5,"npu_load":[10,20,30],"mem_used_pct":63.2,)"
        R"("mem_avail_mb":2797,"disk_free_mb":3803})";
    std::vector<PrometheusTaskSample> tasks(2);
    tasks[0].id = 1;
    tasks[0].task = "detect";
    tasks[0].state = "running";
    tasks[0].uptime_s = 65;
    tasks[0].child_ok = true;
    tasks[0].publish_fps = 60.5;
    tasks[0].published_frames = 523;
    tasks[0].dropped_frames = 2;
    tasks[0].input_reconnects = 1;
    tasks[1].id = 2;
    tasks[1].task = "pose";
    tasks[1].state = "stopped";
    const std::string text = formatPrometheusMetrics(board, tasks);
    CHECK(text.find("rkpipe_board_cpu_percent 12.5\n") != std::string::npos);
    CHECK(text.find("rkpipe_board_temp_c 41.5\n") != std::string::npos);
    CHECK(text.find("rkpipe_board_npu_load_percent{core=\"2\"} 30\n") != std::string::npos);
    CHECK(text.find("rkpipe_board_mem_used_percent 63.2\n") != std::string::npos);
    CHECK(text.find("rkpipe_task_info{task_id=\"1\",task=\"detect\",state=\"running\"} 1") !=
          std::string::npos);
    CHECK(text.find("rkpipe_task_publish_fps{task_id=\"1\"} 60.5\n") != std::string::npos);
    CHECK(text.find("rkpipe_task_published_frames_total{task_id=\"1\"} 523\n") != std::string::npos);
    CHECK(text.find("rkpipe_task_input_reconnect_count{task_id=\"1\"} 1\n") != std::string::npos);
    CHECK(text.find("rkpipe_task_uptime_seconds{task_id=\"2\"}") == std::string::npos);  // child 不可得不输出
    CHECK(text.find("rkpipe_task_info{task_id=\"2\",task=\"pose\",state=\"stopped\"} 1") !=
          std::string::npos);
    // 坏 JSON → 默认 0 值，不崩溃
    const std::string bad = formatPrometheusMetrics("not-json", {});
    CHECK(bad.find("rkpipe_board_cpu_percent 0\n") != std::string::npos);
}


static void testHttpPost() {
    // 本地起一个一次性 listener：收 POST 断言 body，回 200
    const int lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(lfd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    CHECK(::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    CHECK(::listen(lfd, 1) == 0);
    socklen_t len = sizeof(addr);
    ::getsockname(lfd, reinterpret_cast<sockaddr*>(&addr), &len);
    const int port = ntohs(addr.sin_port);

    std::string got_request;
    std::thread server([&lfd, &got_request]() {
        const int c = ::accept(lfd, nullptr, nullptr);
        if (c < 0) {
            return;
        }
        char buf[4096];
        ssize_t n;
        while ((n = ::recv(c, buf, sizeof(buf), 0)) > 0) {
            got_request.append(buf, static_cast<std::size_t>(n));
            if (got_request.find("\r\n\r\n") != std::string::npos &&
                got_request.size() >= got_request.find("\r\n\r\n") + 4 + 9) {
                break;  // 头 + 9 字节 body（{"k":1234}）到齐
            }
        }
        const char* resp = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        ::send(c, resp, std::strlen(resp), 0);
        ::close(c);
    });

    long code = 0;
    const bool ok = httpPost("127.0.0.1", port, "/internal/event", "{\"k\":1234}",
                             "application/json", 2000, &code);
    server.join();
    ::close(lfd);
    CHECK(ok);
    CHECK(code == 200);
    CHECK(got_request.find("POST /internal/event HTTP/1.1") == 0);
    CHECK(got_request.find("Content-Type: application/json") != std::string::npos);
    CHECK(got_request.find("{\"k\":1234}") != std::string::npos);

    // 拒绝连接 → false
    CHECK(!httpPost("127.0.0.1", 1, "/x", "{}", "application/json", 500));
    // 非法参数 → false
    CHECK(!httpPost("", 80, "/x", "{}", "application/json", 500));

    // httpPostUrl(http://) 形式
    const int lfd2 = ::socket(AF_INET, SOCK_STREAM, 0);
    CHECK(lfd2 >= 0);
    sockaddr_in a2{};
    a2.sin_family = AF_INET;
    a2.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a2.sin_port = 0;
    ::bind(lfd2, reinterpret_cast<sockaddr*>(&a2), sizeof(a2));
    ::listen(lfd2, 1);
    socklen_t l2 = sizeof(a2);
    ::getsockname(lfd2, reinterpret_cast<sockaddr*>(&a2), &l2);
    const int port2 = ntohs(a2.sin_port);
    std::thread responder([&lfd2]() {
        const int c = ::accept(lfd2, nullptr, nullptr);
        if (c < 0) {
            return;
        }
        char buf[2048];
        ::recv(c, buf, sizeof(buf), 0);
        const char* resp = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        ::send(c, resp, std::strlen(resp), 0);
        ::close(c);
    });
    long url_code = 0;
    CHECK(httpPostUrl("http://127.0.0.1:" + std::to_string(port2) + "/hook", "{}",
                      "application/json", 2000, &url_code));
    CHECK(url_code == 200);
    responder.join();
    ::close(lfd2);
}


static void testFlywheel() {
    EventStore::Review ok_review;
    ok_review.verdict = "true";
    ok_review.source = "llm";
    ok_review.ts_ms = 1690000000000;
    const std::string body =
        R"({"task":"detect","event":"rule_events","rule_type":"line_cross","detail":"穿越","snapshot":"/tmp/snap.jpg"})";
    // true 结论不归档
    CHECK(!buildFlywheelRecord(7, body, ok_review).ok);
    // false 结论归档
    EventStore::Review bad = ok_review;
    bad.verdict = "false";
    const FlywheelRecord rec = buildFlywheelRecord(7, body, bad);
    CHECK(rec.ok);
    CHECK(rec.rel_dir == "detect/line_cross");
    CHECK(rec.snapshot_src == "/tmp/snap.jpg");
    CHECK(rec.filename.rfind("7_", 0) == 0);
    bool pok = false;
    const mini_json::JsonValue m = mini_json::JsonParser(rec.manifest_json).parse(pok);
    CHECK(pok);
    CHECK(m.type == mini_json::JsonValue::Type::Object);
    CHECK(m.getNumber("id", -1) == 7);
    CHECK(m.getString("task") == "detect");
    CHECK(m.getString("type") == "line_cross");
    CHECK(m.getString("source") == "llm");
    CHECK(m.getString("image").find("detect/line_cross/7_") == 0);
    // 无快照不归档
    const std::string body_nosnap = R"({"task":"detect","event":"count"})";
    CHECK(!buildFlywheelRecord(8, body_nosnap, bad).ok);
    // 非法 body 不归档
    CHECK(!buildFlywheelRecord(9, "not-json", bad).ok);
}


static void testThermalGovernor() {
    ThermalGovernor::Params p;  // high=85 resume=80 step=5 min_ratio=0.25
    ThermalGovernor g(p);
    // 常温：满速
    CHECK(g.update(45.0, 60.0) == 60.0);
    CHECK(g.level() == 0);
    // 过热：85C → level1 = 0.6
    CHECK(g.update(85.0, 60.0) == 36.0);
    CHECK(g.level() == 1);
    // 迟滞区：保持 level1
    CHECK(g.update(82.0, 60.0) == 36.0);
    // 越高越深：95C → 1+(95-85)/5=3 → 封顶 level3 = 0.25
    CHECK(g.update(95.0, 60.0) == 15.0);
    CHECK(g.level() == 3);
    // 读取失败（负温度）：保持现状
    CHECK(g.update(-1.0, 60.0) == 15.0);
    // 低于 resume：逐级恢复（90→82 保持? 82>=80 迟滞保持; <80 才恢复）
    CHECK(g.update(79.9, 60.0) == 24.0);   // level2
    CHECK(g.update(79.9, 60.0) == 36.0);   // level1
    CHECK(g.update(79.9, 60.0) == 60.0);   // level0
    // base_fps<=0：返回 0
    CHECK(g.update(95.0, 0.0) == 0.0);
}


static void testMiniJsonDeepNesting() {
    // fuzz 发现面：递归下降解析器深嵌套必须判非法而非栈溢出（kMaxDepth=64）
    const std::string deep(100000, '[');
    bool ok = true;
    mini_json::JsonParser(deep).parse(ok);
    CHECK(!ok);
    const std::string deep_obj(100000, '{');
    ok = true;
    mini_json::JsonParser(deep_obj).parse(ok);
    CHECK(!ok);
    // 上限内的嵌套正常解析
    const std::string nest = std::string(50, '[') + std::string(50, ']');
    ok = false;
    mini_json::JsonParser(nest).parse(ok);
    CHECK(ok);
}


static void testLlmCircuitBreaker() {
    // provider=cloud + 不可达端点：connect 快速失败（CI 下 NO_CLOUD 桩同样恒失败）
    // 连续 5 次失败应熔断——后续 job 直接记失败（不再发起请求），且无崩溃
    LlmConfig cfg;
    cfg.enabled = true;
    cfg.provider = "cloud";
    cfg.model = "cb-test";
    cfg.endpoint = "http://127.0.0.1:1/v1/chat/completions";
    cfg.timeout_s = 2.0;
    cfg.interval_s = 0.0;   // 不限速，连续提交
    cfg.min_conf = 0.1f;
    cfg.max_results_keep = 50;

    llm::LlmAnalyzer analyzer(cfg);
    analyzer.start();

    TaskResult result = DetectTaskResult{};
    DetectTaskResult* det = std::get_if<DetectTaskResult>(&result);
    det->data.count = 1;
    det->data.results[0].box = {1, 2, 11, 22};
    det->data.results[0].prop = 0.05f;  // 低于 min_conf(0.1) → 通过提交门控
    det->data.results[0].cls_id = 0;

    PipelineFrame frame;
    frame.hasResult = true;
    frame.result = result;

    for (int i = 0; i < 8; ++i) {
        analyzer.submit(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    // 等待熔断生效：连续 5 次失败后 skip 的 job 会带 "circuit open" 错误入 recent
    bool enough = false;
    std::string status;
    for (int i = 0; i < 250 && !enough; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        status = analyzer.statusJson();
        if (status.find("circuit open") != std::string::npos) {
            enough = true;
        }
    }
    CHECK(enough);
    std::fprintf(stderr, "[cb-test] status tail: %s\n", status.substr(status.size() > 300 ? status.size() - 300 : 0).c_str());
    analyzer.stop();
    std::printf("[test] LLM 熔断行为正常（连续失败进入跳过）\n");
}


static void testDaemonAllocPort() {
    // 清场（其他测试可能残留任务）
    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        g_tasks.clear();
    }
    g_opt.base_port = 18090;
    g_opt.api_port = 18099;
    CHECK(allocPort() == 18090);
    // 占用 18090-18099 后，18099 是 api_port 也必须跳过
    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        for (int p = 18090; p <= 18098; ++p) {
            auto t = std::make_unique<Task>();
            t->port = p;
            g_tasks.push_back(std::move(t));
        }
    }
    CHECK(allocPort() == 18100);  // 跳过被占的 90-98 与 api_port 99
    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        g_tasks.clear();
    }
}


static void testEventStatsPersistence() {
    const std::string path = "/tmp/rk_test_event_stats.json";
    std::remove(path.c_str());

    AppConfig cfg;
    cfg.event_line = "100,100,900,900";
    cfg.event_line_cross = 1;
    cfg.event_stats_path = path;

    // 引擎 1：构造→恢复（首跑无文件）→析构
    {
        EventEngine engine(cfg);
        CHECK(!engine.stats().empty());
    }

    // 模拟已有计数的文件（手工构造，验证恢复路径）
    {
        std::ofstream out(path);
        out << R"({"saved_ts":123,"rules":[{"id":"line_cross","type":"line_cross",)"
            << R"("cross_ab":123,"cross_ba":45}]})";
    }
    EventEngine engine2(cfg);
    const auto stats = engine2.stats();
    CHECK(!stats.empty());
    CHECK(stats[0].cross_ab == 123);
    CHECK(stats[0].cross_ba == 45);

    // 坏文件不崩溃
    {
        std::ofstream out(path);
        out << "garbage{{{";
    }
    EventEngine engine3(cfg);
    CHECK(!engine3.stats().empty());
    std::remove(path.c_str());
}


// ---- 0.3.0 二级任务 JSONL 载荷回归：composite_cls / face / action ----
// 覆盖两套序列化器：buildFrameResultJson（JSONL 写入端）与 llm::taskResultToJson（LLM 复查链路）。
// 背景：0.3.0 前这两处缺二级任务分支，composite_cls 等全落 type:"none" 兜底（板上实测踩坑）。
static void testResultJsonStage2() {
    std::printf("[test] ResultJson stage2 (composite_cls/face/action)\n");

    PipelineFrame frame;
    frame.index = 7;
    frame.sourceName = "stage2";
    frame.bufferFrame.width = 1920;
    frame.bufferFrame.height = 1080;
    frame.hasResult = true;

    // 1) composite_cls：一级检测框 + 二级 top-1（sub 字段）
    {
        CompositeClsTaskResult cc;
        cc.data.count = 1;
        cc.data.results[0].box = {10, 20, 110, 120};
        cc.data.results[0].prop = 0.778f;
        cc.data.results[0].cls_id = 2;
        cc.cls_ids = {656};
        cc.cls_scores = {0.4006f};
        cc.cls_labels = {"minivan 40%"};
        frame.result = cc;
        const std::string json = buildFrameResultJson(frame, nullptr);
        CHECK(json.find("\"type\":\"composite_cls\"") != std::string::npos);
        CHECK(json.find("[10,20,100,100]") != std::string::npos);
        CHECK(json.find("\"sub\":{\"cls\":656") != std::string::npos);
        CHECK(json.find("\"label\":\"minivan 40%\"") != std::string::npos);
        const std::string ljson = llm::taskResultToJson(frame.result, {});
        CHECK(ljson.find("\"type\":\"composite_cls\"") != std::string::npos);
        CHECK(ljson.find("\"sub\":{\"cls_id\":656") != std::string::npos);
    }
    // 1b) composite 无检出 → 空数组、无 sub 键
    {
        CompositeClsTaskResult cc;
        frame.result = cc;
        const std::string json = buildFrameResultJson(frame, nullptr);
        CHECK(json.find("\"type\":\"composite_cls\"") != std::string::npos);
        CHECK(json.find("\"dets\":[]") != std::string::npos);
        CHECK(json.find("\"sub\"") == std::string::npos);
    }
    // 2) face：框 + 5 点 landmark
    {
        FaceTaskResult fr;
        FaceItem f;
        f.box = {0, 0, 100, 100};
        f.score = 0.9f;
        f.landmarks[0] = {10.0f, 20.0f};
        fr.faces.push_back(f);
        frame.result = fr;
        const std::string json = buildFrameResultJson(frame, nullptr);
        CHECK(json.find("\"type\":\"face\"") != std::string::npos);
        CHECK(json.find("\"kpts\":[[10,20]") != std::string::npos);
        const std::string ljson = llm::taskResultToJson(frame.result, {});
        CHECK(ljson.find("\"type\":\"face\"") != std::string::npos);
        CHECK(ljson.find("\"landmarks\":[[10.0,20.0]") != std::string::npos);
    }
    // 3) action：pose 主结果原样保留 + 本帧新鲜动作项
    {
        ActionTaskResult act;
        act.data.count = 1;
        act.data.results[0].box = {5, 5, 50, 150};
        act.data.results[0].box_conf = 0.8f;
        act.data.results[0].cls_id = 0;
        act.data.results[0].track_id = 3;
        act.actions.push_back({3, 12, 0.77f});
        frame.result = act;
        const std::string json = buildFrameResultJson(frame, nullptr);
        CHECK(json.find("\"type\":\"action\"") != std::string::npos);
        CHECK(json.find("\"actions\":[{\"track_id\":3,\"action\":12,\"score\":0.77}]") != std::string::npos);
        const std::string ljson = llm::taskResultToJson(frame.result, {});
        CHECK(ljson.find("\"type\":\"action\"") != std::string::npos);
        CHECK(ljson.find("\"keypoints\"") != std::string::npos);
        CHECK(ljson.find("\"actions\":[{\"action_id\":12") != std::string::npos);
    }
    // 3b) NaN/Inf 防护：病态分数钳为 0，不得产出非法 JSON 数字
    {
        ActionTaskResult act;
        act.actions.push_back({1, 2, std::nanf("")});
        frame.result = act;
        const std::string json = buildFrameResultJson(frame, nullptr);
        CHECK(json.find("nan") == std::string::npos);
        CHECK(json.find("NaN") == std::string::npos);
    }
}

int main() {
    testConfigLoad();
    testConfigValidate();
    testCommandLine();
#ifndef RK_PIPE_CI
    testDetectorFactory();
#endif
    testPostprocessMath();
    testDepthDistance();
    testEventEngine();
    testEventEngineMultiRule();
    testEventEngineFall();
    testEventEngineSpeedRef();
    testOutputTuning();
#ifndef RK_PIPE_CI
#endif
    testMiniJson();
    testEventStore();
    testDaemonUrlDecode();
    testDaemonParseQuery();
    testDaemonYamlScalar();
    testDaemonPassthroughKeys();
    testDaemonSpecCompare();
    testDaemonNextTaskId();
    testDaemonTaskSpecJson();
    testDaemonWriteYaml();
    testDaemonLoadConfig();
    testDaemonMemBudget();
    testHttpRequest();
    testDaemonPrometheus();
    testHttpPost();
    testFlywheel();
    testThermalGovernor();
    testMiniJsonDeepNesting();
    testLlmCircuitBreaker();
    testDaemonAllocPort();
    testEventStatsPersistence();
    testCropAffine();
    testCtcDecode();
    testPoseSequence();
    testRetinafaceDecode();
    testDetect3DDecode();
    testPerformanceAverage();
    testDetectionFilter();
    testTurbojpeg();
    testLlmConfig();
    testLlmAnalyzer();
    testResultJsonStage2();
#ifndef RK_PIPE_CI
    testTrackerIds();
    testTrackerOcclusion();
    testTrackerRelink();
    testTrackerModes();
    testTrackerOtherTasks();
    testTrackerPerf();
    testTrackerBench();
#endif

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
