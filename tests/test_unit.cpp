// rk_pipe 最小单元测试集（无硬件依赖，仅覆盖配置解析 / Detector 工厂路由 / 后处理纯函数）
// 构建与运行：cmake --build build --target rk_pipe_unit_tests && ./build/rk_pipe_unit_tests
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "config/app_config.h"
#include "core/detection_filter.h"
#include "core/event_engine.h"
#include "core/performance.h"
#include "postprocess/detect3d_decode.h"
#include "postprocess/postprocess_common.h"
#include "postprocess/retinaface_decode.h"
#include "detection/simple_object_tracker.h"
#include "detection/tracking_runtime.h"
#include "utils/depth_distance.h"
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
            "event_absence_seconds: 45\n")) {
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
    // 未指定 task 且文件名无 yolov5 → 默认 YOLOv8
    {
        auto d = Detector::createForModel("/tmp/yolo11n.rknn", "");
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
    // 目标：3D 底面中心 (0,0,10)，alpha=0 → theta=atan2(10,0)=π/2 → ry=π/2
    Detect3DItem w{};
    w.center_u = 50.f; w.center_v = 50.f; w.depth_m = 10.f;
    w.cos_alpha = 1.f; w.sin_alpha = 0.f;
    w.h3 = 1.5f; w.w3 = 2.0f; w.l3 = 4.0f;
    const float p2[12] = {100.f, 0, 50.f, 0, 0, 100.f, 50.f, 0, 0, 0, 1.f, 0};
    float corners[8][2];
    CHECK(computeDetect3DCorners2D(w, p2, corners));
    // 中心语义（y 角点 ±h/2）；ry=π/2 → c≈0,s≈1：X3=zc, Z3=-xc+10
    // 角点 0（xc=+2, zc=+1, 底面 y=+0.75）：X3=1, Z3=8 → py=(100×0.75+400)/8=59.375
    CHECK(std::fabs(corners[0][0] - 62.5f) < 1e-3f && std::fabs(corners[0][1] - 59.375f) < 1e-3f);
    // 角点 2（xc=-2, zc=-1, 底面）：X3=-1, Z3=12 → px=(-100+600)/12≈41.667, py=(75+600)/12=56.25
    CHECK(std::fabs(corners[2][0] - 41.6667f) < 1e-2f && std::fabs(corners[2][1] - 56.25f) < 1e-3f);
    // 角点 4（xc=+2, zc=+1, 顶面 y=-0.75）：X3=1, Z3=8 → py=(100×(-0.75)+400)/8=40.625
    CHECK(std::fabs(corners[4][0] - 62.5f) < 1e-3f && std::fabs(corners[4][1] - 40.625f) < 1e-3f);
    // 非法输入：深度非正 → false
    Detect3DItem bad = w;
    bad.depth_m = 0.0f;
    CHECK(!computeDetect3DCorners2D(bad, p2, corners));
}

static void testRetinaFaceDecode() {
    std::printf("[test] retinaface priorbox + decode\n");
    // ---- PriorBox:与 zoo examples/RetinaFace/cpp/rknn_box_priors.h 逐项对照 ----
    const std::vector<float> priors = retinafacePriorBoxes(320);
    // 320×320:40²×2 + 20²×2 + 10²×2 = 4200
    CHECK(priors.size() == 4200u * 4u);
    auto anchor = [&priors](int idx, int k) { return priors[static_cast<size_t>(idx) * 4 + k]; };
    // 首 3 锚点:cx=cy=4/320,size=16/320;j 列先于 i 行变化;min_size 16→32
    CHECK(std::fabs(anchor(0, 0) - 0.0125f) < 1e-6f && std::fabs(anchor(0, 1) - 0.0125f) < 1e-6f);
    CHECK(std::fabs(anchor(0, 2) - 0.05f) < 1e-6f && std::fabs(anchor(0, 3) - 0.05f) < 1e-6f);
    CHECK(std::fabs(anchor(1, 2) - 0.1f) < 1e-6f);
    CHECK(std::fabs(anchor(2, 0) - 0.0375f) < 1e-6f && std::fabs(anchor(2, 1) - 0.0125f) < 1e-6f);
    // 层边界:锚点 3199 = stride8 末位（zoo 第 3200 项）;3200 = stride16 首位（64/320=0.2）
    CHECK(std::fabs(anchor(3199, 0) - 0.9875f) < 1e-6f && std::fabs(anchor(3199, 2) - 0.1f) < 1e-6f);
    CHECK(std::fabs(anchor(3200, 0) - 0.025f) < 1e-6f && std::fabs(anchor(3200, 2) - 0.2f) < 1e-6f);
    // 末锚点 = stride32 (9,9) 的 512 档:cx=cy=0.95,size=1.6（zoo 第 4200 项）
    CHECK(std::fabs(anchor(4199, 0) - 0.95f) < 1e-6f && std::fabs(anchor(4199, 2) - 1.6f) < 1e-6f);
    // 中心全部落在 (0,1);尺寸上界 512/320=1.6
    bool centers_ok = true, sizes_ok = true;
    for (size_t a = 0; a < priors.size(); a += 4) {
        if (priors[a] <= 0.f || priors[a] >= 1.f || priors[a + 1] <= 0.f || priors[a + 1] >= 1.f) {
            centers_ok = false;
        }
        if (priors[a + 2] <= 0.f || priors[a + 2] > 1.6f + 1e-6f) {
            sizes_ok = false;
        }
    }
    CHECK(centers_ok);
    CHECK(sizes_ok);
    // 640×640:80²×2 + 40²×2 + 20²×2 = 16800;首锚点 16/640=0.025,中心 4/640=0.00625
    const std::vector<float> priors640 = retinafacePriorBoxes(640);
    CHECK(priors640.size() == 16800u * 4u);
    CHECK(std::fabs(priors640[0] - 0.00625f) < 1e-6f && std::fabs(priors640[2] - 0.025f) < 1e-6f);

    // ---- 解码:合成三头输出 ----
    const int n = 4200;
    std::vector<float> box(n * 4, 0.f), cls(n * 2, 0.f), landm(n * 10, 0.f);
    RetinaFaceRawHeads heads;
    heads.box = box.data();
    heads.cls = cls.data();
    heads.landm = landm.data();
    heads.num_anchors = n;

    // 零回归闭合:offsets=0 → 框还原为先验框矩形（模型坐标系像素）,landmark=先验中心
    cls[3200 * 2 + 1] = 0.9f;  // stride16 首锚点:中心 (8,8),w=h=64 → [-24,-24,40,40]
    std::vector<RetinaFaceItem> items = retinafaceDecode(heads, 0.5f, 0.4f);
    CHECK(items.size() == 1);
    if (items.size() == 1) {
        CHECK(std::fabs(items[0].x1 - (-24.f)) < 1e-2f && std::fabs(items[0].y1 - (-24.f)) < 1e-2f);
        CHECK(std::fabs(items[0].x2 - 40.f) < 1e-2f && std::fabs(items[0].y2 - 40.f) < 1e-2f);
        CHECK(std::fabs(items[0].score - 0.9f) < 1e-6f);
        bool lm_ok = true;
        for (int k = 0; k < 5; ++k) {
            if (std::fabs(items[0].landmarks[k * 2] - 8.f) > 1e-2f ||
                std::fabs(items[0].landmarks[k * 2 + 1] - 8.f) > 1e-2f) {
                lm_ok = false;
            }
        }
        CHECK(lm_ok);
    }
    cls[3200 * 2 + 1] = 0.f;

    // 方差数学（variances [0.1,0.2],landm 0.1）:锚点 0,loc=[10,-5,2,0]
    // cx=0.0125+0.1×0.05×10=0.0625→20px;w=0.05×e^0.4→23.87px;h=0.05→16px
    box[0] = 10.f; box[1] = -5.f; box[2] = 2.f; box[3] = 0.f;
    landm[0] = 5.f; landm[1] = 5.f;  // 点 0:0.0125+0.1×0.05×5=0.0375→12px
    cls[1] = 0.8f;
    items = retinafaceDecode(heads, 0.5f, 0.4f);
    CHECK(items.size() == 1);
    if (items.size() == 1) {
        CHECK(std::fabs(items[0].x1 - (20.f - 23.8692f / 2)) < 1e-2f);
        CHECK(std::fabs(items[0].y1 - (-12.f)) < 1e-2f && std::fabs(items[0].y2 - 4.f) < 1e-2f);
        CHECK(std::fabs(items[0].x2 - (20.f + 23.8692f / 2)) < 1e-2f);
        CHECK(std::fabs(items[0].landmarks[0] - 12.f) < 1e-2f &&
              std::fabs(items[0].landmarks[1] - 12.f) < 1e-2f);
    }
    box[0] = box[1] = box[2] = box[3] = 0.f;
    landm[0] = landm[1] = 0.f;
    cls[1] = 0.f;

    // cls 取第 1 列（图内已 Softmax,列 0 是背景分,不得混入）
    cls[0] = 0.99f;   // 背景列高分不得触发检出
    items = retinafaceDecode(heads, 0.5f, 0.4f);
    CHECK(items.empty());
    cls[0] = 0.f;
    cls[2 + 1] = 0.7f;  // 锚点 1:人脸列 0.7 → 检出且 score 原样透传
    items = retinafaceDecode(heads, 0.5f, 0.4f);
    CHECK(items.size() == 1 && std::fabs(items[0].score - 0.7f) < 1e-6f);
    cls[3] = 0.f;

    // conf 阈值过滤:0.9/0.4/0.6 @ conf=0.5 → 只留 2 个（均写 cls 列 1 = 2*anchor+1）
    cls[2 * 10 + 1] = 0.9f;
    cls[2 * 100 + 1] = 0.4f;
    cls[2 * 200 + 1] = 0.6f;
    items = retinafaceDecode(heads, 0.5f, 0.99f);
    CHECK(items.size() == 2);
    // 分数降序 + max_before_nms 截断（锚点 4199 = stride32 末位,分高且远离前两者）
    cls[2 * 4199 + 1] = 0.85f;
    items = retinafaceDecode(heads, 0.5f, 0.99f, 320, 2);
    CHECK(items.size() == 2);
    CHECK(std::fabs(items[0].score - 0.9f) < 1e-6f && std::fabs(items[1].score - 0.85f) < 1e-6f);
    cls[2 * 10 + 1] = cls[2 * 100 + 1] = cls[2 * 200 + 1] = cls[2 * 4199 + 1] = 0.f;

    // NMS:锚点 0/1 同中心 16px/32px 方框,IoU=0.25 → 阈 0.2 抑制、0.3 双留
    cls[1] = 0.9f;
    cls[3] = 0.9f;
    items = retinafaceDecode(heads, 0.5f, 0.2f);
    CHECK(items.size() == 1);
    items = retinafaceDecode(heads, 0.5f, 0.3f);
    CHECK(items.size() == 2);
    cls[1] = cls[3] = 0.f;

    // max_results 截断:3 个两两 IoU<0.4 的高分框,上限 2 → 只出前 2
    // 锚点 0(16px@(4,4)) vs 1600(stride16,64px@(8,8)):IoU=0.0625;4198(stride32,256px@(304,304)) 远离两者
    cls[2 * 0 + 1] = 0.9f;
    cls[2 * 1600 + 1] = 0.8f;
    cls[2 * 4198 + 1] = 0.75f;
    items = retinafaceDecode(heads, 0.5f, 0.4f, 320, 256, 2);
    CHECK(items.size() == 2 && std::fabs(items[0].score - 0.9f) < 1e-6f);
    CHECK(std::fabs(items[1].score - 0.8f) < 1e-6f);
    cls[1] = cls[2 * 1600 + 1] = cls[2 * 4198 + 1] = 0.f;

    // 病态输入防护
    RetinaFaceRawHeads bad{};
    CHECK(retinafaceDecode(bad, 0.5f, 0.4f).empty());  // 空指针
    heads.num_anchors = 0;
    CHECK(retinafaceDecode(heads, 0.5f, 0.4f).empty());
    heads.num_anchors = 4201;  // 超出先验框数量 → 拒绝解码防越界
    CHECK(retinafaceDecode(heads, 0.5f, 0.4f).empty());
    heads.num_anchors = n;
    box[2] = 1e9f;  // exp 溢出 → 非有限框被丢弃
    cls[1] = 0.9f;
    CHECK(retinafaceDecode(heads, 0.5f, 0.4f).empty());
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
    testDetect3DDecode();
    testRetinaFaceDecode();
    testPerformanceAverage();
    testDetectionFilter();
    testTurbojpeg();
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
