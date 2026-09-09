// rk_pipe 最小单元测试集（无硬件依赖，仅覆盖配置解析 / Detector 工厂路由 / 后处理纯函数）
// 构建与运行：cmake --build build --target rk_pipe_unit_tests && ./build/rk_pipe_unit_tests
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>  // getpid: 测试临时文件按进程区分,支持 ctest 并行

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "config/app_config.h"
#include "core/detection_filter.h"
#include "core/event_engine.h"
#include "core/performance.h"
#include "core/task_result_json.h"
#include "io/result_sink.h"
#include "postprocess/postprocess_common.h"
#include "detection/simple_object_tracker.h"
#include "detection/tracking_runtime.h"
#ifndef RK_PIPE_CI
#include <thread>

#include "detection/detector.h"
#include "io/web_preview_server.h"
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

static const char* kTestYamlTemplate = "/tmp/rk_pipe_unit_test_%d_cfg.yaml";

static std::string g_test_yaml_path;

static const char* testYamlPath() {
    if (g_test_yaml_path.empty()) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), kTestYamlTemplate, (int)getpid());
        g_test_yaml_path = buf;
    }
    return g_test_yaml_path.c_str();
}

static bool writeTestYaml(const std::string& content) {
    std::ofstream out(testYamlPath());
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
    CHECK(cfg.loadFromFile(testYamlPath()));
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

// ---- 命名多规则(RK_PIPE_EVENT_RULES,8 类规则+方向/ref_point/多实例) -------------

static void testEventRules() {
    std::printf("[test] EventEngine named rules\n");
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
    const char* kEnv = "RK_PIPE_EVENT_RULES";

    const std::string path =
        std::string("/tmp/rk_pipe_unit_") + std::to_string(getpid()) + "_rules.yaml";
    {
        std::ofstream out(path, std::ios::binary);
        out << "%YAML:1.0\n---\nevent_rules:\n"
            "  - {id: \"gate\", type: \"line_cross\", line: \"0,100,200,100\", classes: \"0\","
            " direction: \"A2B\"}\n"
            "  - {id: \"gate2\", type: \"line_cross\", line: \"0,100,200,100\","
            " ref_point: \"bottom\"}\n"
            "  - {id: \"yard\", type: \"intrusion\", region: \"0,0,100,0,100,100,0,100\","
            " classes: \"0\"}\n"
            "  - {id: \"crowd1\", type: \"crowd\", region: \"0,0,100,0,100,100,0,100\","
            " min_count: 3}\n"
            "  - {id: \"fall1\", type: \"fall\", fall_aspect: 1.2, fall_seconds: 2.0,"
            " classes: \"0\"}\n"
            "  - {id: \"spd1\", type: \"speed\", speed_limit: 500, speed_seconds: 1.0,"
            " llm_hint: \"check speed limit\"}\n"
            "  - {id: \"left1\", type: \"abandoned\", region: \"0,0,100,0,100,100,0,100\","
            " seconds: 60}\n"
            "  - {id: \"bad\", type: \"ghost\"}\n";
    }

    setenv(kEnv, path.c_str(), 1);
    AppConfig cfg;  // 扁平键全部关闭:命名模式仍须激活
    EventEngine eng(cfg);
    CHECK(eng.active());
    CHECK(!eng.lineCrossEnabled() && !eng.intrusionEnabled());  // 扁平内联开关保持关闭

    // 1) 绊线方向过滤(gate, direction=A2B):上方(B 侧)下行 = A->B,上报且带 [gate] 前缀;
    //    回上行 = B->A,仅计数不上报(逆行过滤)
    auto evs = eng.update({mk(1, 0, 90, 40, 110, 60)}, t0);  // 线上方(B 侧)建立轨迹
    evs = eng.update({mk(1, 0, 90, 140, 110, 160)}, t0 + ms(33));
    // 同帧 gate2(方向不限)可能随 track1 的 bottom 跨线额外上报,故按内容断言
    bool gate_ab = false;
    for (const auto& e : evs) {
        if (e.type == "line_cross" && e.detail == "[gate] A->B") {
            gate_ab = true;
        }
    }
    CHECK(gate_ab);
    evs = eng.update({mk(1, 0, 90, 40, 110, 60)}, t0 + ms(66));
    {
        bool gate_back = false;
        for (const auto& e : evs) {
            if (e.type == "line_cross" && e.detail.find("[gate]") == 0) {
                gate_back = true;  // B->A 被 direction=A2B 过滤,不得上报
            }
        }
        CHECK(!gate_back);  // 计数仍累计(下方 stats 断言 1/1)
    }

    // 2) ref_point=bottom(gate2, 类别不限):track2 用 cls 2 避开 gate 的类别过滤。
    //    高框 center 跨线而 bottom 未跨 → gate2 不触发;bottom 随后跨线 → 触发
    evs = eng.update({mk(2, 2, 90, 95, 110, 195)}, t0 + ms(500));
    evs = eng.update({mk(2, 2, 90, 5, 110, 105)}, t0 + ms(533));
    CHECK(evs.empty());  // bottom 195→105 未跨线(center 跨线不作数)
    evs = eng.update({mk(2, 2, 90, -5, 110, 95)}, t0 + ms(566));  // bottom 105→95 跨线
    CHECK(evs.size() == 1 && evs[0].type == "line_cross" &&
          evs[0].detail.find("[gate2]") == 0);

    // 3) 入侵+类别过滤(yard, classes=0):cls 2 进入不触发;cls 0 进入触发
    evs = eng.update({mk(3, 2, 40, 40, 60, 60)}, t0 + ms(600));
    CHECK(evs.empty());
    evs = eng.update({mk(3, 0, 190, 40, 210, 60)}, t0 + ms(700));   // 区域外,建立轨迹
    evs = eng.update({mk(3, 0, 40, 40, 60, 60)}, t0 + ms(733));     // 进入区域
    bool yard_hit = false;
    for (const auto& e : evs) {
        if (e.type == "intrusion" && e.detail == "[yard] ") {
            yard_hit = true;
        }
    }
    CHECK(yard_hit);

    // 4) 聚集(crowd1, min_count=3):3 人进入触发一次,回落重新武装后再触发
    evs = eng.update({mk(11, 0, 10, 10, 30, 30), mk(12, 0, 40, 10, 60, 30),
                      mk(13, 0, 70, 10, 90, 30)}, t0 + ms(800));
    bool crowd_hit = false;
    for (const auto& e : evs) {
        if (e.type == "crowd" && e.detail == "[crowd1] 3人") {
            crowd_hit = true;
        }
    }
    CHECK(crowd_hit);
    evs = eng.update({mk(11, 0, 10, 10, 30, 30), mk(12, 0, 40, 10, 60, 30)}, t0 + ms(833));
    CHECK(evs.empty());
    evs = eng.update({mk(11, 0, 10, 10, 30, 30), mk(12, 0, 40, 10, 60, 30),
                      mk(13, 0, 70, 10, 90, 30)}, t0 + ms(866));
    CHECK(evs.size() == 1 && evs[0].type == "crowd");  // 迟滞重武装后再触发

    // 5) 跌倒(fall1, w/h>=1.2 持续 2s):躺倒→触发;起身复位;再躺倒再触发
    evs = eng.update({mk(20, 0, 200, 200, 250, 240)}, t0);  // 站立 w/h=1.25? 50/40=1.25 躺倒起步
    evs = eng.update({mk(20, 0, 200, 200, 250, 240)}, t0 + ms(1000));
    CHECK(evs.empty());  // 躺倒 1s,不足 2s
    evs = eng.update({mk(20, 0, 200, 200, 250, 240)}, t0 + ms(2200));
    bool fall_hit = false;
    for (const auto& e : evs) {
        if (e.type == "fall" && e.detail.find("[fall1]") == 0) {
            fall_hit = true;
        }
    }
    CHECK(fall_hit);
    evs = eng.update({mk(20, 0, 200, 150, 240, 260)}, t0 + ms(2400));  // 起身 w/h=0.28
    CHECK(evs.empty());
    evs = eng.update({mk(20, 0, 200, 200, 250, 240)}, t0 + ms(3000));  // 重新躺倒(重新计时)
    evs = eng.update({mk(20, 0, 200, 200, 250, 240)}, t0 + ms(4200));
    evs = eng.update({mk(20, 0, 200, 200, 250, 240)}, t0 + ms(5300));
    CHECK(evs.size() == 1 && evs[0].type == "fall");  // 复位后可重复触发

    // 6) 测速(spd1, EMA>=500px/s 持续 1s):100px/100ms=1000px/s → 触发
    evs = eng.update({mk(30, 0, 10, 10, 30, 30)}, t0);
    bool speed_hit = false;
    for (int step = 1; step <= 12; ++step) {
        const int x = 10 + step * 100;
        evs = eng.update({mk(30, 0, x, 10, x + 20, 30)}, t0 + ms(step * 100));
        for (const auto& e : evs) {
            if (e.type == "speed" && e.detail.find("[spd1]") == 0 &&
                e.detail.find("px/s") != std::string::npos) {
                speed_hit = true;  // 触发发生在持续超限 1s 的中间帧,需循环内累计
            }
        }
    }
    CHECK(speed_hit);
    CHECK(eng.ruleHint("spd1") == "check speed limit");
    CHECK(eng.ruleHint("nope").empty());

    // 7) 遗留物(left1, seconds=60):静止 60s 触发一次;周期喂帧防轨迹老化(5s)
    bool abandoned_hit = false;
    for (int s = 1; s <= 61; ++s) {
        evs = eng.update({mk(40, 0, 40, 40, 60, 60)}, t0 + ms(s * 1000));
        for (const auto& e : evs) {
            if (e.type == "abandoned" && e.detail.find("[left1]") == 0) {
                abandoned_hit = true;
            }
        }
    }
    CHECK(abandoned_hit);

    // 8) stats/takeStats:快照含各规则 id/type,穿越计数与瞬时值透出
    auto st = eng.stats();
    CHECK(st.size() >= 7);  // bad(ghost) 已跳过
    size_t gate_idx = st.size();
    size_t crowd_idx = st.size();
    for (size_t i = 0; i < st.size(); ++i) {
        if (st[i].id == "gate") {
            gate_idx = i;
        }
        if (st[i].id == "crowd1") {
            crowd_idx = i;
        }
    }
    CHECK(gate_idx < st.size() && st[gate_idx].type == "line_cross");
    // gate 计数:track1 双向各一次;track2(cls2)被 gate 类别过滤不计
    CHECK(st[gate_idx].cross_ab == 1 && st[gate_idx].cross_ba == 1);
    // crowd_now 瞬时值:遗留物用例的 track40 停在 crowd 区域内 → 当前 1 人
    CHECK(crowd_idx < st.size() && st[crowd_idx].crowd_now == 1);
    st = eng.takeStats();
    // takeStats 语义:返回快照携带清零前累计值;引擎内计数已清零
    CHECK(st[gate_idx].cross_ab == 1 && st[gate_idx].cross_ba == 1);
    eng.update({}, t0 + ms(9000));  // 喂一帧刷新缓存后再验证清零
    CHECK(eng.stats()[gate_idx].cross_ab == 0 && eng.stats()[gate_idx].cross_ba == 0);

    // 9) 轨迹老化:停喂 6s(>5s)后轨迹状态被清除,重新出现视为新轨迹
    evs = eng.update({mk(3, 0, 190, 40, 210, 60)}, t0 + ms(10000));
    evs = eng.update({mk(3, 0, 40, 40, 60, 60)}, t0 + ms(10100));
    CHECK(evs.size() == 1 && evs[0].type == "intrusion");

    unsetenv(kEnv);
    std::remove(path.c_str());
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

// ---- 逐帧结果 JSON 序列化(schema v1,docs/event_payload.md) --------------------

static void testResultJson() {
    std::printf("[test] task_result_json schema v1\n");
    deinit_post_process();  // 归一化 labels 全局状态,独立/全量运行口径一致

    PipelineFrame frame;
    frame.index = 42;
    frame.sourceName = "/tmp/rkpipe_test.mp4";
    frame.matFrame = cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(0));

    // 信封 + 无结果 → type none
    {
        const std::string json = buildFrameResultJson(frame);
        CHECK(json.find("\"schema_version\":1") != std::string::npos);
        CHECK(json.find("\"frame\":42") != std::string::npos);
        CHECK(json.find("\"width\":1920") != std::string::npos);
        CHECK(json.find("\"height\":1080") != std::string::npos);
        CHECK(json.find("\"source\":\"/tmp/rkpipe_test.mp4\"") != std::string::npos);
        CHECK(json.find("\"type\":\"none\"") != std::string::npos);
    }

    frame.hasResult = true;

    // detect:未加载 labels → "null";缺省不带 track_id
    {
        DetectTaskResult det;
        det.data.count = 2;
        det.data.results[0].box = {10, 20, 110, 220};
        det.data.results[0].prop = 0.871f;
        det.data.results[0].cls_id = 0;
        det.data.results[1].box = {300, 400, 340, 440};
        det.data.results[1].prop = 0.5f;
        det.data.results[1].cls_id = 15;
        frame.result = det;
        const std::string json = buildFrameResultJson(frame);
        CHECK(json.find("\"type\":\"detect\"") != std::string::npos);
        CHECK(json.find("\"bbox\":[10,20,100,200]") != std::string::npos);
        CHECK(json.find("\"bbox\":[300,400,40,40]") != std::string::npos);
        CHECK(json.find("\"score\":0.871") != std::string::npos);
        CHECK(json.find("\"cls\":15") != std::string::npos);
        CHECK(json.find("\"label\":\"null\"") != std::string::npos);
        CHECK(json.find("track_id") == std::string::npos);
    }

    // detect + 事件缝预留的 track ids 参数
    {
        const std::vector<int> ids = {7, 0};
        const std::string json = buildFrameResultJson(frame, &ids);
        CHECK(json.find("\"track_id\":7") != std::string::npos);
        CHECK(json.find("\"track_id\":0") != std::string::npos);
    }

    // label 解析(临时 labels 文件,cls 0 → "person")
    {
        const std::string labels_path =
            std::string("/tmp/rk_pipe_unit_") + std::to_string(getpid()) + "_labels.txt";
        {
            std::ofstream out(labels_path);
            out << "person\ncar\n";
        }
        CHECK(init_post_process(labels_path.c_str()) == 0);
        const std::string json = buildFrameResultJson(frame);
        CHECK(json.find("\"label\":\"person\"") != std::string::npos);
        deinit_post_process();
        std::remove(labels_path.c_str());
    }

    // pose:17 关键点 + track_id
    {
        PoseTaskResult pose;
        pose.data.count = 1;
        pose_detect_result& p = pose.data.results[0];
        p.box = {0, 0, 50, 100};
        p.box_conf = 0.9f;
        p.cls_id = 0;
        p.track_id = 3;
        for (int k = 0; k < KEYPOINT_NUM; ++k) {
            p.keypoints[k] = {10.0f * k, 20.0f + k, k == 0 ? 0.95f : 0.0f};
        }
        frame.result = pose;
        const std::string json = buildFrameResultJson(frame);
        CHECK(json.find("\"type\":\"pose\"") != std::string::npos);
        CHECK(json.find("\"track_id\":3") != std::string::npos);
        CHECK(json.find("\"kpts\":[[0,20,0.95]") != std::string::npos);
        // 17 组关键点 → kpts 数组内应有 16 个 "],[" 分隔
        const size_t kpts_pos = json.find("\"kpts\":[");
        CHECK(kpts_pos != std::string::npos);
        int kpt_count = 0;
        for (size_t pos = kpts_pos; (pos = json.find("],[", pos)) != std::string::npos; ++kpt_count) {
            ++pos;
        }
        CHECK(kpt_count == KEYPOINT_NUM - 1);
    }

    // obb:外接框左上 + 弧度角 + track_id
    {
        OBBTaskResult obb;
        obb.data.count = 1;
        obb_detect_result& b = obb.data.results[0];
        b.box = {5, 6, 100, 50, 0.785398f};
        b.prop = 0.8f;
        b.cls_id = 1;
        b.track_id = 0;
        frame.result = obb;
        const std::string json = buildFrameResultJson(frame);
        CHECK(json.find("\"type\":\"obb\"") != std::string::npos);
        CHECK(json.find("\"box\":[5,6,100,50,0.785398]") != std::string::npos);
        CHECK(json.find("\"score\":0.8") != std::string::npos);
        CHECK(json.find("\"track_id\":0") != std::string::npos);
    }

    // seg:RLE 编码与解码约定闭合;空掩膜 → mask:0 无 rle
    {
        SegTaskResult seg;
        seg.data.boxes.push_back(cv::Rect(2, 1, 4, 3));
        seg.data.scores.push_back(0.7f);
        seg.data.class_ids.push_back(0);
        seg.track_ids.push_back(9);
        // 4x3 掩膜:row0=1,1,0,0 row1=0,0,0,0 row2=0,0,0,1 → 行主序 12 值
        // [1,1,0,0,0,0,0,0,0,0,0,1] → RLE 从 0 游程起:[0,2,9,1]
        cv::Mat mask(3, 4, CV_8UC1, cv::Scalar(0));
        mask.at<uint8_t>(0, 0) = 255;
        mask.at<uint8_t>(0, 1) = 255;
        mask.at<uint8_t>(2, 3) = 255;
        seg.data.masks.push_back(mask);
        frame.result = seg;
        std::string json = buildFrameResultJson(frame);
        CHECK(json.find("\"type\":\"seg\"") != std::string::npos);
        CHECK(json.find("\"bbox\":[2,1,4,3]") != std::string::npos);
        CHECK(json.find("\"score\":0.7") != std::string::npos);
        CHECK(json.find("\"track_id\":9") != std::string::npos);
        CHECK(json.find("\"mask\":1") != std::string::npos);
        CHECK(json.find("\"rle\":[0,2,9,1]") != std::string::npos);

        // variant 按值拷贝:改掩膜后需重新赋给 frame.result
        seg.data.masks[0] = cv::Mat();
        frame.result = seg;
        json = buildFrameResultJson(frame);
        CHECK(json.find("\"mask\":0") != std::string::npos);
        CHECK(json.find("\"rle\"") == std::string::npos);
    }

    // depth:仅元数据,不携带栅格
    {
        DepthTaskResult depth;
        depth.roi = cv::Rect(8, 8, 632, 352);
        depth.depth_lo = 1.5f;
        depth.depth_hi = 72.25f;
        frame.result = depth;
        std::string json = buildFrameResultJson(frame);
        CHECK(json.find("\"depth\":{\"roi\":[8,8,632,352],\"range\":[1.5,72.25]}") != std::string::npos);
    }



    // jsonEscape 边界 + 非有限浮点钳为合法 JSON
    {
        CHECK(jsonEscape("a\"b\\c\nd\te") == "a\\\"b\\\\c\\nd\\te");
        CHECK(jsonEscape("中文") == "中文");
        CHECK(jsonEscape(std::string(1, '\x01')) == "\\u0001");

        DetectTaskResult det;
        det.data.count = 1;
        det.data.results[0].box = {0, 0, 1, 1};
        det.data.results[0].prop = std::sqrt(-1.0f);  // NaN
        det.data.results[0].cls_id = 0;
        frame.result = det;
        const std::string json = buildFrameResultJson(frame);
        CHECK(json.find("\"score\":0,") != std::string::npos);
        CHECK(json.find("nan") == std::string::npos);
        CHECK(json.find("inf") == std::string::npos);
    }
}

// ---- 逐帧结果 JSONL 汇(io/result_sink.h) --------------------------------------

static void testResultSink() {
    std::printf("[test] result_sink JSONL 汇\n");
    const std::string path =
        std::string("/tmp/rk_pipe_unit_") + std::to_string(getpid()) + "_result.jsonl";
    std::remove(path.c_str());
    const char* kEnv = "RK_PIPE_RESULT_JSONL";

    PipelineFrame frame;
    frame.index = 1;
    frame.sourceName = "s1";
    frame.matFrame = cv::Mat(8, 8, CV_8UC3, cv::Scalar(0));

    DetectTaskResult det;
    det.data.count = 1;
    det.data.results[0].box = {0, 0, 4, 4};
    det.data.results[0].prop = 0.5f;
    det.data.results[0].cls_id = 0;
    frame.result = det;

    // 未配置时写入为 no-op
    unsetenv(kEnv);
    shutdownFrameResultSink();
    writeFrameResult(frame);

    AppConfig cfg;
    setenv(kEnv, path.c_str(), 1);
    configureFrameResultSink(cfg);

    writeFrameResult(frame);  // hasResult=false → 不落盘
    frame.hasResult = true;
    writeFrameResult(frame);
    frame.index = 2;
    frame.sourceName = "s2";
    writeFrameResult(frame);
    shutdownFrameResultSink();

    auto countLines = [&path]() {
        std::ifstream in(path);
        if (!in.is_open()) {
            return -1;
        }
        int lines = 0;
        std::string line;
        while (std::getline(in, line)) {
            ++lines;
            if (line.find("{\"schema_version\":1,") != 0 || line.empty() || line.back() != '}') {
                return -2;  // 行格式坏
            }
        }
        return lines;
    };
    CHECK(countLines() == 2);

    // 重新 configure → truncate 重写
    setenv(kEnv, path.c_str(), 1);
    configureFrameResultSink(cfg);
    writeFrameResult(frame);
    shutdownFrameResultSink();
    CHECK(countLines() == 1);

    // 打不开的路径:只失败不崩溃,流水线继续
    setenv(kEnv, "/nonexistent_dir_xyz/result.jsonl", 1);
    configureFrameResultSink(cfg);
    writeFrameResult(frame);
    shutdownFrameResultSink();

    unsetenv(kEnv);
    std::remove(path.c_str());
}


static std::string webHttpGet(int port, const char* target, const char* extra_header = nullptr) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return "socket-error";
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return "connect-error";
    }
    std::string req = std::string("GET ") + target + " HTTP/1.0\r\nHost: 127.0.0.1\r\n";
    if (extra_header) {
        req += std::string(extra_header) + "\r\n";
    }
    req += "\r\n";
    if (::send(fd, req.c_str(), req.size(), 0) < 0) {
        ::close(fd);
        return "send-error";
    }
    std::string resp;
    char buf[1024];
    ssize_t n = 0;
    while ((n = ::recv(fd, buf, sizeof(buf), 0)) > 0) {
        resp.append(buf, static_cast<std::size_t>(n));
    }
    ::close(fd);
    return resp;
}

#ifndef RK_PIPE_CI
static void testWebTokenAuth() {
    std::printf("[test] web preview token auth\n");
    const int port = 18000 + static_cast<int>(getpid() % 2000);

    WebPreviewServer::Config cfg;
    cfg.bind_address = "127.0.0.1";
    cfg.port = port;

    // 未配置令牌:鉴权关闭,直接 200(向后兼容)
    unsetenv("RK_PIPE_WEB_TOKEN");
    WebPreviewServer open_server;
    if (open_server.start(cfg)) {
        const std::string resp = webHttpGet(port, "/status.json");
        CHECK(resp.find("200") == 0 || resp.find("HTTP/1") == 0);
        open_server.stop();
    }

    // 配置令牌:无令牌/错令牌 → 401;查询参数/请求头 → 200;/healthz 始终开放
    setenv("RK_PIPE_WEB_TOKEN", "unit_test_tok", 1);
    WebPreviewServer server;
    CHECK(server.start(cfg));
    CHECK(server.running());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const std::string no_tok = webHttpGet(port, "/status.json");
    CHECK(no_tok.find("401") != std::string::npos);
    const std::string bad_tok = webHttpGet(port, "/status.json?token=wrong");
    CHECK(bad_tok.find("401") != std::string::npos);
    const std::string qry_ok = webHttpGet(port, "/status.json?token=unit_test_tok");
    CHECK(qry_ok.find("200") != std::string::npos);
    const std::string hdr_ok =
        webHttpGet(port, "/status.json", "X-Auth-Token: unit_test_tok");
    CHECK(hdr_ok.find("200") != std::string::npos);
    const std::string healthz = webHttpGet(port, "/healthz");
    CHECK(healthz.find("200") != std::string::npos);
    // 主页面:令牌已嵌入端点 URL
    const std::string page = webHttpGet(port, "/?token=unit_test_tok");
    CHECK(page.find("token=unit_test_tok") != std::string::npos);

    server.stop();
    unsetenv("RK_PIPE_WEB_TOKEN");
}
#endif  // !RK_PIPE_CI


int main(int argc, char** argv) {
    // 用例注册表:与 CMakeLists.txt 中 RK_PIPE_CI_TEST_NAMES / RK_PIPE_FULL_TEST_NAMES 保持一致
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    static const TestCase kTests[] = {
        {"config_load", testConfigLoad},
        {"config_validate", testConfigValidate},
        {"command_line", testCommandLine},
#ifndef RK_PIPE_CI
        {"detector_factory", testDetectorFactory},
#endif
        {"postprocess_math", testPostprocessMath},
        {"event_engine", testEventEngine},
        {"event_rules", testEventRules},
        {"performance_average", testPerformanceAverage},
        {"detection_filter", testDetectionFilter},
        {"turbojpeg", testTurbojpeg},
        {"result_json", testResultJson},
        {"result_sink", testResultSink},
#ifndef RK_PIPE_CI
        {"tracker_ids", testTrackerIds},
        {"tracker_occlusion", testTrackerOcclusion},
        {"tracker_relink", testTrackerRelink},
        {"tracker_modes", testTrackerModes},
        {"tracker_other_tasks", testTrackerOtherTasks},
        {"tracker_perf", testTrackerPerf},
        {"tracker_bench", testTrackerBench},
        {"web_token_auth", testWebTokenAuth},
#endif
    };
    constexpr int kTestCount = sizeof(kTests) / sizeof(kTests[0]);
    (void)kTestCount;

    auto runOne = [&](const char* wanted) -> bool {
        for (const auto& t : kTests) {
            if (std::strcmp(t.name, wanted) == 0) {
                t.fn();
                return true;
            }
        }
        return false;
    };

    if (argc > 1) {
        // 按名运行(ctest 为每个用例注册独立条目,单用例崩溃不影响其余结果)
        for (int i = 1; i < argc; ++i) {
            if (!runOne(argv[i])) {
                std::printf("unknown test: %s\n", argv[i]);
                return 2;
            }
        }
    } else {
        for (const auto& t : kTests) {
            t.fn();
        }
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
