// rknn_eval —— RK3588 板端模型精度/性能一键评测工具。
//
// 编排流程：配置校验 → 逐模型推理（经 C ABI 驱动流水线 + dump_detections 导出）
// → 调 Python 评测脚本（pycocotools / 旋转 IoU）→ 解析摘要 → 论文风格报告（MD + LaTeX + CSV）
// → 可选断言（CI 退出码）。
//
// 用法示例：
//   rknn_eval --task detect --dataset coco \
//     --model model/yolo26n.rknn --compare fp16=model/yolo26n_fp16.rknn \
//     --images data/coco/images/val2017 \
//     --ann data/coco/annotations/instances_val2017.json \
//     --label assets/labels/coco_80_labels_list.txt --obj-num 80 \
//     --out-dir eval_runs/detect --threads 4
//
// 评测口径（论文式汇报）: conf=0.001 全量导出，COCOeval maxDets=100（kpts=20），
// 指标 ×100 保留 1 位小数；imgIds 限定为实际推理图集合。

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <unistd.h>
#include <vector>

#include "rkpipe/rkpipe.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#ifndef RK_EVAL_SCRIPTS_DIR
#define RK_EVAL_SCRIPTS_DIR "tools/eval"
#endif

// ---------------------------------------------------------------------------
// 极简 JSON 解析（对象/数组/字符串/数字/bool/null），读评测摘要用。
// ---------------------------------------------------------------------------
struct JValue {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JValue> arr;
    std::map<std::string, JValue> obj;

    const JValue* find(const std::string& key) const {
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
    }
    double num_or(double dflt) const { return type == Num ? num : dflt; }
    std::string str_or(const std::string& dflt) const { return type == Str ? str : dflt; }
};

namespace mini_json {

struct Parser {
    const std::string& s;
    size_t i = 0;
    explicit Parser(const std::string& src) : s(src) {}

    void skip_ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    [[noreturn]] void fail(const std::string& msg) {
        throw std::runtime_error("[rknn_eval][json] 解析失败@" + std::to_string(i) + ": " + msg);
    }
    char peek() {
        if (i >= s.size()) fail("意外结尾");
        return s[i];
    }
    void expect(char c) {
        skip_ws();
        if (peek() != c) fail(std::string("期望 '").append(1, c).append("'"));
        ++i;
    }
    JValue parse() {
        skip_ws();
        JValue v = parse_value();
        return v;
    }
    JValue parse_value() {
        skip_ws();
        switch (peek()) {
            case '{': return parse_obj();
            case '[': return parse_arr();
            case '"': {
                JValue v; v.type = JValue::Str; v.str = parse_str(); return v;
            }
            case 't': i += 4; { JValue v; v.type = JValue::Bool; v.b = true; return v; }
            case 'f': i += 5; { JValue v; v.type = JValue::Bool; v.b = false; return v; }
            case 'n': i += 4; return JValue{};
            default: {
                JValue v; v.type = JValue::Num; v.num = parse_num(); return v;
            }
        }
    }
    JValue parse_obj() {
        expect('{');
        JValue v; v.type = JValue::Obj;
        skip_ws();
        if (peek() == '}') { ++i; return v; }
        for (;;) {
            skip_ws();
            std::string key = parse_str();
            expect(':');
            v.obj[key] = parse_value();
            skip_ws();
            char c = peek();
            if (c == ',') { ++i; continue; }
            if (c == '}') { ++i; return v; }
            fail("对象分隔符");
        }
    }
    JValue parse_arr() {
        expect('[');
        JValue v; v.type = JValue::Arr;
        skip_ws();
        if (peek() == ']') { ++i; return v; }
        for (;;) {
            v.arr.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ',') { ++i; continue; }
            if (c == ']') { ++i; return v; }
            fail("数组分隔符");
        }
    }
    std::string parse_str() {
        expect('"');
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[++i];
                switch (c) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    default: out.push_back(c); break;  // \u 简化处理
                }
            } else {
                out.push_back(s[i]);
            }
            ++i;
        }
        expect('"');
        return out;
    }
    double parse_num() {
        skip_ws();
        size_t start = i;
        while (i < s.size() && (isdigit(s[i]) || s[i] == '-' || s[i] == '+' ||
                                s[i] == '.' || s[i] == 'e' || s[i] == 'E')) ++i;
        if (start == i) fail("数字");
        return std::atof(s.substr(start, i - start).c_str());
    }
};

JValue parse(const std::string& text) { return Parser(text).parse(); }
}  // namespace mini_json

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------
static bool file_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && !S_ISDIR(st.st_mode);
}
static bool dir_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
// 递归建目录（mkdir 不建父目录，out-dir 带多级时必须逐级创建）
static void mkdirs(const std::string& path) {
    std::string cur;
    if (!path.empty() && path[0] == '/') cur = "/";
    size_t pos = 0;
    while (pos <= path.size()) {
        size_t next = path.find('/', pos);
        std::string seg = path.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        if (!seg.empty()) {
            if (!cur.empty() && cur.back() != '/') cur += "/";
            cur += seg;
            if (!dir_exists(cur)) ::mkdir(cur.c_str(), 0755);
        }
        if (next == std::string::npos) break;
        pos = next + 1;
    }
}
static std::string basename_of(const std::string& p) {
    auto pos = p.find_last_of('/');
    return pos == std::string::npos ? p : p.substr(pos + 1);
}
static std::string stem_of(const std::string& p) {
    std::string b = basename_of(p);
    auto pos = b.find_last_of('.');
    return pos == std::string::npos ? b : b.substr(0, pos);
}
static std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
// 流式查找文本 token（标注 JSON 可达数百 MB，不全量载入内存）
static bool file_contains_token(const std::string& path, const std::string& token) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    const size_t keep = token.size() - 1;
    std::string buf(keep, '\0');
    std::vector<char> chunk(1 << 20);
    while (f.read(chunk.data(), chunk.size()) || f.gcount() > 0) {
        size_t n = static_cast<size_t>(f.gcount());
        std::string window = buf + std::string(chunk.data(), n);
        if (window.find(token) != std::string::npos) return true;
        buf = window.substr(window.size() - keep);
    }
    return false;
}
static std::string exec_capture(const std::string& cmd) {
    std::string out;
    FILE* fp = ::popen(cmd.c_str(), "r");
    if (!fp) return "";
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) out.append(buf, n);
    ::pclose(fp);
    return out;
}
// shell 单引号转义，防路径中的空格/特殊字符破坏拼出的命令
static std::string shq(const std::string& p) {
    std::string out = "'";
    for (char c : p) {
        if (c == '\'') out += "'\\''";
        else out.push_back(c);
    }
    out += "'";
    return out;
}
static std::string fmt1(double v) {  // ×100, 1 位小数（论文口径）
    if (!std::isfinite(v)) return "n/a";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", v * 100.0);
    return buf;
}
static std::string fmt_fps(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    return buf;
}
static std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        if (c == '\n') { out += "\\n"; continue; }
        out.push_back(c);
    }
    return out;
}

// ---------------------------------------------------------------------------
// 任务注册表
// ---------------------------------------------------------------------------
enum class Task { Detect, Pose, Seg, Obb };
enum class Dataset { Coco, Dota };

struct TaskSpec {
    Task task;
    const char* name;
    Dataset dataset;
    const char* evaluator;        // tools/eval 下脚本名
    std::vector<std::string> cols;  // 主表列
};

static const TaskSpec& task_spec(Task t) {
    static const std::map<int, TaskSpec> kSpecs = {
        {static_cast<int>(Task::Detect),
         {Task::Detect, "detect", Dataset::Coco, "eval_coco.py",
          {"mAP", "AP50", "AP75", "APs", "APm", "APl", "AR", "FPS"}}},
        {static_cast<int>(Task::Pose),
         {Task::Pose, "pose", Dataset::Coco, "eval_coco.py",
          {"AP", "AP50", "AP75", "APm", "APl", "AR20", "FPS"}}},
        {static_cast<int>(Task::Seg),
         {Task::Seg, "seg", Dataset::Coco, "eval_coco.py",
          {"mAP", "AP50", "AP75", "APs", "APm", "APl", "AR", "FPS"}}},
        {static_cast<int>(Task::Obb),
         {Task::Obb, "obb", Dataset::Dota, "eval_dota.py",
          {"mAP50", "mAP50-95", "FPS"}}},
    };
    return kSpecs.at(static_cast<int>(t));
}

// ---------------------------------------------------------------------------
// 配置
// ---------------------------------------------------------------------------
struct Config {
    Task task = Task::Detect;
    Dataset dataset = Dataset::Coco;
    std::string task_str;        // 透传给推理器的 task 字符串
    std::string model;
    std::string images;
    std::string ann;
    std::string label;
    std::string gt;              // obb：切片级 GT jsonl（省去切片步骤）
    std::string dota_labels;     // obb：原始 labelTxt 目录（触发切片）
    int obj_num = 0;
    int threads = 4;
    double conf = 0.001;
    double nms = 0.65;
    std::string out_dir = "eval_runs";
    std::string name;            // 主模型 tag
    std::vector<std::pair<std::string, std::string>> compare;  // tag -> model
    std::vector<int> threads_sweep;
    std::vector<double> conf_sweep;
    double assert_ap = -1.0;     // 断言主指标（detect/seg/pose=AP(50-95)，obb=mAP50）
    double assert_ap50 = -1.0;
    std::string reuse_dump;      // 跳过主模型推理，直接用已有 dump 评测
    std::string vis_dir;         // 非空时：渲染主模型检测结果图（单图即"开图出图"）
    int vis_sample = 0;          // >0 时可视化抽帧（每 N 帧渲 1 张），0=全渲
    bool preview = false;        // 推理时打开引擎 Web MJPEG 预览（带叠加，不影响 dump 正确性）
    int preview_port = 8090;     // 预览端口（--compare 串行复用）
    bool dump_only = false;      // 只推理导出 dump+metrics，跳过评测/报告（远程模式用）
    bool dry_run = false;
    std::string scripts_dir;
};

struct RunResult {
    std::string tag;
    std::string model;
    std::string dump;
    std::string summary_path;
    JValue summary;
    double fps = 0, infer_ms = 0, pre_ms = 0;
    bool ok = false;
};

// ---------------------------------------------------------------------------
// CLI 解析（支持 key=value 简易配置文件）
// ---------------------------------------------------------------------------
static void usage_and_exit() {
    std::printf(
        "rknn_eval —— RK3588 板端模型精度/性能一键评测\n\n"
        "用法: rknn_eval --task <detect|pose|seg|obb> --model <rknn> --images <dir> [选项]\n\n"
        "必选:\n"
        "  --task TASK           detect | pose | seg | obb\n"
        "  --model FILE          主模型 .rknn\n"
        "  --images DIR          图片目录（必须含 '/'；obb 传切片目录或原图目录）\n"
        "  --ann FILE            COCO 任务: instances/person_keypoints 标注\n"
        "  --gt FILE             obb: 切片级 GT jsonl（或用 --dota-labels 触发自动切片）\n"
        "  --dota-labels DIR     obb: 原始 labelTxt 目录，自动 1024/200 切片\n"
        "可选:\n"
        "  --label FILE          类别表（label 顺序校验）\n"
        "  --obj-num N           类别数（detect=80, obb=15）\n"
        "  --threads N           推理线程数（默认 4）\n"
        "  --conf / --nms        导出与 NMS 阈值（默认 0.001 / 0.65）\n"
        "  --compare tag=FILE    对照模型，可多次（如 fp16=model/yolo26n_fp16.rknn）\n"
        "  --threads-sweep a,b   线程扫描（只测 FPS，不重评测精度）\n"
        "  --conf-sweep a,b,c    conf 扫描（同一 dump 离线重评）\n"
        "  --assert-ap X         CI 断言: 主指标(50-95) x100 >= X，否则退出码 1\n"
        "  --assert-ap50 X       CI 断言: AP50 x100 >= X\n"
        "  --reuse-dump FILE     跳过推理，直接评测已有 dump\n"
        "  --vis-dir DIR         渲染主模型检测结果图到该目录（单图输入 = 开图出图）\n"
        "  --vis-sample N        可视化抽帧：每 N 帧渲 1 张（0=全渲，默认 0）\n"
        "  --preview             推理时打开板端 Web MJPEG 实时预览（带叠加，不影响 dump）\n"
        "  --preview-port N      预览端口（默认 8090；--compare 串行复用）\n"
        "  --dump-only           只推理导出 dump+metrics，跳过评测/报告（远程模式用）\n"
        "  --out-dir DIR         产物目录（默认 eval_runs）\n"
        "  --name STR            主模型 tag（默认模型文件名主干）\n"
        "  --config FILE         key=value 配置文件（# 注释；CLI 参数覆盖）\n"
        "  --scripts-dir DIR     评测脚本目录（默认随二进制定位 tools/eval）\n"
        "  --dry-run             只打印计划不执行\n");
    std::exit(0);
}

static Task parse_task(const std::string& s, std::string* task_str_out = nullptr) {
    if (s == "detect") return Task::Detect;
    if (s == "pose") return Task::Pose;
    if (s == "seg") return Task::Seg;
    if (s == "obb") return Task::Obb;
    std::fprintf(stderr, "[rknn_eval] 未知任务: %s（detect/pose/seg/obb）\n", s.c_str());
    std::exit(2);
}

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

static void apply_arg(Config& cfg, const std::string& k, const std::string& v) {
    if (k == "task") {
        cfg.task_str.clear();
        cfg.task = parse_task(v, &cfg.task_str);
        if (cfg.task_str.empty()) cfg.task_str = v;  // 常规任务也记录（pose/detect/...）
    }
    else if (k == "dataset") cfg.dataset = (v == "dota") ? Dataset::Dota : Dataset::Coco;
    else if (k == "model") cfg.model = v;
    else if (k == "images") cfg.images = v;
    else if (k == "ann") cfg.ann = v;
    else if (k == "label") cfg.label = v;
    else if (k == "gt") cfg.gt = v;
    else if (k == "dota-labels" || k == "dota_labels") cfg.dota_labels = v;
    else if (k == "obj-num" || k == "obj_num") cfg.obj_num = std::atoi(v.c_str());
    else if (k == "threads") cfg.threads = std::atoi(v.c_str());
    else if (k == "conf") cfg.conf = std::atof(v.c_str());
    else if (k == "nms") cfg.nms = std::atof(v.c_str());
    else if (k == "out-dir" || k == "out_dir") cfg.out_dir = v;
    else if (k == "name") cfg.name = v;
    else if (k == "compare") {
        auto pos = v.find('=');
        if (pos != std::string::npos)
            cfg.compare.emplace_back(v.substr(0, pos), v.substr(pos + 1));
        else
            cfg.compare.emplace_back(stem_of(v), v);
    }
    else if (k == "threads-sweep" || k == "threads_sweep") {
        for (auto& t : split_csv(v)) cfg.threads_sweep.push_back(std::atoi(t.c_str()));
    }
    else if (k == "conf-sweep" || k == "conf_sweep") {
        for (auto& c : split_csv(v)) cfg.conf_sweep.push_back(std::atof(c.c_str()));
    }
    else if (k == "assert-ap" || k == "assert_ap") cfg.assert_ap = std::atof(v.c_str());
    else if (k == "assert-ap50" || k == "assert_ap50") cfg.assert_ap50 = std::atof(v.c_str());
    else if (k == "reuse-dump" || k == "reuse_dump") cfg.reuse_dump = v;
    else if (k == "vis-dir" || k == "vis_dir") cfg.vis_dir = v;
    else if (k == "vis-sample" || k == "vis_sample") cfg.vis_sample = std::atoi(v.c_str());
    else if (k == "preview") cfg.preview = (v == "1" || v == "true");
    else if (k == "preview-port" || k == "preview_port") cfg.preview_port = std::atoi(v.c_str());
    else if (k == "dump-only" || k == "dump_only") cfg.dump_only = (v == "1" || v == "true");
    else if (k == "scripts-dir" || k == "scripts_dir") cfg.scripts_dir = v;
    else if (k == "dry-run" || k == "dry_run") cfg.dry_run = (v == "1" || v == "true");
    else {
        std::fprintf(stderr, "[rknn_eval] 未知参数: --%s\n", k.c_str());
        std::exit(2);
    }
}

// 无值开关（出现即置 "1"），不消费下一个 argv
static bool is_valueless_flag(const std::string& k) {
    return k == "dry-run" || k == "dry_run" ||
           k == "dump-only" || k == "dump_only" ||
           k == "preview";
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    std::vector<std::pair<std::string, std::string>> args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { usage_and_exit(); }
        if (a.rfind("--", 0) != 0) {
            std::fprintf(stderr, "[rknn_eval] 无法识别的参数: %s\n", a.c_str());
            std::exit(2);
        }
        std::string body = a.substr(2);
        auto eq = body.find('=');
        if (eq != std::string::npos) {
            args.emplace_back(body.substr(0, eq), body.substr(eq + 1));
        } else if (is_valueless_flag(body)) {
            args.emplace_back(body, "1");  // 无值开关
        } else {
            if (i + 1 >= argc) { std::fprintf(stderr, "[rknn_eval] --%s 缺值\n", body.c_str()); std::exit(2); }
            args.emplace_back(body, argv[++i]);
        }
    }
    // 先处理 config 文件（key=value 每行），CLI 覆盖文件
    for (auto& [k, v] : args) {
        if (k == "config") {
            std::ifstream f(v);
            std::string line;
            while (std::getline(f, line)) {
                auto hash = line.find('#');
                if (hash != std::string::npos) line = line.substr(0, hash);
                // trim
                auto issp = [](unsigned char c) { return isspace(c); };
                while (!line.empty() && issp(line.front())) line.erase(line.begin());
                while (!line.empty() && issp(line.back())) line.pop_back();
                if (line.empty()) continue;
                auto eq = line.find('=');
                if (eq == std::string::npos) continue;
                auto trim = [](std::string s) {
                    auto issp = [](unsigned char c) { return isspace(c); };
                    while (!s.empty() && issp(s.front())) s.erase(s.begin());
                    while (!s.empty() && issp(s.back())) s.pop_back();
                    return s;
                };
                apply_arg(cfg, trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
            }
        }
    }
    for (auto& [k, v] : args) {
        if (k != "config") apply_arg(cfg, k, v);
    }
    if (cfg.name.empty()) cfg.name = cfg.model.empty() ? "main" : stem_of(cfg.model);
    return cfg;
}

// ---------------------------------------------------------------------------
// 推理执行：复用流水线，导出 dump + metrics
// ---------------------------------------------------------------------------
static bool run_inference(const Config& cfg, const std::string& model,
                          const std::string& tag, const std::string& dump_path,
                          const std::string& metrics_path, RunResult* out) {
    if (dump_path != "/dev/null" && file_exists(dump_path)) std::remove(dump_path.c_str());
    ::setenv("RK_PIPE_METRICS_FILE", metrics_path.c_str(), 1);
    // seg 评测标准口径：掩膜二值阈值取概率 0.5（logit 0）；引擎默认 -0.5 偏宽松
    if (cfg.task == Task::Seg) ::setenv("RK_PIPE_SEG_MASK_THRESH", "0", 1);

    // 配置即契约：写临时 YAML，经 C ABI 驱动与 console_detector 完全一致的流水线。
    // 工具自身只依赖 rkpipe/rkpipe.h，可在开源仓（预编译核心）环境编译运行。
    const std::string yaml_path = cfg.out_dir + "/rknn_eval_" + tag + ".yaml";
    {
        std::string y = "%YAML:1.0\n---\n";
        y += "model_path: \"" + model + "\"\n";
        y += "input_path: \"" + cfg.images + "\"\n";
        y += std::string("task: \"") +
             (cfg.task_str.empty() ? task_spec(cfg.task).name : cfg.task_str) + "\"\n";
        y += "mode: \"pipeline\"\n";
        y += "thread_count: " + std::to_string(cfg.threads) + "\n";
        y += "label_path: \"" + cfg.label + "\"\n";
        y += "obj_class_num: " + std::to_string(cfg.obj_num) + "\n";
        y += "dump_detections_path: \"" + dump_path + "\"\n";
        char conf_buf[32], nms_buf[32];
        std::snprintf(conf_buf, sizeof(conf_buf), "%g", cfg.conf);
        std::snprintf(nms_buf, sizeof(nms_buf), "%g", cfg.nms);
        y += "conf_threshold: " + std::string(conf_buf) + "\n";
        y += "nms_threshold: " + std::string(nms_buf) + "\n";
        // 评测口径：关闭一切输出/追踪（配置缺省即关，显式声明防漂移）
        y += "output_video_path: \"\"\n";
        y += std::string("web_preview: ") + (cfg.preview ? "1" : "0") + "\n";
        if (cfg.preview) {
            // 实时预览：MJPEG 带叠加渲染；dump 在叠加之前写，不影响评测正确性
            y += "web_preview_bind: \"0.0.0.0\"\n";
            y += "web_preview_port: " + std::to_string(cfg.preview_port) + "\n";
        }
        y += "enable_tracking: 0\n";
        std::ofstream f(yaml_path);
        if (!f) {
            std::fprintf(stderr, "[rknn_eval][%s] 写临时配置失败: %s\n", tag.c_str(), yaml_path.c_str());
            return false;
        }
        f << y;
    }

    rkpipe_handle h = rkpipe_create(yaml_path.c_str());
    if (!h) {
        std::fprintf(stderr, "[rknn_eval][%s] rkpipe_create 失败（配置加载/初始化）\n", tag.c_str());
        return false;
    }
    const int rc_start = rkpipe_start(h);
    if (rc_start != RK_PIPE_OK) {
        std::fprintf(stderr, "[rknn_eval][%s] rkpipe_start 失败: %d\n", tag.c_str(), rc_start);
        rkpipe_destroy(h);
        return false;
    }
    const int rc_wait = rkpipe_wait(h, -1);
    const int exit_code = rkpipe_get_exit_code(h);
    rkpipe_destroy(h);
    if (rc_wait != RK_PIPE_OK || exit_code != RK_PIPE_OK) {
        std::fprintf(stderr, "[rknn_eval][%s] 推理异常（wait=%d, exit_code=%d）\n",
                     tag.c_str(), rc_wait, exit_code);
        return false;
    }
    // 读 metrics（损坏/不完整时按 0 处理，不阻断评测）
    if (file_exists(metrics_path)) {
        try {
            JValue m = mini_json::parse(read_file(metrics_path));
            if (const JValue* rp = m.find("rk_pipe")) {
                if (const JValue* pl = rp->find("pipeline")) {
                    out->fps = pl->find("avg_fps") ? pl->find("avg_fps")->num_or(0) : 0;
                }
                if (const JValue* st = rp->find("stage_avg_ms")) {
                    out->infer_ms = st->find("inference") ? st->find("inference")->num_or(0) : 0;
                    out->pre_ms = st->find("preprocess") ? st->find("preprocess")->num_or(0) : 0;
                }
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rknn_eval][%s] metrics 解析失败，按 0 处理: %s\n", tag.c_str(), e.what());
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 评测脚本调用
// ---------------------------------------------------------------------------
static std::string resolve_scripts_dir(const Config& cfg) {
    if (!cfg.scripts_dir.empty()) return cfg.scripts_dir;
    if (const char* env = ::getenv("RK_EVAL_SCRIPTS")) return env;
    char buf[4096];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = 0;
        std::string exe_dir = std::string(buf);
        auto pos = exe_dir.find_last_of('/');
        exe_dir = pos == std::string::npos ? "." : exe_dir.substr(0, pos);
        std::string cand = exe_dir + "/../" RK_EVAL_SCRIPTS_DIR;
        if (file_exists(cand + "/eval_coco.py")) return cand;
    }
    return RK_EVAL_SCRIPTS_DIR;
}

static bool run_evaluator(const Config& cfg, const TaskSpec& spec, const std::string& scripts,
                          const std::string& dump, const std::string& summary_path,
                          double conf, JValue* summary_out, std::string* log_out) {
    std::ostringstream cmd;
    cmd << "python3 " << shq(scripts + "/" + spec.evaluator)
        << " --dump " << shq(dump)
        << " --out " << shq(summary_path)
        << " --conf " << conf;
    if (spec.task == Task::Obb) {
        cmd << " --gt " << shq(cfg.gt);
    } else {
        cmd << " --ann " << shq(cfg.ann);
        if (!cfg.label.empty()) cmd << " --label " << shq(cfg.label);
    }
    std::string log = exec_capture(cmd.str() + " 2>&1");
    if (log_out) *log_out = log;
    if (!file_exists(summary_path)) {
        std::fprintf(stderr, "[rknn_eval] 评测脚本失败:\n%s\n", log.c_str());
        return false;
    }
    try {
        *summary_out = mini_json::parse(read_file(summary_path));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rknn_eval] 摘要解析失败: %s\n", e.what());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// 检测结果可视化：渲染主模型 dump 到原图，输出 *_vis.jpg。
// 单张图片输入时即"开图出图"。行解析仅接受以 '}' 结尾的完整行（容忍被截断的尾行）。
// ---------------------------------------------------------------------------
static const cv::Scalar kPalette[16] = {
    {60, 76, 231},   {28, 185, 69},  {180, 51, 59},  {163, 255, 10},
    {0, 160, 255},   {255, 0, 121},  {226, 43, 138}, {255, 191, 0},
    {0, 215, 255},   {128, 0, 128},  {203, 192, 255},{50, 205, 50},
    {255, 165, 0},   {255, 0, 255},  {30, 105, 210}, {222, 196, 176}};

static cv::Scalar cls_color(int cls) { return kPalette[cls % 16]; }

static std::vector<std::string> load_label_names(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream f(path);
    std::string l;
    while (std::getline(f, l)) {
        while (!l.empty() && (l.back() == '\r' || l.back() == ' ')) l.pop_back();
        if (!l.empty()) names.push_back(l);
    }
    return names;
}

// dump 的行级 RLE（框内裁剪）-> 全帧二值掩膜
static cv::Mat decode_rle_mask(const int* counts, int n_counts, const cv::Rect& box,
                               const cv::Size& frame) {
    cv::Mat crop = cv::Mat::zeros(box.height, box.width, CV_8UC1);
    int total = box.width * box.height, pos = 0, val = 0;
    for (int i = 0; i < n_counts; ++i) {
        int c = counts[i];
        if (val) {
            for (int k = pos; k < std::min(pos + c, total); ++k)
                crop.at<uchar>(k / box.width, k % box.width) = 255;
        }
        pos += c;
        val ^= 1;
    }
    cv::Mat full = cv::Mat::zeros(frame.height, frame.width, CV_8UC1);
    cv::Rect roi = box & cv::Rect(0, 0, frame.width, frame.height);
    if (roi.area() > 0) {
        crop(cv::Rect(roi.x - box.x, roi.y - box.y, roi.width, roi.height)).copyTo(full(roi));
    }
    return full;
}

static void draw_tag(cv::Mat& img, const std::string& text, const cv::Point& tl,
                     const cv::Scalar& color) {
    int base = 0;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &base);
    cv::Point p(std::max(0, tl.x), std::max(ts.height + 6, tl.y));
    cv::rectangle(img, {p.x, p.y - ts.height - 6, ts.width + 6, ts.height + 6}, color, -1);
    cv::putText(img, text, {p.x + 3, p.y - 3}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {255, 255, 255}, 1);
}

static std::string det_tag(const std::vector<std::string>& names, int cls, double score) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), " %.2f", score);
    return (cls >= 0 && cls < static_cast<int>(names.size()) ? names[cls]
            : "cls" + std::to_string(cls)) + buf;
}

static void render_vis(const Config& cfg, const TaskSpec& spec, const std::string& dump,
                       const std::string& vis_dir, int vis_sample) {
    mkdirs(vis_dir);
    std::vector<std::string> names = load_label_names(cfg.label);

    // ---- 选帧 ----
    // pose 任务优先渲"有人"的图：单人/多人交替覆盖，无人帧兜底补齐，
    // 输出密度与原 every-N 抽帧一致；其余任务保持固定间隔抽帧。
    struct FrameRef { std::streamoff off; int cnt; };
    std::vector<FrameRef> refs;  // 全部有效帧（按文件顺序）
    if (spec.task == Task::Pose) {
        std::ifstream fr(dump);
        std::string ln;
        while (std::getline(fr, ln)) {
            if (ln.size() < 5 || ln.back() != '}') continue;
            std::streamoff off = (std::streamoff)fr.tellg() -
                                 (std::streamoff)ln.size() - 1;
            int cnt = 0;
            try {
                JValue x = mini_json::parse(ln);
                if (const JValue* arr = x.find("poses")) cnt = (int)arr->arr.size();
            } catch (const std::exception&) {
                continue;
            }
            refs.push_back({off, cnt});
        }
    }
    std::vector<std::string> sel;  // 待渲染的 dump 行
    if (spec.task == Task::Pose && vis_sample > 0) {
        int target = (int)((refs.size() + vis_sample - 1) / vis_sample);
        std::vector<FrameRef> one, multi, none;
        for (auto& r : refs) {
            if (r.cnt >= 2) multi.push_back(r);
            else if (r.cnt == 1) one.push_back(r);
            else none.push_back(r);
        }
        std::vector<FrameRef> pick;
        size_t a = 0, b = 0, c = 0;
        bool turn = true;  // 单人/多人交替
        while ((int)pick.size() < target &&
               (a < one.size() || b < multi.size() || c < none.size())) {
            if (turn && a < one.size()) pick.push_back(one[a++]);
            else if (!turn && b < multi.size()) pick.push_back(multi[b++]);
            else if (a < one.size()) pick.push_back(one[a++]);
            else if (b < multi.size()) pick.push_back(multi[b++]);
            else if (c < none.size()) pick.push_back(none[c++]);
            turn = !turn;
        }
        std::ifstream fr(dump);
        for (auto& p : pick) {
            std::string ln;
            fr.seekg(p.off);
            if (std::getline(fr, ln)) sel.push_back(ln);
        }
    } else {
        std::ifstream fr(dump);
        std::string ln;
        int frame_no = -1;
        while (std::getline(fr, ln)) {
            ++frame_no;
            if (vis_sample > 0 && (frame_no % vis_sample) != 0) continue;
            if (ln.size() < 5 || ln.back() != '}') continue;
            sel.push_back(ln);
        }
    }

    int rendered = 0;
    for (const auto& line : sel) {
        JValue fr;
        try {
            fr = mini_json::parse(line);
        } catch (const std::exception&) {
            continue;  // 坏行跳过，不中断可视化
        }
        const JValue* jfile = fr.find("file");
        if (!jfile || jfile->str.empty()) continue;
        cv::Mat img = cv::imread(jfile->str);
        if (img.empty()) continue;
        cv::Size frame(img.cols, img.rows);
        const JValue* jw = fr.find("width");
        const JValue* jh = fr.find("height");
        if (jw && jh && (static_cast<int>(jw->num) != frame.width ||
                         static_cast<int>(jh->num) != frame.height)) {
            cv::resize(img, img, {static_cast<int>(jw->num), static_cast<int>(jh->num)});
            frame = img.size();
        }

        if (spec.task == Task::Detect) {
            const JValue* arr = fr.find("dets");
            for (const auto& d : arr ? arr->arr : std::vector<JValue>{}) {
                const JValue* bb = d.find("bbox");
                if (!bb || bb->arr.size() < 4) continue;
                cv::Rect r((int)bb->arr[0].num, (int)bb->arr[1].num,
                           (int)bb->arr[2].num, (int)bb->arr[3].num);
                int cls = d.find("cls") ? (int)d.find("cls")->num : 0;
                double sc = d.find("score") ? d.find("score")->num_or(0) : 0;
                cv::Scalar col = cls_color(cls);
                cv::rectangle(img, r & cv::Rect(0, 0, frame.width, frame.height), col, 2);
                draw_tag(img, det_tag(names, cls, sc), {r.x, r.y}, col);
            }
        } else if (spec.task == Task::Pose) {
            // COCO 17 点骨架连接（0 基）
            static const int kSkel[][2] = {{15,13},{13,11},{16,14},{14,12},{11,12},
                                           {5,11},{6,12},{5,6},{5,7},{6,8},{7,9},{8,10},
                                           {1,2},{0,1},{0,2},{1,3},{2,4},{3,5},{4,6}};
            const JValue* arr = fr.find("poses");
            for (const auto& p : arr ? arr->arr : std::vector<JValue>{}) {
                const JValue* bb = p.find("bbox");
                if (bb && bb->arr.size() >= 4) {
                    cv::rectangle(img, {(int)bb->arr[0].num, (int)bb->arr[1].num,
                                        (int)bb->arr[2].num, (int)bb->arr[3].num},
                                  {180, 51, 59}, 2);
                }
                const JValue* kp = p.find("kpts");
                if (!kp || kp->arr.size() < 17) continue;
                std::vector<cv::Point> pts(17, {-1, -1});
                for (int k = 0; k < 17; ++k) {
                    const JValue& j = kp->arr[k];
                    if (j.arr.size() < 3 || j.arr[2].num < 1) continue;
                    pts[k] = {(int)j.arr[0].num, (int)j.arr[1].num};
                }
                for (auto& e : kSkel) {
                    if (pts[e[0]].x >= 0 && pts[e[1]].x >= 0)
                        cv::line(img, pts[e[0]], pts[e[1]], {28, 185, 69}, 2);
                }
                for (auto& pt : pts)
                    if (pt.x >= 0) cv::circle(img, pt, 3, {0, 0, 255}, -1);
            }
        } else if (spec.task == Task::Obb) {
            const JValue* arr = fr.find("obbs");
            for (const auto& d : arr ? arr->arr : std::vector<JValue>{}) {
                const JValue* bb = d.find("box");
                if (!bb || bb->arr.size() < 5) continue;
                double x = bb->arr[0].num, y = bb->arr[1].num, w = bb->arr[2].num,
                       h = bb->arr[3].num, a = bb->arr[4].num;
                double cx = x + w / 2, cy = y + h / 2, c = std::cos(a), s = std::sin(a);
                double dx = w / 2, dy = h / 2;
                cv::Point2f quad[4] = {
                    {float(cx - dx * c + dy * s), float(cy - dx * s - dy * c)},
                    {float(cx + dx * c + dy * s), float(cy + dx * s - dy * c)},
                    {float(cx + dx * c - dy * s), float(cy + dx * s + dy * c)},
                    {float(cx - dx * c - dy * s), float(cy - dx * s + dy * c)}};
                int cls = d.find("cls") ? (int)d.find("cls")->num : 0;
                double sc = d.find("score") ? d.find("score")->num_or(0) : 0;
                cv::Scalar col = cls_color(cls);
                for (int e = 0; e < 4; ++e) cv::line(img, quad[e], quad[(e + 1) % 4], col, 2);
                draw_tag(img, det_tag(names, cls, sc),
                         {(int)quad[0].x, (int)quad[0].y}, col);
            }
        } else if (spec.task == Task::Seg) {
            const JValue* arr = fr.find("segs");
            for (const auto& d : arr ? arr->arr : std::vector<JValue>{}) {
                const JValue* bb = d.find("bbox");
                const JValue* rle = d.find("rle");
                int cls = d.find("cls") ? (int)d.find("cls")->num : 0;
                double sc = d.find("score") ? d.find("score")->num_or(0) : 0;
                cv::Scalar col = cls_color(cls);
                if (rle && bb && bb->arr.size() >= 4 && !rle->arr.empty()) {
                    cv::Rect box((int)bb->arr[0].num, (int)bb->arr[1].num,
                                 std::max(1, (int)bb->arr[2].num), std::max(1, (int)bb->arr[3].num));
                    std::vector<int> counts;
                    counts.reserve(rle->arr.size());
                    for (const auto& j : rle->arr) counts.push_back((int)j.num);
                    cv::Mat mask = decode_rle_mask(counts.data(), (int)counts.size(), box, frame);
                    cv::Mat col_img(frame, CV_8UC3, col);
                    cv::Mat overlay = img.clone();
                    col_img.copyTo(overlay, mask);  // 掩膜区填类色
                    cv::addWeighted(overlay, 0.5, img, 0.5, 0, img);
                    std::vector<std::vector<cv::Point>> cs;
                    cv::findContours(mask, cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                    cv::drawContours(img, cs, -1, col, 1);
                }
                if (bb && bb->arr.size() >= 4) {
                    cv::rectangle(img, {(int)bb->arr[0].num, (int)bb->arr[1].num,
                                        (int)bb->arr[2].num, (int)bb->arr[3].num}, col, 1);
                    draw_tag(img, det_tag(names, cls, sc),
                             {(int)bb->arr[0].num, (int)bb->arr[1].num}, col);
                }
            }
        }

        std::string stem = stem_of(jfile->str);
        cv::imwrite(vis_dir + "/" + stem + "_vis.jpg", img, {cv::IMWRITE_JPEG_QUALITY, 95});
        ++rendered;
    }
    std::printf("[rknn_eval] 可视化 %d 张 -> %s\n", rendered, vis_dir.c_str());
}

// ---------------------------------------------------------------------------
// 报告生成（论文风格）
// ---------------------------------------------------------------------------
static const JValue* pick(const JValue& s, const std::vector<std::string>& keys) {
    for (const auto& k : keys) {
        if (const JValue* v = s.find(k)) return v;
    }
    return nullptr;
}

// 主表某列取值：列名 -> summary 键映射
static double metric_value(const JValue& s, Task task, const std::string& col) {
    if (col == "FPS") return -1;  // FPS 来自 metrics，单独处理
    std::string k;
    if (task == Task::Obb) {
        k = (col == "mAP50") ? "mAP50" : "mAP50_95";
    } else if (col == "mAP" || col == "AP") {
        k = "AP";
    } else if (col == "APs") {
        k = "AP_small";
    } else if (col == "APm") {
        k = "AP_medium";
    } else if (col == "APl") {
        k = "AP_large";
    } else if (col == "AR") {
        k = "AR_max";
    } else if (col == "AR20") {
        k = "AR_20";
    } else {
        k = col;  // AP50 / AP75
    }
    const JValue* v = pick(s, {k});
    return v ? v->num_or(0.0) : 0.0;
}

static void write_report(const Config& cfg, const TaskSpec& spec,
                         const std::vector<RunResult>& runs, const std::string& path) {
    std::ostringstream md;
    time_t now = ::time(nullptr);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M", localtime(&now));

    md << "# " << spec.name << " 板端评测报告（RK3588）\n\n";
    md << "- 生成时间: " << tbuf << "\n";
    md << "- 数据集: " << (spec.dataset == Dataset::Dota ? "DOTA v1.0 val（1024/200 切片）" : "COCO val2017")
       << " | 推理输入: " << cfg.images << "\n";
    md << "- 评测参数: 置信度阈值 conf=" << cfg.conf << ", 线程数 threads=" << cfg.threads
       << "（均为提交页可调项）, 检测框按该阈值导出后离线评测, 指标 ×100\n\n";

    // 主表
    md << "## Table 1: 主要结果\n\n";
    md << "| Model |";
    for (auto& c : spec.cols) md << " " << c << " |";
    md << "\n|---";
    for (size_t i = 0; i < spec.cols.size(); ++i) md << "|---";
    md << "|\n";
    for (auto& r : runs) {
        md << "| " << r.tag << " |";
        for (auto& c : spec.cols) {
            if (c == "FPS") md << " " << fmt_fps(r.fps) << " |";
            else if (!r.ok) md << " - |";  // 评测失败：指标缺失，FPS 仍真实
            else md << " " << fmt1(metric_value(r.summary, cfg.task, c)) << " |";
        }
        md << "\n";
        if (!r.ok) md << "  （⚠ " << r.tag << " 评测失败，指标缺失，详见 run.log）\n";
    }

    // OBB 逐类表
    if (cfg.task == Task::Obb) {
        md << "\n## Table 2: 逐类 AP50\n\n| Class |";
        for (auto& r : runs) md << " " << r.tag << " |";
        md << "\n|---";
        for (size_t i = 0; i < runs.size(); ++i) md << "|---";
        md << "|\n";
        std::map<std::string, std::vector<double>> per_cls;  // class -> 每行 AP50
        for (auto& r : runs) {
            if (const JValue* pc = r.summary.find("AP50_per_class")) {
                for (auto& [cls, v] : pc->obj) per_cls[cls].push_back(v.num_or(0));
            }
        }
        for (auto& [cls, vals] : per_cls) {
            md << "| " << cls << " |";
            for (double v : vals) md << " " << fmt1(v) << " |";
            md << "\n";
        }
    }

    // conf / 线程扫描在主流程追加（见 sweep 输出文件），这里只留位置说明。

    md << "\n## Implementation Details\n\n";
    md << "- 推理: RK3588 NPU, pipeline " << cfg.threads << " 线程, letterbox 预处理\n";
    md << "- 评测: ";
    if (cfg.task == Task::Obb) {
        md << "旋转 IoU（凸多边形）, IoU=0.5:0.95 十档 101 点插值 AP, difficult 忽略, GT 相交占比>=0.5\n";
    } else {
        md << "pycocotools COCOeval, maxDets=" << (cfg.task == Task::Pose ? 20 : 100)
           << (cfg.task == Task::Pose ? ", 仅 person 类" : "") << ", imgIds=实际推理图集合\n";
    }
    md << "- 导出: 每帧检测/关键点/掩膜经后处理逆映射回原图坐标后落盘\n";

    std::ofstream f(path);
    f << md.str() << "\n";
}

static void write_summary_json(const Config& cfg, const TaskSpec& spec,
                               const std::vector<RunResult>& runs,
                               const std::vector<std::pair<int, double>>& thread_sweep,
                               const std::vector<std::pair<double, double>>& conf_sweep_res,
                               const std::string& path) {
    std::ostringstream j;
    j << "{\n  \"task\": \"" << spec.name << "\",\n  \"runs\": [\n";
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto& r = runs[i];
        j << "    {\"tag\": \"" << json_escape(r.tag) << "\", \"model\": \"" << json_escape(r.model)
          << "\", \"fps\": " << r.fps << ", \"infer_ms\": " << r.infer_ms
          << ", \"summary_file\": \"" << json_escape(r.summary_path) << "\"}"
          << (i + 1 < runs.size() ? "," : "") << "\n";
    }
    j << "  ],\n  \"threads_sweep\": [";
    for (size_t i = 0; i < thread_sweep.size(); ++i)
        j << "{\"threads\": " << thread_sweep[i].first << ", \"fps\": " << thread_sweep[i].second << "}"
          << (i + 1 < thread_sweep.size() ? ", " : "");
    j << "],\n  \"conf_sweep\": [";
    for (size_t i = 0; i < conf_sweep_res.size(); ++i)
        j << "{\"conf\": " << conf_sweep_res[i].first << ", \"main_metric\": " << conf_sweep_res[i].second << "}"
          << (i + 1 < conf_sweep_res.size() ? ", " : "");
    j << "],\n  \"config\": {\"conf\": " << cfg.conf << ", \"threads\": " << cfg.threads
      << ", \"images\": \"" << json_escape(cfg.images) << "\", \"ann\": \"" << json_escape(cfg.ann)
      << "\"}\n}\n";
    std::ofstream f(path);
    f << j.str();
}

// ---------------------------------------------------------------------------
// 主流程
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    const TaskSpec& spec = task_spec(cfg.task);

    // ---- 校验 ----
    if (cfg.model.empty() && cfg.reuse_dump.empty()) {
        std::fprintf(stderr, "[rknn_eval] 需要 --model（或 --reuse-dump）\n");
        return 2;
    }
    if (!cfg.model.empty() && !file_exists(cfg.model)) {
        std::fprintf(stderr, "[rknn_eval] 模型不存在: %s\n", cfg.model.c_str());
        return 2;
    }
    // 只有需要推理时才强制 images 含 '/'；--reuse-dump 模式可省略
    if (!cfg.reuse_dump.empty() && cfg.images.empty()) {
        // reuse 模式不推理：images 仅用于报告展示，可缺省
    } else if (!cfg.images.empty() && cfg.images.find('/') == std::string::npos) {
        std::fprintf(stderr, "[rknn_eval] --images 路径必须包含 '/'（输入源把无斜杠路径当单文件）: %s\n",
                     cfg.images.c_str());
        return 2;
    } else if (cfg.images.empty()) {
        std::fprintf(stderr, "[rknn_eval] 需要 --images（图片目录，须含 '/'）\n");
        return 2;
    }
    if (cfg.dump_only) {
        // 远程模式：标注/评测脚本在 PC 端，板端只推理导出
    } else if (cfg.task != Task::Obb) {
        if (cfg.ann.empty() || !file_exists(cfg.ann)) {
            std::fprintf(stderr, "[rknn_eval] --ann 不存在: %s\n", cfg.ann.c_str());
            return 2;
        }
        if (cfg.task == Task::Pose) {
            // 内容校验而非文件名：COCO 关键点标注的 annotation/category 带
            // "keypoints" 字段，instances 标注没有；网页上传等流程可能把文件
            // 存成 ann.json，按文件名判断会误伤合法标注。
            if (!file_contains_token(cfg.ann, "\"keypoints\"")) {
                std::fprintf(stderr,
                             "[rknn_eval] pose 任务的标注不含 keypoints 字段（%s）——"
                             "请用 person_keypoints 类 COCO 标注，而不是 instances 标注\n",
                             cfg.ann.c_str());
                return 2;
            }
        }
    } else if (cfg.gt.empty() && cfg.dota_labels.empty()) {
        std::fprintf(stderr, "[rknn_eval] obb 需要 --gt <patches_gt.jsonl> 或 --dota-labels <labelTxt目录>\n");
        return 2;
    }
    std::string scripts = resolve_scripts_dir(cfg);
    if (!cfg.dump_only && !file_exists(scripts + "/" + std::string(spec.evaluator))) {
        std::fprintf(stderr, "[rknn_eval] 评测脚本缺失: %s/%s（可用 --scripts-dir 或 RK_EVAL_SCRIPTS 指定）\n",
                     scripts.c_str(), spec.evaluator);
        return 2;
    }
    ::mkdirs(cfg.out_dir);

    if (cfg.dry_run) {
        std::printf("[dry-run] task=%s scripts=%s out=%s\n", spec.name, scripts.c_str(), cfg.out_dir.c_str());
        return 0;
    }

    // ---- obb 数据准备：必要时切片 ----
    if (cfg.task == Task::Obb && cfg.gt.empty()) {
        std::string gt = cfg.out_dir + "/patches_gt.jsonl";
        std::string patches = cfg.out_dir + "/patches";
        std::printf("[rknn_eval] DOTA 切片: %s + %s -> %s\n", cfg.images.c_str(),
                    cfg.dota_labels.c_str(), patches.c_str());
        std::string cmd = "python3 " + shq(scripts + "/split_dota.py") + " --images " + shq(cfg.images) +
                          " --labels " + shq(cfg.dota_labels) + " --out " + shq(patches) +
                          " --gt " + shq(gt);
        std::string log = exec_capture(cmd + " 2>&1");
        std::printf("%s", log.c_str());
        if (!file_exists(gt)) {
            std::fprintf(stderr, "[rknn_eval] 切片失败\n");
            return 3;
        }
        cfg.gt = gt;
        cfg.images = patches;
    }

    // ---- 逐模型：推理 + 评测 ----
    std::vector<RunResult> runs;
    struct Job { std::string tag, model, dump_override; };
    std::vector<Job> jobs;
    if (!cfg.reuse_dump.empty()) {
        jobs.push_back({cfg.name, "", ""});
        // --compare tag=dump:PATH：评测已拉回的 dump（远程模式 PC 端汇总），不推理
        for (auto& [tag, m] : cfg.compare) {
            if (m.rfind("dump:", 0) == 0) jobs.push_back({tag, "", m.substr(5)});
        }
    } else {
        jobs.push_back({cfg.name, cfg.model, ""});
        for (auto& [tag, m] : cfg.compare) jobs.push_back({tag, m, ""});
    }

    for (auto& job : jobs) {
        RunResult r;
        r.tag = job.tag;
        r.model = job.model;
        r.dump = !job.dump_override.empty() ? job.dump_override
                 : !cfg.reuse_dump.empty()  ? cfg.reuse_dump
                                            : cfg.out_dir + "/dump_" + job.tag + ".jsonl";
        r.summary_path = cfg.out_dir + "/summary_" + job.tag + ".json";
        std::string metrics = cfg.out_dir + "/metrics_" + job.tag + ".json";
        if (!cfg.reuse_dump.empty() || job.model.empty()) {
            // reuse 模式无 metrics，读同名旧 metrics（如有）
            if (file_exists(metrics)) {
                try {
                    JValue m = mini_json::parse(read_file(metrics));
                    if (const JValue* rp = m.find("rk_pipe")) {
                        if (const JValue* pl = rp->find("pipeline"))
                            r.fps = pl->find("avg_fps") ? pl->find("avg_fps")->num_or(0) : 0;
                        if (const JValue* st = rp->find("stage_avg_ms")) {
                            r.infer_ms = st->find("inference") ? st->find("inference")->num_or(0) : 0;
                            r.pre_ms = st->find("preprocess") ? st->find("preprocess")->num_or(0) : 0;
                        }
                    }
                } catch (const std::exception&) {
                }
            }
        } else {
            std::printf("\n[rknn_eval] ===== 推理: %s (%s) =====\n", job.tag.c_str(), job.model.c_str());
            if (!run_inference(cfg, job.model, job.tag, r.dump, metrics, &r)) {
                return 3;
            }
        }
        if (cfg.dump_only) {
            // 远程模式：只推理导出 dump+metrics，评测/报告由 PC 端完成
            std::printf("[rknn_eval][dump-only] %s done: fps=%.1f infer=%.1fms dump=%s\n",
                        job.tag.c_str(), r.fps, r.infer_ms, r.dump.c_str());
            runs.push_back(std::move(r));
            continue;
        }
        std::printf("[rknn_eval] 评测: %s\n", job.tag.c_str());
        std::string log;
        r.ok = run_evaluator(cfg, spec, scripts, r.dump, r.summary_path, cfg.conf, &r.summary, &log);
        if (!r.ok) {
            // 单模型评测失败（如类别数填错/标注问题）不应拖垮其他模型的对比
            // 报告：标记后继续评下一个，报告里该行指标显示缺失（166e7c97 教训）。
            std::printf("[rknn_eval] ⚠ %s 评测失败，跳过（其余模型继续出报告）\n", job.tag.c_str());
        }
        // 摘要行
        std::printf("[rknn_eval] %s done: fps=%.1f infer=%.1fms\n", job.tag.c_str(), r.fps, r.infer_ms);
        runs.push_back(std::move(r));
    }

    // 全部模型评测失败 → 报告没有任何可用指标，按失败退出；部分失败则继续出报告
    {
        bool any_eval_ok = false;
        for (auto& r : runs) any_eval_ok = r.ok || any_eval_ok;
        if (!runs.empty() && !any_eval_ok) {
            std::fprintf(stderr, "[rknn_eval] 所有模型评测均失败，退出\n");
            return 3;
        }
    }

    if (cfg.dump_only) {
        // dump-only 也允许 --vis-dir（板端本地出图场景）
        if (!cfg.vis_dir.empty()) {
            render_vis(cfg, spec, runs[0].dump, cfg.vis_dir, cfg.vis_sample);
        }
        std::printf("\n[rknn_eval] dump-only 完成: %d 个模型导出于 %s\n",
                    (int)runs.size(), cfg.out_dir.c_str());
        return 0;
    }

    // ---- conf 扫描（主模型 dump 离线重评，不重推理）----
    std::vector<std::pair<double, double>> conf_sweep_res;
    for (double c : cfg.conf_sweep) {
        std::string sp = cfg.out_dir + "/summary_conf" + std::to_string(c).substr(0, 5) + ".json";
        JValue s;
        std::string log;
        std::printf("[rknn_eval] conf 扫描: %.3f\n", c);
        if (!run_evaluator(cfg, spec, scripts, runs[0].dump, sp, c, &s, &log)) continue;
        double v = 0;
        if (cfg.task == Task::Obb) {
            if (const JValue* m = s.find("mAP50")) v = m->num_or(0);
        } else if (const JValue* m = s.find("AP")) {
            v = m->num_or(0);
        }
        conf_sweep_res.emplace_back(c, v);
    }

    // ---- 线程扫描（仅主模型重推理测 FPS，精度不变）----
    std::vector<std::pair<int, double>> thread_sweep;
    for (int t : cfg.threads_sweep) {
        std::printf("[rknn_eval] 线程扫描: %d\n", t);
        Config tc = cfg;
        tc.threads = t;
        RunResult r;
        std::string dump = "/dev/null";  // 扫描只测速，不落盘
        std::string metrics = cfg.out_dir + "/metrics_threads" + std::to_string(t) + ".json";
        if (run_inference(tc, jobs[0].model, jobs[0].tag, dump, metrics, &r))
            thread_sweep.emplace_back(t, r.fps);
    }

    // ---- 可视化（主模型 dump -> 检测结果图；vis-sample >0 时抽帧）----
    if (!cfg.vis_dir.empty()) {
        render_vis(cfg, spec, runs[0].dump, cfg.vis_dir, cfg.vis_sample);
    }

    // ---- 报告 ----
    std::string report_md = cfg.out_dir + "/report.md";
    write_report(cfg, spec, runs, report_md);
    write_summary_json(cfg, spec, runs, thread_sweep, conf_sweep_res, cfg.out_dir + "/summary_all.json");

    // 逐类 CSV（COCO 系任务）
    if (cfg.task != Task::Obb) {
        std::ofstream csv(cfg.out_dir + "/per_class.csv");
        csv << "category_id,name,AP,AP50\n";
        for (auto& r : runs) {
            const JValue* pc = r.summary.find("per_class");
            if (!pc) continue;
            for (auto& [cid, v] : pc->obj) {
                std::string name;
                if (const JValue* n = v.find("name")) name = n->str_or("");
                double ap = v.find("AP") ? v.find("AP")->num_or(0) : 0;
                double ap50 = v.find("AP50") ? v.find("AP50")->num_or(0) : 0;
                csv << cid << "," << name << "," << ap << "," << ap50 << "\n";
            }
        }
    }

    // ---- 断言 ----
    int exit_code = 0;
    const bool main_ok = !runs.empty() && runs[0].ok;
    double main_metric = (cfg.task == Task::Obb)
                             ? (main_ok && runs[0].summary.find("mAP50") ? runs[0].summary.find("mAP50")->num_or(0) : 0)
                             : (main_ok && runs[0].summary.find("AP") ? runs[0].summary.find("AP")->num_or(0) : 0);
    double main_ap50 = (cfg.task == Task::Obb)
                           ? main_metric
                           : (main_ok && runs[0].summary.find("AP50") ? runs[0].summary.find("AP50")->num_or(0) : 0);
    if (cfg.assert_ap > 0 && main_ok && main_metric * 100.0 < cfg.assert_ap) {
        std::fprintf(stderr, "[rknn_eval][ASSERT] 主指标 %.2f < 阈值 %.2f\n",
                     main_metric * 100.0, cfg.assert_ap);
        exit_code = 1;
    }
    if (cfg.assert_ap50 > 0 && main_ok && main_ap50 * 100.0 < cfg.assert_ap50) {
        std::fprintf(stderr, "[rknn_eval][ASSERT] AP50 %.2f < 阈值 %.2f\n",
                     main_ap50 * 100.0, cfg.assert_ap50);
        exit_code = 1;
    }

    int n_eval_fail = 0;
    for (auto& r : runs) if (!r.ok) ++n_eval_fail;
    std::printf("\n[rknn_eval] 完成: %s | 主指标 %.2f | 报告 %s%s\n",
                report_md.c_str(), main_metric * 100.0, report_md.c_str(),
                n_eval_fail ? "（⚠ 有模型评测失败，见 run.log）" : "");
    return exit_code;
}
