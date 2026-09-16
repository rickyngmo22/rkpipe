// llm_report：工业质检离线报告工具（LLM PoC 附属）
//
// 输入：
//   --detections <file>   逐帧结果 JSONL（schema v1，result_jsonl_path / docs/event_payload.md）
//   --llm-results <file>  LLM 复检结论 JSONL（LlmConfig.results_path 落盘）
// 输出：markdown 质检报告（-o 指定；缺省打印 stdout）。
// 报告末节可调 LLM 生成叙述性分析（--no-llm 跳过；provider 由 --config/--provider 决定）。
//
// 独立小工具：只链 llm 模块纯逻辑源，不依赖 RKNN/闭源核心，x86/板端均可构建。

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "llm/llm_config.h"
#include "llm/llm_provider.h"

namespace {

struct DetectionStats {
    long long lines = 0;              // JSONL 行数（= 帧数）
    long long frames_with_dets = 0;   // 有检出的帧数
    long long total_dets = 0;
    std::map<std::string, long long> by_class;      // label/cls -> 计数
    std::map<std::string, long long> by_source;     // source -> 计数
    double score_sum = 0.0;
    double score_min = 1.0;
    double score_max = 0.0;
    long long borderline = 0;         // score < 0.5 的检出数（复检重点关注区）

    void addDet(const nlohmann::json& det) {
        ++total_dets;
        std::string label = "class_" + std::to_string(det.value("cls", -1));
        if (det.contains("label") && det["label"].is_string() && !det["label"].get<std::string>().empty()
            && det["label"].get<std::string>() != "null") {
            label = det["label"].get<std::string>();
        }
        ++by_class[label];
        const double score = det.value("score", 0.0);
        score_sum += score;
        score_min = std::min(score_min, score);
        score_max = std::max(score_max, score);
        if (score < 0.5) {
            ++borderline;
        }
    }
};

struct VerdictStats {
    long long ok = 0;
    long long failed = 0;
    std::map<std::string, long long> by_verdict;  // real_defect/false_positive/uncertain
    std::vector<std::string> recent_descriptions;
};

// 逐行容错解析（同 result_jsonl 语义：意外终止可能丢尾部，逐行独立）
std::vector<nlohmann::json> readJsonlLines(const std::string& path) {
    std::vector<nlohmann::json> lines;
    std::ifstream in(path);
    if (!in.is_open()) {
        return lines;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        nlohmann::json parsed = nlohmann::json::parse(line, nullptr, false);
        if (!parsed.is_discarded() && parsed.is_object()) {
            lines.push_back(std::move(parsed));
        }
    }
    return lines;
}

DetectionStats aggregateDetections(const std::vector<nlohmann::json>& lines) {
    DetectionStats stats;
    for (const nlohmann::json& frame : lines) {
        ++stats.lines;
        if (frame.value("type", "") != "detect") {
            continue;
        }
        const nlohmann::json dets = frame.value("dets", nlohmann::json::array());
        if (!dets.empty()) {
            ++stats.frames_with_dets;
        }
        for (const nlohmann::json& det : dets) {
            stats.addDet(det);
        }
        const std::string source = frame.value("source", "");
        if (!source.empty()) {
            ++stats.by_source[source];
        }
    }
    return stats;
}

VerdictStats aggregateVerdicts(const std::vector<nlohmann::json>& records) {
    VerdictStats stats;
    for (const nlohmann::json& record : records) {
        if (!record.value("ok", false)) {
            ++stats.failed;
            continue;
        }
        ++stats.ok;
        // verdict 字段优先按 JSON 解析（云端 VLM 约定格式）；板端小模型输出自然语言
        // （"结论：真实缺陷/疑似误报"），按关键词归类兜底
        nlohmann::json verdict = nlohmann::json::parse(record.value("verdict", ""), nullptr, false);
        const std::string raw = record.value("verdict", "");
        std::string kind = "unparsed";
        if (!verdict.is_discarded() && verdict.is_object() && verdict.contains("verdict") &&
            verdict["verdict"].is_string()) {
            kind = verdict["verdict"].get<std::string>();
        } else if (raw.find("误报") != std::string::npos) {
            kind = "false_positive";
        } else if (raw.find("缺陷") != std::string::npos) {
            kind = "real_defect";
        } else if (!raw.empty()) {
            kind = "text";
        }
        ++stats.by_verdict[kind];
        if (verdict.is_object() && verdict.contains("description") && verdict["description"].is_string()) {
            stats.recent_descriptions.push_back(verdict["description"].get<std::string>());
        } else if (!raw.empty() && raw.size() < 300) {
            stats.recent_descriptions.push_back(raw);
        }
    }
    while (stats.recent_descriptions.size() > 10) {
        stats.recent_descriptions.erase(stats.recent_descriptions.begin());
    }
    return stats;
}

std::string statsToMarkdown(const DetectionStats& det, const VerdictStats& verdict,
                            const std::string& detections_path, const std::string& llm_results_path) {
    std::ostringstream md;
    md << "# 工业质检报告\n\n"
       << "- 检测数据: " << (detections_path.empty() ? "（未提供）" : detections_path) << "\n"
       << "- 复检数据: " << (llm_results_path.empty() ? "（未提供）" : llm_results_path) << "\n\n";

    md << "## 检出统计\n\n";
    if (det.total_dets == 0) {
        md << "无检出记录。\n";
    } else {
        md << "| 指标 | 数值 |\n|---|---|\n"
           << "| 总帧数 | " << det.lines << " |\n"
           << "| 有检出帧数 | " << det.frames_with_dets << " |\n"
           << "| 检出总数 | " << det.total_dets << " |\n"
           << "| 平均置信度 | " << (det.score_sum / det.total_dets) << " |\n"
           << "| 置信度范围 | [" << det.score_min << ", " << det.score_max << "] |\n"
           << "| 低置信度(<0.5)检出 | " << det.borderline << " |\n\n";
        md << "### 按类别\n\n| 类别 | 数量 |\n|---|---|\n";
        for (const auto& kv : det.by_class) {
            md << "| " << kv.first << " | " << kv.second << " |\n";
        }
        if (det.by_source.size() > 1) {
            md << "\n### 按数据源\n\n| 源 | 检出数 |\n|---|---|\n";
            for (const auto& kv : det.by_source) {
                md << "| " << kv.first << " | " << kv.second << " |\n";
            }
        }
        md << "\n";
    }

    md << "## 复检结论（LLM）\n\n";
    if (verdict.ok == 0 && verdict.failed == 0) {
        md << "无复检记录（未启用 LLM 分析或未提供 --llm-results）。\n";
    } else {
        md << "- 复检调用: " << (verdict.ok + verdict.failed) << " 次（成功 " << verdict.ok
           << "，失败 " << verdict.failed << "）\n\n";
        if (!verdict.by_verdict.empty()) {
            md << "| 结论 | 次数 |\n|---|---|\n";
            for (const auto& kv : verdict.by_verdict) {
                md << "| " << kv.first << " | " << kv.second << " |\n";
            }
            md << "\n";
        }
        if (!verdict.recent_descriptions.empty()) {
            md << "### 近期说明摘要\n\n";
            for (const std::string& desc : verdict.recent_descriptions) {
                md << "- " << desc << "\n";
            }
            md << "\n";
        }
    }
    return md.str();
}

std::string buildLlmPrompt(const DetectionStats& det, const VerdictStats& verdict) {
    nlohmann::json by_class = nlohmann::json::object();
    for (const auto& kv : det.by_class) {
        by_class[kv.first] = kv.second;
    }
    nlohmann::json context = {
        {"total_frames", det.lines},
        {"frames_with_detections", det.frames_with_dets},
        {"total_detections", det.total_dets},
        {"avg_score", det.total_dets ? det.score_sum / det.total_dets : 0.0},
        {"borderline_count", det.borderline},
        {"by_class", by_class},
        {"recheck", {{"ok", verdict.ok}, {"failed", verdict.failed}, {"by_verdict", verdict.by_verdict}}},
    };
    return
        "以下是工业缺陷检测系统一个批次的检出与复检统计(JSON)。请生成 markdown 格式的质量分析"
        "（中文），包含：1) 整体质量状况评估；2) 各缺陷类别的分布解读与可能的工艺原因假设；"
        "3) 低置信度检出占比是否需要人工复核的建议；4) 后续改进建议。直接输出 markdown 正文，"
        "不要重复原始数据。\n\n统计 JSON：\n" + context.dump();
}

void printUsage(const char* program) {
    std::printf(
        "用法: %s [选项]\n"
        "  --detections <file>   逐帧结果 JSONL（schema v1）\n"
        "  --llm-results <file>  LLM 复检结论 JSONL\n"
        "  --config <llm.yaml>   LLM 配置（缺省按 RKPIPE_LLM_CONFIG / ./llm.yaml 发现）\n"
        "  --provider <name>     覆盖配置中的 provider（mock|cloud|rkllm）\n"
        "  --no-llm              只汇总统计，不调用大模型\n"
        "  --title <text>        报告标题后缀\n"
        "  -o <file>             输出文件（缺省 stdout）\n",
        program);
}

}  // namespace

int main(int argc, char** argv) {
    std::string detections_path, llm_results_path, config_path, provider_override, output_path, title_suffix;
    bool no_llm = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg == "--detections") detections_path = next();
        else if (arg == "--llm-results") llm_results_path = next();
        else if (arg == "--config") config_path = next();
        else if (arg == "--provider") provider_override = next();
        else if (arg == "--no-llm") no_llm = true;
        else if (arg == "--title") title_suffix = next();
        else if (arg == "-o" || arg == "--output") output_path = next();
        else { printUsage(argv[0]); return arg == "-h" || arg == "--help" ? 0 : 1; }
    }

    if (detections_path.empty() && llm_results_path.empty()) {
        std::fprintf(stderr, "至少提供 --detections 或 --llm-results 之一\n\n");
        printUsage(argv[0]);
        return 1;
    }

    const DetectionStats det = aggregateDetections(readJsonlLines(detections_path));
    const VerdictStats verdict = aggregateVerdicts(readJsonlLines(llm_results_path));

    std::string report = statsToMarkdown(det, verdict, detections_path, llm_results_path);
    if (!title_suffix.empty()) {
        report.insert(report.find("\n"), " — " + title_suffix);
    }

    if (!no_llm) {
        LlmConfig config = config_path.empty() ? LlmConfig::fromDefaultLocation()
                                               : ([&]() {
                                                     LlmConfig c;
                                                     c.loadFromFile(config_path);
                                                     return c;
                                                 })();
        if (!provider_override.empty()) {
            config.provider = provider_override;
        }
        if (!config.enabled) {
            // 报告工具默认宽松：配置文件缺失也允许用 mock 出报告结构
            config.enabled = true;
            if (config.provider.empty()) {
                config.provider = "mock";
            }
        }
        std::printf("[llm_report] 生成分析段落: provider=%s model=%s\n", config.provider.c_str(),
                    config.model.c_str());
        std::unique_ptr<llm::LlmProvider> provider(llm::createProvider(config));
        llm::LlmRequest request;
        request.system =
            "你是一名工业质量工程师，基于质检统计数据撰写简洁、专业的分析报告。";
        request.prompt = buildLlmPrompt(det, verdict);
        const llm::LlmResponse response = provider->generate(request);
        report += "## 大模型分析（provider=" + config.provider + "）\n\n";
        report += response.ok ? response.text + "\n"
                              : "> 生成失败: " + response.error + "\n";
    }

    if (output_path.empty()) {
        std::fputs(report.c_str(), stdout);
    } else {
        std::ofstream out(output_path);
        if (!out.is_open()) {
            std::fprintf(stderr, "无法写输出文件: %s\n", output_path.c_str());
            return 1;
        }
        out << report;
        std::printf("[llm_report] 报告已写入 %s\n", output_path.c_str());
    }
    return 0;
}
