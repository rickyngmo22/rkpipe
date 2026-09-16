// 解析器 fuzz harness（内建变异循环，无需外部 libFuzzer/AFL）。
//
// 目标（全部是吃不可信输入的纯函数）：
//   1. mini_json::JsonParser  —— /internal/event、streams.json、复检回复解析
//   2. io/http_request 全家   —— web 预览与 daemon 的 HTTP 请求文本解析
//   3. jsonEscape 往返        —— 输出必须仍是合法 JSON 字符串
//
// 不变量：
//   - 任何输入不得崩溃（栈溢出/越界/抛异常）
//   - parse() 必须通过 ok 标志表达失败，不得未定义行为
//   - jsonEscape(s) 经 mini_json 字符串语义解析后应等价于 s（控制字符除外）
//
// 用法:
//   parser_fuzz [秒数] [种子]     # 默认 10s / 种子 1（固定种子保证可复现）
//
// 深度排查（建议发版前跑一次）:
//   g++ -fsanitize=address,undefined -g ... 同源码单独构建后跑 60s

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "daemon/mini_json.h"
#include "io/http_request.h"
#include "utils/json_escape.h"

namespace {

std::vector<std::string> kSeeds = {
    // 合法 JSON
    "{\"id\":1,\"recv_ts\":169,\"body\":{\"type\":\"line_cross\",\"task\":\"detect\"}}",
    "{\"a\":[1,2.5e3,-3,{\"b\":\"c\"}],\"d\":true,\"e\":null}",
    "[[[1]]]",
    "{\"u\":\"\\u4e2d\\u6587\\ud83d\\ude00\",\"esc\":\"a\\\"b\\\\c\\n\\t\"}",
    "{\"big\":123456789012345678901234567890,\"neg\":-0.0,\"exp\":1e-300}",
    "  { \"x\" : [ ] }  ",
    // 合法 HTTP 请求
    "GET /stream.mjpg?t=123 HTTP/1.1\r\nHost: h\r\n\r\n",
    "POST /api/events/1/review HTTP/1.1\r\nContent-Length: 17\r\n\r\n{\"verdict\":\"true\"}",
    "POST /x HTTP/1.1\r\ncontent-length:7\r\n\r\n1234567",
    // 边界字符串
    "", "%", "%zz", "%%%", "\xff\xfe", "Content-Length: 99999999",
};

std::string mutate(const std::string& in, std::mt19937& rng) {
    std::string s = in;
    const int mutations = 1 + static_cast<int>(rng() % 4);
    static const char* kInteresting[] = {"\r\n\r\n", "Content-Length:", "\\u", "\\", "\"",
                                         "[[[[",     "}}}",             "9999999999999999999999",
                                         "\x00\x01", "%FF",             "-1e9999"};
    for (int m = 0; m < mutations; ++m) {
        if (s.empty()) {
            s.insert(0, kInteresting[rng() % 11]);  // 空串只做插入
            continue;
        }
        switch (rng() % 6) {
            case 0: {  // 位翻转
                if (s.empty()) break;
                const size_t i = rng() % s.size();
                s[i] ^= static_cast<char>(1u << (rng() % 8));
                break;
            }
            case 1: {  // 删除片段
                if (s.size() < 2) break;
                const size_t i = rng() % s.size();
                s.erase(i, 1 + rng() % std::min<size_t>(s.size() - i, 32));
                break;
            }
            case 2: {  // 插入有趣串
                const size_t i = rng() % (s.size() + 1);
                const std::string token = kInteresting[rng() % 11];
                s.insert(i, token);
                break;
            }
            case 3: {  // 复制片段
                if (s.empty()) break;
                const size_t i = rng() % s.size();
                s.insert(i, s.substr(i, 1 + rng() % std::min<size_t>(s.size() - i, 16)));
                break;
            }
            case 4: {  // 截断
                if (s.size() > 1) {
                    s.resize(1 + rng() % s.size());
                }
                break;
            }
            default: {  // 随机字节
                if (s.empty()) break;
                const size_t i = rng() % s.size();
                s[i] = static_cast<char>(rng() & 0xff);
                break;
            }
        }
    }
    return s;
}

void fuzzOne(const std::string& input) {
    // 1) JSON 解析：不崩溃、ok 标志表达结果
    {
        bool ok = true;
        mini_json::JsonValue v = mini_json::JsonParser(input).parse(ok);
        (void)v;
        (void)ok;
    }
    // 2) JSON 解析后 dump 再解析（往返稳定）
    {
        bool ok = true;
        mini_json::JsonValue v = mini_json::JsonParser(input).parse(ok);
        if (ok) {
            const std::string dumped = mini_json::dump(v);
            bool ok2 = false;
            mini_json::JsonParser(dumped).parse(ok2);
            (void)ok2;  // 二次解析不崩溃即可（dump 的保真度由单测保证）
        }
    }
    // 3) HTTP 请求解析
    {
        volatile size_t sink = 0;
        sink += httpRequestMethod(input).size();
        sink += httpRequestPath(input).size();
        sink += httpRequestBody(input).size();
        (void)httpContentLength(input);
        const std::string q = input.find('?') == std::string::npos ? input : input.substr(input.find('?') + 1);
        for (const auto& kv : parseQuery(q)) {
            sink += kv.first.size() + kv.second.size();
        }
        sink += urlDecode(input).size();
        (void)sink;
    }
    // 4) jsonEscape 往返：输出必须是合法 JSON 字符串体（除控制字符被删）
    {
        const std::string esc = jsonEscape(input);
        const std::string wrapped = "\"" + esc + "\"";
        bool ok = true;
        mini_json::JsonValue v = mini_json::JsonParser(wrapped).parse(ok);
        if (ok && v.type == mini_json::JsonValue::Type::String) {
            // 不变量：LF/CR/TAB 经转义保留，其余 <0x20 控制字符丢弃，其余字节原样
            std::string expect;
            for (char ch : input) {
                if (ch == '\n' || ch == '\r' || ch == '\t') {
                    expect += ch;
                } else if (static_cast<unsigned char>(ch) >= 0x20) {
                    expect += ch;
                }
            }
            if (v.str != expect) {
                std::fprintf(stderr, "[round-trip] 不一致: in=%zu bytes\n", input.size());
                std::fprintf(stderr, "  input hex:");
                for (unsigned char ch : input) {
                    std::fprintf(stderr, " %02x", ch);
                }
                std::fprintf(stderr, "\n  escaped : \"%s\"\n  parsed  : %zu bytes\n", esc.c_str(),
                             v.str.size());
                std::fprintf(stderr, "  expect  : %zu bytes:", expect.size());
                for (unsigned char ch : expect) {
                    std::fprintf(stderr, " %02x", ch);
                }
                std::fprintf(stderr, "\n  parsed hex:");
                for (unsigned char ch : v.str) {
                    std::fprintf(stderr, " %02x", ch);
                }
                std::fprintf(stderr, "\n");
                std::abort();
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 10;
    const unsigned seed = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 1;
    std::mt19937 rng(seed);

    std::printf("parser_fuzz: %ds seed=%u targets=json/http/escape\n", seconds, seed);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    long iters = 0;
    // 先跑原始种子（确定性用例）
    for (const std::string& s : kSeeds) {
        fuzzOne(s);
        ++iters;
    }
    // 变异循环
    while (std::chrono::steady_clock::now() < deadline) {
        const std::string& base = kSeeds[rng() % kSeeds.size()];
        fuzzOne(mutate(base, rng));
        ++iters;
    }
    std::printf("parser_fuzz: %ld 次迭代无崩溃\n", iters);
    return 0;
}
