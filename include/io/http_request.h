#pragma once

// HTTP 请求文本解析纯函数（web 预览服务 / daemon REST 共用，无 socket 依赖、可单测）。

#include <map>
#include <string>

// URL 解码：%XX 十六进制转义 + '+' → 空格；非法转义原样保留
std::string urlDecode(const std::string& s);

// query string 解析（"a=1&b=2"）；无 '=' 的段忽略；键值均经 URL 解码
std::map<std::string, std::string> parseQuery(const std::string& query);

// 请求行方法（无空格时默认 GET）
std::string httpRequestMethod(const std::string& request);

// 请求行 path（剥离 query string；解析失败返回 "/"）
std::string httpRequestPath(const std::string& request);

// body（"\r\n\r\n" 之后的部分；无则空串）
std::string httpRequestBody(const std::string& request);

// Content-Length 头解析；缺失/非法返回 -1
long httpContentLength(const std::string& request);
