#include "utils/http_post.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <cstdlib>
#include <cstdio>
#include <sstream>

namespace {

// 解析状态行 "HTTP/1.1 200 OK" → 200；读不到返回 0
long parseStatusCode(const std::string& response) {
    if (response.rfind("HTTP/", 0) != 0) {
        return 0;
    }
    const size_t sp = response.find(' ');
    if (sp == std::string::npos) {
        return 0;
    }
    return std::strtol(response.c_str() + sp + 1, nullptr, 10);
}

}  // namespace

bool httpPost(const std::string& host, int port, const std::string& path,
              const std::string& body, const std::string& content_type,
              int timeout_ms, long* status_code) {
    if (status_code) {
        *status_code = 0;
    }
    if (host.empty() || port <= 0 || path.empty()) {
        return false;
    }
    // getaddrinfo：兼容域名与 IPv4/IPv6 点分地址
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* list = nullptr;
    if (::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &list) != 0 || !list) {
        return false;
    }
    // connect + send 全程受 deadline 约束：黑洞地址 timeout_ms 内放弃（阻塞式可达 ~2min）
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    int fd = -1;
    for (addrinfo* ai = list; ai != nullptr && fd < 0; ai = ai->ai_next) {
        const int cand = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (cand < 0) {
            continue;
        }
        const int flags = ::fcntl(cand, F_GETFL, 0);
        ::fcntl(cand, F_SETFL, flags | O_NONBLOCK);
        const int cr = ::connect(cand, ai->ai_addr, ai->ai_addrlen);
        bool connected = cr == 0;
        if (cr != 0 && errno == EINPROGRESS) {
            while (!connected) {
                const auto left_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                         deadline - std::chrono::steady_clock::now())
                                         .count();
                if (left_ms <= 0) {
                    break;
                }
                pollfd pfd{cand, POLLOUT, 0};
                if (::poll(&pfd, 1, static_cast<int>(left_ms)) <= 0) {
                    break;  // 超时或 poll 出错
                }
                int so_err = 0;
                socklen_t len = sizeof(so_err);
                ::getsockopt(cand, SOL_SOCKET, SO_ERROR, &so_err, &len);
                if (so_err == EINPROGRESS) {
                    continue;
                }
                connected = so_err == 0;
            }
        }
        if (!connected) {
            ::close(cand);
            continue;
        }
        // 恢复阻塞模式，剩余超时预算作为发送超时
        ::fcntl(cand, F_SETFL, flags);
        const auto left_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 deadline - std::chrono::steady_clock::now())
                                 .count();
        if (left_ms <= 0) {
            ::close(cand);
            continue;
        }
        timeval tv{};
        tv.tv_sec = static_cast<time_t>(left_ms / 1000);
        tv.tv_usec = static_cast<suseconds_t>((left_ms % 1000) * 1000);
        ::setsockopt(cand, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        fd = cand;
    }
    ::freeaddrinfo(list);
    if (fd < 0) {
        return false;
    }
    std::ostringstream request;
    request << "POST " << path << " HTTP/1.1\r\n"
            << "Host: " << host << ":" << port << "\r\n"
            << "Content-Type: " << content_type << "\r\n"
            << "Content-Length: " << body.size() << "\r\n"
            << "Connection: close\r\n\r\n"
            << body;
    const std::string req = request.str();
    const ssize_t sent = ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
    // 读回状态行受剩余预算约束（对端不响应时不得永久阻塞）
    const auto read_left_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  deadline - std::chrono::steady_clock::now())
                                  .count();
    timeval rtv{};
    rtv.tv_sec = static_cast<time_t>(std::max<long>(read_left_ms, 0) / 1000);
    rtv.tv_usec = static_cast<suseconds_t>((std::max<long>(read_left_ms, 0) % 1000) * 1000);
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rtv, sizeof(rtv));
    // 尽力读回状态行（对端 Connection: close 需要读完才不会收到 RST）
    std::string response;
    char buf[1024];
    while (response.size() < 4096) {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        response.append(buf, static_cast<std::size_t>(n));
        if (response.find("\r\n") != std::string::npos) {
            break;  // 状态行已到手即可
        }
    }
    ::close(fd);
    if (status_code) {
        *status_code = parseStatusCode(response);
    }
    return sent == static_cast<ssize_t>(req.size());
}

#ifdef RK_PIPE_HAVE_CURL
#include <curl/curl.h>

namespace {
std::once_flag g_curl_post_once;

size_t curlAppend(void* ptr, size_t size, size_t nmemb, void* userdata) {
    static_cast<std::string*>(userdata)->append(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}
}  // namespace

// https POST：libcurl 直连（TLS 由 curl 处理；证书校验按 curl 默认开启）
static bool httpPostCurl(const std::string& url, const std::string& body,
                         const std::string& content_type, int timeout_ms, long* status_code) {
    std::call_once(g_curl_post_once, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }
    std::string response;
    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, ("Content-Type: " + content_type).c_str());
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlAppend);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 5000L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    const CURLcode code = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (status_code) {
        *status_code = code == CURLE_OK ? http_code : 0;
    }
    return code == CURLE_OK;
}
#endif

bool httpPostUrl(const std::string& url, const std::string& body,
                 const std::string& content_type, int timeout_ms,
                 long* status_code) {
    if (status_code) {
        *status_code = 0;
    }
    if (url.rfind("https://", 0) == 0) {
#ifdef RK_PIPE_HAVE_CURL
        return httpPostCurl(url, body, content_type, timeout_ms, status_code);
#else
        std::fprintf(stderr, "[http] https webhook unavailable: built without libcurl\n");
        return false;
#endif
    }
    // http://（或无前缀）→ socket 实现
    std::string rest = url;
    if (rest.rfind("http://", 0) == 0) {
        rest = rest.substr(7);
    }
    const std::size_t slash = rest.find('/');
    const std::string host_port = slash == std::string::npos ? rest : rest.substr(0, slash);
    const std::string path = slash == std::string::npos ? "/" : rest.substr(slash);
    std::string host = host_port;
    int port = 80;
    const std::size_t colon = host_port.rfind(':');
    if (colon != std::string::npos && host_port.find(']') == std::string::npos) {
        host = host_port.substr(0, colon);
        port = std::atoi(host_port.substr(colon + 1).c_str());
    }
    return httpPost(host, port, path, body, content_type, timeout_ms, status_code);
}
