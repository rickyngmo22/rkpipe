#include "../../include/io/web_preview_server.h"

#include <arpa/inet.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <netinet/in.h>
#include <poll.h>
#include <sstream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <turbojpeg.h>

#include "core/runtime_affinity.h"

namespace {

constexpr const char* kBoundary = "frame";

// 兼容旧调试开关：RK_PIPE_REPLAY_PACE=1 时仍按原视频帧率预览
bool replayPaceEnvEnabled() {
    const char* v = std::getenv("RK_PIPE_REPLAY_PACE");
    return v && *v && std::string(v) != "0";
}

std::string jsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 16);
    for (char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

// ---- 访问令牌鉴权(RK_PIPE_WEB_TOKEN;FAQ 曾自认"明文 HTTP、无鉴权") ----------------
// 环境变量非空时启用:除 /healthz(探针用,无信息泄露)外所有路径要求令牌,来源二选一:
//   ?token=<t> 查询参数(浏览器 <img>/fetch 直接可用) 或 X-Auth-Token 请求头。
// Config 为闭源核心按布局填充的类型,不可加字段,令牌经环境变量下发。
// 令牌建议使用 URL 安全字符(A-Za-z0-9._-~);嵌入页面时做白名单净化。

const char* webToken() {
    // 逐次读取:env 视为进程内常量,这里不缓存以便测试在单进程内切换开关
    return std::getenv("RK_PIPE_WEB_TOKEN");
}

// 常时比较(避免逐字节提前返回的时序侧信道;LAN 场景属纵深防御)
bool constantTimeEquals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) {
        return false;
    }
    unsigned char diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
    }
    return diff == 0;
}

// 请求行/头里取令牌:优先查询参数 ?token=,其次 X-Auth-Token 头
std::string requestToken(const std::string& request) {
    // 查询参数:请求行 "GET /path?token=x HTTP/1.1" 的 target 段
    const std::size_t line_end = request.find("\r\n");
    const std::string line = line_end == std::string::npos ? request : request.substr(0, line_end);
    const std::size_t sp1 = line.find(' ');
    const std::size_t sp2 = sp1 == std::string::npos ? std::string::npos : line.find(' ', sp1 + 1);
    if (sp1 != std::string::npos && sp2 != std::string::npos) {
        const std::string target = line.substr(sp1 + 1, sp2 - sp1 - 1);
        const std::size_t q = target.find('?');
        if (q != std::string::npos) {
            const std::string query = target.substr(q + 1);
            std::size_t pos = 0;
            while ((pos = query.find("token=", pos)) != std::string::npos) {
                if (pos == 0 || query[pos - 1] == '&' || query[pos - 1] == ';') {
                    const std::size_t vstart = pos + 6;
                    const std::size_t vend = query.find('&', vstart);
                    return query.substr(vstart,
                                        vend == std::string::npos ? std::string::npos
                                                                  : vend - vstart);
                }
                ++pos;
            }
        }
    }
    // 请求头:逐行找 x-auth-token(大小写不敏感前缀)
    std::size_t pos = 0;
    while ((pos = request.find('\n', pos)) != std::string::npos) {
        const std::size_t eol = request.find('\r', pos);
        const std::size_t len = (eol == std::string::npos ? request.size() : eol) - (pos + 1);
        if (len > 13) {
            const std::string hdr = request.substr(pos + 1, len);
            std::string lower;
            lower.reserve(13);
            for (std::size_t i = 0; i < 13; ++i) {
                lower.push_back(
                    static_cast<char>(std::tolower(static_cast<unsigned char>(hdr[i]))));
            }
            if (lower.compare(0, 13, "x-auth-token:") == 0) {
                std::string value = hdr.substr(13);
                while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
                    value.erase(value.begin());
                }
                while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
                    value.pop_back();
                }
                return value;
            }
        }
        pos = eol;
    }
    return std::string();
}

bool requestAuthorized(const std::string& request) {
    const char* token = webToken();
    if (token == nullptr || *token == '\0') {
        return true;  // 未配置 = 鉴权关闭(向后兼容)
    }
    return constantTimeEquals(requestToken(request), token);
}

// 令牌嵌入页面/JS 前的白名单净化(防经令牌值的 HTML/JS 注入)
std::string sanitizeTokenForEmbedding(const std::string& token) {
    std::string out;
    for (const char ch : token) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                        (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == '-' ||
                        ch == '~';
        if (ok) {
            out.push_back(ch);
        }
    }
    return out;
}

std::string makeHtmlPage(const std::string& title,
                         const std::vector<int>& extra_stream_ports,
                         const std::vector<std::string>& webrtc_urls,
                         const std::string& token) {
    // 页面内端点统一追加令牌(鉴权关闭时为空串,URL 不变)
    const std::string tok = sanitizeTokenForEmbedding(token);
    const std::string tokq = tok.empty() ? "" : "?token=" + tok;
    // 窗格媒体元素：有 WebRTC URL 用 iframe（经 MediaMTX 观看推流），否则 MJPEG <img>
    auto pane_media = [&](std::size_t pane_index, const std::string& img_tag) {
        const bool has_wrtc = webrtc_urls.size() > pane_index && !webrtc_urls[pane_index].empty();
        if (!has_wrtc) {
            return img_tag;
        }
        return std::string("<iframe class=\"webrtc-stream\" src=\"") + webrtc_urls[pane_index] +
               "\" data-wrtc=\"1\" allow=\"autoplay; encrypted-media\" "
               "style=\"width:100%;aspect-ratio:16/9;border:0;background:#000;border-radius:8px;display:block;\" "
               "allowfullscreen></iframe>";
    };

    std::ostringstream html;
    html << "<!doctype html><html><head><meta charset=\"utf-8\">"
         << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
         << "<title>" << title << "</title>"
         << "<style>"
         << "body{margin:0;background:#111;color:#eee;font-family:sans-serif;}"
         << ".wrap{padding:12px;max-width:1600px;margin:0 auto;}"
         << ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:12px;}"
         << ".card{background:#1a1a1a;padding:10px;border-radius:8px;}"
         << ".label{margin:0 0 8px 0;opacity:.85;font-size:14px;}"
         << ".stat{margin:6px 0 0 0;font-size:12px;opacity:.6;font-family:monospace;}"
         << "img{width:100%;height:auto;background:#000;border-radius:8px;display:block;}"
         << "p{opacity:.8;}"
         << "</style></head><body><div class=\"wrap\">"
         << "<h2>" << title << "</h2>"
         << "<p>实时预览流 / Real-time preview</p>"
         << "<div class=\"grid\">"
         << "<div class=\"card\"><p class=\"label\">stream_1"
         << (webrtc_urls.size() > 0 && !webrtc_urls[0].empty() ? " (WebRTC)" : "")
         << "</p>"
         << pane_media(0, std::string("<img src=\"/stream.mjpg") + tokq +
                              "\" alt=\"rkpipe_preview_stream_1\">")
         << "<p class=\"stat\" id=\"stm_1\">--</p></div>";
    for (std::size_t i = 0; i < extra_stream_ports.size(); ++i) {
        const int port = extra_stream_ports[i];
        if (port <= 0) {
            continue;
        }
        const bool has_wrtc = webrtc_urls.size() > (i + 1) && !webrtc_urls[i + 1].empty();
        html << "<div class=\"card\"><p class=\"label\">stream_" << (i + 2)
             << (has_wrtc ? " (WebRTC)" : "") << "</p>"
             << pane_media(i + 1,
                           "<img class=\"remote-stream\" data-port=\"" + std::to_string(port) +
                               "\" alt=\"rkpipe_preview_stream_" + std::to_string(i + 2) + "\">")
             << "<p class=\"stat\" id=\"stm_" << port << "\">--</p></div>";
    }
    html << "</div>"
         << "<script>"
         << "const tok='" << tok << "';const tokq=tok?('?token='+tok):'';"
         << "const host=window.location.hostname||'127.0.0.1';"
         << "document.querySelectorAll('img.remote-stream').forEach((img)=>{"
         << "const port=img.dataset.port;"
         << "img.src=`http://${host}:${port}/stream.mjpg${tokq}`;"
         << "});"
         << "function fmtStat(d){"
         << "let s=(d.source_fps>0?'Video '+d.source_fps+' fps \\u00b7 ':'')"
         << "+'Stream '+d.publish_fps.toFixed(1)+' fps';"
         << "if(d.encode_ms>0)s+=' \\u00b7 enc '+d.encode_ms.toFixed(1)+'ms';"
         << "if(d.dropped_frames>0)s+=' \\u00b7 drop '+d.dropped_frames;"
         << "if(d.clients>0)s+=' \\u00b7 '+d.clients+' cli';"
         << "return s;} "
         << "function refreshStats(){"
         << "fetch('/status.json'+tokq).then(r=>r.json()).then(d=>{"
         << "const el=document.getElementById('stm_1');if(el)el.textContent=fmtStat(d);"
         << "}).catch(()=>{});"
         << "document.querySelectorAll('[data-port]').forEach((el)=>{"
         << "const port=el.dataset.port;"
         << "fetch(`http://${host}:${port}/status.json${tokq}`).then(r=>r.json()).then(d=>{"
         << "const id=document.getElementById('stm_'+port);if(id)id.textContent=fmtStat(d);"
         << "}).catch(()=>{});"
         << "});}"
         << "refreshStats();setInterval(refreshStats,1000);"
         << "</script>"
         << "</div></body></html>";
    return html.str();
}

std::string httpHeader(const std::string& status,
                       const std::string& content_type,
                       std::size_t content_length) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << "\r\n"
        << "Connection: close\r\n"
        << "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
        << "Pragma: no-cache\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << content_length << "\r\n\r\n";
    return oss.str();
}

std::string streamHeader() {
    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n"
        << "Connection: close\r\n"
        << "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
        << "Pragma: no-cache\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Content-Type: multipart/x-mixed-replace; boundary=" << kBoundary << "\r\n\r\n";
    return oss.str();
}

std::string readRequest(int fd) {
    std::string request;
    char buffer[4096];
    while (request.find("\r\n\r\n") == std::string::npos && request.size() < 16384) {
        const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            break;
        }
        request.append(buffer, static_cast<std::size_t>(n));
    }
    return request;
}

std::string requestPath(const std::string& request) {
    const std::size_t line_end = request.find("\r\n");
    const std::string line = request.substr(0, line_end);
    const std::size_t method_end = line.find(' ');
    if (method_end == std::string::npos) {
        return "/";
    }
    const std::size_t path_end = line.find(' ', method_end + 1);
    if (path_end == std::string::npos) {
        return "/";
    }
    std::string path = line.substr(method_end + 1, path_end - method_end - 1);
    // 剥离 query string（允许 /stream.mjpg?t=123 这种强制刷新 URL）
    const std::size_t qmark = path.find('?');
    if (qmark != std::string::npos) {
        path = path.substr(0, qmark);
    }
    return path;
}

}  // namespace

WebPreviewServer::~WebPreviewServer() {
    stop();
}

bool WebPreviewServer::start(const Config& config) {
    stop();
    config_ = config;

    server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::perror("[WebPreview] socket");
        return false;
    }

    int opt = 1;
    ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(config_.port));
    if (config_.bind_address.empty() || config_.bind_address == "0.0.0.0") {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, config_.bind_address.c_str(), &addr.sin_addr) != 1) {
        std::fprintf(stderr, "[WebPreview] invalid bind address: %s\n", config_.bind_address.c_str());
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (::bind(server_fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::perror("[WebPreview] bind");
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    if (::listen(server_fd_, 8) != 0) {
        std::perror("[WebPreview] listen");
        ::close(server_fd_);
        server_fd_ = -1;
        return false;
    }

    running_.store(true);
    current_publish_fps_.store(0.0);
    {
        std::lock_guard<std::mutex> lock(fps_mutex_);
        fps_window_start_ = std::chrono::steady_clock::now();
        fps_window_count_ = 0;
    }
    replay_pace_enabled_ = config_.replay_pace || replayPaceEnvEnabled();
    replay_started_ = false;
    replay_frame_index_ = 0;
    replay_start_time_ = std::chrono::steady_clock::time_point{};
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_frames_.clear();
    }
    // turbojpeg 压缩器在 encodeLoop 线程复用；初始化失败时回退 cv::imencode
    if (!tj_compressor_) {
        tj_compressor_ = tjInitCompress();
    }
    accept_thread_ = std::thread(&WebPreviewServer::acceptLoop, this);
    encode_thread_ = std::thread(&WebPreviewServer::encodeLoop, this);
    std::printf("[WebPreview] listening on %s (mode=%s fps=%.2f pace=%d)\n", url().c_str(),
                config_.preview_mode.c_str(), config_.replay_fps, replay_pace_enabled_ ? 1 : 0);
        return true;
}

void WebPreviewServer::stop() {
    if (!running_.load()) {
        return;
    }

    if (config_.preview_mode == "replay" && config_.eof_linger_ms > 0 && replay_started_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.eof_linger_ms));
    }

    if (!running_.exchange(false)) {
        return;
    }

    frame_cv_.notify_all();
    queue_cv_.notify_all();

    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }
    if (tj_compressor_) {
        tjDestroy(tj_compressor_);
        tj_compressor_ = nullptr;
    }

    if (server_fd_ >= 0) {
        ::shutdown(server_fd_, SHUT_RDWR);
        ::close(server_fd_);
        server_fd_ = -1;
    }
    if (accept_thread_.joinable()) {
        accept_thread_.join();  // 此后不再产生新的客户端线程
    }

    // 唤醒阻塞在 recv/send 上的客户端线程(shutdown 使其立即失败返回)。
    // 这里只 shutdown 不 close:close 由各客户端线程经 closeClient 恰好执行一次,
    // 避免双重 close(误伤复用后的无关句柄)。
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (int fd : client_fds_) {
            ::shutdown(fd, SHUT_RDWR);
        }
    }
    frame_cv_.notify_all();

    // 等待全部客户端线程退出后才允许析构,消除 detached 线程解引用 this 的问题
    for (;;) {
        reapClientThreads();
        {
            std::lock_guard<std::mutex> lock(client_threads_mutex_);
            if (client_threads_.empty()) {
                break;
            }
        }
        frame_cv_.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // 兜底:线程全部退出后,仍登记在册的 fd 已无人持有,统一关闭
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        for (int fd : client_fds_) {
            ::close(fd);
        }
        client_fds_.clear();
    }
}

std::string WebPreviewServer::url() const {
    return std::string("http://") + (config_.bind_address.empty() ? "0.0.0.0" : config_.bind_address) +
           ":" + std::to_string(config_.port) + "/";
}

int WebPreviewServer::clientCount() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    return static_cast<int>(client_fds_.size());
}

std::string WebPreviewServer::makeStatusJson() const {
    std::uint64_t input_dropped = 0;
    std::uint64_t input_reconnects = 0;
    if (input_stat_source_) {
        auto stats = input_stat_source_();
        input_dropped = stats.first;
        input_reconnects = stats.second;
    }
    std::ostringstream json;
    json << "{\n"
         << "  \"title\": \"" << jsonEscape(config_.title) << "\",\n"
         << "  \"task\": \"" << jsonEscape(config_.task) << "\",\n"
         << "  \"input\": \"" << jsonEscape(config_.input_path) << "\",\n"
         << "  \"port\": " << config_.port << ",\n"
         << "  \"source_fps\": " << config_.source_fps << ",\n"
         << "  \"frame_width\": " << config_.frame_width << ",\n"
         << "  \"frame_height\": " << config_.frame_height << ",\n"
         << "  \"running\": " << (running_.load() ? "true" : "false") << ",\n"
         << "  \"preview_mode\": \"" << jsonEscape(config_.preview_mode) << "\",\n"
         << "  \"publish_fps\": " << publishFPS() << ",\n"
         << "  \"published_frames\": " << publishedFrames() << ",\n"
         << "  \"dropped_frames\": " << droppedFrames() << ",\n"
         << "  \"encode_ms\": " << encodeMs() << ",\n"
         << "  \"clients\": " << clientCount() << ",\n"
         << "  \"input_dropped_frames\": " << input_dropped << ",\n"
         << "  \"input_reconnect_count\": " << input_reconnects << "\n"
         << "}\n";
    return json.str();
}

void WebPreviewServer::publishFrame(cv::Mat frame, bool pre_scaled) {
    if (!running_.load() || frame.empty()) {
        return;
    }

    const bool paced = replay_pace_enabled_ && config_.preview_mode == "replay" && config_.replay_fps > 0.0;

    // 原帧率播放（web_preview_replay_pace=1）：在生产端按源帧率节流，通过
    // backpressure 让解码/NPU 流水线整体按源帧率推进，避免"显示帧率对但内容跳帧快进"。
    if (paced) {
        if (!replay_started_) {
            replay_started_ = true;
            replay_frame_index_ = 0;
            replay_start_time_ = std::chrono::steady_clock::now();
        }
        const double frame_interval_us = 1000000.0 / config_.replay_fps;
        const auto desired_offset = std::chrono::microseconds(
            static_cast<long long>(std::llround(static_cast<double>(replay_frame_index_) * frame_interval_us)));
        const auto desired_time = replay_start_time_ + desired_offset;
        const auto now = std::chrono::steady_clock::now();
        if (desired_time > now) {
            std::this_thread::sleep_until(desired_time);
        }
        ++replay_frame_index_;
    }

    // 异步编码：输出线程只做格式转换 + 入队，JPEG 编码放到 encodeLoop 独立线程，
    // 避免 cv::imencode 阻塞输出线程（当前主要输出开销）。帧以 cv::Mat 引用计数
    // 保活，入队即持有数据，编码完成后由队列释放。
    if (frame.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
        frame = std::move(bgr);
    } else if (frame.channels() == 1) {
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
        frame = std::move(bgr);
    } else if (frame.channels() != 3) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (paced) {
            // 原帧率模式：编码队列满时阻塞生产端（背压到上游流水线），
            // 保证帧不丢弃、内容不跳帧（丢最旧帧会导致浏览器看到"快进"）。
            queue_cv_.wait(lock, [&]() { return !running_.load() || pending_frames_.size() < kMaxPendingFrames; });
            if (!running_.load()) {
                return;
            }
        } else if (pending_frames_.size() >= kMaxPendingFrames) {
            // 非节流（实时预览）模式：队满丢最旧帧，保持低延迟预览并限制内存占用，不阻塞输出线程
            pending_frames_.pop_front();
            dropped_frames_.fetch_add(1);
        }
        PendingFrame pending;
        pending.bgr = std::move(frame);
        pending.pre_scaled = pre_scaled;
        pending_frames_.push_back(std::move(pending));
        published_frames_.fetch_add(1);
    }
    queue_cv_.notify_one();
}

void WebPreviewServer::encodeLoop() {
    pinCurrentThreadFromEnv("RK_PIPE_PIN_ENCODE");
    for (;;) {
        PendingFrame pending;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [&]() { return !running_.load() || !pending_frames_.empty(); });
            if (!running_.load()) {
                pending_frames_.clear();
                return;
            }
            pending = std::move(pending_frames_.front());
            pending_frames_.pop_front();
        }
        queue_cv_.notify_one();  // 唤醒因原帧率模式排队满而阻塞的生产端
        reapClientThreads();

        // 无 MJPEG 客户端时跳过 JPEG 编码（窗格用 WebRTC iframe / 无人观看时），
        // 释放 CPU 给推理；有人请求 /stream.mjpg 时自动恢复编码。
        {
            std::lock_guard<std::mutex> lock(clients_mutex_);
            if (client_fds_.empty()) {
                continue;
            }
        }
        encodeAndPublish(std::move(pending.bgr), pending.pre_scaled);
    }
}

void WebPreviewServer::encodeAndPublish(cv::Mat bgr, bool pre_scaled) {
    // 编码前降采样（preview_scale<1.0）：减小 JPEG 像素量，降低编码耗时与预览带宽。
    // 若帧已在生产端（RGA 硬件）完成降采样（pre_scaled），则不再二次缩放。
    if (!pre_scaled && config_.preview_scale > 0.1 && config_.preview_scale < 1.0) {
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(), config_.preview_scale, config_.preview_scale, cv::INTER_AREA);
        bgr = std::move(resized);
    }

    const auto encode_begin = std::chrono::steady_clock::now();
    std::vector<unsigned char> jpeg;
    // 优先 turbojpeg（NEON SIMD，比 cv::imencode 快 1.5~2.5x），失败回退 OpenCV。
    if (tj_compressor_) {
        unsigned char* jpeg_buf = nullptr;
        unsigned long jpeg_size = 0;
        if (tjCompress2(tj_compressor_, bgr.data, bgr.cols, bgr.step, bgr.rows,
                        TJPF_BGR, &jpeg_buf, &jpeg_size, TJSAMP_420,
                        config_.jpeg_quality, TJFLAG_FASTDCT) == 0) {
            jpeg.assign(jpeg_buf, jpeg_buf + jpeg_size);
        }
        tjFree(jpeg_buf);
    }
    if (jpeg.empty()) {
        std::vector<int> params{cv::IMWRITE_JPEG_QUALITY, config_.jpeg_quality};
        if (!cv::imencode(".jpg", bgr, jpeg, params)) {
            return;
        }
    }
    const auto encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - encode_begin)
                               .count();
    encode_ms_.store(static_cast<double>(encode_ms));

    std::size_t jpeg_size = 0;
    std::uint64_t frame_version = 0;
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_jpeg_ = std::move(jpeg);
        ++frame_version_;
        jpeg_size = latest_jpeg_.size();
        frame_version = frame_version_;
    }
        frame_cv_.notify_all();

    {
        std::lock_guard<std::mutex> lock(fps_mutex_);
        ++fps_window_count_;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - fps_window_start_).count();
        if (elapsed_ms >= 1000) {
            current_publish_fps_.store(fps_window_count_ * 1000.0 / static_cast<double>(elapsed_ms));
            fps_window_start_ = now;
            fps_window_count_ = 0;
        }
    }
}

void WebPreviewServer::acceptLoop() {
    while (running_.load()) {
        // close()/shutdown() 并不能可靠唤醒阻塞在 accept() 的线程,
        // 改用 poll 超时轮询,让本循环周期性回到循环头检查 running_
        pollfd pfd{};
        pfd.fd = server_fd_;
        pfd.events = POLLIN;
        if (::poll(&pfd, 1, 200) <= 0) {
            continue;
        }
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd = ::accept(server_fd_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd < 0) {
            if (!running_.load()) {
                break;
            }
            if (errno == EINTR) {
                continue;
            }
            std::perror("[WebPreview] accept");
            continue;
        }
        spawnClientThread(client_fd);
        reapClientThreads();
    }
}

void WebPreviewServer::spawnClientThread(int client_fd) {
    auto done = std::make_shared<std::atomic<bool>>(false);
    std::thread th([this, client_fd, done]() {
        handleClient(client_fd);
        done->store(true);  // 线程函数最后动作:reapClientThreads 据此 join 回收
    });
    std::lock_guard<std::mutex> lock(client_threads_mutex_);
    client_threads_.push_back(ClientThread{std::move(th), std::move(done)});
}

void WebPreviewServer::reapClientThreads() {
    std::lock_guard<std::mutex> lock(client_threads_mutex_);
    for (int i = static_cast<int>(client_threads_.size()) - 1; i >= 0; --i) {
        if (client_threads_[i].done->load()) {
            client_threads_[i].th.join();
            client_threads_.erase(client_threads_.begin() + i);
        }
    }
}

void WebPreviewServer::handleClient(int client_fd) {
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        if (client_fds_.size() >= kMaxClients) {
            ::close(client_fd);
            return;
        }
        client_fds_.insert(client_fd);
    }

    // 收发超时:静默/慢速连接不能长期占用有限的客户端槽位
    timeval tv{};
    tv.tv_sec = 10;
    ::setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    const std::string request = readRequest(client_fd);
    if (request.empty()) {
        closeClient(client_fd);
        return;
    }
    const std::string path = requestPath(request);

    // 访问令牌鉴权(RK_PIPE_WEB_TOKEN 非空时启用;/healthz 保持开放供存活探针)
    if (path != "/healthz" && !requestAuthorized(request)) {
        sendTextResponse(client_fd, "401 Unauthorized", "text/plain; charset=utf-8",
                         "unauthorized: missing or invalid token\n");
        closeClient(client_fd);
        return;
    }
    const std::string config_token =
        sanitizeTokenForEmbedding(webToken() ? webToken() : "");
    
    if (path == "/" || path == "/index.html") {
        sendTextResponse(client_fd,
                         "200 OK",
                         "text/html; charset=utf-8",
                         makeHtmlPage(config_.title, config_.extra_stream_ports, config_.webrtc_urls,
                         config_token));
        closeClient(client_fd);
        return;
    }

    if (path == "/healthz") {
        sendTextResponse(client_fd, "200 OK", "text/plain; charset=utf-8", "ok\n");
        closeClient(client_fd);
        return;
    }

    if (path == "/status.json") {
        sendTextResponse(client_fd, "200 OK", "application/json; charset=utf-8", makeStatusJson());
        closeClient(client_fd);
        return;
    }

    if (path == "/snapshot.jpg") {
        std::vector<unsigned char> jpeg;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            jpeg = latest_jpeg_;
        }
        if (jpeg.empty()) {
            sendTextResponse(client_fd, "503 Service Unavailable", "text/plain; charset=utf-8", "frame not ready\n");
        } else {
            sendBinaryResponse(client_fd, "200 OK", "image/jpeg", jpeg);
        }
        closeClient(client_fd);
        return;
    }

    if (path == "/stream.mjpg") {
        streamClient(client_fd);
        return;
    }

    sendTextResponse(client_fd, "404 Not Found", "text/plain; charset=utf-8", "not found\n");
    closeClient(client_fd);
}

void WebPreviewServer::streamClient(int client_fd) {
    const std::string header = streamHeader();
    if (!sendAll(client_fd, header.data(), header.size())) {
        closeClient(client_fd);
        return;
    }

    std::uint64_t last_version = 0;
    int sent_frames = 0;
    while (running_.load()) {
        std::vector<unsigned char> jpeg;
        std::uint64_t version = 0;

        {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait(lock, [&]() { return !running_.load() || frame_version_ > last_version; });
            if (!running_.load()) {
                break;
            }
            jpeg = latest_jpeg_;
            version = frame_version_;
        }

        if (jpeg.empty()) {
            continue;
        }

        std::ostringstream part;
        part << "--" << kBoundary << "\r\n"
             << "Content-Type: image/jpeg\r\n"
             << "Content-Length: " << jpeg.size() << "\r\n\r\n";
        const std::string part_header = part.str();
        if (!sendAll(client_fd, part_header.data(), part_header.size()) ||
            !sendAll(client_fd, jpeg.data(), jpeg.size()) ||
            !sendAll(client_fd, "\r\n", 2)) {
            break;
        }
        last_version = version;
        ++sent_frames;
            }
    
    closeClient(client_fd);
}

void WebPreviewServer::closeClient(int client_fd) {
    // erase 的返回值保证 close 恰好发生一次;与 stop() 的 shutdown 扫描互斥
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        if (client_fds_.erase(client_fd) == 0) {
            return;
        }
    }
    ::shutdown(client_fd, SHUT_RDWR);
    ::close(client_fd);
}

bool WebPreviewServer::sendAll(int fd, const void* data, std::size_t size) {
    const char* ptr = static_cast<const char*>(data);
    std::size_t sent = 0;
    while (sent < size) {
        const ssize_t n = ::send(fd, ptr + sent, size - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    return true;
}

bool WebPreviewServer::sendTextResponse(int fd,
                                        const std::string& status,
                                        const std::string& content_type,
                                        const std::string& body) {
    const std::string header = httpHeader(status, content_type, body.size());
    return sendAll(fd, header.data(), header.size()) && sendAll(fd, body.data(), body.size());
}

bool WebPreviewServer::sendBinaryResponse(int fd,
                                          const std::string& status,
                                          const std::string& content_type,
                                          const std::vector<unsigned char>& body) {
    const std::string header = httpHeader(status, content_type, body.size());
    return sendAll(fd, header.data(), header.size()) &&
           (body.empty() || sendAll(fd, body.data(), body.size()));
}
