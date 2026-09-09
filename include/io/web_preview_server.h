#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <opencv2/opencv.hpp>

class WebPreviewServer {
public:
    struct Config {
        std::string bind_address = "0.0.0.0";
        int port = 8080;
        int jpeg_quality = 80;
        std::string preview_mode = "live";
        double replay_fps = 0.0;
        int eof_linger_ms = 1500;
        bool replay_pace = false;
        double preview_scale = 1.0;  // 编码前降采样 (0.1~1.0)，降低 JPEG 编码开销与带宽
        double source_fps = 0.0;     // 输入源帧率（静态信息，用于 /status.json 卡片显示）
        int frame_width = 0;         // 输入源分辨率宽（静态信息，用于 /status.json 卡片显示）
        int frame_height = 0;        // 输入源分辨率高
        std::vector<int> extra_stream_ports;
        // 各窗格对应的 WebRTC 播放 URL（经 MediaMTX 等网关观看推流）；空=该窗格用 MJPEG。
        // 下标对齐：0=本服务自身窗格，1..N=extra_stream_ports 对应窗格。
        std::vector<std::string> webrtc_urls;
        std::string title = "rk_pipe preview";
        std::string task;        // 静态信息，用于 /status.json
        std::string input_path;
    };

    WebPreviewServer() = default;
    ~WebPreviewServer();

    bool start(const Config& config);
    void stop();
    void publishFrame(cv::Mat frame, bool pre_scaled = false);

    bool running() const { return running_.load(); }
    std::string url() const;
    double publishFPS() const { return current_publish_fps_.load(); }
    double previewScale() const { return config_.preview_scale; }

    // 预览量化指标
    std::uint64_t publishedFrames() const { return published_frames_.load(); }
    std::uint64_t droppedFrames() const { return dropped_frames_.load(); }
    double encodeMs() const { return encode_ms_.load(); }
    int clientCount() const;

    // 输入侧统计源（丢帧数, 重连次数）——由主流程注入，/status.json 实时展示
    using InputStatSource = std::function<std::pair<std::uint64_t, std::uint64_t>()>;
    void setInputStatSource(InputStatSource source) { input_stat_source_ = std::move(source); }

private:
    // 异步 JPEG 编码的队列元素：BGR(3ch) 帧 + 是否已由生产端完成 preview_scale 降采样
    struct PendingFrame {
        cv::Mat bgr;
        bool pre_scaled = false;
    };

    // 客户端连接线程句柄：done 由线程函数最后一动作置位，reap 据此 join 并回收,
    // 保证 stop() 返回后没有任何线程再引用 this(可安全析构)
    struct ClientThread {
        std::thread th;
        std::shared_ptr<std::atomic<bool>> done;
    };

    void acceptLoop();
    void spawnClientThread(int client_fd);
    void reapClientThreads();
    void handleClient(int client_fd);
    void streamClient(int client_fd);
    void closeClient(int client_fd);
    void encodeLoop();
    void encodeAndPublish(cv::Mat bgr, bool pre_scaled);
    std::string makeStatusJson() const;

    // 并发客户端上限：超过则直接拒绝新连接,防止线程无限增长
    static constexpr int kMaxClients = 8;

    bool sendAll(int fd, const void* data, std::size_t size);
    bool sendTextResponse(int fd,
                          const std::string& status,
                          const std::string& content_type,
                          const std::string& body);
    bool sendBinaryResponse(int fd,
                            const std::string& status,
                            const std::string& content_type,
                            const std::vector<unsigned char>& body);

    Config config_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread accept_thread_;

    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    std::vector<unsigned char> latest_jpeg_;
    std::uint64_t frame_version_ = 0;

    // 异步 JPEG 编码：输出线程只入队 BGR/I420 帧，encodeLoop 独立线程取帧编码。
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<PendingFrame> pending_frames_;
    std::thread encode_thread_;
    static constexpr std::size_t kMaxPendingFrames = 2;

    // turbojpeg 压缩器 handle（encodeLoop 线程内复用）；nullptr 时回退 cv::imencode
    void* tj_compressor_ = nullptr;

    mutable std::mutex clients_mutex_;
    std::unordered_set<int> client_fds_;

    // 客户端连接线程登记(stop/join 依据);由 spawnClientThread 压入、reapClientThreads 回收
    std::mutex client_threads_mutex_;
    std::vector<ClientThread> client_threads_;

    std::mutex fps_mutex_;
    std::chrono::steady_clock::time_point fps_window_start_{};
    int fps_window_count_ = 0;
    std::atomic<double> current_publish_fps_{0.0};

    // 量化指标：已发布/丢弃帧数、最近一次 JPEG 编码耗时(ms)
    std::atomic<std::uint64_t> published_frames_{0};
    std::atomic<std::uint64_t> dropped_frames_{0};
    std::atomic<double> encode_ms_{0.0};

    // 输入侧统计源（丢帧数 / 重连次数），由主流程在 start 后注入
    InputStatSource input_stat_source_;

    bool replay_pace_enabled_ = false;
    bool replay_started_ = false;
    std::uint64_t replay_frame_index_ = 0;
    std::chrono::steady_clock::time_point replay_start_time_{};
};
