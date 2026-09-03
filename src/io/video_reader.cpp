#include "io/video_reader.h"
#include <cstdio>
#include <cstdlib>
#include <array>
#include <vector>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <linux/dma-heap.h>
#include <linux/videodev2.h>
#include <cerrno>
#include <cstring>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/imgutils.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>
}

namespace {

bool debugYuvEnabled() {
    const char* value = std::getenv("RK_PIPE_DEBUG_YUV");
    return value && *value && std::string(value) != "0";
}

bool isNetworkSource(const std::string& source) {
    std::string lower = source;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower.rfind("rtsp://", 0) == 0 ||
           lower.rfind("rtsps://", 0) == 0 ||
           lower.rfind("rtmp://", 0) == 0 ||
           lower.rfind("rtmps://", 0) == 0 ||
           lower.rfind("http://", 0) == 0 ||
           lower.rfind("https://", 0) == 0 ||
           lower.rfind("udp://", 0) == 0 ||
           lower.rfind("srt://", 0) == 0;
}

// 网络输入打开参数：核心是超时，避免死地址/断网时 avformat_open_input 无限阻塞。
// 时长由 RK_PIPE_STREAM_TIMEOUT_MS 控制（默认 5000ms）。
void setNetworkInputOptions(const std::string& source, AVDictionary** opts) {
    if (!opts || !isNetworkSource(source)) {
        return;
    }
    int timeout_ms = 5000;
    const char* env = std::getenv("RK_PIPE_STREAM_TIMEOUT_MS");
    if (env && *env) {
        const int v = std::atoi(env);
        if (v > 0) {
            timeout_ms = v;
        }
    }

    std::string lower = source;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // rw_timeout 为所有协议通用的读写超时（微秒）
    av_dict_set(opts, "rw_timeout", std::to_string(timeout_ms * 1000).c_str(), 0);
    if (lower.rfind("rtsp://", 0) == 0 || lower.rfind("rtsps://", 0) == 0) {
        // RTSP 走 TCP 更稳定，stimeout 兜底 socket 超时（微秒）
        av_dict_set(opts, "rtsp_transport", "tcp", 0);
        av_dict_set(opts, "stimeout", std::to_string(timeout_ms * 1000).c_str(), 0);
    } else if (lower.rfind("rtmp://", 0) == 0 || lower.rfind("rtmps://", 0) == 0) {
        // RTMP 连接超时（毫秒）
        av_dict_set(opts, "timeout", std::to_string(timeout_ms).c_str(), 0);
    } else if (lower.rfind("http://", 0) == 0 || lower.rfind("https://", 0) == 0) {
        // HTTP 分片流断开可协议层续拉（应用层重连仍保留）
        av_dict_set(opts, "reconnect", "1", 0);
        av_dict_set(opts, "reconnect_streamed", "1", 0);
        av_dict_set(opts, "reconnect_delay_max", "5", 0);
    }
}

// Helper functions (unused for now)
/*
std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::string item;
    std::istringstream iss(s);
    while (std::getline(iss, item, delim)) {
        parts.push_back(item);
    }
    return parts;
}

double parseRational(const std::string& s) {
    auto pos = s.find('/');
    if (pos == std::string::npos) {
        return std::stod(s);
    }
    double num = std::stod(s.substr(0, pos));
    double den = std::stod(s.substr(pos + 1));
    if (den == 0.0) return 0.0;
    return num / den;
}
*/
// FFmpegRKMPP 零拷贝帧引用：NV12 帧被导入自有 dma-buf，priv_data 指向本结构，
// releaseFrame 按 magic 识别后把槽位归还池。
constexpr uint32_t kFFmpegRkmppFrameMagic = 0x464B4D50;  // 'FKMP'

struct FFmpegRkmppDmaPool;

struct FFmpegRkmppFrameRef {
    uint32_t magic;
    FFmpegRkmppDmaPool* pool;
    int slot;
};

struct FFmpegRkmppDmaSlot {
    int fd = -1;
    unsigned char* addr = nullptr;
    size_t size = 0;
    bool in_use = false;
};

struct FFmpegRkmppDmaPool {
    std::vector<FFmpegRkmppDmaSlot> slots;
    std::mutex mutex;
    std::condition_variable cv;
    int stride = 0;
    int height = 0;
    size_t buf_size = 0;

    int acquireSlot() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]() {
            for (const auto& s : slots) {
                if (!s.in_use) return true;
            }
            return false;
        });
        for (size_t i = 0; i < slots.size(); ++i) {
            if (!slots[i].in_use) {
                slots[i].in_use = true;
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void releaseSlot(int slot) {
        std::lock_guard<std::mutex> lock(mutex);
        if (slot >= 0 && slot < static_cast<int>(slots.size())) {
            slots[slot].in_use = false;
        }
        cv.notify_one();
    }
};

// 从 dma-heap 分配 dma32 内存（rga/rknn 可用的 dma-buf，uncached 保证 CPU 写入对硬件可见）。
static bool ffmpegRkmppDmaHeapAlloc(size_t size, int& out_fd, void*& out_addr) {
    static const char* kHeaps[] = {
        "/dev/dma_heap/system-uncached-dma32",
        "/dev/dma_heap/system-dma32",
        "/dev/dma_heap/dma32_uncached",
        "/dev/dma_heap/dma32",
    };
    for (const char* hp : kHeaps) {
        int hfd = open(hp, O_RDONLY | O_CLOEXEC);
        if (hfd < 0) {
            continue;
        }
        dma_heap_allocation_data data{};
        data.len = size;
        data.fd_flags = O_RDWR | O_CLOEXEC;
        data.heap_flags = 0;
        int ret = ioctl(hfd, DMA_HEAP_IOCTL_ALLOC, &data);
        close(hfd);
        if (ret != 0 || data.fd <= 0) {
            continue;
        }
        void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, data.fd, 0);
        if (addr == MAP_FAILED) {
            close(data.fd);
            continue;
        }
        out_fd = data.fd;
        out_addr = addr;
        return true;
    }
    return false;
}

static FFmpegRkmppDmaPool* ffmpegRkmppCreatePool(int width, int height) {
    int stride = (width + 15) & ~15;
    size_t buf_size = static_cast<size_t>(stride) * height * 3 / 2;
    auto* pool = new FFmpegRkmppDmaPool();
    pool->stride = stride;
    pool->height = height;
    pool->buf_size = buf_size;
    // 槽位要覆盖流水线在途帧（task 队列 + worker + 结果队列），太小会阻塞 producer。
    // 默认 16 槽（×1.38MB ≈ 22MB），可用 RK_PIPE_RKMPP_POOL_SLOTS 覆盖。
    int slot_count = 16;
    if (const char* v = std::getenv("RK_PIPE_RKMPP_POOL_SLOTS")) {
        int n = std::atoi(v);
        if (n > 0 && n <= 64) {
            slot_count = n;
        }
    }
    pool->slots.resize(slot_count);
    for (int i = 0; i < slot_count; ++i) {
        int fd = -1;
        void* addr = nullptr;
        if (!ffmpegRkmppDmaHeapAlloc(buf_size, fd, addr)) {
            for (int j = 0; j < i; ++j) {
                munmap(pool->slots[j].addr, pool->slots[j].size);
                close(pool->slots[j].fd);
            }
            delete pool;
            return nullptr;
        }
        pool->slots[i].fd = fd;
        pool->slots[i].addr = static_cast<unsigned char*>(addr);
        pool->slots[i].size = buf_size;
    }
    return pool;
}

static void ffmpegRkmppDestroyPool(FFmpegRkmppDmaPool* pool) {
    if (!pool) {
        return;
    }
    for (auto& s : pool->slots) {
        if (s.addr && s.addr != MAP_FAILED) {
            munmap(s.addr, s.size);
        }
        if (s.fd > 0) {
            close(s.fd);
        }
    }
    delete pool;
}

// 摄像头设备（UVC / V4L2）路径判断
bool isCameraDevice(const std::string& source) {
    return source.rfind("/dev/video", 0) == 0;
}

int v4l2Ioctl(int fd, unsigned long req, void* arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

// V4L2 摄像头候选：像素格式 + 分辨率 + 表内最大帧率（×1000）
struct CameraCandidate {
    unsigned int fourcc;
    unsigned int width;
    unsigned int height;
    long area;
    long max_fps1000;
};

// 枚举摄像头全部格式/分辨率/帧率组合。返回 false 表示枚举失败（设备不可用/无权限）。
bool enumerateCameraCandidates(const std::string& path, std::vector<CameraCandidate>& out) {
    int fd = ::open(path.c_str(), O_RDWR);
    if (fd < 0) {
        return false;
    }
    struct v4l2_fmtdesc fmtdesc;
    std::memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    while (v4l2Ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
        struct v4l2_frmsizeenum frmsize;
        std::memset(&frmsize, 0, sizeof(frmsize));
        frmsize.index = 0;
        frmsize.pixel_format = fmtdesc.pixelformat;
        while (v4l2Ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0) {
            unsigned int w = 0;
            unsigned int h = 0;
            if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                w = frmsize.discrete.width;
                h = frmsize.discrete.height;
            } else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE) {
                w = frmsize.stepwise.max_width;
                h = frmsize.stepwise.max_height;
            }
            if (w > 0 && h > 0) {
                long max_fps1000 = 0;
                struct v4l2_frmivalenum frmival;
                std::memset(&frmival, 0, sizeof(frmival));
                frmival.index = 0;
                frmival.pixel_format = fmtdesc.pixelformat;
                frmival.width = w;
                frmival.height = h;
                while (v4l2Ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &frmival) == 0) {
                    if (frmival.type == V4L2_FRMIVAL_TYPE_DISCRETE && frmival.discrete.denominator > 0) {
                        long fps1000 = static_cast<long>(static_cast<double>(frmival.discrete.denominator) /
                                                         frmival.discrete.numerator * 1000.0);
                        if (fps1000 > max_fps1000) {
                            max_fps1000 = fps1000;
                        }
                    } else if (frmival.type == V4L2_FRMIVAL_TYPE_STEPWISE &&
                               frmival.stepwise.min.denominator > 0) {
                        long fps1000 = static_cast<long>(static_cast<double>(frmival.stepwise.min.denominator) /
                                                         frmival.stepwise.min.numerator * 1000.0);
                        if (fps1000 > max_fps1000) {
                            max_fps1000 = fps1000;
                        }
                    }
                    frmival.index++;
                }
                CameraCandidate c;
                c.fourcc = fmtdesc.pixelformat;
                c.width = w;
                c.height = h;
                c.area = static_cast<long>(w) * h;
                c.max_fps1000 = max_fps1000;
                out.push_back(c);
            }
            frmsize.index++;
        }
        fmtdesc.index++;
    }
    ::close(fd);
    return !out.empty();
}

// 选出"面积最大，同面积下帧率最高"的组合。
// 输出 fourcc 为 V4L2 像素格式（与 cv::VideoWriter::fourcc 数值一致，可直接作 CAP_PROP_FOURCC）。
bool pickBestCameraFormat(const std::string& path, int& fourcc, int& width, int& height, int& fps) {
    std::vector<CameraCandidate> candidates;
    if (!enumerateCameraCandidates(path, candidates)) {
        return false;
    }
    const CameraCandidate* best = nullptr;
    for (const auto& c : candidates) {
        if (!best || c.area > best->area || (c.area == best->area && c.max_fps1000 > best->max_fps1000)) {
            best = &c;
        }
    }
    if (!best) {
        return false;
    }
    fourcc = static_cast<int>(best->fourcc);
    width = static_cast<int>(best->width);
    height = static_cast<int>(best->height);
    fps = static_cast<int>((best->max_fps1000 + 500) / 1000);
    return true;
}

// 实际起流测帧率：短时间 mmap 抓 N 帧，按 buffer 时间戳算平均帧率。
// 慢格式的启动爬坡期（从低 fps 爬升）会压低平均帧率，恰好用于把跑不满 30fps 的档位筛掉。
// 返回 -1 表示无法按该格式起流。
double measureCameraStreamFps(const std::string& path, unsigned int fourcc,
                              unsigned int width, unsigned int height, int frames) {
    int fd = ::open(path.c_str(), O_RDWR);
    if (fd < 0) {
        return -1.0;
    }
    double result = -1.0;

    struct v4l2_format fmt;
    std::memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = fourcc;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (v4l2Ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        ::close(fd);
        return -1.0;
    }

    struct v4l2_requestbuffers req;
    std::memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (v4l2Ioctl(fd, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) {
        ::close(fd);
        return -1.0;
    }

    std::vector<void*> bufs(req.count, nullptr);
    std::vector<size_t> lens(req.count, 0);
    for (unsigned int i = 0; i < req.count; ++i) {
        struct v4l2_buffer buf;
        std::memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (v4l2Ioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) {
            goto cleanup;
        }
        lens[i] = buf.length;
        bufs[i] = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (bufs[i] == MAP_FAILED) {
            bufs[i] = nullptr;
            goto cleanup;
        }
    }
    for (unsigned int i = 0; i < req.count; ++i) {
        struct v4l2_buffer buf;
        std::memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        v4l2Ioctl(fd, VIDIOC_QBUF, &buf);
    }
    if (v4l2Ioctl(fd, VIDIOC_STREAMON, &req.type) < 0) {
        goto cleanup;
    }

    {
        int got = 0;
        double first_ts = 0.0;
        double last_ts = 0.0;
        while (got < frames) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd, &fds);
            struct timeval tv;
            tv.tv_sec = 3;
            tv.tv_usec = 0;
            const int sel = select(fd + 1, &fds, nullptr, nullptr, &tv);
            if (sel <= 0) {
                break;
            }
            struct v4l2_buffer buf;
            std::memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            if (v4l2Ioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
                break;
            }
            const double ts = static_cast<double>(buf.timestamp.tv_sec) + buf.timestamp.tv_usec * 1e-6;
            if (got == 0) {
                first_ts = ts;
            }
            last_ts = ts;
            ++got;
            v4l2Ioctl(fd, VIDIOC_QBUF, &buf);
        }
        if (got >= 3) {
            const double dur = last_ts - first_ts;
            if (dur > 0) {
                result = static_cast<double>(got - 1) / dur;
            }
        }
    }

cleanup:
    for (unsigned int i = 0; i < bufs.size(); ++i) {
        if (bufs[i]) {
            munmap(bufs[i], lens[i]);
            bufs[i] = nullptr;
        }
    }
    ::close(fd);
    return result;
}

// RK_PIPE_CAM_PREFER_30FPS=1：选"格式表支持 30fps、且实测能稳 30fps"的最大分辨率。
// 全部不达标则退回面积最大组合（等同 RK_PIPE_CAM_MAX）。
bool pickPreferred30FpsCameraFormat(const std::string& path, int& fourcc, int& width, int& height, int& fps) {
    std::vector<CameraCandidate> candidates;
    if (!enumerateCameraCandidates(path, candidates)) {
        return false;
    }
    std::sort(candidates.begin(), candidates.end(), [](const CameraCandidate& a, const CameraCandidate& b) {
        if (a.area != b.area) {
            return a.area > b.area;
        }
        return a.max_fps1000 > b.max_fps1000;
    });

    int tested = 0;
    for (const auto& c : candidates) {
        if (c.max_fps1000 < 30000) {
            continue;  // 格式表里就不支持 30fps
        }
        // 起流可能瞬时失败（首次首帧慢），重试一次再判死刑
        double measured = -1.0;
        for (int attempt = 0; attempt < 2 && measured < 0; ++attempt) {
            measured = measureCameraStreamFps(path, c.fourcc, c.width, c.height, 12);
        }
        if (measured < 0) {
            std::cout << "[VideoInput] 30fps probe " << c.width << "x" << c.height
                      << " fourcc=" << c.fourcc << " FAILED (cannot start stream)" << std::endl;
            continue;
        }
        std::cout << "[VideoInput] 30fps probe " << c.width << "x" << c.height
                  << " fourcc=" << c.fourcc << " -> " << measured << "fps" << std::endl;
        if (measured >= 27.0) {
            fourcc = static_cast<int>(c.fourcc);
            width = static_cast<int>(c.width);
            height = static_cast<int>(c.height);
            fps = static_cast<int>((c.max_fps1000 + 500) / 1000);
            return true;
        }
        if (++tested >= 8) {
            break;
        }
    }

    return pickBestCameraFormat(path, fourcc, width, height, fps);
}

}  // namespace

// VideoReader base implementation
void VideoReader::releaseFrame(image_buffer_t& image) {
    if (image.priv_data) {
        auto* ref = static_cast<FFmpegRkmppFrameRef*>(image.priv_data);
        if (ref->magic == kFFmpegRkmppFrameMagic) {
            if (ref->pool) {
                ref->pool->releaseSlot(ref->slot);
            }
            delete ref;
        } else if (image.format == IMAGE_FORMAT_YUV420SP_NV12 ||
                   image.format == IMAGE_FORMAT_YUV420SP_NV21) {
            MppFrame frame = (MppFrame)image.priv_data;
            mpp_frame_deinit(&frame);
        }
        image.priv_data = nullptr;
    } else if (image.format == IMAGE_FORMAT_BGR888 && image.virt_addr) {
        free(image.virt_addr);
        image.virt_addr = nullptr;
    }
    image.fd = 0;
    image.size = 0;
    image.width = 0;
    image.height = 0;
    image.width_stride = 0;
    image.height_stride = 0;
}

// OpenCVVideoReader implementation
OpenCVVideoReader::OpenCVVideoReader()
    : width(0), height(0), fps(0), total_frames(0), flip_code_(0) {
}

OpenCVVideoReader::~OpenCVVideoReader() {
    release();
}

bool OpenCVVideoReader::open(const std::string& source) {
    release();

    // 摄像头（/dev/video*）可配置采集参数：
    //   RK_PIPE_CAM_PREFER_30FPS=1 实测能稳 30fps 的最大分辨率（优先于 CAM_MAX）
    //   RK_PIPE_CAM_MAX=1       自动枚举并选用面积最大、同面积下帧率最高的格式
    //   RK_PIPE_CAM_WIDTH/HEIGHT 手动指定分辨率
    //   RK_PIPE_CAM_FPS          手动指定帧率
    //   RK_PIPE_CAM_FORMAT       手动指定 fourcc（如 MJPG / YUYV）
    //   RK_PIPE_CAM_FLIP         画面翻转：1=水平镜像，2=垂直，3=180°（倒装）
    int cam_w = 0;
    int cam_h = 0;
    int cam_fps = 0;
    int cam_fourcc = -1;
    bool configured = false;

    if (isCameraDevice(source)) {
        const char* flip_env = std::getenv("RK_PIPE_CAM_FLIP");
        if (flip_env && *flip_env) {
            const int v = std::atoi(flip_env);
            // cv::flip flipCode：>0 水平，0 垂直，<0 180°
            if (v == 1) {
                flip_code_ = 1;
            } else if (v == 2) {
                flip_code_ = 0;
            } else if (v == 3) {
                flip_code_ = -1;
            } else {
                flip_code_ = 0;
            }
            std::cout << "[VideoInput] camera flip: " << flip_code_ << std::endl;
        }

        // 自动选档：PREFER_30FPS 优先（实测能稳 30fps 的最大分辨率），否则 CAM_MAX（面积最大）。
        // 必须在 cap.open 之前探测，否则设备被 OpenCV 占住 REQBUFS 会失败。
        const char* pref_env = std::getenv("RK_PIPE_CAM_PREFER_30FPS");
        if (pref_env && *pref_env && std::string(pref_env) != "0") {
            if (pickPreferred30FpsCameraFormat(source, cam_fourcc, cam_w, cam_h, cam_fps)) {
                configured = true;
                std::cout << "[VideoInput] camera prefer-30fps mode: " << cam_w << "x" << cam_h
                          << "@" << cam_fps << "fps fourcc=" << cam_fourcc << std::endl;
            }
        } else {
            const char* max_env = std::getenv("RK_PIPE_CAM_MAX");
            if (max_env && *max_env && std::string(max_env) != "0") {
                if (pickBestCameraFormat(source, cam_fourcc, cam_w, cam_h, cam_fps)) {
                    configured = true;
                    std::cout << "[VideoInput] camera max mode: " << cam_w << "x" << cam_h
                              << "@" << cam_fps << "fps fourcc=" << cam_fourcc << std::endl;
                }
            }
        }

        const char* w_env = std::getenv("RK_PIPE_CAM_WIDTH");
        if (w_env && *w_env) {
            const int v = std::atoi(w_env);
            if (v > 0) {
                cam_w = v;
                configured = true;
            }
        }
        const char* h_env = std::getenv("RK_PIPE_CAM_HEIGHT");
        if (h_env && *h_env) {
            const int v = std::atoi(h_env);
            if (v > 0) {
                cam_h = v;
                configured = true;
            }
        }
        const char* fps_env = std::getenv("RK_PIPE_CAM_FPS");
        if (fps_env && *fps_env) {
            const int v = std::atoi(fps_env);
            if (v > 0) {
                cam_fps = v;
                configured = true;
            }
        }
        const char* fmt_env = std::getenv("RK_PIPE_CAM_FORMAT");
        if (fmt_env && *fmt_env) {
            std::string s(fmt_env);
            if (s.size() >= 4) {
                cam_fourcc = cv::VideoWriter::fourcc(s[0], s[1], s[2], s[3]);
                configured = true;
            }
        }
    }

    if (isNetworkSource(source)) {
        int timeout_ms = 5000;
        const char* env = std::getenv("RK_PIPE_STREAM_TIMEOUT_MS");
        if (env && *env) {
            const int v = std::atoi(env);
            if (v > 0) {
                timeout_ms = v;
            }
        }
        // OpenCV 打开/读取网络流同样加超时，避免死地址阻塞
        cap.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms);
        cap.set(cv::CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms * 2);
    }
    if (!cap.open(source)) {
        return false;
    }

    if (isCameraDevice(source) && configured) {
        // V4L2 协商顺序：fourcc 优先，随后宽高，最后帧率
        if (cam_fourcc > 0) {
            cap.set(cv::CAP_PROP_FOURCC, cam_fourcc);
        }
        if (cam_w > 0) {
            cap.set(cv::CAP_PROP_FRAME_WIDTH, cam_w);
        }
        if (cam_h > 0) {
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, cam_h);
        }
        if (cam_fps > 0) {
            cap.set(cv::CAP_PROP_FPS, cam_fps);
        }
    }

    width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps = cap.get(cv::CAP_PROP_FPS);
    total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    if (isCameraDevice(source)) {
        std::cout << "[VideoInput] camera opened " << width << "x" << height << "@" << fps
                  << "fps" << std::endl;
    }
    return true;
}

bool OpenCVVideoReader::read(cv::Mat& frame) {
    if (!cap.isOpened()) {
        return false;
    }
    if (!cap.read(frame)) {
        return false;
    }
    // 倒装/镜像摄像头：采集侧全局翻转（影响检测、叠加、预览全部环节）
    if (flip_code_ != 0 && !frame.empty()) {
        cv::flip(frame, frame, flip_code_);
    }
    return true;
}

bool OpenCVVideoReader::read(image_buffer_t& image) {
    cv::Mat frame;
    if (!read(frame)) return false;

    // Allocate memory if needed
    size_t required_size = frame.total() * frame.elemSize();
    if (image.virt_addr == nullptr || image.size < (int)required_size) {
        if (image.virt_addr) free(image.virt_addr);
        image.virt_addr = (unsigned char*)malloc(required_size);
        image.size = (int)required_size;
    }

    memcpy(image.virt_addr, frame.data, required_size);
    image.width = frame.cols;
    image.height = frame.rows;
    image.width_stride = frame.step[0]; // Row stride
    image.height_stride = frame.rows;
    image.format = IMAGE_FORMAT_BGR888;
    image.fd = 0; // CPU memory

    return true;
}

int OpenCVVideoReader::getWidth() const {
    return width;
}

int OpenCVVideoReader::getHeight() const {
    return height;
}

double OpenCVVideoReader::getFPS() const {
    return fps;
}

int OpenCVVideoReader::getTotalFrames() const {
    return total_frames;
}

void OpenCVVideoReader::release() {
    if (cap.isOpened()) {
        cap.release();
    }
    width = 0;
    height = 0;
    fps = 0;
    total_frames = 0;
    flip_code_ = 0;
}

bool OpenCVVideoReader::isOpened() const {
    return cap.isOpened();
}

// RKMPPVideoReader implementation

struct RKMPPInternalState {
    AVFormatContext* fmt_ctx = nullptr;
    int video_stream_index = -1;
    MppCtx mpp_ctx = nullptr;
    MppApi* mpp_api = nullptr;
    MppPacket packet = nullptr;
    MppFrame frame = nullptr; // Holds the current frame reference
    AVBSFContext* bsf_ctx = nullptr;
    AVPacket* pending_pkt = nullptr;
    bool has_pending_pkt = false;
    bool eof_reached = false;
    uint64_t discard_count = 0;
    uint64_t errinfo_count = 0;
    uint64_t nobuffer_count = 0;
    bool err_summary_logged = false;
    uint64_t decoded_count = 0;
    uint64_t invalid_frame_count = 0;
    uint64_t info_change_count = 0;
    uint64_t last_log_count = 0;
    bool debug = false;
    // 背压节流：本地文件开启，避免 MPP 全速解码导致下游跟不上而丢帧。
    bool pacing_enabled = false;
    // 节流窗口内已喂未取帧的包数（每次 read() 重置）
    int paced_feed_count = 0;
    // 打开网络输入的超时中断：rw_timeout/stimeout 对 avformat_open_input 的
    // connect/握手阶段不保证生效（尤其黑洞路由会阻塞至内核 TCP 超时约 2 分钟），
    // 用 interrupt_callback 兜底，使打开阶段在可配置秒数内中断。
    std::atomic<bool> interrupt_enabled{false};
    std::chrono::steady_clock::time_point io_deadline;
    int open_timeout_ms = 5000;

    RKMPPInternalState() {
        pending_pkt = av_packet_alloc();
    }

    ~RKMPPInternalState() {
        if (pending_pkt) {
            av_packet_free(&pending_pkt);
            pending_pkt = nullptr;
        }
        if (bsf_ctx) {
            av_bsf_free(&bsf_ctx);
            bsf_ctx = nullptr;
        }
        if (frame) {
            mpp_frame_deinit(&frame);
            frame = nullptr;
        }
        if (packet) {
            mpp_packet_deinit(&packet);
            packet = nullptr;
        }
        if (mpp_ctx) {
             mpp_destroy(mpp_ctx);
             mpp_ctx = nullptr;
        }
        if (fmt_ctx) {
            avformat_close_input(&fmt_ctx);
            fmt_ctx = nullptr;
        }
    }
};

// avformat_open_input 的 I/O 中断回调：超过截止时间即返回 1 中断打开，
// 避免死地址/黑洞路由下 connect/握手无限阻塞（rw_timeout 不覆盖此阶段）。
static int rkmppInterruptCallback(void* opaque) {
    auto* state = static_cast<RKMPPInternalState*>(opaque);
    if (!state || !state->interrupt_enabled.load()) {
        return 0;
    }
    return std::chrono::steady_clock::now() > state->io_deadline ? 1 : 0;
}

RKMPPVideoReader::RKMPPVideoReader()
    : pipe_(nullptr), width(0), height(0), fps(0), total_frames(0), frame_bytes(0), internal_state_(nullptr) {
    internal_state_ = new RKMPPInternalState();
}

RKMPPVideoReader::~RKMPPVideoReader() {
    release();
    if (internal_state_) {
        delete static_cast<RKMPPInternalState*>(internal_state_);
        internal_state_ = nullptr;
    }
}

bool RKMPPVideoReader::open(const std::string& source) {
    release();
    auto state = static_cast<RKMPPInternalState*>(internal_state_);
    state->discard_count = 0;
    state->errinfo_count = 0;
    state->nobuffer_count = 0;
    state->err_summary_logged = false;
    state->decoded_count = 0;
    state->invalid_frame_count = 0;
    state->info_change_count = 0;
    state->last_log_count = 0;
    const char* debug_env = std::getenv("YOLOV8_MPP_DEBUG");
    state->debug = debug_env && debug_env[0] == '1';

    // Open input file
    AVDictionary* input_opts = nullptr;
    setNetworkInputOptions(source, &input_opts);
    if (isNetworkSource(source)) {
        // rw_timeout/stimeout 不保证覆盖 connect/握手阶段（黑洞路由可阻塞至内核
        // TCP 超时约 2 分钟），挂 interrupt_callback 使打开阶段在可配置秒数内中断。
        const char* timeout_env = std::getenv("RK_PIPE_STREAM_TIMEOUT_MS");
        if (timeout_env && *timeout_env) {
            const int v = std::atoi(timeout_env);
            if (v > 0) {
                state->open_timeout_ms = v;
            }
        }
        state->fmt_ctx = avformat_alloc_context();
        if (state->fmt_ctx) {
            state->fmt_ctx->interrupt_callback.callback = &rkmppInterruptCallback;
            state->fmt_ctx->interrupt_callback.opaque = state;
            state->interrupt_enabled.store(true);
            state->io_deadline = std::chrono::steady_clock::now() +
                                 std::chrono::milliseconds(state->open_timeout_ms);
        }
    }
    if (avformat_open_input(&state->fmt_ctx, source.c_str(), nullptr, &input_opts) != 0) {
        av_dict_free(&input_opts);
        state->interrupt_enabled.store(false);
        state->fmt_ctx = nullptr;  // avformat_open_input 失败时已释放内部 context
        std::cerr << "Failed to open video file: " << source << std::endl;
        return false;
    }
    state->interrupt_enabled.store(false);
    av_dict_free(&input_opts);

    if (avformat_find_stream_info(state->fmt_ctx, NULL) < 0) {
        std::cerr << "Could not find stream info" << std::endl;
        return false;
    }

    av_dump_format(state->fmt_ctx, 0, source.c_str(), 0);

    state->video_stream_index = -1;
    for (unsigned int i = 0; i < state->fmt_ctx->nb_streams; i++) {
        if (state->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            state->video_stream_index = i;
            break;
        }
    }

    if (state->video_stream_index == -1) {
        std::cerr << "No video stream found" << std::endl;
        return false;
    }

    AVCodecParameters* codecpar = state->fmt_ctx->streams[state->video_stream_index]->codecpar;
    std::cout << "Codec Extradata Size: " << codecpar->extradata_size << std::endl;
    width = codecpar->width;
    height = codecpar->height;

    AVStream* stream = state->fmt_ctx->streams[state->video_stream_index];
    if (stream->avg_frame_rate.den > 0) {
        fps = av_q2d(stream->avg_frame_rate);
    } else {
        fps = 30.0;
    }
    total_frames = stream->nb_frames;

    // Initialize MPP
    MppCodingType type = MPP_VIDEO_CodingUnused;
    const AVBitStreamFilter* bsf = nullptr;
    if (codecpar->codec_id == AV_CODEC_ID_H264) {
        type = MPP_VIDEO_CodingAVC;
        bsf = av_bsf_get_by_name("h264_mp4toannexb");
        std::cout << "Detected H264 codec" << std::endl;
    } else if (codecpar->codec_id == AV_CODEC_ID_HEVC) {
        type = MPP_VIDEO_CodingHEVC;
        bsf = av_bsf_get_by_name("hevc_mp4toannexb");
        std::cout << "Detected HEVC codec" << std::endl;
    } else {
        std::cerr << "Unsupported codec for MPP: " << codecpar->codec_id << std::endl;
        return false;
    }

    if (bsf) {
        if (av_bsf_alloc(bsf, &state->bsf_ctx) < 0) {
            std::cerr << "Failed to alloc bsf" << std::endl;
            return false;
        }
        avcodec_parameters_copy(state->bsf_ctx->par_in, codecpar);
        if (av_bsf_init(state->bsf_ctx) < 0) {
            std::cerr << "Failed to init bsf" << std::endl;
            return false;
        }
    }

    MPP_RET ret = mpp_create(&state->mpp_ctx, &state->mpp_api);
    if (ret != MPP_OK) {
        std::cerr << "mpp_create failed" << std::endl;
        return false;
    }

    ret = mpp_init(state->mpp_ctx, MPP_CTX_DEC, type);
    if (ret != MPP_OK) {
        std::cerr << "mpp_init failed" << std::endl;
        return false;
    }

    if (state->bsf_ctx && state->bsf_ctx->par_out->extradata_size > 0) {
        MppPacket extra_pkt = nullptr;
        mpp_packet_init(&extra_pkt, state->bsf_ctx->par_out->extradata, state->bsf_ctx->par_out->extradata_size);
        mpp_packet_set_data(extra_pkt, state->bsf_ctx->par_out->extradata);
        mpp_packet_set_size(extra_pkt, state->bsf_ctx->par_out->extradata_size);
        mpp_packet_set_extra_data(extra_pkt);

        ret = state->mpp_api->decode_put_packet(state->mpp_ctx, extra_pkt);
        if (ret != MPP_OK) {
            std::cerr << "Failed to send extradata: " << ret << std::endl;
        }
        mpp_packet_deinit(&extra_pkt);
    }

    // Enable split mode
    RK_U32 need_split = 1;
    ret = state->mpp_api->control(state->mpp_ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &need_split);
    if (ret != MPP_OK) {
        std::cerr << "Failed to set split mode: " << ret << std::endl;
        return false;
    }

    RK_U32 immediate = codecpar->video_delay > 0 ? 0 : 1;
    ret = state->mpp_api->control(state->mpp_ctx, MPP_DEC_SET_IMMEDIATE_OUT, &immediate);
    if (ret != MPP_OK) {
        std::cerr << "Failed to set immediate out: " << ret << std::endl;
        return false;
    }

    MppFrameFormat output_fmt = MPP_FMT_YUV420SP;
    ret = state->mpp_api->control(state->mpp_ctx, MPP_DEC_SET_OUTPUT_FORMAT, &output_fmt);
    if (ret != MPP_OK) {
        std::cerr << "Failed to set output format: " << ret << std::endl;
    }

    // 背压节流（实验性，默认关闭）：YOLOV8_MPP_PACING=1 显式启用。
    // 实测对本地文件无效：仍会 discard/errinfo 丢帧，且轮询把解码器饿死，
    // 平均 FPS 从 ~43 暴跌到 ~6。本地文件请走 OpenCV 软解（input_source 默认分流）。
    const char* pacing_env = std::getenv("YOLOV8_MPP_PACING");
    state->pacing_enabled = pacing_env && pacing_env[0] == '1';
    if (state->pacing_enabled) {
        std::cout << "[VideoInput] MPP pacing enabled (experimental, slow & may drop)" << std::endl;
    }

    mpp_packet_init(&state->packet, nullptr, 0);

    return true;
}

bool RKMPPVideoReader::read(image_buffer_t& image) {
    VideoReader::releaseFrame(image);
    auto state = static_cast<RKMPPInternalState*>(internal_state_);
    if (!state->mpp_ctx) return false;
    state->paced_feed_count = 0;

    int eof_drain_retries = 0;

    while (true) {
        // Release previous frame if any
        if (state->frame) {
            mpp_frame_deinit(&state->frame);
            state->frame = nullptr;
        }

        // Helper to try getting a frame from MPP
        auto try_get_frame = [&]() -> bool {
            RK_S32 ret = state->mpp_api->decode_get_frame(state->mpp_ctx, &state->frame);
            if (ret != MPP_OK) {
                if (ret != MPP_ERR_BUFFER_FULL) {
                    std::cerr << "decode_get_frame failed: " << ret << std::endl;
                }
                return false;
            }

            if (!state->frame) {
                return false;
            }

            if (mpp_frame_get_info_change(state->frame)) {
                state->info_change_count++;
                state->mpp_api->control(state->mpp_ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
                mpp_frame_deinit(&state->frame);
                state->frame = nullptr;
                return false;
            }
            if (mpp_frame_get_discard(state->frame)) {
                state->discard_count++;
                if (state->debug && state->discard_count % 60 == 0) {
                    std::cerr << "MPP discard count=" << state->discard_count << std::endl;
                }
                mpp_frame_deinit(&state->frame);
                state->frame = nullptr;
                return false;
            }
            if (mpp_frame_get_errinfo(state->frame)) {
                state->errinfo_count++;
                if (state->debug && state->errinfo_count % 60 == 0) {
                    std::cerr << "MPP errinfo count=" << state->errinfo_count << std::endl;
                }
                mpp_frame_deinit(&state->frame);
                state->frame = nullptr;
                return false;
            }

            MppBuffer buffer = mpp_frame_get_buffer(state->frame);
            if (!buffer) {
                state->nobuffer_count++;
                if (state->debug && state->nobuffer_count % 60 == 0) {
                    std::cerr << "MPP nobuffer count=" << state->nobuffer_count << std::endl;
                }
                mpp_frame_deinit(&state->frame);
                state->frame = nullptr;
                return false;
            }

            image.fd = mpp_buffer_get_fd(buffer);
            image.virt_addr = (unsigned char*)mpp_buffer_get_ptr(buffer);
            image.width = mpp_frame_get_width(state->frame);
            image.height = mpp_frame_get_height(state->frame);
            image.width_stride = mpp_frame_get_hor_stride(state->frame);
            image.height_stride = mpp_frame_get_ver_stride(state->frame);
            MppFrameFormat fmt = mpp_frame_get_fmt(state->frame);
            if (fmt == MPP_FMT_YUV420SP_VU) {
                image.format = IMAGE_FORMAT_YUV420SP_NV21;
            } else if (fmt == MPP_FMT_YUV420SP) {
                image.format = IMAGE_FORMAT_YUV420SP_NV12;
            } else {
                image.format = IMAGE_FORMAT_YUV420SP_NV12;
                std::cerr << "Unsupported MPP format: " << fmt << ", fallback to NV12" << std::endl;
            }
            image.size = image.width_stride * image.height_stride * 3 / 2;
            // #region debug-point C:rkmpp-frame-format
            if (debugYuvEnabled()) {
                std::fprintf(stderr,
                             "[DEBUG][YUV][C] RKMPP read fmt=%d mapped=%d w=%d h=%d ws=%d hs=%d fd=%d ptr=%p\n",
                             static_cast<int>(fmt),
                             static_cast<int>(image.format),
                             image.width,
                             image.height,
                             image.width_stride,
                             image.height_stride,
                             image.fd,
                             static_cast<void*>(image.virt_addr));
            }
            // #endregion

            if (image.width <= 0 || image.height <= 0 || image.width_stride <= 0 || image.height_stride <= 0 ||
                (image.fd <= 0 && image.virt_addr == nullptr)) {
                state->invalid_frame_count++;
                if (state->debug && state->invalid_frame_count % 30 == 0) {
                    std::cerr << "MPP invalid frame count=" << state->invalid_frame_count
                              << " w=" << image.width
                              << " h=" << image.height
                              << " ws=" << image.width_stride
                              << " hs=" << image.height_stride
                              << " fd=" << image.fd
                              << " ptr=" << image.virt_addr
                              << std::endl;
                }
                mpp_frame_deinit(&state->frame);
                state->frame = nullptr;
                return false;
            }

            // Transfer ownership to image
            image.priv_data = state->frame;
            state->frame = nullptr;

            state->decoded_count++;
            if (state->debug && state->decoded_count - state->last_log_count >= 120) {
                state->last_log_count = state->decoded_count;
                std::cerr << "MPP decoded=" << state->decoded_count
                          << " discard=" << state->discard_count
                          << " errinfo=" << state->errinfo_count
                          << " nobuffer=" << state->nobuffer_count
                          << " info_change=" << state->info_change_count
                          << " invalid=" << state->invalid_frame_count
                          << std::endl;
            }

            return true;
        };

        auto fill_mpp_packet = [&](AVPacket* pkt) {
            mpp_packet_set_data(state->packet, pkt->data);
            mpp_packet_set_size(state->packet, pkt->size);
            mpp_packet_set_pos(state->packet, pkt->data);
            mpp_packet_set_length(state->packet, pkt->size);

            AVStream* stream = state->fmt_ctx->streams[state->video_stream_index];
            if (pkt->pts != AV_NOPTS_VALUE) {
                int64_t pts_us = av_rescale_q(pkt->pts, stream->time_base, AV_TIME_BASE_Q);
                mpp_packet_set_pts(state->packet, pts_us);
            }
            if (pkt->dts != AV_NOPTS_VALUE) {
                int64_t dts_us = av_rescale_q(pkt->dts, stream->time_base, AV_TIME_BASE_Q);
                mpp_packet_set_dts(state->packet, dts_us);
            } else if (pkt->pts != AV_NOPTS_VALUE) {
                int64_t pts_us = av_rescale_q(pkt->pts, stream->time_base, AV_TIME_BASE_Q);
                mpp_packet_set_dts(state->packet, pts_us);
            }
        };

        // Helper to process a packet with retry logic for BUFFER_FULL
        auto process_pkt = [&](AVPacket* pkt) -> bool {
            while (true) {
                fill_mpp_packet(pkt);

                RK_S32 ret = state->mpp_api->decode_put_packet(state->mpp_ctx, state->packet);

                if (ret == MPP_OK) {
                    if (try_get_frame()) return true;
                    return false;
                } else if (ret == MPP_ERR_BUFFER_FULL) {
                    if (try_get_frame()) {
                        av_packet_unref(state->pending_pkt);
                        av_packet_ref(state->pending_pkt, pkt);
                        state->has_pending_pkt = true;
                        return true;
                    }
                    usleep(1000);
                } else {
                    return false;
                }
            }
        };

        // 背压节流：先取已解码帧，队列空时才继续喂包。
        // 实测 MPP 解码超前时，积压帧数超过解码缓冲容量（约 60 帧）后会开始
        // discard/errinfo 级联丢弃；先取后喂把积压压在重排序深度附近，避免溢出。
        if (state->pacing_enabled && try_get_frame()) {
            return true;
        }

        // 0. If EOF reached previously, keep draining until MPP is really empty.
        if (state->eof_reached) {
            if (try_get_frame()) return true;
            if (eof_drain_retries < 200) {
                usleep(10000);
                eof_drain_retries++;
                continue;
            }
            if (!state->err_summary_logged && (state->discard_count || state->errinfo_count || state->nobuffer_count)) {
                std::cerr << "MPP dropped frames summary: discard=" << state->discard_count
                          << " errinfo=" << state->errinfo_count
                          << " nobuffer=" << state->nobuffer_count << std::endl;
                state->err_summary_logged = true;
            }
            return false;
        }

        // 1. Handle pending packet
        if (state->has_pending_pkt) {
            while (true) {
                fill_mpp_packet(state->pending_pkt);

                RK_S32 ret = state->mpp_api->decode_put_packet(state->mpp_ctx, state->packet);
                if (ret == MPP_OK) {
                    av_packet_unref(state->pending_pkt);
                    state->has_pending_pkt = false;
                    if (try_get_frame()) return true;
                    break;
                } else if (ret == MPP_ERR_BUFFER_FULL) {
                    if (try_get_frame()) return true;
                    usleep(2000);
                } else {
                    av_packet_unref(state->pending_pkt);
                    state->has_pending_pkt = false;
                    break;
                }
            }
        }

        // 2. Drain BSF
        if (state->bsf_ctx) {
            AVPacket* bsf_pkt = av_packet_alloc();
            if (!bsf_pkt) {
                return false;
            }
            while (av_bsf_receive_packet(state->bsf_ctx, bsf_pkt) == 0) {
                if (process_pkt(bsf_pkt)) {
                    av_packet_unref(bsf_pkt);
                    av_packet_free(&bsf_pkt);
                    return true;
                }
                av_packet_unref(bsf_pkt);
            }
            av_packet_free(&bsf_pkt);
        }

        // 3. Read new packets
        AVPacket* av_pkt = av_packet_alloc();
        if (!av_pkt) {
            return false;
        }

        int read_ret = 0;
        while ((read_ret = av_read_frame(state->fmt_ctx, av_pkt)) >= 0) {
            if (av_pkt->stream_index != state->video_stream_index) {
                av_packet_unref(av_pkt);
                continue;
            }

            if (state->bsf_ctx) {
                int send_ret = av_bsf_send_packet(state->bsf_ctx, av_pkt);
                if (send_ret == 0) {
                    AVPacket* bsf_pkt = av_packet_alloc();
                    while (av_bsf_receive_packet(state->bsf_ctx, bsf_pkt) == 0) {
                        if (process_pkt(bsf_pkt)) {
                            av_packet_unref(bsf_pkt);
                            av_packet_unref(av_pkt);
                            av_packet_free(&bsf_pkt);
                            av_packet_free(&av_pkt);
                            return true;
                        }
                        av_packet_unref(bsf_pkt);
                    }
                    av_packet_free(&bsf_pkt);
                } else {
                    std::cerr << "BSF send failed: " << send_ret << std::endl;
                }
            } else {
                if (process_pkt(av_pkt)) {
                    av_packet_unref(av_pkt);
                    av_packet_free(&av_pkt);
                    return true;
                }
            }

            // 背压节流（实验性）：小窗口喂包 + 短轮询。
            // MPP 解码是流水线式，喂 1 个包不会立刻出帧（需超前 2~3 包），
            // 因此每次喂一批（4 包）再短轮询：既不让输出缓冲溢出（避免丢帧），
            // 也不把解码器饿死（避免掉速）。稳态下帧随喂随出，几乎不进入轮询。
            if (state->pacing_enabled) {
                state->paced_feed_count++;
                if (state->paced_feed_count >= 4) {
                    int paced_polls = 0;
                    while (paced_polls < 15) {  // 15*2ms = 30ms
                        if (try_get_frame()) {
                            state->paced_feed_count = 0;
                            av_packet_unref(av_pkt);
                            av_packet_free(&av_pkt);
                            return true;
                        }
                        usleep(2000);
                        paced_polls++;
                    }
                    state->paced_feed_count = 0;  // 窗口期仍无帧，重置后继续下一批
                }
            }

            av_packet_unref(av_pkt);
        }

        av_packet_free(&av_pkt);

        if (read_ret == AVERROR_EOF) {
            if (!state->eof_reached) {
                std::cerr << "EOF reached, sending EOS packet" << std::endl;
                mpp_packet_set_eos(state->packet);
                mpp_packet_set_size(state->packet, 0);
                mpp_packet_set_data(state->packet, NULL);
                RK_S32 ret = state->mpp_api->decode_put_packet(state->mpp_ctx, state->packet);
                if (ret != MPP_OK) {
                    std::cerr << "Failed to send EOS packet: " << ret << std::endl;
                }
                state->eof_reached = true;
                eof_drain_retries = 0;
            }
            if (try_get_frame()) return true;
            usleep(2000);
            continue;
        }

        if (read_ret == AVERROR(EAGAIN) || read_ret == AVERROR(EINTR)) {
            if (try_get_frame()) return true;
            usleep(2000);
            continue;
        }

        if (try_get_frame()) return true;
        usleep(1000);
    }
}

bool RKMPPVideoReader::read(cv::Mat& frame) {
    image_buffer_t img;
    memset(&img, 0, sizeof(img));

    if (!read(img)) return false;

    if (img.fd > 0) {
        // Need to map memory and convert to BGR
        // This is slow path
        MppFrame frame_ptr = (MppFrame)img.priv_data;
        MppBuffer buffer = mpp_frame_get_buffer(frame_ptr);
        void* ptr = mpp_buffer_get_ptr(buffer);

        if (ptr) {
            // Handle stride gaps correctly
            if (img.height_stride > img.height) {
                // Copy Y and UV planes to a contiguous buffer to ensure correct conversion
                cv::Mat contiguous_yuv(img.height * 3 / 2, img.width, CV_8UC1);

                uint8_t* src_y = (uint8_t*)ptr;
                uint8_t* dst_y = contiguous_yuv.data;
                for (int y = 0; y < img.height; ++y) {
                    memcpy(dst_y + y * img.width, src_y + y * img.width_stride, img.width);
                }

                uint8_t* src_uv = (uint8_t*)ptr + img.width_stride * img.height_stride;
                uint8_t* dst_uv = contiguous_yuv.data + img.width * img.height;
                for (int y = 0; y < img.height / 2; ++y) {
                    memcpy(dst_uv + y * img.width, src_uv + y * img.width_stride, img.width);
                }

                int cvt_code = img.format == IMAGE_FORMAT_YUV420SP_NV21 ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12;
                // #region debug-point D:reader-bgr-convert-strided
                if (debugYuvEnabled()) {
                    std::fprintf(stderr,
                                 "[DEBUG][YUV][D] reader strided fmt=%d w=%d h=%d ws=%d hs=%d cvt=%s\n",
                                 static_cast<int>(img.format),
                                 img.width,
                                 img.height,
                                 img.width_stride,
                                 img.height_stride,
                                 cvt_code == cv::COLOR_YUV2BGR_NV21 ? "NV21" : "NV12");
                }
                // #endregion
                cv::cvtColor(contiguous_yuv, frame, cvt_code);
            } else {
                // Fast path for contiguous memory
                cv::Mat yuv(img.height * 3 / 2, img.width, CV_8UC1, ptr, img.width_stride);
                int cvt_code = img.format == IMAGE_FORMAT_YUV420SP_NV21 ? cv::COLOR_YUV2BGR_NV21 : cv::COLOR_YUV2BGR_NV12;
                // #region debug-point E:reader-bgr-convert-fast
                if (debugYuvEnabled()) {
                    std::fprintf(stderr,
                                 "[DEBUG][YUV][E] reader fast fmt=%d w=%d h=%d ws=%d hs=%d cvt=%s\n",
                                 static_cast<int>(img.format),
                                 img.width,
                                 img.height,
                                 img.width_stride,
                                 img.height_stride,
                                 cvt_code == cv::COLOR_YUV2BGR_NV21 ? "NV21" : "NV12");
                }
                // #endregion
                cv::cvtColor(yuv, frame, cvt_code);
            }

            // Adjust frame size to actual size (stride might be larger)
            if (frame.cols != img.width || frame.rows != img.height) {
                frame = frame(cv::Rect(0, 0, img.width, img.height)).clone();
            }
            VideoReader::releaseFrame(img);
            return true;
        }
    }

    VideoReader::releaseFrame(img);
    return false;
}

int RKMPPVideoReader::getWidth() const {
    return width;
}

int RKMPPVideoReader::getHeight() const {
    return height;
}

double RKMPPVideoReader::getFPS() const {
    return fps;
}

int RKMPPVideoReader::getTotalFrames() const {
    return total_frames;
}

void RKMPPVideoReader::release() {
    auto state = static_cast<RKMPPInternalState*>(internal_state_);
    if (state->frame) {
        mpp_frame_deinit(&state->frame);
        state->frame = nullptr;
    }
    if (state->packet) {
        mpp_packet_deinit(&state->packet);
        state->packet = nullptr;
    }
    if (state->mpp_ctx) {
         mpp_destroy(state->mpp_ctx);
         state->mpp_ctx = nullptr;
    }
    if (state->fmt_ctx) {
        avformat_close_input(&state->fmt_ctx);
        state->fmt_ctx = nullptr;
    }
    width = 0;
    height = 0;
    fps = 0;
    total_frames = 0;
}

bool RKMPPVideoReader::isOpened() const {
    auto state = static_cast<RKMPPInternalState*>(internal_state_);
    return state->mpp_ctx != nullptr;
}

bool RKMPPVideoReader::isEofReached() const {
    auto state = static_cast<RKMPPInternalState*>(internal_state_);
    return state && state->eof_reached;
}

std::uint64_t RKMPPVideoReader::droppedFrameCount() const {
    auto state = static_cast<RKMPPInternalState*>(internal_state_);
    return state ? state->discard_count + state->errinfo_count : 0;
}

// FFmpegRKMPPVideoReader implementation
// 复用 ffmpeg 的 h264_rkmpp / hevc_rkmpp 解码器（已验证能完整解码本地文件），
// 通过标准 avcodec_send_packet / avcodec_receive_frame 循环驱动。

struct FFmpegRKMPPState {
    AVFormatContext* fmt_ctx = nullptr;
    int video_stream_index = -1;
    AVCodecContext* codec_ctx = nullptr;
    const AVCodec* codec = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    FFmpegRkmppDmaPool* dma_pool = nullptr;  // 零拷贝 dma-buf 池（可为空）

    FFmpegRKMPPState() {
        frame = av_frame_alloc();
        packet = av_packet_alloc();
    }
    ~FFmpegRKMPPState() {
        if (frame) av_frame_free(&frame);
        if (packet) av_packet_free(&packet);
        if (codec_ctx) avcodec_free_context(&codec_ctx);
        if (fmt_ctx) avformat_close_input(&fmt_ctx);
        ffmpegRkmppDestroyPool(dma_pool);
    }
};

namespace {
// 喂一个数据包给解码器。到达 EOF 时发送 flush 空包让解码器输出剩余帧。
bool ffmpegRkmppFeedNext(FFmpegRKMPPState* state) {
    while (true) {
        int r = av_read_frame(state->fmt_ctx, state->packet);
        if (r < 0) {
            if (r == AVERROR_EOF) {
                return avcodec_send_packet(state->codec_ctx, nullptr) >= 0;
            }
            return false;
        }
        if (state->packet->stream_index != state->video_stream_index) {
            av_packet_unref(state->packet);
            continue;
        }
        int sret = avcodec_send_packet(state->codec_ctx, state->packet);
        av_packet_unref(state->packet);
        if (sret == AVERROR(EAGAIN)) {
            return true;  // 解码器输入满，先取帧再继续喂
        }
        if (sret < 0) {
            return false;
        }
        return true;
    }
}

// DRM 描述符中 NV21 的 fourcc（避免引入 libdrm 头依赖）。
constexpr uint32_t kDrmFmtNV21 = 0x3132564e;  // 'NV21'

// NV12/NV21 双平面 → BGR。用 cvtColorTwoPlane 正确处理各自步长，避免单 Mat
// 拼接对 UV 平面偏移/对齐的错误假设导致颜色错乱。
bool ffmpegRkmppNv12ToBgr(const AVFrame* sw, cv::Mat& out, int cvt_code) {
    if (!sw->data[0] || !sw->data[1]) {
        return false;
    }
    cv::Mat y(sw->height, sw->width, CV_8UC1, sw->data[0], sw->linesize[0]);
    cv::Mat uv(sw->height / 2, sw->width / 2, CV_8UC2, sw->data[1], sw->linesize[1]);
    cv::cvtColorTwoPlane(y, uv, out, cvt_code);
    return !out.empty();
}
}  // namespace

FFmpegRKMPPVideoReader::FFmpegRKMPPVideoReader()
    : impl_(nullptr), width(0), height(0), fps(0), total_frames(0) {}

FFmpegRKMPPVideoReader::~FFmpegRKMPPVideoReader() {
    release();
}

bool FFmpegRKMPPVideoReader::open(const std::string& source) {
    release();
    impl_ = new FFmpegRKMPPState();
    auto state = static_cast<FFmpegRKMPPState*>(impl_);

    AVDictionary* input_opts = nullptr;
    setNetworkInputOptions(source, &input_opts);
    if (avformat_open_input(&state->fmt_ctx, source.c_str(), nullptr, &input_opts) != 0) {
        av_dict_free(&input_opts);
        std::cerr << "Failed to open video file: " << source << std::endl;
        return false;
    }
    av_dict_free(&input_opts);
    if (avformat_find_stream_info(state->fmt_ctx, nullptr) < 0) {
        std::cerr << "Could not find stream info" << std::endl;
        return false;
    }
    av_dump_format(state->fmt_ctx, 0, source.c_str(), 0);

    for (unsigned int i = 0; i < state->fmt_ctx->nb_streams; i++) {
        if (state->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            state->video_stream_index = static_cast<int>(i);
            break;
        }
    }
    if (state->video_stream_index < 0) {
        std::cerr << "No video stream found" << std::endl;
        return false;
    }

    AVCodecParameters* par = state->fmt_ctx->streams[state->video_stream_index]->codecpar;
    const char* decoder_name = nullptr;
    if (par->codec_id == AV_CODEC_ID_H264) {
        decoder_name = "h264_rkmpp";
    } else if (par->codec_id == AV_CODEC_ID_HEVC) {
        decoder_name = "hevc_rkmpp";
    } else {
        std::cerr << "Unsupported codec for FFmpegRKMPP reader: " << par->codec_id << std::endl;
        return false;
    }

    state->codec = avcodec_find_decoder_by_name(decoder_name);
    if (!state->codec) {
        std::cerr << "Decoder not found: " << decoder_name << std::endl;
        return false;
    }

    state->codec_ctx = avcodec_alloc_context3(state->codec);
    if (!state->codec_ctx) {
        return false;
    }
    if (avcodec_parameters_to_context(state->codec_ctx, par) < 0) {
        std::cerr << "Failed to copy codec parameters" << std::endl;
        return false;
    }
    state->codec_ctx->thread_count = 1;  // 硬件解码器无需多线程
    if (avcodec_open2(state->codec_ctx, state->codec, nullptr) < 0) {
        std::cerr << "Failed to open decoder: " << decoder_name << std::endl;
        return false;
    }

    width = par->width;
    height = par->height;
    AVStream* stream = state->fmt_ctx->streams[state->video_stream_index];
    if (stream->avg_frame_rate.den > 0) {
        fps = av_q2d(stream->avg_frame_rate);
    } else {
        fps = 30.0;
    }
    total_frames = static_cast<int>(stream->nb_frames);

    // 零拷贝 dma-buf 池：把解码出的 NV12 帧导入自有 dma-buf，producer 只传 fd
    state->dma_pool = ffmpegRkmppCreatePool(width, height);
    if (!state->dma_pool) {
        std::cerr << "[FFmpegRKMPP] dma-heap unavailable, falling back to BGR copy path" << std::endl;
    }

    std::cout << "[VideoInput] FFmpegRKMPP reader (decoder=" << decoder_name << ")" << std::endl;
    return true;
}

bool FFmpegRKMPPVideoReader::read(cv::Mat& frame) {
    auto state = static_cast<FFmpegRKMPPState*>(impl_);
    if (!state || !state->codec_ctx) {
        return false;
    }

    while (true) {
        int ret = avcodec_receive_frame(state->codec_ctx, state->frame);
        if (ret == 0) {
            bool ok = false;
            AVFrame* f = state->frame;
            if (f->format == AV_PIX_FMT_DRM_PRIME) {
                // 硬件帧：下载到 NV12 软件帧再转 BGR
                int cvt_code = cv::COLOR_YUV2BGR_NV12;
                if (f->data[0]) {
                    AVDRMFrameDescriptor* desc =
                        reinterpret_cast<AVDRMFrameDescriptor*>(f->data[0]);
                    if (desc->nb_layers > 0) {
                        uint32_t drm_fmt = desc->layers[0].format;
                        if (drm_fmt == kDrmFmtNV21) {
                            cvt_code = cv::COLOR_YUV2BGR_NV21;
                        }
                    }
                }
                AVFrame* sw = av_frame_alloc();
                sw->format = AV_PIX_FMT_NV12;
                sw->width = f->width;
                sw->height = f->height;
                if (av_hwframe_transfer_data(sw, f, 0) == 0) {
                    ok = ffmpegRkmppNv12ToBgr(sw, frame, cvt_code);
                } else {
                    std::cerr << "[FFmpegRKMPP] hwframe transfer failed" << std::endl;
                }
                av_frame_free(&sw);
            } else if (f->format == AV_PIX_FMT_NV12 || f->format == AV_PIX_FMT_NV21) {
                int cvt_code = (f->format == AV_PIX_FMT_NV21)
                                   ? cv::COLOR_YUV2BGR_NV21
                                   : cv::COLOR_YUV2BGR_NV12;
                ok = ffmpegRkmppNv12ToBgr(f, frame, cvt_code);
            } else {
                std::cerr << "[FFmpegRKMPP] unsupported frame format: " << f->format << std::endl;
            }
            av_frame_unref(state->frame);
            return ok;
        }
        if (ret == AVERROR(EAGAIN)) {
            if (!ffmpegRkmppFeedNext(state)) {
                return false;
            }
            continue;
        }
        if (ret == AVERROR_EOF) {
            return false;
        }
        std::cerr << "[FFmpegRKMPP] avcodec_receive_frame error: " << ret << std::endl;
        return false;
    }
}

bool FFmpegRKMPPVideoReader::read(image_buffer_t& image) {
    auto state = static_cast<FFmpegRKMPPState*>(impl_);
    if (!state || !state->codec_ctx) {
        return false;
    }

    while (true) {
        int ret = avcodec_receive_frame(state->codec_ctx, state->frame);
        if (ret == 0) {
            AVFrame* f = state->frame;
            // NV12/NV21 帧：优先导入自有 dma-buf 实现零拷贝
            // （producer 只传 fd，跳过 cvtColor/clone；worker 的 rga 直接 NV12 letterbox）
            if (f->format == AV_PIX_FMT_NV12 || f->format == AV_PIX_FMT_NV21) {
                if (state->dma_pool) {
                    int slot = state->dma_pool->acquireSlot();
                    FFmpegRkmppDmaSlot& s = state->dma_pool->slots[slot];
                    int stride = state->dma_pool->stride;
                    int height = state->dma_pool->height;
                    int width = f->width;
                    unsigned char* dst = s.addr;
                    const uint8_t* srcY = f->data[0];
                    const uint8_t* srcUV = f->data[1];
                    int srcYLine = f->linesize[0];
                    int srcUVLine = f->linesize[1];
                    for (int y = 0; y < height; ++y) {
                        memcpy(dst + static_cast<size_t>(y) * stride,
                               srcY + static_cast<size_t>(y) * srcYLine, width);
                    }
                    unsigned char* dstUV = dst + static_cast<size_t>(stride) * height;
                    for (int y = 0; y < height / 2; ++y) {
                        memcpy(dstUV + static_cast<size_t>(y) * stride,
                               srcUV + static_cast<size_t>(y) * srcUVLine, width);
                    }
                    image.fd = s.fd;
                    image.virt_addr = s.addr;
                    image.width = width;
                    image.height = height;
                    image.width_stride = stride;
                    image.height_stride = height;
                    image.format = (f->format == AV_PIX_FMT_NV21)
                                       ? IMAGE_FORMAT_YUV420SP_NV21
                                       : IMAGE_FORMAT_YUV420SP_NV12;
                    image.size = static_cast<int>(state->dma_pool->buf_size);
                    image.priv_data =
                        new FFmpegRkmppFrameRef{kFFmpegRkmppFrameMagic, state->dma_pool, slot};
                    av_frame_unref(state->frame);
                    return true;
                }

                // 无池（dma-heap 不可用）：退回直接转 BGR
                int cvt_code = (f->format == AV_PIX_FMT_NV21)
                                   ? cv::COLOR_YUV2BGR_NV21
                                   : cv::COLOR_YUV2BGR_NV12;
                cv::Mat bgr;
                if (ffmpegRkmppNv12ToBgr(f, bgr, cvt_code) && !bgr.empty()) {
                    size_t required = bgr.total() * bgr.elemSize();
                    if (image.virt_addr == nullptr || image.size < static_cast<int>(required)) {
                        if (image.virt_addr) free(image.virt_addr);
                        image.virt_addr = static_cast<unsigned char*>(malloc(required));
                        image.size = static_cast<int>(required);
                    }
                    memcpy(image.virt_addr, bgr.data, required);
                    image.width = bgr.cols;
                    image.height = bgr.rows;
                    image.width_stride = bgr.step[0];
                    image.height_stride = bgr.rows;
                    image.format = IMAGE_FORMAT_BGR888;
                    image.fd = 0;
                    av_frame_unref(state->frame);
                    return true;
                }
                av_frame_unref(state->frame);
                return false;
            }

            // 其它格式：下载 + NV12->BGR
            bool ok = false;
            cv::Mat bgr;
            AVFrame* sw = av_frame_alloc();
            sw->format = AV_PIX_FMT_NV12;
            sw->width = f->width;
            sw->height = f->height;
            if (av_hwframe_transfer_data(sw, f, 0) == 0) {
                ok = ffmpegRkmppNv12ToBgr(sw, bgr, cv::COLOR_YUV2BGR_NV12);
            }
            av_frame_free(&sw);
            av_frame_unref(state->frame);
            if (!ok || bgr.empty()) {
                return false;
            }
            size_t required = bgr.total() * bgr.elemSize();
            if (image.virt_addr == nullptr || image.size < static_cast<int>(required)) {
                if (image.virt_addr) free(image.virt_addr);
                image.virt_addr = static_cast<unsigned char*>(malloc(required));
                image.size = static_cast<int>(required);
            }
            memcpy(image.virt_addr, bgr.data, required);
            image.width = bgr.cols;
            image.height = bgr.rows;
            image.width_stride = bgr.step[0];
            image.height_stride = bgr.rows;
            image.format = IMAGE_FORMAT_BGR888;
            image.fd = 0;
            return true;
        }
        if (ret == AVERROR(EAGAIN)) {
            if (!ffmpegRkmppFeedNext(state)) {
                return false;
            }
            continue;
        }
        if (ret == AVERROR_EOF) {
            return false;
        }
        std::cerr << "[FFmpegRKMPP] avcodec_receive_frame error: " << ret << std::endl;
        return false;
    }
}

int FFmpegRKMPPVideoReader::getWidth() const {
    return width;
}

int FFmpegRKMPPVideoReader::getHeight() const {
    return height;
}

double FFmpegRKMPPVideoReader::getFPS() const {
    return fps;
}

int FFmpegRKMPPVideoReader::getTotalFrames() const {
    return total_frames;
}

void FFmpegRKMPPVideoReader::release() {
    delete static_cast<FFmpegRKMPPState*>(impl_);
    impl_ = nullptr;
    width = 0;
    height = 0;
    fps = 0;
    total_frames = 0;
}

bool FFmpegRKMPPVideoReader::isOpened() const {
    auto state = static_cast<FFmpegRKMPPState*>(impl_);
    return state && state->codec_ctx;
}

// VideoReaderFactory implementation
std::unique_ptr<VideoReader> VideoReaderFactory::createReader(ReaderType type) {
    switch (type) {
        case ReaderType::OPENCV:
            return std::make_unique<OpenCVVideoReader>();
        case ReaderType::FFMPEG:
            throw std::runtime_error("FFMPEG video reader not implemented yet");
        case ReaderType::GSTREAMER:
            throw std::runtime_error("GStreamer video reader not implemented yet");
        case ReaderType::RKMPP:
            return std::make_unique<RKMPPVideoReader>();
        case ReaderType::FFMPEG_RKMPP:
            return std::make_unique<FFmpegRKMPPVideoReader>();
        default:
            throw std::invalid_argument("Invalid video reader type");
    }
}
