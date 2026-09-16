#include "utils/draw_utils.h"
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <rga/im2d.h>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <atomic>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include "postprocess/postprocess.h"

// 显示阈值：与默认 conf_threshold(0.25) 对齐。原 0.5 会把低分检测(0.25~0.5)全过滤，
// 导致分数天然偏低/量化的模型(如 yolo26 INT8)几乎不画框。RK_PIPE_DISPLAY_THRESH 可覆盖。
static const float DISPLAY_THRESH = []() {
    const char* v = getenv("RK_PIPE_DISPLAY_THRESH");
    if (v && *v && strcmp(v, "0") != 0) {
        float t = strtof(v, nullptr);
        if (t > 0.0f && t <= 1.0f) return t;
    }
    return 0.3f;
}();

float displayThreshold() {
    return DISPLAY_THRESH;
}

// 是否隐藏所有检测框（RK_PIPE_HIDE_BOXES=1）：只画掩膜/骨架/深度伪彩，不画任何框
static bool hideBoxesEnabled() {
    const char* v = std::getenv("RK_PIPE_HIDE_BOXES");
    return v && *v && std::string(v) != "0";
}

// seg 黑底视图（RK_PIPE_SEG_BLACK_BG=1）：整帧背景置黑，只显示掩膜区域。
// 适合 seg 单独视图或 seg 主任务 + 后画层（骨架/伪彩在掩膜之后画，仍可见）。
static bool segBlackBgEnabled() {
    const char* v = std::getenv("RK_PIPE_SEG_BLACK_BG");
    return v && *v && std::string(v) != "0";
}

static float calcOverlayScaleFactor(const image_buffer_t& frame) {
    if (frame.width <= 0 || frame.height <= 0) {
        return 1.0f;
    }
    float scale_w = static_cast<float>(frame.width) / 1280.0f;
    float scale_h = static_cast<float>(frame.height) / 720.0f;
    float scale = std::min(scale_w, scale_h);
    if (scale < 0.5f) {
        scale = 0.5f;
    }
    return scale;
}

static int calcOverlayTextScale(const image_buffer_t& frame) {
    float scale = calcOverlayScaleFactor(frame);
    return std::max(1, static_cast<int>(std::round(scale)));
}

static int calcOverlaySize(const image_buffer_t& frame, int base) {
    float scale = calcOverlayScaleFactor(frame);
    return std::max(1, static_cast<int>(std::round(base * scale)));
}

static bool fillRectNV12CPU(image_buffer_t& frame, int x, int y, int width, int height, const cv::Scalar& color);

static bool ensureRga() {
    static bool configured = false;
    if (configured) {
        return true;
    }
    configured = true;
    return true;
}

static bool rgaStatusOk(IM_STATUS status) {
    return status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR;
}

static void logRgaFailure(const char* tag, IM_STATUS status) {
    std::cerr << "[DrawUtils] " << tag << " failed (ret=" << status << ", " << imStrError(status) << ")" << std::endl;
}

struct DmaMappedBuffer {
    int fd = -1;
    void* addr = nullptr;
    size_t size = 0;
};

static void releaseDmaMappedBuffer(DmaMappedBuffer& buf) {
    if (buf.addr && buf.addr != MAP_FAILED && buf.size > 0) {
        munmap(buf.addr, buf.size);
    }
    if (buf.fd >= 0) {
        close(buf.fd);
    }
    buf.fd = -1;
    buf.addr = nullptr;
    buf.size = 0;
}

static bool dmaHeapAllocFrom(const char* heap_path, size_t size, int& out_fd, void*& out_addr) {
    int heap_fd = open(heap_path, O_RDONLY | O_CLOEXEC);
    if (heap_fd < 0) {
        return false;
    }

    dma_heap_allocation_data data;
    std::memset(&data, 0, sizeof(data));
    data.len = size;
    data.fd_flags = O_RDWR | O_CLOEXEC;
    data.heap_flags = 0;

    int ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data);
    close(heap_fd);
    if (ret != 0 || data.fd < 0) {
        return false;
    }

    void* addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, data.fd, 0);
    if (addr == MAP_FAILED) {
        close(data.fd);
        return false;
    }

    out_fd = data.fd;
    out_addr = addr;
    return true;
}

static bool dmaHeapAllocDma32(size_t size, int& out_fd, void*& out_addr) {
    static const char* kHeaps[] = {
        "/dev/dma_heap/system-uncached-dma32",
        "/dev/dma_heap/system-dma32",
        "/dev/dma_heap/dma32_uncached",
        "/dev/dma_heap/dma32",
    };

    for (const char* heap_path : kHeaps) {
        if (dmaHeapAllocFrom(heap_path, size, out_fd, out_addr)) {
            return true;
        }
    }
    return false;
}

static bool ensureDma32MappedBuffer(size_t required_size, DmaMappedBuffer& buf) {
    size_t size = (required_size + 4095u) & ~4095u;
    if (buf.addr && buf.addr != MAP_FAILED && buf.fd >= 0 && buf.size >= size) {
        return true;
    }
    releaseDmaMappedBuffer(buf);

    int fd = -1;
    void* addr = nullptr;
    if (!dmaHeapAllocDma32(size, fd, addr)) {
        return false;
    }
    buf.fd = fd;
    buf.addr = addr;
    buf.size = size;
    return true;
}

static bool dmaBufSync(int fd, bool start) {
    dma_buf_sync sync;
    std::memset(&sync, 0, sizeof(sync));
    sync.flags = (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) == 0;
}

static size_t nv12BytesByStride(const image_buffer_t& frame) {
    int wstride = frame.width_stride > 0 ? frame.width_stride : frame.width;
    int hstride = frame.height_stride > 0 ? frame.height_stride : frame.height;
    if (wstride <= 0 || hstride <= 0) {
        return 0;
    }
    return static_cast<size_t>(wstride) * static_cast<size_t>(hstride) * 3u / 2u;
}

static bool mapNv12FrameForCpu(image_buffer_t& frame, void*& out_addr, size_t& out_size) {
    if (frame.virt_addr != nullptr || frame.fd <= 0) {
        return false;
    }

    size_t bytes = nv12BytesByStride(frame);
    if (bytes == 0) {
        return false;
    }

    if (!dmaBufSync(frame.fd, true)) {
        return false;
    }

    void* addr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, frame.fd, 0);
    if (addr == MAP_FAILED) {
        dmaBufSync(frame.fd, false);
        return false;
    }

    out_addr = addr;
    out_size = bytes;
    return true;
}

static void unmapNv12FrameForCpu(int fd, void*& addr, size_t& size) {
    if (addr && addr != MAP_FAILED && size > 0) {
        munmap(addr, size);
    }
    addr = nullptr;
    size = 0;
    dmaBufSync(fd, false);
}

cv::Scalar classColor(int cls_id);  // 定义在本文件下方(供 tracking overlay 等复用)

// 语义分割掩膜/框色:与检测共用 Ultralytics 调色板(掩膜、框、标签条三者同色)
static cv::Scalar getSegColor(int class_id) {
    return classColor(class_id);
}

// 类别颜色:黄金角 HSV 均匀分布 + 惰性缓存(BGR 语义)。
// 同一类别在任何任务/任何帧颜色一致(便于跨帧追踪辨识),替代原先全绿画法。
// Ultralytics 风格标签:置信度两位小数("person 0.92"),不带百分号
static std::string fmtConf(double conf) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%.2f", conf);
    return buf;
}

// Ultralytics 风格调色板(官方 20 色,RGB 转 BGR 存储)。
// 同一类别/跟踪 ID 的颜色全局一致;标签条与框同色(Ultralytics 样式)。
cv::Scalar classColor(int cls_id) {
    // RGB 转 BGR:每项按 R,G,B 书写
    static const cv::Scalar kPalette[] = {
        {255, 42, 4},      {235, 219, 11},    // 0 蓝  1 青
        {243, 243, 243},   {183, 223, 0},     // 2 白  3 亮绿
        {104, 31, 17},     {221, 111, 255},   // 4 深蓝  5 粉
        {79, 68, 255},     {0, 237, 204},     // 6 红  7 黄绿
        {68, 243, 0},      {255, 0, 189},     // 8 绿  9 紫
        {255, 180, 0},     {186, 0, 221},     // 10 天蓝  11 品红
        {255, 255, 0},     {0, 192, 26},      // 12 黄  13 深绿
        {179, 255, 1},     {255, 36, 125},    // 14 青绿  15 蓝紫
        {104, 0, 123},     {108, 27, 255},    // 16 深紫  17 玫红
        {47, 109, 252},    {11, 255, 162},    // 18 橙  19 黄绿
    };
    const int idx = cls_id < 0 ? 0 : cls_id % 20;
    return kPalette[idx];
}

// BGR 帧标签:实底条 + 白字(抗锯齿),贴框顶边上方;顶边出画面时贴进框内
void putClassLabelBGR(cv::Mat& frame, const std::string& label, int x, int y1,
                     const cv::Scalar& bar_color, double scale, int thick) {
    int base = 0;
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    cv::Size ts = cv::getTextSize(label, font, scale, std::max(1, thick - 1), &base);
    const int bw = ts.width + 6;
    const int bar_h = ts.height + 4;
    int bar_x = x;
    if (bar_x + bw > frame.cols) bar_x = std::max(0, frame.cols - bw);
    int bar_y = y1 - bar_h;
    if (bar_y < 0) bar_y = y1;
    cv::rectangle(frame, cv::Rect(bar_x, bar_y, bw, bar_h), bar_color, cv::FILLED, cv::LINE_AA);
    cv::putText(frame, label, cv::Point(bar_x + 3, bar_y + ts.height - 1), font, scale,
                cv::Scalar(255, 255, 255), std::max(1, thick - 1), cv::LINE_AA);
}

// 掩膜绘制最小框面积（像素²）：过小目标（<28x28 量级）掩膜基本不可见，跳过混合省 resize+blend 开销。
// RK_PIPE_SEG_MASK_MIN_AREA 覆盖（默认 800，0=不跳过）。
static int segMaskMinArea() {
    const char* v = std::getenv("RK_PIPE_SEG_MASK_MIN_AREA");
    if (!v || !*v) {
        return 800;
    }
    const int out = std::atoi(v);
    return out < 0 ? 800 : out;
}

static int getRgaNvFormat(const image_buffer_t& frame) {
    if (frame.format == IMAGE_FORMAT_YUV420SP_NV21) {
        return RK_FORMAT_YCrCb_420_SP;
    }
    return RK_FORMAT_YCbCr_420_SP;
}

static bool rgaFillRectNV12(image_buffer_t& frame, int x, int y, int width, int height, const cv::Scalar& color) {
    static std::atomic<bool> rga_fill_failed{false};
    if (frame.virt_addr == nullptr && frame.fd <= 0) {
        return false;
    }

    if (frame.fd > 0 && frame.virt_addr != nullptr) {
        return fillRectNV12CPU(frame, x, y, width, height, color);
    }

    if (frame.virt_addr == nullptr && frame.fd > 0) {
        void* addr = nullptr;
        size_t bytes = 0;
        if (!mapNv12FrameForCpu(frame, addr, bytes)) {
            return true;
        }
        image_buffer_t mapped = frame;
        mapped.virt_addr = static_cast<unsigned char*>(addr);
        bool ok = fillRectNV12CPU(mapped, x, y, width, height, color);
        unmapNv12FrameForCpu(frame.fd, addr, bytes);
        return ok;
    }

    if (rga_fill_failed.load()) {
        if (frame.virt_addr != nullptr) {
            return fillRectNV12CPU(frame, x, y, width, height, color);
        }
        return false;
    }

    if (width <= 0 || height <= 0) {
        return true;
    }

    // Always fallback to CPU if no RGA capability or previous failure handling?
    // For now, we try RGA every time, but if it fails, we fallback to CPU.

    int frame_width = frame.width;
    int frame_height = frame.height;
    if (x < 0) {
        width += x;
        x = 0;
    }
    if (y < 0) {
        height += y;
        y = 0;
    }
    if (x >= frame_width || y >= frame_height) {
        return true;
    }
    if (x + width > frame_width) {
        width = frame_width - x;
    }
    if (y + height > frame_height) {
        height = frame_height - y;
    }
    if (width <= 0 || height <= 0) {
        return true;
    }
    if (width < 2 || height < 2) {
        if (frame.virt_addr != nullptr) {
            return fillRectNV12CPU(frame, x, y, width, height, color);
        }
        return true;
    }

    int aligned_x = x & ~1;
    int aligned_y = y & ~1;
    int aligned_w = width + (x - aligned_x);
    int aligned_h = height + (y - aligned_y);
    aligned_w = (aligned_w + 1) & ~1;
    aligned_h = (aligned_h + 1) & ~1;
    if (aligned_x + aligned_w > frame_width) {
        aligned_w = frame_width - aligned_x;
    }
    if (aligned_y + aligned_h > frame_height) {
        aligned_h = frame_height - aligned_y;
    }
    aligned_w &= ~1;
    aligned_h &= ~1;
    if (aligned_w <= 0 || aligned_h <= 0) {
        return true;
    }
    if (aligned_w < 2 || aligned_h < 2) {
        if (frame.virt_addr != nullptr) {
            return fillRectNV12CPU(frame, x, y, width, height, color);
        }
        return true;
    }
    if (aligned_w <= 8 || aligned_h <= 8) {
        if (frame.virt_addr != nullptr) {
            return fillRectNV12CPU(frame, x, y, width, height, color);
        }
        return true;
    }

    int wstride = frame.width_stride > 0 ? frame.width_stride : frame.width;
    int hstride = frame.height_stride > 0 ? frame.height_stride : frame.height;
    bool use_fd = frame.fd > 0 && frame.virt_addr == nullptr;
    rga_buffer_t dst;
    if (use_fd) {
        dst = wrapbuffer_fd(frame.fd, frame.width, frame.height, getRgaNvFormat(frame), wstride, hstride);
    } else {
        dst = wrapbuffer_virtualaddr(frame.virt_addr, frame.width, frame.height, getRgaNvFormat(frame), wstride, hstride);
    }

    ensureRga();

    rga_buffer_t src;
    static thread_local DmaMappedBuffer tl_src_dma;
    if (use_fd) {
        if (ensureDma32MappedBuffer(4096, tl_src_dma)) {
            auto* p = static_cast<uint8_t*>(tl_src_dma.addr);
            p[0] = static_cast<uint8_t>(color[0]);
            p[1] = static_cast<uint8_t>(color[1]);
            p[2] = static_cast<uint8_t>(color[2]);
            src = wrapbuffer_fd(tl_src_dma.fd, 1, 1, RK_FORMAT_BGR_888, 1, 1);
        } else {
            uint8_t color_pixel[3];
            color_pixel[0] = static_cast<uint8_t>(color[0]);
            color_pixel[1] = static_cast<uint8_t>(color[1]);
            color_pixel[2] = static_cast<uint8_t>(color[2]);
            src = wrapbuffer_virtualaddr(color_pixel, 1, 1, RK_FORMAT_BGR_888, 1, 1);
        }
    } else {
        uint8_t color_pixel[3];
        color_pixel[0] = static_cast<uint8_t>(color[0]);
        color_pixel[1] = static_cast<uint8_t>(color[1]);
        color_pixel[2] = static_cast<uint8_t>(color[2]);
        src = wrapbuffer_virtualaddr(color_pixel, 1, 1, RK_FORMAT_BGR_888, 1, 1);
    }

    im_rect srect = {0, 0, 1, 1};
    im_rect drect = {aligned_x, aligned_y, aligned_w, aligned_h};
    im_rect prect = {0, 0, 0, 0};
    rga_buffer_t pat = {};

    IM_STATUS check = imcheck(src, dst, srect, drect);
    if (check != IM_STATUS_SUCCESS && check != IM_STATUS_NOERROR) {
        if (frame.virt_addr != nullptr) {
            return fillRectNV12CPU(frame, x, y, width, height, color);
        }
        return false;
    }
    IM_STATUS status = improcess(src, dst, pat, srect, drect, prect, -1, NULL, NULL, IM_SYNC);
    if (!rgaStatusOk(status) && use_fd && frame.virt_addr != nullptr) {
        dst = wrapbuffer_virtualaddr(frame.virt_addr, frame.width, frame.height, getRgaNvFormat(frame), wstride, hstride);
        status = improcess(src, dst, pat, srect, drect, prect, -1, NULL, NULL, IM_SYNC);
    }
    if (!rgaStatusOk(status)) {
        char buf[256];
        snprintf(buf, sizeof(buf), "rgaFillRectNV12 (src=[1,1], dst=[%d,%d])", aligned_w, aligned_h);
        logRgaFailure(buf, status);
        rga_fill_failed.store(true);
        if (frame.virt_addr != nullptr) {
            return fillRectNV12CPU(frame, x, y, width, height, color);
        }
        return false;
    }

    return true;
}

static bool drawLineNV12CPU(image_buffer_t& frame, int x0, int y0, int x1, int y1, int thickness,
                             const cv::Scalar& color);  // 前向声明(描边用,定义在下方)

static bool blendMaskNV12CPU(image_buffer_t& frame, const cv::Mat& mask, const cv::Rect& box, const cv::Scalar& color) {
    if (frame.virt_addr == nullptr) {
        return false;
    }
    if (mask.empty()) {
        return true;
    }
    if (frame.format != IMAGE_FORMAT_YUV420SP_NV12 && frame.format != IMAGE_FORMAT_YUV420SP_NV21) {
        return false;
    }

    int frame_width = frame.width;
    int frame_height = frame.height;
    cv::Rect rect = box & cv::Rect(0, 0, frame_width, frame_height);
    if (rect.width <= 0 || rect.height <= 0) {
        return true;
    }

    cv::Mat local_mask = mask;
    if (mask.cols != rect.width || mask.rows != rect.height) {
        cv::resize(mask, local_mask, rect.size(), 0, 0, cv::INTER_LINEAR);
        // Threshold mask to sharpen edges after linear resize
        cv::threshold(local_mask, local_mask, 127, 255, cv::THRESH_BINARY);
    }

    int wstride = frame.width_stride > 0 ? frame.width_stride : frame.width;
    int hstride = frame.height_stride > 0 ? frame.height_stride : frame.height;

    int b = static_cast<int>(color[0]);
    int g = static_cast<int>(color[1]);
    int r = static_cast<int>(color[2]);
    int yv = (77 * r + 150 * g + 29 * b) >> 8;
    int uv = ((-43 * r - 85 * g + 128 * b) >> 8) + 128;
    int vv = ((128 * r - 107 * g - 21 * b) >> 8) + 128;
    uint8_t y_val = static_cast<uint8_t>(std::min(255, std::max(0, yv)));
    uint8_t u_val = static_cast<uint8_t>(std::min(255, std::max(0, uv)));
    uint8_t v_val = static_cast<uint8_t>(std::min(255, std::max(0, vv)));

    // 只混合掩膜非零包围盒，避免扫整个 box（人形掩膜通常只占框的一部分）
    cv::Rect mrect = cv::boundingRect(local_mask);
    if (mrect.width <= 0 || mrect.height <= 0) {
        return true;
    }
    cv::Rect blend(rect.x + mrect.x, rect.y + mrect.y, mrect.width, mrect.height);
    blend &= cv::Rect(0, 0, frame_width, frame_height);
    if (blend.width <= 0 || blend.height <= 0) {
        return true;
    }
    const uint8_t* mask_base = local_mask.ptr<uint8_t>(mrect.y) + mrect.x;

    uint8_t* y_plane = frame.virt_addr;
    uint8_t* uv_plane = frame.virt_addr + wstride * hstride;

    // Y 平面：按 16 像素向量化（alpha = mask>>1，约 0.5 不透明度，掩膜饱和醒目）
#if defined(__ARM_NEON)
    const uint8x8_t vy = vdup_n_u8(y_val);
    const uint16x8_t v127 = vdupq_n_u16(127);
    for (int yy = 0; yy < blend.height; ++yy) {
        uint8_t* y_row = y_plane + (blend.y + yy) * wstride + blend.x;
        const uint8_t* m_row = mask_base + yy * local_mask.cols;
        int xx = 0;
        for (; xx + 16 <= blend.width; xx += 16) {
            uint8x16_t m = vld1q_u8(m_row + xx);
            uint8x16_t o = vld1q_u8(y_row + xx);
            uint8x16_t alpha = vshrq_n_u8(m, 1);
            uint8x16_t inv = vsubq_u8(vdupq_n_u8(255), alpha);
            uint16x8_t lo = vaddq_u16(vaddq_u16(vmull_u8(vget_low_u8(alpha), vy),
                                                vmull_u8(vget_low_u8(inv), vget_low_u8(o))),
                                      v127);
            uint16x8_t hi = vaddq_u16(vaddq_u16(vmull_u8(vget_high_u8(alpha), vy),
                                                vmull_u8(vget_high_u8(inv), vget_high_u8(o))),
                                      v127);
            vst1q_u8(y_row + xx, vcombine_u8(vshrn_n_u16(lo, 8), vshrn_n_u16(hi, 8)));
        }
        for (; xx < blend.width; ++xx) {
            int alpha = m_row[xx] >> 1;
            if (alpha == 0) {
                continue;
            }
            int orig = y_row[xx];
            y_row[xx] = static_cast<uint8_t>((alpha * y_val + (255 - alpha) * orig + 127) / 255);
        }
    }
#else
    for (int yy = 0; yy < blend.height; ++yy) {
        uint8_t* y_row = y_plane + (blend.y + yy) * wstride + blend.x;
        const uint8_t* m_row = mask_base + yy * local_mask.cols;
        for (int xx = 0; xx < blend.width; ++xx) {
            int alpha = m_row[xx] >> 1;
            if (alpha == 0) {
                continue;
            }
            int orig = y_row[xx];
            y_row[xx] = static_cast<uint8_t>((alpha * y_val + (255 - alpha) * orig + 127) / 255);
        }
    }
#endif

    // UV 平面：按掩膜非零区域做 2x2 平均混合
    int uv_x_start = blend.x & ~1;
    int uv_y_start = blend.y & ~1;
    int uv_x_end = (blend.x + blend.width + 1) & ~1;
    int uv_y_end = (blend.y + blend.height + 1) & ~1;
    if (uv_x_end > frame_width) uv_x_end = frame_width;
    if (uv_y_end > frame_height) uv_y_end = frame_height;

    bool is_nv21 = frame.format == IMAGE_FORMAT_YUV420SP_NV21;
    const int mask_cols = local_mask.cols;
    for (int yy = uv_y_start; yy < uv_y_end; yy += 2) {
        uint8_t* uv_row = uv_plane + (yy / 2) * wstride;
        const int m_yy0 = yy - blend.y;
        for (int xx = uv_x_start; xx < uv_x_end; xx += 2) {
            int sum = 0;
            int cnt = 0;
            const int m_xx0 = xx - blend.x;
            for (int dy = 0; dy < 2; ++dy) {
                int my = m_yy0 + dy;
                if (my < 0 || my >= blend.height) {
                    continue;
                }
                const uint8_t* m_row = mask_base + my * mask_cols;
                for (int dx = 0; dx < 2; ++dx) {
                    int mx = m_xx0 + dx;
                    if (mx < 0 || mx >= blend.width) {
                        continue;
                    }
                    sum += m_row[mx];
                    cnt++;
                }
            }
            if (cnt == 0) {
                continue;
            }
            int alpha = (sum / cnt) >> 1;
            if (alpha == 0) {
                continue;
            }
            int u_index = is_nv21 ? xx + 1 : xx;
            int v_index = is_nv21 ? xx : xx + 1;
            int orig_u = uv_row[u_index];
            int orig_v = uv_row[v_index];
            uv_row[u_index] = static_cast<uint8_t>((alpha * u_val + (255 - alpha) * orig_u + 127) / 255);
            uv_row[v_index] = static_cast<uint8_t>((alpha * v_val + (255 - alpha) * orig_v + 127) / 255);
        }
    }

    // 掩膜边缘描边(同色细线,NV12 直绘):实例边界清晰
    {
        cv::Mat bw;
        cv::threshold(local_mask, bw, 127, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        const cv::Point origin(rect.x + mrect.x, rect.y + mrect.y);
        for (const auto& c : contours) {
            if (c.size() < 3) {
                continue;
            }
            for (size_t k = 0; k < c.size(); ++k) {
                cv::Point a = c[k] + origin;
                cv::Point b = c[(k + 1) % c.size()] + origin;
                drawLineNV12CPU(frame, a.x, a.y, b.x, b.y, 1, color);
            }
        }
    }

    return true;
}

static bool fillRectNV12CPU(image_buffer_t& frame, int x, int y, int width, int height, const cv::Scalar& color) {
    if (frame.virt_addr == nullptr) {
        return false;
    }

    if (width <= 0 || height <= 0) {
        return true;
    }

    int frame_width = frame.width;
    int frame_height = frame.height;
    if (x < 0) {
        width += x;
        x = 0;
    }
    if (y < 0) {
        height += y;
        y = 0;
    }
    if (x >= frame_width || y >= frame_height) {
        return true;
    }
    if (x + width > frame_width) {
        width = frame_width - x;
    }
    if (y + height > frame_height) {
        height = frame_height - y;
    }
    if (width <= 0 || height <= 0) {
        return true;
    }

    int wstride = frame.width_stride > 0 ? frame.width_stride : frame.width;
    int hstride = frame.height_stride > 0 ? frame.height_stride : frame.height;

    int b = static_cast<int>(color[0]);
    int g = static_cast<int>(color[1]);
    int r = static_cast<int>(color[2]);
    int yv = (77 * r + 150 * g + 29 * b) >> 8;
    int uv = ((-43 * r - 85 * g + 128 * b) >> 8) + 128;
    int vv = ((128 * r - 107 * g - 21 * b) >> 8) + 128;
    uint8_t y_val = static_cast<uint8_t>(std::min(255, std::max(0, yv)));
    uint8_t u_val = static_cast<uint8_t>(std::min(255, std::max(0, uv)));
    uint8_t v_val = static_cast<uint8_t>(std::min(255, std::max(0, vv)));

    uint8_t* y_plane = frame.virt_addr;
    uint8_t* uv_plane = frame.virt_addr + wstride * hstride;

    int x_end = x + width;
    int y_end = y + height;

    for (int yy = y; yy < y_end; ++yy) {
        uint8_t* row = y_plane + yy * wstride + x;
        memset(row, y_val, width);
    }

    int uv_x_start = x & ~1;
    int uv_y_start = y & ~1;
    int uv_x_end = (x_end + 1) & ~1;
    int uv_y_end = (y_end + 1) & ~1;
    if (uv_x_end > frame_width) uv_x_end = frame_width;
    if (uv_y_end > frame_height) uv_y_end = frame_height;

    bool is_nv21 = frame.format == IMAGE_FORMAT_YUV420SP_NV21;
    for (int yy = uv_y_start; yy < uv_y_end; yy += 2) {
        uint8_t* uv_row = uv_plane + (yy / 2) * wstride;
        for (int xx = uv_x_start; xx < uv_x_end; xx += 2) {
            if (is_nv21) {
                uv_row[xx] = v_val;
                uv_row[xx + 1] = u_val;
            } else {
                uv_row[xx] = u_val;
                uv_row[xx + 1] = v_val;
            }
        }
    }

    return true;
}

static bool rgaDrawRectangleNV12(image_buffer_t& frame, int x1, int y1, int x2, int y2, int thickness, const cv::Scalar& color) {
    if (thickness <= 0) {
        thickness = 1;
    }

    if (x2 < x1 || y2 < y1) {
        return false;
    }

    int width = x2 - x1 + 1;
    int height = y2 - y1 + 1;
    bool ok = true;

    ok = ok && rgaFillRectNV12(frame, x1, y1, width, thickness, color);
    ok = ok && rgaFillRectNV12(frame, x1, y2 - thickness + 1, width, thickness, color);
    ok = ok && rgaFillRectNV12(frame, x1, y1, thickness, height, color);
    ok = ok && rgaFillRectNV12(frame, x2 - thickness + 1, y1, thickness, height, color);

    return ok;
}

static bool drawRectangleNV12CPU(image_buffer_t& frame, int x1, int y1, int x2, int y2, int thickness, const cv::Scalar& color) {
    if (frame.virt_addr == nullptr) {
        return false;
    }
    if (thickness <= 0) {
        thickness = 1;
    }
    if (x2 < x1 || y2 < y1) {
        return false;
    }

    int width = x2 - x1 + 1;
    int height = y2 - y1 + 1;
    bool ok = true;

    ok = ok && fillRectNV12CPU(frame, x1, y1, width, thickness, color);
    ok = ok && fillRectNV12CPU(frame, x1, y2 - thickness + 1, width, thickness, color);
    ok = ok && fillRectNV12CPU(frame, x1, y1, thickness, height, color);
    ok = ok && fillRectNV12CPU(frame, x2 - thickness + 1, y1, thickness, height, color);

    return ok;
}

static bool drawLineNV12CPU(image_buffer_t& frame, int x0, int y0, int x1, int y1, int thickness, const cv::Scalar& color) {
    if (frame.virt_addr == nullptr) {
        return false;
    }
    if (thickness <= 0) {
        thickness = 1;
    }

    int w = frame.width;
    int h = frame.height;
    if (w <= 0 || h <= 0) {
        return false;
    }

    auto clampi = [](int v, int lo, int hi) -> int {
        return std::max(lo, std::min(hi, v));
    };
    x0 = clampi(x0, 0, w - 1);
    y0 = clampi(y0, 0, h - 1);
    x1 = clampi(x1, 0, w - 1);
    y1 = clampi(y1, 0, h - 1);

    int dx = std::abs(x1 - x0);
    int dy = std::abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;

    int half = thickness / 2;
    while (true) {
        int px = x0 - half;
        int py = y0 - half;
        int pw = thickness;
        int ph = thickness;
        fillRectNV12CPU(frame, px, py, pw, ph, color);

        if (x0 == x1 && y0 == y1) {
            break;
        }
        int e2 = err * 2;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
    return true;
}

static const uint8_t* getFontRows(char c) {
    switch (c) {
        case '0': {
            static const uint8_t rows[] = {0x1E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x1E};
            return rows;
        }
        case '1': {
            static const uint8_t rows[] = {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
            return rows;
        }
        case '2': {
            static const uint8_t rows[] = {0x1E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
            return rows;
        }
        case '3': {
            static const uint8_t rows[] = {0x1E, 0x11, 0x01, 0x0E, 0x01, 0x11, 0x1E};
            return rows;
        }
        case '4': {
            static const uint8_t rows[] = {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
            return rows;
        }
        case '5': {
            static const uint8_t rows[] = {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x1E};
            return rows;
        }
        case '6': {
            static const uint8_t rows[] = {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x1E};
            return rows;
        }
        case '7': {
            static const uint8_t rows[] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
            return rows;
        }
        case '8': {
            static const uint8_t rows[] = {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
            return rows;
        }
        case '9': {
            static const uint8_t rows[] = {0x1E, 0x11, 0x11, 0x1F, 0x01, 0x02, 0x0C};
            return rows;
        }
        case 'A': {
            static const uint8_t rows[] = {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
            return rows;
        }
        case 'B': {
            static const uint8_t rows[] = {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
            return rows;
        }
        case 'C': {
            static const uint8_t rows[] = {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
            return rows;
        }
        case 'D': {
            static const uint8_t rows[] = {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E};
            return rows;
        }
        case 'E': {
            static const uint8_t rows[] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
            return rows;
        }
        case 'F': {
            static const uint8_t rows[] = {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
            return rows;
        }
        case 'G': {
            static const uint8_t rows[] = {0x0E, 0x11, 0x10, 0x10, 0x13, 0x11, 0x0E};
            return rows;
        }
        case 'H': {
            static const uint8_t rows[] = {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
            return rows;
        }
        case 'I': {
            static const uint8_t rows[] = {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E};
            return rows;
        }
        case 'J': {
            static const uint8_t rows[] = {0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E};
            return rows;
        }
        case 'K': {
            static const uint8_t rows[] = {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
            return rows;
        }
        case 'L': {
            static const uint8_t rows[] = {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
            return rows;
        }
        case 'N': {
            static const uint8_t rows[] = {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
            return rows;
        }
        case 'O': {
            static const uint8_t rows[] = {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
            return rows;
        }
        case 'P': {
            static const uint8_t rows[] = {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
            return rows;
        }
        case 'Q': {
            static const uint8_t rows[] = {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
            return rows;
        }
        case 'R': {
            static const uint8_t rows[] = {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
            return rows;
        }
        case 'S': {
            static const uint8_t rows[] = {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
            return rows;
        }
        case 'T': {
            static const uint8_t rows[] = {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
            return rows;
        }
        case 'U': {
            static const uint8_t rows[] = {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
            return rows;
        }
        case 'V': {
            static const uint8_t rows[] = {0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04};
            return rows;
        }
        case 'W': {
            static const uint8_t rows[] = {0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11};
            return rows;
        }
        case 'X': {
            static const uint8_t rows[] = {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11};
            return rows;
        }
        case 'Y': {
            static const uint8_t rows[] = {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04};
            return rows;
        }
        case 'Z': {
            static const uint8_t rows[] = {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};
            return rows;
        }
        case 'M': {
            static const uint8_t rows[] = {0x11, 0x1B, 0x15, 0x11, 0x11, 0x11, 0x11};
            return rows;
        }
        case '%': {
            static const uint8_t rows[] = {0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13};
            return rows;
        }
        case ':': {
            static const uint8_t rows[] = {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00};
            return rows;
        }
        case '.': {
            static const uint8_t rows[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C};
            return rows;
        }
        case ' ': {
            static const uint8_t rows[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
            return rows;
        }
        default:
            break;
    }
    static const uint8_t rows[] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    return rows;
}

void drawOverlayText(image_buffer_t& frame, const std::string& text, int x, int y, const cv::Scalar& color, int scale,
                     const cv::Scalar& bg_color) {
    if (frame.virt_addr == nullptr && frame.fd <= 0) {
        return;
    }
    if (scale <= 0) {
        scale = 1;
    }

    bool is_nv = frame.format == IMAGE_FORMAT_YUV420SP_NV12 || frame.format == IMAGE_FORMAT_YUV420SP_NV21;
    if (frame.fd > 0 && frame.virt_addr != nullptr) {
        std::string upper = text;
        std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });

        int cursor_x = x;
        int cursor_y = y;
        for (char ch : upper) {
            const uint8_t* rows = getFontRows(ch);
            for (int row = 0; row < 7; ++row) {
                uint8_t bits = rows[row] & 0x1F;
                int col = 0;
                while (col < 5) {
                    while (col < 5 && ((bits & (1 << (4 - col))) == 0)) {
                        col++;
                    }
                    if (col >= 5) {
                        break;
                    }
                    int run_start = col;
                    while (col < 5 && (bits & (1 << (4 - col)))) {
                        col++;
                    }
                    int run_len = col - run_start;
                    int px = cursor_x + run_start * scale;
                    int py = cursor_y + row * scale;
                    fillRectNV12CPU(frame, px, py, run_len * scale, scale, color);
                }
            }
            cursor_x += 6 * scale;
        }
        return;
    }

    if (frame.virt_addr == nullptr && frame.fd > 0) {
        void* addr = nullptr;
        size_t bytes = 0;
        if (!mapNv12FrameForCpu(frame, addr, bytes)) {
            return;
        }

        image_buffer_t mapped = frame;
        mapped.virt_addr = static_cast<unsigned char*>(addr);
        if (!is_nv) {
            mapped.format = IMAGE_FORMAT_YUV420SP_NV12;
        }

        std::string upper = text;
        std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });

        int cursor_x = x;
        int cursor_y = y;
        for (char ch : upper) {
            const uint8_t* rows = getFontRows(ch);
            for (int row = 0; row < 7; ++row) {
                uint8_t bits = rows[row] & 0x1F;
                int col = 0;
                while (col < 5) {
                    while (col < 5 && ((bits & (1 << (4 - col))) == 0)) {
                        col++;
                    }
                    if (col >= 5) {
                        break;
                    }
                    int run_start = col;
                    while (col < 5 && (bits & (1 << (4 - col)))) {
                        col++;
                    }
                    int run_len = col - run_start;
                    int px = cursor_x + run_start * scale;
                    int py = cursor_y + row * scale;
                    fillRectNV12CPU(mapped, px, py, run_len * scale, scale, color);
                }
            }
            cursor_x += 6 * scale;
        }

        unmapNv12FrameForCpu(frame.fd, addr, bytes);
        return;
    }

    std::string upper = text;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });

    int char_width = 6 * scale;
    int char_height = 8 * scale;
    int text_width = upper.length() * char_width;
    int text_height = char_height;

    // Ensure text is within frame bounds (simple check)
    if (x + text_width > frame.width) text_width = frame.width - x;
    if (y + text_height > frame.height) text_height = frame.height - y;
    if (text_width <= 0 || text_height <= 0) return;

    // Use exact size to avoid ROI copy issues and simplify stride handling
    // Align width to 4 to match RGA requirements for BGRA (stride alignment)
    // Align height to 2 for general video format requirements
    int aligned_text_width = (text_width + 3) & ~3;
    int aligned_text_height = (text_height + 1) & ~1;
    cv::Mat text_img = cv::Mat::zeros(aligned_text_height, aligned_text_width, CV_8UC4);

    int cursor_x = 0;
    int cursor_y = 0;
    cv::Scalar color_bgra(color[0], color[1], color[2], 255);

    for (char ch : upper) {
        const uint8_t* rows = getFontRows(ch);
        for (int row = 0; row < 7; ++row) {
            uint8_t bits = rows[row] & 0x1F;
            int col = 0;
            while (col < 5) {
                while (col < 5 && ((bits & (1 << (4 - col))) == 0)) {
                    col++;
                }
                if (col >= 5) {
                    break;
                }
                int run_start = col;
                while (col < 5 && (bits & (1 << (4 - col)))) {
                    col++;
                }
                int run_len = col - run_start;
                int px = cursor_x + run_start * scale;
                int py = cursor_y + row * scale;

                // Check bounds before drawing rectangle
                if (px + run_len * scale <= text_img.cols && py + scale <= text_img.rows) {
                    cv::rectangle(text_img, cv::Rect(px, py, run_len * scale, scale), color_bgra, -1);
                }
            }
        }
        cursor_x += 6 * scale;
    }

    is_nv = frame.format == IMAGE_FORMAT_YUV420SP_NV12 || frame.format == IMAGE_FORMAT_YUV420SP_NV21;
    if (is_nv && frame.virt_addr != nullptr) {
        // Try to use CPU drawing if virtual address is available and RGA is not preferred or failed before
        // But here we want to prioritize RGA for zero-copy if possible.
    }
    int src_width = aligned_text_width;
    int src_height = aligned_text_height;
    int aligned_w = (src_width + 3) & ~3;
    cv::Mat bgra_aligned;
    const cv::Mat* src_mat = &text_img;

    // Always copy to aligned buffer if width is not aligned or if we want to ensure contiguous memory for RGA
    // Since text_img is created with exact width, its step is aligned to 4 bytes (CV_8UC4),
    // but RGA might require 4-pixel alignment for width.
    if (aligned_w != src_width) {
        bgra_aligned = cv::Mat(src_height, aligned_w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        // Use ROI for destination to avoid size mismatch
        text_img.copyTo(bgra_aligned(cv::Rect(0, 0, src_width, src_height)));
        src_mat = &bgra_aligned;
    }
    int src_wstride = aligned_w;
    bool dst_use_fd = frame.fd > 0 && frame.virt_addr == nullptr;
    rga_buffer_t src;
    static thread_local DmaMappedBuffer tl_osd_dma;
    if (dst_use_fd) {
        size_t bytes = static_cast<size_t>(src_wstride) * static_cast<size_t>(src_height) * 4u;
        if (ensureDma32MappedBuffer(bytes, tl_osd_dma)) {
            if (src_mat->isContinuous() && src_mat->step == static_cast<size_t>(src_wstride) * 4u) {
                std::memcpy(tl_osd_dma.addr, src_mat->data, bytes);
            } else {
                auto* dst_row = static_cast<uint8_t*>(tl_osd_dma.addr);
                const uint8_t* src_row = src_mat->data;
                size_t row_bytes = static_cast<size_t>(src_wstride) * 4u;
                for (int row = 0; row < src_height; ++row) {
                    std::memcpy(dst_row, src_row, row_bytes);
                    dst_row += row_bytes;
                    src_row += src_mat->step;
                }
            }
            src = wrapbuffer_fd(tl_osd_dma.fd, src_width, src_height, RK_FORMAT_BGRA_8888, src_wstride, src_height);
        } else {
            src = wrapbuffer_virtualaddr((void*)src_mat->data, src_width, src_height, RK_FORMAT_BGRA_8888, src_wstride, src_height);
        }
    } else {
        src = wrapbuffer_virtualaddr((void*)src_mat->data, src_width, src_height, RK_FORMAT_BGRA_8888, src_wstride, src_height);
    }

    int dst_x = x & ~1;
    int dst_y = y & ~1;
    int dst_w = src_width;
    int dst_h = src_height;
    dst_w = (dst_w + 1) & ~1;
    dst_h = (dst_h + 1) & ~1;

    int dst_wstride = frame.width_stride > 0 ? frame.width_stride : frame.width;
    int dst_hstride = frame.height_stride > 0 ? frame.height_stride : frame.height;

    rga_buffer_t dst;
    if (frame.fd > 0) {
        dst = wrapbuffer_fd(frame.fd, frame.width, frame.height, getRgaNvFormat(frame), dst_wstride, dst_hstride);
    } else {
        dst = wrapbuffer_virtualaddr(frame.virt_addr, frame.width, frame.height, getRgaNvFormat(frame), dst_wstride, dst_hstride);
    }

    ensureRga();

    im_rect srect = {0, 0, src_width, src_height};
    im_rect drect = {dst_x, dst_y, dst_w, dst_h};

    static bool rga_osd_failed = false;

    if (!rga_osd_failed) {
        rga_buffer_t pat = {};
        im_rect prect = {};
        IM_STATUS check = imcheck_composite(src, dst, pat, srect, drect, prect, IM_ALPHA_BLEND_SRC_OVER);
        if (check != IM_STATUS_SUCCESS && check != IM_STATUS_NOERROR) {
            rga_osd_failed = true;
        } else {
            IM_STATUS status = improcess(src, dst, pat, srect, drect, prect, -1, NULL, NULL, IM_ALPHA_BLEND_SRC_OVER | IM_SYNC);
            if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
                std::cerr << "[DrawUtils] RGA OSD failed (ret=" << status << "), falling back to CPU" << std::endl;
                rga_osd_failed = true;
            }
        }
    }

    if (rga_osd_failed) {
        bool use_cpu = is_nv && frame.virt_addr != nullptr;
        if (use_cpu) {
             cursor_x = x;
             cursor_y = y;
             for (char ch : upper) {
                const uint8_t* rows = getFontRows(ch);
                for (int row = 0; row < 7; ++row) {
                    uint8_t bits = rows[row] & 0x1F;
                    int col = 0;
                    while (col < 5) {
                        while (col < 5 && ((bits & (1 << (4 - col))) == 0)) {
                            col++;
                        }
                        if (col >= 5) {
                            break;
                        }
                        int run_start = col;
                        while (col < 5 && (bits & (1 << (4 - col)))) {
                            col++;
                        }
                        int run_len = col - run_start;
                        int px = cursor_x + run_start * scale;
                        int py = cursor_y + row * scale;
                        fillRectNV12CPU(frame, px, py, run_len * scale, scale, color);
                    }
                }
                cursor_x += 6 * scale;
            }
        }
    }
}

void drawDetectionResults(image_buffer_t& frame, const object_detect_result_list& results) {
    bool rga_ok = true;
    int box_thickness = calcOverlaySize(frame, 2);
    int text_scale = calcOverlayTextScale(frame);
    int text_offset = calcOverlaySize(frame, 12);
    for (int i = 0; i < results.count; ++i) {
        const object_detect_result* det_result = &(results.results[i]);
        if (det_result->prop < DISPLAY_THRESH) {
            continue;
        }
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(frame.width - 1, x2);
        y2 = std::min(frame.height - 1, y2);

        rga_ok = rga_ok && rgaDrawRectangleNV12(frame, x1, y1, x2, y2, box_thickness, classColor(det_result->cls_id));
    }

    if (rga_ok) {
        for (int i = 0; i < results.count; ++i) {
            const object_detect_result* det_result = &(results.results[i]);
            if (det_result->prop < DISPLAY_THRESH) {
                continue;
            }
            int x1 = std::max(0, det_result->box.left);
            int y1 = std::max(0, det_result->box.top);
            std::string label = "CLS" + std::to_string(det_result->cls_id) + " " + fmtConf(det_result->prop);
            drawOverlayText(frame, label, x1, std::max(0, y1 - text_offset), cv::Scalar(255, 255, 255), text_scale,
                            classColor(det_result->cls_id));
        }
    }
}

// ---- OverlayPrim：任务逻辑与绘制后端分离（draw_utils 重构阶段1）----
// 任务逻辑（遍历检出/裁剪/配色/标签文本）只写一份，产出 Prim 列表；
// BGR 与 NV12 后端各自按原有语义回放。像素行为不变（回归工具护航）。
struct OverlayPrim {
    enum class Kind : unsigned char { Box, Line, Point, Label };
    int radius = 0;  // Point 半径（filled）
    Kind kind;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // Box/Line（Line=线段端点）
    int anchor_x = 0, anchor_y = 0;      // Label 锚点
    std::string text;                     // Label
    cv::Scalar color;                     // 框/线色或标签主色（类别色）
};

static std::vector<OverlayPrim> buildDetectionOverlayPrims(const object_detect_result_list& results,
                                                           int frame_w, int frame_h) {
    std::vector<OverlayPrim> prims;
    prims.reserve(static_cast<size_t>(results.count) * 2);
    for (int i = 0; i < results.count; ++i) {
        const object_detect_result* det = &(results.results[i]);
        if (det->prop < DISPLAY_THRESH) {
            continue;
        }
        OverlayPrim box;
        box.kind = OverlayPrim::Kind::Box;
        box.x1 = std::max(0, det->box.left);
        box.y1 = std::max(0, det->box.top);
        box.x2 = std::min(frame_w - 1, det->box.right);
        box.y2 = std::min(frame_h - 1, det->box.bottom);
        box.color = classColor(det->cls_id);
        prims.push_back(std::move(box));

        OverlayPrim label;
        label.kind = OverlayPrim::Kind::Label;
        label.anchor_x = box.x1;
        label.anchor_y = box.y1;
        label.text = std::string(coco_cls_to_name(det->cls_id)) + " " + fmtConf(det->prop);
        label.color = box.color;
        prims.push_back(std::move(label));
    }
    return prims;
}

// BGR 后端：按 Prim 序回放（与原 drawDetectionResultsBGR 逐调用一致）
static void renderOverlayPrimsBgr(cv::Mat& frame, const std::vector<OverlayPrim>& prims,
                                  int thickness_div = 320, double text_scale_div = 900.0,
                                  bool box_anti_aliased = true) {
    const int thickness = std::max(1, std::min(frame.cols, frame.rows) / thickness_div);
    const double text_scale = std::max(0.4, std::min(frame.cols, frame.rows) / text_scale_div);
    for (const auto& p : prims) {
        if (p.kind == OverlayPrim::Kind::Box) {
            cv::rectangle(frame, cv::Rect(cv::Point(p.x1, p.y1), cv::Point(p.x2, p.y2)),
                          p.color, thickness, box_anti_aliased ? cv::LINE_AA : 0);
        } else if (p.kind == OverlayPrim::Kind::Line) {
            cv::line(frame, cv::Point(p.x1, p.y1), cv::Point(p.x2, p.y2), p.color,
                     p.radius > 0 ? p.radius : thickness, cv::LINE_AA);
        } else if (p.kind == OverlayPrim::Kind::Point) {
            cv::circle(frame, cv::Point(p.x1, p.y1), p.radius, p.color, cv::FILLED, cv::LINE_AA);
        } else {
            putClassLabelBGR(frame, p.text, p.anchor_x, p.anchor_y, p.color, text_scale, thickness);
        }
    }
}

// NV12 后端：保持原三阶段语义——RGA 框 →（fd-only 且 RGA 失败时）mmap CPU
// 框+文本 →（RGA 成功时）virt_addr 文本；返回 rga_ok 供上层 BGR 回退判断
static bool renderOverlayPrimsNv12(image_buffer_t& frame, const std::vector<OverlayPrim>& prims) {
    const int box_thickness = calcOverlaySize(frame, 2);
    const int text_scale = calcOverlayTextScale(frame);
    const int text_offset = calcOverlaySize(frame, 12);

    bool rga_ok = true;
    for (const auto& p : prims) {
        if (p.kind != OverlayPrim::Kind::Box) {
            continue;
        }
        rga_ok = rga_ok && rgaDrawRectangleNV12(frame, p.x1, p.y1, p.x2, p.y2, box_thickness, p.color);
    }

    if (!rga_ok && frame.virt_addr == nullptr && frame.fd > 0 &&
        (frame.format == IMAGE_FORMAT_YUV420SP_NV12 || frame.format == IMAGE_FORMAT_YUV420SP_NV21)) {
        void* addr = nullptr;
        size_t bytes = 0;
        if (!mapNv12FrameForCpu(frame, addr, bytes)) {
            return false;
        }
        image_buffer_t mapped = frame;
        mapped.virt_addr = static_cast<unsigned char*>(addr);
        for (const auto& p : prims) {
            if (p.kind == OverlayPrim::Kind::Box) {
                drawRectangleNV12CPU(mapped, p.x1, p.y1, p.x2, p.y2, box_thickness, p.color);
            } else if (p.kind == OverlayPrim::Kind::Line) {
                drawLineNV12CPU(mapped, p.x1, p.y1, p.x2, p.y2, box_thickness, p.color);
            }
        }
        for (const auto& p : prims) {
            if (p.kind != OverlayPrim::Kind::Label) {
                continue;
            }
            drawOverlayText(mapped, p.text, p.anchor_x, std::max(0, p.anchor_y - text_offset),
                            cv::Scalar(255, 255, 255), text_scale, p.color);
        }
        unmapNv12FrameForCpu(frame.fd, addr, bytes);
        return true;
    }

    if (rga_ok) {
        for (const auto& p : prims) {
            if (p.kind != OverlayPrim::Kind::Label) {
                continue;
            }
            drawOverlayText(frame, p.text, p.anchor_x, std::max(0, p.anchor_y - text_offset),
                            cv::Scalar(255, 255, 255), text_scale, p.color);
        }
    }
    return rga_ok;
}

bool drawDetectionResultsZeroCopy(image_buffer_t& frame, const object_detect_result_list& results) {
    if (hideBoxesEnabled()) {
        return true;  // 隐藏所有框：不画也不触发 BGR 回退
    }
    return renderOverlayPrimsNv12(frame, buildDetectionOverlayPrims(results, frame.width, frame.height));
}

static std::vector<OverlayPrim> buildOBBOverlayPrims(const obb_detect_result_list& results) {
    std::vector<OverlayPrim> prims;
    prims.reserve(static_cast<size_t>(results.count) * 5);
    for (int i = 0; i < results.count; ++i) {
        const obb_detect_result* det = &(results.results[i]);
        if (det->prop < DISPLAY_THRESH) {
            continue;
        }
        const auto& box = det->box;
        const float cx = box.x + box.w * 0.5f;
        const float cy = box.y + box.h * 0.5f;
        cv::RotatedRect rect(cv::Point2f(cx, cy), cv::Size2f(box.w, box.h),
                             box.angle * 57.2957795f);
        cv::Point2f pts[4];
        rect.points(pts);
        const cv::Scalar color = classColor(det->cls_id);

        float min_x = pts[0].x;
        float min_y = pts[0].y;
        for (int k = 1; k < 4; ++k) {
            min_x = std::min(min_x, pts[k].x);
            min_y = std::min(min_y, pts[k].y);
        }

        for (int k = 0; k < 4; ++k) {
            OverlayPrim line;
            line.kind = OverlayPrim::Kind::Line;
            line.x1 = static_cast<int>(pts[k].x);
            line.y1 = static_cast<int>(pts[k].y);
            line.x2 = static_cast<int>(pts[(k + 1) % 4].x);
            line.y2 = static_cast<int>(pts[(k + 1) % 4].y);
            line.color = color;
            prims.push_back(std::move(line));
        }

        OverlayPrim label;
        label.kind = OverlayPrim::Kind::Label;
        label.anchor_x = std::max(0, static_cast<int>(min_x));
        label.anchor_y = std::max(0, static_cast<int>(min_y));
        label.text = std::string(coco_cls_to_name(det->cls_id)) + " " + fmtConf(det->prop);
        label.color = color;
        prims.push_back(std::move(label));
    }
    return prims;
}

bool drawOBBResultsZeroCopy(image_buffer_t& frame, const obb_detect_result_list& results) {
    if (hideBoxesEnabled()) {
        return true;  // 隐藏所有框
    }
    if (frame.virt_addr == nullptr && frame.fd <= 0) {
        return false;
    }
    if (frame.width <= 0 || frame.height <= 0) {
        return false;
    }
    if (frame.format != IMAGE_FORMAT_YUV420SP_NV12 && frame.format != IMAGE_FORMAT_YUV420SP_NV21) {
        return false;
    }

    // OBB 的 Prim 全部为 CPU 绘制（线段/文本）：fd-only 时懒 mmap
    void* cpu_addr = nullptr;
    size_t cpu_bytes = 0;
    image_buffer_t cpu_frame = frame;
    if (frame.virt_addr == nullptr) {
        if (!mapNv12FrameForCpu(frame, cpu_addr, cpu_bytes)) {
            return false;
        }
        cpu_frame.virt_addr = static_cast<unsigned char*>(cpu_addr);
    }

    bool ok = true;
    const int line_thickness = calcOverlaySize(frame, 2);
    const int text_scale = calcOverlayTextScale(frame);
    const int text_offset = calcOverlaySize(frame, 12);
    image_buffer_t* target = frame.virt_addr ? &frame : &cpu_frame;
    for (const auto& p : buildOBBOverlayPrims(results)) {
        if (p.kind == OverlayPrim::Kind::Line) {
            ok = ok && drawLineNV12CPU(*target, p.x1, p.y1, p.x2, p.y2, line_thickness, p.color);
        } else if (p.kind == OverlayPrim::Kind::Label) {
            const int lx = std::min(frame.width - 1, p.anchor_x);
            const int ly = std::min(frame.height - 1, p.anchor_y - text_offset);
            drawOverlayText(*target, p.text, std::max(0, lx), std::max(0, ly),
                            cv::Scalar(255, 255, 255), text_scale, p.color);
        }
    }

    if (cpu_addr) {
        unmapNv12FrameForCpu(frame.fd, cpu_addr, cpu_bytes);
    }
    return ok;
}

bool drawSegResultsZeroCopy(image_buffer_t& frame, const seg_detect_result_list& results, bool draw_box) {
    static bool rga_seg_failed = false;
    if (frame.virt_addr == nullptr && frame.fd <= 0) {
        return false;
    }
    if (frame.width <= 0 || frame.height <= 0) {
        return false;
    }
    if (frame.format != IMAGE_FORMAT_YUV420SP_NV12 && frame.format != IMAGE_FORMAT_YUV420SP_NV21) {
        return false;
    }
    int dst_wstride = frame.width_stride > 0 ? frame.width_stride : frame.width;
    int dst_hstride = frame.height_stride > 0 ? frame.height_stride : frame.height;
    if (dst_wstride < frame.width || dst_hstride < frame.height) {
        return false;
    }
    ensureRga();

    void* cpu_addr = nullptr;
    size_t cpu_bytes = 0;
    bool mapped_for_cpu = false;
    image_buffer_t cpu_frame = frame;
    auto ensureCpuFrame = [&]() -> image_buffer_t* {
        if (frame.virt_addr != nullptr) {
            return &frame;
        }
        if (frame.fd <= 0) {
            return nullptr;
        }
        if (!mapped_for_cpu) {
            if (!mapNv12FrameForCpu(frame, cpu_addr, cpu_bytes)) {
                return nullptr;
            }
            cpu_frame.virt_addr = static_cast<unsigned char*>(cpu_addr);
            mapped_for_cpu = true;
        }
        return &cpu_frame;
    };

    int count = static_cast<int>(results.boxes.size());
    // 黑底视图：先整帧置黑（NV12），掩膜区域随后以颜色混合到黑底上
    if (segBlackBgEnabled()) {
        image_buffer_t* bg = ensureCpuFrame();
        if (!bg) {
            return false;
        }
        fillRectNV12CPU(*bg, 0, 0, frame.width, frame.height, cv::Scalar(0, 0, 0));
    }
    int text_scale = calcOverlayTextScale(frame);
    int text_offset = calcOverlaySize(frame, 12);
    count = std::min(count, static_cast<int>(results.class_ids.size()));
    count = std::min(count, static_cast<int>(results.scores.size()));
    for (int i = 0; i < count; ++i) {
        float score = results.scores[i];
        if (score < DISPLAY_THRESH) {
            continue;
        }
        cv::Rect box = results.boxes[i];
        box &= cv::Rect(0, 0, frame.width, frame.height);
        if (box.width <= 0 || box.height <= 0) {
            continue;
        }
        if (segMaskMinArea() > 0 && box.width * box.height < segMaskMinArea()) {
            continue;  // 小目标跳过掩膜混合（仍画框和标签）
        }
        if (i < static_cast<int>(results.masks.size())) {
            cv::Mat mask = results.masks[i];
            if (!mask.empty()) {
                if (mask.type() != CV_8U) {
                    mask.convertTo(mask, CV_8U);
                }
                if (mask.size() != box.size()) {
                    cv::resize(mask, mask, box.size(), 0, 0, cv::INTER_LINEAR);
                    cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
                }
                cv::Scalar color = getSegColor(results.class_ids[i]);
                image_buffer_t* cpu = ensureCpuFrame();
                if (!cpu || !blendMaskNV12CPU(*cpu, mask, box, color)) {
                    if (mapped_for_cpu) {
                        unmapNv12FrameForCpu(frame.fd, cpu_addr, cpu_bytes);
                    }
                    return false;
                }
            }
        }
    }

    // Draw boxes and text（多任务叠加时 draw_box=false 跳过；RK_PIPE_HIDE_BOXES=1 全部隐藏）
    if (draw_box && !hideBoxesEnabled()) {
    for (int i = 0; i < count; ++i) {
        float score = results.scores[i];
        if (score < DISPLAY_THRESH) {
            continue;
        }
        cv::Rect box = results.boxes[i];
        box &= cv::Rect(0, 0, frame.width, frame.height);
        if (box.width <= 0 || box.height <= 0) {
            continue;
        }
        cv::Scalar color = getSegColor(results.class_ids[i]);

        image_buffer_t* target = &frame;
        if (rga_seg_failed && frame.virt_addr == nullptr) {
            target = ensureCpuFrame();
            if (!target) {
                if (mapped_for_cpu) {
                    unmapNv12FrameForCpu(frame.fd, cpu_addr, cpu_bytes);
                }
                return false;
            }
        }

        // 掩膜描边已提供实例轮廓,不画外接框;标签条照常(白字 + 同色底)
        // Draw text
        std::string className = coco_cls_to_name(results.class_ids[i]);
        std::string label = className + " " + fmtConf(score);
        int label_x = std::max(0, std::min(frame.width - 1, box.x));
        int label_y = std::max(0, std::min(frame.height - 1, box.y - text_offset));
        drawOverlayText(*target, label, label_x, label_y, cv::Scalar(255, 255, 255), text_scale, color);
    }
    }

    if (mapped_for_cpu) {
        unmapNv12FrameForCpu(frame.fd, cpu_addr, cpu_bytes);
    }
    return true;
}

bool drawOCRResultsZeroCopy(image_buffer_t& frame, const OCRDetectTaskResult& results) {
    if (frame.virt_addr == nullptr && frame.fd <= 0) {
        return false;
    }
    if (frame.width <= 0 || frame.height <= 0) {
        return false;
    }
    if (frame.format != IMAGE_FORMAT_YUV420SP_NV12 && frame.format != IMAGE_FORMAT_YUV420SP_NV21) {
        return false;
    }

    void* cpu_addr = nullptr;
    size_t cpu_bytes = 0;
    bool mapped_for_cpu = false;
    image_buffer_t cpu_frame = frame;
    auto ensureCpuFrame = [&]() -> image_buffer_t* {
        if (frame.virt_addr != nullptr) {
            return &frame;
        }
        if (frame.fd <= 0) {
            return nullptr;
        }
        if (!mapped_for_cpu) {
            if (!mapNv12FrameForCpu(frame, cpu_addr, cpu_bytes)) {
                return nullptr;
            }
            cpu_frame.virt_addr = static_cast<unsigned char*>(cpu_addr);
            mapped_for_cpu = true;
        }
        return &cpu_frame;
    };

    image_buffer_t* target = ensureCpuFrame();
    if (!target) {
        return false;
    }

    bool ok = true;
    int line_thickness = calcOverlaySize(frame, 2);
    int text_scale = calcOverlayTextScale(frame);
    int text_offset = calcOverlaySize(frame, 12);
    for (const OCRPolygon& polygon : results.polygons) {
        if (polygon.score < DISPLAY_THRESH) {
            continue;
        }

        float min_x = polygon.points[0].x;
        float min_y = polygon.points[0].y;
        for (size_t idx = 0; idx < polygon.points.size(); ++idx) {
            const cv::Point2f& start = polygon.points[idx];
            const cv::Point2f& end = polygon.points[(idx + 1) % polygon.points.size()];
            ok = ok && drawLineNV12CPU(*target,
                                       static_cast<int>(std::round(start.x)),
                                       static_cast<int>(std::round(start.y)),
                                       static_cast<int>(std::round(end.x)),
                                       static_cast<int>(std::round(end.y)),
                                       line_thickness,
                                       cv::Scalar(0, 255, 255));
            min_x = std::min(min_x, start.x);
            min_y = std::min(min_y, start.y);
        }

        std::string label = "TEXT " + std::to_string(static_cast<int>(polygon.score * 100)) + "%";
        int label_x = std::max(0, std::min(frame.width - 1, static_cast<int>(min_x)));
        int label_y = std::max(0, std::min(frame.height - 1, static_cast<int>(min_y) - text_offset));
        drawOverlayText(*target, label, label_x, label_y, cv::Scalar(0, 255, 255), text_scale);
    }

    if (mapped_for_cpu) {
        unmapNv12FrameForCpu(frame.fd, cpu_addr, cpu_bytes);
    }
    return ok;
}

void drawPoseResults(image_buffer_t& frame, const pose_detect_result_list& results) {
    bool rga_ok = true;
    int box_thickness = calcOverlaySize(frame, 1);
    int keypoint_size = std::max(3, calcOverlaySize(frame, 5));
    int keypoint_half = keypoint_size / 2;
    int text_scale = calcOverlayTextScale(frame);
    int text_offset = calcOverlaySize(frame, 12);
    for (int i = 0; i < results.count; ++i) {
        const pose_detect_result* det_result = &(results.results[i]);
        if (det_result->box_conf < DISPLAY_THRESH) {
            continue;
        }
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(frame.width - 1, x2);
        y2 = std::min(frame.height - 1, y2);

        const cv::Scalar person_color = classColor(det_result->track_id > 0 ? det_result->track_id : (i + 1));
        rga_ok = rga_ok && rgaDrawRectangleNV12(frame, x1, y1, x2, y2, box_thickness, person_color);

        for (int k = 0; k < KEYPOINT_NUM; k++) {
            const pose_keypoint& kp = det_result->keypoints[k];
            if (kp.conf > KEYPOINT_THRESH) {
                int kx = std::max(0, std::min(frame.width - 1, static_cast<int>(kp.x)));
                int ky = std::max(0, std::min(frame.height - 1, static_cast<int>(kp.y)));
                rga_ok = rga_ok && rgaFillRectNV12(frame, kx - keypoint_half, ky - keypoint_half, keypoint_size, keypoint_size, cv::Scalar(255, 255, 255));
            }
        }
    }

    if (rga_ok) {
        for (int i = 0; i < results.count; ++i) {
            const pose_detect_result* det_result = &(results.results[i]);
            if (det_result->box_conf < DISPLAY_THRESH) {
                continue;
            }
            int x1 = std::max(0, det_result->box.left);
            int y1 = std::max(0, det_result->box.top);
            std::string confText = fmtConf(det_result->box_conf);
            drawOverlayText(frame, confText, x1, std::max(0, y1 - text_offset), cv::Scalar(255, 255, 255), text_scale,
                            classColor(det_result->track_id > 0 ? det_result->track_id : (i + 1)));
        }
    }
}

bool drawPoseResultsZeroCopy(image_buffer_t& frame, const pose_detect_result_list& results, bool draw_box) {
    if (frame.virt_addr == nullptr && frame.fd <= 0) {
        return false;
    }
    if (frame.width <= 0 || frame.height <= 0) {
        return false;
    }
    if (frame.format != IMAGE_FORMAT_YUV420SP_NV12 && frame.format != IMAGE_FORMAT_YUV420SP_NV21) {
        return false;
    }

    void* cpu_addr = nullptr;
    size_t cpu_bytes = 0;
    bool mapped_for_cpu = false;
    image_buffer_t cpu_frame = frame;
    auto ensureCpuFrame = [&]() -> image_buffer_t* {
        if (frame.virt_addr != nullptr) {
            return &frame;
        }
        if (frame.fd <= 0) {
            return nullptr;
        }
        if (!mapped_for_cpu) {
            if (!mapNv12FrameForCpu(frame, cpu_addr, cpu_bytes)) {
                return nullptr;
            }
            cpu_frame.virt_addr = static_cast<unsigned char*>(cpu_addr);
            mapped_for_cpu = true;
        }
        return &cpu_frame;
    };

    image_buffer_t* target = ensureCpuFrame();
    if (!target) {
        return false;
    }

    bool ok = true;
    int box_thickness = calcOverlaySize(frame, 1);
    int line_thickness = calcOverlaySize(frame, 2);
    int keypoint_size = std::max(3, calcOverlaySize(frame, 5));
    int keypoint_half = keypoint_size / 2;
    int text_scale = calcOverlayTextScale(frame);
    int text_offset = calcOverlaySize(frame, 12);
    const int skeleton[][2] = {
        {0, 1}, {0, 2},
        {1, 3}, {2, 4},
        {5, 6},
        {5, 11}, {6, 12},
        {11, 12},
        {6, 8}, {8, 10},
        {5, 7}, {7, 9},
        {12, 14}, {14, 16},
        {11, 13}, {13, 15}
    };

    for (int i = 0; i < results.count; ++i) {
        const pose_detect_result* det_result = &(results.results[i]);
        if (det_result->box_conf < DISPLAY_THRESH) {
            continue;
        }
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(frame.width - 1, x2);
        y2 = std::min(frame.height - 1, y2);

        if (draw_box && !hideBoxesEnabled()) {
            ok = ok && drawRectangleNV12CPU(*target, x1, y1, x2, y2, box_thickness, classColor(det_result->track_id > 0 ? det_result->track_id : (i + 1)));
        }

        for (size_t k = 0; k < sizeof(skeleton) / sizeof(skeleton[0]); k++) {
            int idx1 = skeleton[k][0];
            int idx2 = skeleton[k][1];
            const pose_keypoint& kp1 = det_result->keypoints[idx1];
            const pose_keypoint& kp2 = det_result->keypoints[idx2];
            if (kp1.conf > KEYPOINT_THRESH && kp2.conf > KEYPOINT_THRESH) {
                // 骨架线加粗一档，深红更醒目（多任务叠加时盖在淡掩膜之上）
                ok = ok && drawLineNV12CPU(*target, static_cast<int>(kp1.x), static_cast<int>(kp1.y), static_cast<int>(kp2.x), static_cast<int>(kp2.y), line_thickness + 1, classColor(det_result->track_id > 0 ? det_result->track_id : (i + 1)));
            }
        }

        for (int k = 0; k < KEYPOINT_NUM; k++) {
            const pose_keypoint& kp = det_result->keypoints[k];
            if (kp.conf > KEYPOINT_THRESH) {
                int kx = std::max(0, std::min(frame.width - 1, static_cast<int>(kp.x)));
                int ky = std::max(0, std::min(frame.height - 1, static_cast<int>(kp.y)));
                fillRectNV12CPU(*target, kx - keypoint_half, ky - keypoint_half, keypoint_size, keypoint_size, cv::Scalar(255, 255, 255));
            }
        }

        if (draw_box && !hideBoxesEnabled()) {
            std::string confText = fmtConf(det_result->box_conf);
            drawOverlayText(*target, confText, x1, std::max(0, y1 - text_offset), cv::Scalar(255, 255, 255), text_scale,
                            classColor(det_result->track_id > 0 ? det_result->track_id : (i + 1)));
        }
    }

    if (mapped_for_cpu) {
        unmapNv12FrameForCpu(frame.fd, cpu_addr, cpu_bytes);
    }
    return ok;
}

static std::vector<OverlayPrim> buildPoseOverlayPrims(const pose_detect_result_list& results,
                                                      bool draw_box, int frame_cols, int frame_rows) {
    static const int kSkeleton[][2] = {
        {0, 1}, {0, 2},  {1, 3},  {2, 4},   {5, 6},   {5, 11}, {6, 12}, {11, 12},
        {6, 8}, {8, 10}, {5, 7},  {7, 9},   {12, 14}, {14, 16}, {11, 13}, {13, 15}};
    std::vector<OverlayPrim> prims;
    prims.reserve(static_cast<size_t>(results.count) * 60);
    const int thickness = std::max(1, std::min(frame_cols, frame_rows) / 320);
    const int keypoint_radius = std::max(2, std::min(frame_cols, frame_rows) / 160);
    const double text_scale = std::max(0.4, std::min(frame_cols, frame_rows) / 900.0);

    for (int i = 0; i < results.count; ++i) {
        const pose_detect_result* det = &(results.results[i]);
        if (det->box_conf < DISPLAY_THRESH) {
            continue;
        }
        const int x1 = std::max(0, std::min(frame_cols - 1, det->box.left));
        const int y1 = std::max(0, std::min(frame_rows - 1, det->box.top));
        const int x2 = std::max(0, std::min(frame_cols - 1, det->box.right));
        const int y2 = std::max(0, std::min(frame_rows - 1, det->box.bottom));
        // 每人一色(黄金角调色板),框/骨架/标签条同色便于多人辨识
        const cv::Scalar person_color =
            classColor(det->track_id > 0 ? det->track_id : (i + 1));
        if (draw_box && !hideBoxesEnabled()) {
            OverlayPrim box;
            box.kind = OverlayPrim::Kind::Box;
            box.x1 = x1; box.y1 = y1; box.x2 = x2; box.y2 = y2;
            box.color = person_color;
            prims.push_back(std::move(box));
        }
        for (const auto& bone : kSkeleton) {
            const pose_keypoint& kp1 = det->keypoints[bone[0]];
            const pose_keypoint& kp2 = det->keypoints[bone[1]];
            if (kp1.conf > KEYPOINT_THRESH && kp2.conf > KEYPOINT_THRESH) {
                OverlayPrim line;
                line.kind = OverlayPrim::Kind::Line;
                line.x1 = static_cast<int>(kp1.x);
                line.y1 = static_cast<int>(kp1.y);
                line.x2 = static_cast<int>(kp2.x);
                line.y2 = static_cast<int>(kp2.y);
                line.color = person_color;
                line.radius = thickness + 1;  // Line 复用 radius 字段带线宽
                prims.push_back(std::move(line));
            }
        }
        for (int k = 0; k < KEYPOINT_NUM; ++k) {
            const pose_keypoint& kp = det->keypoints[k];
            if (kp.conf > KEYPOINT_THRESH) {
                // 白心 + 人色描边,层次清晰：先人色大圆(filled)，后白心小圆
                OverlayPrim halo;
                halo.kind = OverlayPrim::Kind::Point;
                halo.x1 = static_cast<int>(kp.x);
                halo.y1 = static_cast<int>(kp.y);
                halo.radius = keypoint_radius + 1;
                halo.color = person_color;
                prims.push_back(std::move(halo));
                OverlayPrim core;
                core.kind = OverlayPrim::Kind::Point;
                core.x1 = static_cast<int>(kp.x);
                core.y1 = static_cast<int>(kp.y);
                core.radius = keypoint_radius;
                core.color = cv::Scalar(255, 255, 255);
                prims.push_back(std::move(core));
            }
        }
        if (draw_box && !hideBoxesEnabled()) {
            OverlayPrim label;
            label.kind = OverlayPrim::Kind::Label;
            label.anchor_x = x1;
            label.anchor_y = y1;
            label.text = fmtConf(det->box_conf);
            label.color = classColor(det->track_id > 0 ? det->track_id : (i + 1));
            prims.push_back(std::move(label));
        }
    }
    return prims;
}

void drawPoseResultsBGR(cv::Mat& frame, const pose_detect_result_list& results, bool draw_box) {
    if (frame.empty()) {
        return;
    }
    renderOverlayPrimsBgr(frame, buildPoseOverlayPrims(results, draw_box, frame.cols, frame.rows),
                          320, 900.0, false);
}

void drawOBBResultsBGR(cv::Mat& frame, const obb_detect_result_list& results) {
    if (frame.empty() || hideBoxesEnabled()) {
        return;
    }
    renderOverlayPrimsBgr(frame, buildOBBOverlayPrims(results));
}

void drawSegResultsBGR(cv::Mat& frame, const seg_detect_result_list& results, double scale, bool draw_box) {
    if (frame.empty()) {
        return;
    }
    // 黑底视图：先整帧置黑，掩膜区域随后以颜色混合到黑底上
    if (segBlackBgEnabled()) {
        frame.setTo(cv::Scalar(0, 0, 0));
    }
    if (scale <= 0.0) {
        scale = 1.0;
    }
    int thickness = std::max(1, std::min(frame.cols, frame.rows) / 320);
    const cv::Rect frame_rect(0, 0, frame.cols, frame.rows);

    int count = static_cast<int>(results.boxes.size());
    count = std::min(count, static_cast<int>(results.class_ids.size()));
    count = std::min(count, static_cast<int>(results.scores.size()));

    // 掩膜叠加（只处理掩膜非零区域，减少 resize/addWeighted 的 CPU 开销）
    for (int i = 0; i < count; ++i) {
        float score = results.scores[i];
        if (score < DISPLAY_THRESH) {
            continue;
        }
        const cv::Rect orig = results.boxes[i];
        cv::Rect box(static_cast<int>(orig.x * scale),
                     static_cast<int>(orig.y * scale),
                     static_cast<int>(orig.width * scale),
                     static_cast<int>(orig.height * scale));
        box &= frame_rect;
        if (box.width <= 0 || box.height <= 0) {
            continue;
        }
        // 用原始(未缩放)框面积判断，避免预览降采样导致中距离目标被误判为小目标
        if (segMaskMinArea() > 0 && orig.width * orig.height < segMaskMinArea()) {
            continue;  // 小目标跳过掩膜混合（仍画框和标签）
        }
        if (i >= static_cast<int>(results.masks.size())) {
            continue;
        }
        cv::Mat mask = results.masks[i];
        if (mask.empty()) {
            continue;
        }
        if (mask.type() != CV_8U) {
            mask.convertTo(mask, CV_8U);
        }
        if (mask.size() != box.size()) {
            cv::resize(mask, mask, box.size(), 0, 0, cv::INTER_LINEAR);
            cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
        }
        // 掩膜通常只覆盖框内一小片区域，只在其非零包围盒内做混合
        cv::Rect sub_rect = cv::boundingRect(mask);
        if (sub_rect.width <= 0 || sub_rect.height <= 0) {
            continue;
        }
        cv::Rect roi(box.x + sub_rect.x, box.y + sub_rect.y, sub_rect.width, sub_rect.height);
        cv::Mat sub_mask = mask(sub_rect);
        cv::Mat overlay(sub_rect.height, sub_rect.width, CV_8UC3, getSegColor(results.class_ids[i]));

        // 以掩膜强度为 alpha 混合，支持 INTER_LINEAR 平滑边缘：
        // 颜色权重 = mask*alpha，帧权重 = 255-颜色权重。
        // 普通视图 alpha=0.55（饱和醒目）；黑底视图 alpha=0.75（纯黑底上掩膜更实）
        const double mask_alpha = segBlackBgEnabled() ? 0.75 : 0.55;
        cv::Mat mask3;
        cv::cvtColor(sub_mask, mask3, cv::COLOR_GRAY2BGR);
        cv::Mat color_w, frame_w;
        cv::multiply(mask3, cv::Scalar::all(mask_alpha), color_w);
        cv::subtract(cv::Scalar::all(255), color_w, frame_w);
        cv::Mat blended;
        cv::multiply(frame(roi), frame_w, blended, 1.0 / 255.0);
        cv::Mat color_part;
        cv::multiply(overlay, color_w, color_part, 1.0 / 255.0);
        cv::add(blended, color_part, blended);
        blended.copyTo(frame(roi));

        // 掩膜边缘描边(同色细线抗锯齿):实例边界清晰
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(sub_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        const cv::Point origin(roi.x, roi.y);
        for (const auto& c : contours) {
            if (c.size() < 3) {
                continue;
            }
            for (size_t k = 0; k < c.size(); ++k) {
                cv::line(frame, c[k] + origin, c[(k + 1) % c.size()] + origin,
                         getSegColor(results.class_ids[i]), 1, cv::LINE_AA);
            }
        }
    }

    // 框与文字（多任务叠加时 draw_box=false 跳过，框由主任务画；RK_PIPE_HIDE_BOXES=1 全部隐藏）
    if (draw_box && !hideBoxesEnabled()) {
        for (int i = 0; i < count; ++i) {
            float score = results.scores[i];
            if (score < DISPLAY_THRESH) {
                continue;
            }
            const cv::Rect orig = results.boxes[i];
            cv::Rect box(static_cast<int>(orig.x * scale),
                         static_cast<int>(orig.y * scale),
                         static_cast<int>(orig.width * scale),
                         static_cast<int>(orig.height * scale));
            box &= frame_rect;
            if (box.width <= 0 || box.height <= 0) {
                continue;
            }
            cv::Scalar color = getSegColor(results.class_ids[i]);

            // 标签条：同色实底 + 白字（Ultralytics 样式），与掩膜同色；掩膜描边已提供轮廓,不画外接框
            std::string label = std::string(coco_cls_to_name(results.class_ids[i])) + " " +
                                fmtConf(score);
            const double label_scale = std::max(0.4, std::min(frame.cols, frame.rows) / 900.0);
            putClassLabelBGR(frame, label, box.x, box.y, color, label_scale, std::max(1, thickness - 1));
        }
    }
}

void drawDepthResultsBGR(cv::Mat& frame, const cv::Mat& depth, double alpha, const cv::Rect* roi) {
    if (frame.empty() || depth.empty()) {
        return;
    }
    // RK_PIPE_DEPTH_REPLACE=1：整帧 replace 深度视图（alpha 强制 1.0，纯深度伪彩，不再与原始帧混合）
    const char* replace_env = getenv("RK_PIPE_DEPTH_REPLACE");
    if (replace_env && *replace_env && strcmp(replace_env, "0") != 0) {
        alpha = 1.0;
    }
    cv::Rect target;
    if (roi && roi->width > 0 && roi->height > 0) {
        target = *roi & cv::Rect(0, 0, frame.cols, frame.rows);
        if (target.width <= 0 || target.height <= 0) {
            return;
        }
    } else {
        target = cv::Rect(0, 0, frame.cols, frame.rows);
    }
    cv::Mat depth_sized;
    if (depth.size() != cv::Size(target.width, target.height)) {
        cv::resize(depth, depth_sized, cv::Size(target.width, target.height), 0, 0, cv::INTER_LINEAR);
    } else {
        depth_sized = depth;
    }
    cv::Mat color;
    cv::applyColorMap(depth_sized, color, cv::COLORMAP_JET);
    if (alpha >= 1.0) {
        // replace：直接把伪彩写入目标区域（不混合）
        color.copyTo(frame(target));
    } else {
        cv::addWeighted(frame(target), 1.0 - alpha, color, alpha, 0, frame(target));
    }
}

std::vector<DepthDistanceTarget> drawDepthDistanceOverlayBGR(
    cv::Mat& frame, const cv::Mat& depth, const cv::Rect* roi,
    float depth_lo, float depth_hi, const object_detect_result_list& dets,
    float near_m, float scale, bool draw_text) {
    std::vector<DepthDistanceTarget> near_targets;
    if (frame.empty() || depth.empty() || !roi || roi->width <= 0 || roi->height <= 0 ||
        depth_hi <= depth_lo || dets.count <= 0) {
        return near_targets;
    }
    const double frame_w = static_cast<double>(frame.cols);
    for (int i = 0; i < dets.count; ++i) {
        const object_detect_result& d = dets.results[i];
        const int bx1 = std::max(0, std::min(d.box.left, frame.cols - 1));
        const int by1 = std::max(0, std::min(d.box.top, frame.rows - 1));
        const int bx2 = std::max(0, std::min(d.box.right, frame.cols - 1));
        const int by2 = std::max(0, std::min(d.box.bottom, frame.rows - 1));
        if (bx2 <= bx1 || by2 <= by1) {
            continue;
        }
        // 框内深度中值 → 米（纯函数见 depth_distance.h）
        float meters = 0.0f;
        if (!boxDepthMeters(depth, *roi, d.box, depth_lo, depth_hi, scale, &meters)) {
            continue;
        }
        const bool near = near_m > 0.0f && meters <= near_m;
        if (near) {
            near_targets.push_back({d.cls_id, meters});
        }
        if (!draw_text) {
            continue;
        }
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1fm", meters);
        const double label_scale = frame_w > 1280.0 ? 0.6 : 0.5;
        const int thickness = frame_w > 1280.0 ? 2 : 1;
        // 近距目标红框 + 红字；普通目标黄字
        if (near) {
            cv::rectangle(frame, cv::Rect(bx1, by1, bx2 - bx1, by2 - by1),
                          cv::Scalar(0, 0, 255), frame_w > 1280.0 ? 3 : 2);
        }
        // 深色底增强可读性
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, label_scale, thickness, &baseline);
        const int tx = std::max(0, std::min(bx1, frame.cols - text_size.width - 4));
        const int ty = std::max(text_size.height + 4, by1);
        cv::rectangle(frame, cv::Rect(tx - 2, ty - text_size.height - 2, text_size.width + 4, text_size.height + 4),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, buf, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX, label_scale,
                    near ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255), thickness, cv::LINE_AA);
    }
    return near_targets;
}

// Detect3D 线框投影的 P2 矩阵（启动时 set_detect3d_p2 一次性设置，绘制线程只读）
static float g_detect3d_p2[12] = {0};
static bool g_detect3d_p2_valid = false;
static float g_detect3d_depth_scale = 1.0f;  // 模型深度全局缩放（场景尺度校准）

void set_detect3d_depth_scale(float scale) {
    g_detect3d_depth_scale = scale > 0.0f ? scale : 1.0f;
}

void set_detect3d_p2(const std::string& p2_spec) {
    g_detect3d_p2_valid = false;
    if (p2_spec.empty()) {
        return;
    }
    // 分隔符兼容逗号/空格/tab，数值支持科学计数法——KITTI calib 文件的 P2 行可直接粘贴
    float vals[12] = {0};
    int count = 0;
    std::string token;
    auto flush = [&]() {
        if (!token.empty() && count < 12) {
            vals[count++] = static_cast<float>(std::atof(token.c_str()));
            token.clear();
        }
    };
    for (const char ch : p2_spec) {
        if (ch == ',' || ch == ' ' || ch == '\t') {
            flush();
        } else {
            token.push_back(ch);
        }
    }
    flush();
    if (count != 12) {
        std::fprintf(stderr, "[rk_pipe][d3d] detect3d_p2 需要 12 个值（3x4 行主序），实际 %d 个，线框不启用\n", count);
        return;
    }
    if (!(vals[0] > 0.0f && vals[5] > 0.0f)) {
        std::fprintf(stderr, "[rk_pipe][d3d] detect3d_p2 的 fx/fy 非正，线框不启用\n");
        return;
    }
    std::memcpy(g_detect3d_p2, vals, sizeof(vals));
    g_detect3d_p2_valid = true;
    std::printf("[rk_pipe][d3d] P2 已配置，启用 3D 线框投影\n");
}

namespace {

// 12 条边：底面 4 + 顶面 4 + 立柱 4（角点序与 computeDetect3DCorners2D 一致）
void drawDetect3DWireframe(cv::Mat& frame, const float corners[8][2]) {
    const int thickness = std::max(1, std::min(frame.cols, frame.rows) / 400);
    auto edge = [&](int a, int b) {
        cv::Point p1(static_cast<int>(std::lround(corners[a][0])),
                     static_cast<int>(std::lround(corners[a][1])));
        cv::Point p2(static_cast<int>(std::lround(corners[b][0])),
                     static_cast<int>(std::lround(corners[b][1])));
        cv::clipLine(cv::Rect(0, 0, frame.cols, frame.rows), p1, p2);
        cv::line(frame, p1, p2, cv::Scalar(255, 200, 0), thickness, cv::LINE_AA);
    };
    for (int i = 0; i < 4; ++i) {
        edge(i, (i + 1) % 4);          // 底面
        edge(i + 4, (i + 1) % 4 + 4);  // 顶面
        edge(i, i + 4);                // 立柱
    }
}

}  // namespace

void drawDetect3DResultsBGR(cv::Mat& frame, const Detect3DTaskResult& results) {
    static const bool dbg = []() {
        const char* e = getenv("RK_PIPE_DEBUG_D3D");
        return e && *e && strcmp(e, "0") != 0;
    }();
    if (dbg) {
        std::printf("[d3d-draw] items=%zu frame=%dx%d\n", results.items.size(), frame.cols, frame.rows);
        for (const auto& it : results.items) {
            std::printf("[d3d-draw]   cls=%d conf=%.3f box=(%d,%d,%d,%d) d=%.1fm hwl=(%.2f,%.2f,%.2f)\n",
                        it.cls_id, (double)it.conf, it.box.left, it.box.top, it.box.right, it.box.bottom,
                        (double)it.depth_m, (double)it.h3, (double)it.w3, (double)it.l3);
        }
    }
    if (frame.empty() || results.items.empty() || hideBoxesEnabled()) {
        return;
    }
    const int thickness = std::max(1, std::min(frame.cols, frame.rows) / 400);
    const double text_scale = std::max(0.4, std::min(frame.cols, frame.rows) / 1000.0);
    const int text_thickness = std::max(1, thickness);
    for (const auto& it : results.items) {
        if (it.conf < DISPLAY_THRESH) {
            continue;
        }
        const int x1 = std::max(0, std::min(static_cast<int>(it.box.left), frame.cols - 1));
        const int y1 = std::max(0, std::min(static_cast<int>(it.box.top), frame.rows - 1));
        const int x2 = std::max(0, std::min(static_cast<int>(it.box.right), frame.cols - 1));
        const int y2 = std::max(0, std::min(static_cast<int>(it.box.bottom), frame.rows - 1));
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }
        const cv::Scalar color = getSegColor(it.cls_id);
        cv::rectangle(frame, cv::Rect(x1, y1, x2 - x1, y2 - y1), color, thickness);

        // 标签：类别+置信度。模型输出的深度/尺寸在未标定相机上为场景级粗估，默认不显示；
        // 需要时打开 RK_PIPE_DEBUG_D3D 或配置 detect3d_p2（线框）查看
        const std::string label = std::string(coco_cls_to_name(it.cls_id)) + " " +
                                  std::to_string(static_cast<int>(it.conf * 100)) + "%";

        int baseline = 0;
        const cv::Size s1 = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, text_scale, text_thickness, &baseline);
        const int label_w = s1.width + 8;
        const int label_h = s1.height + 8;
        const int lx = std::max(0, std::min(x1, frame.cols - label_w - 1));
        const int ly = std::max(label_h + 2, y1);
        cv::rectangle(frame, cv::Rect(lx, ly - label_h, label_w, label_h), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, label, cv::Point(lx + 4, ly - 4),
                    cv::FONT_HERSHEY_SIMPLEX, text_scale, color, text_thickness, cv::LINE_AA);

        // 3D 线框投影（仅在显式配置 detect3d_p2，即相机已标定时绘制）。
        // 深度乘 detect3d_depth_scale 做场景尺度校准（单目模型在非训练场景上深度常有整体偏差）
        if (g_detect3d_p2_valid) {
            Detect3DItem it_scaled = it;
            it_scaled.depth_m *= g_detect3d_depth_scale;
            float corners[8][2];
            if (computeDetect3DCorners2D(it_scaled, g_detect3d_p2, corners)) {
                drawDetect3DWireframe(frame, corners);
            }
        }
    }
}

namespace {

// Cityscapes 19 类标准色（RGB，绘制时转 BGR）；前 19 类与之对应
const int kCityscapesColors[19][3] = {
    {128, 64, 128},   // 0 road
    {244, 35, 232},   // 1 sidewalk
    {70, 70, 70},     // 2 building
    {102, 102, 156},  // 3 wall
    {190, 153, 153},  // 4 fence
    {153, 153, 153},  // 5 pole
    {250, 170, 30},   // 6 traffic light
    {220, 220, 0},    // 7 traffic sign
    {107, 142, 35},   // 8 vegetation
    {152, 251, 152},  // 9 terrain
    {70, 130, 180},   // 10 sky
    {220, 20, 60},    // 11 person
    {255, 0, 0},      // 12 rider
    {0, 0, 142},      // 13 car
    {0, 0, 70},       // 14 truck
    {0, 60, 100},     // 15 bus
    {0, 80, 100},     // 16 train
    {0, 0, 230},      // 17 motorcycle
    {119, 11, 32},    // 18 bicycle
};

inline void hsvToRgb(float h, float s, float v, int& r, int& g, int& b) {
    const float hh = std::fmod(h, 360.0f);
    const int i = static_cast<int>(hh / 60.0f) % 6;
    const float f = hh / 60.0f - static_cast<float>(i);
    const float p = v * (1.0f - s);
    const float q = v * (1.0f - f * s);
    const float t = v * (1.0f - (1.0f - f) * s);
    float rr = 0, gg = 0, bb = 0;
    switch (i) {
        case 0: rr = v; gg = t; bb = p; break;
        case 1: rr = q; gg = v; bb = p; break;
        case 2: rr = p; gg = v; bb = t; break;
        case 3: rr = p; gg = q; bb = v; break;
        case 4: rr = t; gg = p; bb = v; break;
        default: rr = v; gg = p; bb = q; break;
    }
    r = static_cast<int>(rr * 255.0f + 0.5f);
    g = static_cast<int>(gg * 255.0f + 0.5f);
    b = static_cast<int>(bb * 255.0f + 0.5f);
}

// 语义分割色表：前 19 类 Cityscapes 标准色，其余按黄金角 HSV 生成（避免相邻类同色）。
// cv::LUT 要求色表固定 256 项（CV_8U 输入），超出 class_num 的类别用中性色（深灰）占位。
cv::Mat buildSemColorLut(int class_num) {
    cv::Mat lut(1, 256, CV_8UC3);
    for (int c = 0; c < 256; ++c) {
        int r, g, b;
        if (c < class_num && c < 19) {
            r = kCityscapesColors[c][0];
            g = kCityscapesColors[c][1];
            b = kCityscapesColors[c][2];
        } else if (c < class_num) {
            hsvToRgb(c * 137.508f, 0.7f, 0.9f, r, g, b);
        } else {
            r = g = b = 45;  // 越界类别索引不会出现，置深灰占位
        }
        lut.at<cv::Vec3b>(0, c) = cv::Vec3b(b, g, r);  // 转 BGR
    }
    return lut;
}

}  // namespace

void drawSemResultsBGR(cv::Mat& frame, const cv::Mat& class_map, int class_num, const cv::Rect* roi, double alpha) {
    if (frame.empty() || class_map.empty()) {
        return;
    }
    cv::Rect target;
    if (roi && roi->width > 0 && roi->height > 0) {
        target = *roi & cv::Rect(0, 0, frame.cols, frame.rows);
        if (target.width <= 0 || target.height <= 0) {
            return;
        }
    } else {
        target = cv::Rect(0, 0, frame.cols, frame.rows);
    }
    // 混合透明度：RK_PIPE_SEM_ALPHA 覆盖（0~1），默认 0.5（越透出原画面，色块越不明显）
    static const double env_alpha = []() {
        const char* v = std::getenv("RK_PIPE_SEM_ALPHA");
        if (v && *v && strcmp(v, "0") != 0) {
            const double a = strtod(v, nullptr);
            if (a > 0.0 && a <= 1.0) return a;
        }
        return 0.5;
    }();
    if (env_alpha > 0.0) {
        alpha = env_alpha;
    }
    // 在低分辨率上先着色（cv::LUT 要求 3 通道 LUT 配 3 通道输入，先 GRAY2BGR），
    // 再放大到目标区域：彩色图用 INTER_LINEAR 产生过渡色 + 轻高斯模糊消除块状硬边，
    // 避免 80x80 最近邻放大成"大片马赛克"。
    const cv::Mat lut = buildSemColorLut(class_num);
    cv::Mat map3;
    cv::cvtColor(class_map, map3, cv::COLOR_GRAY2BGR);
    cv::Mat color_low;
    cv::LUT(map3, lut, color_low);
    cv::Mat color;
    if (color_low.size() != cv::Size(target.width, target.height)) {
        cv::resize(color_low, color, cv::Size(target.width, target.height), 0, 0, cv::INTER_LINEAR);
    } else {
        color = color_low;
    }
    cv::GaussianBlur(color, color, cv::Size(3, 3), 0.8);
    cv::addWeighted(frame(target), 1.0 - alpha, color, alpha, 0, frame(target));
}

void drawDetectionResultsBGR(cv::Mat& frame, const object_detect_result_list& results) {
    if (frame.empty() || hideBoxesEnabled()) {
        return;
    }
    renderOverlayPrimsBgr(frame, buildDetectionOverlayPrims(results, frame.cols, frame.rows));
}

// M6 文字识别结果：多边形描边（青色）+ 识别文本标签（多边形左上角实底条白字）。
// cv::putText 不支持 CJK：非 ASCII 字符以 '?' 占位（文本仍完整进 webhook/事件链路）。
void drawOCRTextBGR(cv::Mat& frame, const OCRDetectTaskResult& results) {
    if (frame.empty()) {
        return;
    }
    const cv::Scalar cyan(255, 200, 0);
    const int thickness = std::max(1, frame.cols / 800);
    const double text_scale = std::max(0.4, frame.cols / 1600.0);
    const size_t n = results.polygons.size();
    for (size_t i = 0; i < n; ++i) {
        const OCRPolygon& polygon = results.polygons[i];
        std::vector<std::vector<cv::Point>> outline;
        outline.emplace_back();
        for (const auto& p : polygon.points) {
            outline.back().emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
        }
        cv::polylines(frame, outline, true, cyan, thickness, cv::LINE_AA);

        std::string text;
        float score = polygon.score;
        if (i < results.lines.size()) {
            const OCRTextLine& line = results.lines[i];
            score = line.score > 0.0f ? line.score : score;
            for (const char c : line.text) {
                text += (static_cast<unsigned char>(c) >= 0x20 && static_cast<unsigned char>(c) < 0x7F)
                            ? std::string(1, c)
                            : "?";
            }
        }
        if (text.empty()) {
            continue;
        }
        char buf[32];
        std::snprintf(buf, sizeof(buf), " %.2f", score);
        const std::string label = text + buf;
        int top_y = frame.rows;
        int left_x = frame.cols;
        for (const auto& p : polygon.points) {
            top_y = std::min(top_y, static_cast<int>(p.y));
            left_x = std::min(left_x, static_cast<int>(p.x));
        }
        putClassLabelBGR(frame, label, left_x, top_y, cv::Scalar(40, 40, 40), text_scale,
                         thickness + 1);
    }
}

// M0 两阶级联：检测框沿用原类别配色，二级 top-1 标签画在检测框标签下方一行
void drawCompositeClsBGR(cv::Mat& frame, const CompositeClsTaskResult& results) {
    if (frame.empty() || results.data.count <= 0) {
        return;
    }
    const int thickness = std::max(1, frame.cols / 800);
    const double text_scale = std::max(0.4, frame.cols / 1600.0);
    const int n = std::min<int>(results.data.count,
                                static_cast<int>(results.cls_labels.size()));
    for (int i = 0; i < results.data.count; ++i) {
        const auto& det = results.data.results[i];
        const int x1 = std::max(0, det.box.left);
        const int y1 = std::max(0, det.box.top);
        const cv::Scalar color = classColor(det.cls_id);
        cv::rectangle(frame, cv::Rect(x1, y1, det.box.right - x1, det.box.bottom - y1), color,
                      thickness, cv::LINE_AA);
        const int label_count = i < n && !results.cls_labels[i].empty() ? 2 : 1;
        // 第一行：一级检测类名+分数；第二行：二级 top-1
        putClassLabelBGR(frame,
                         std::string(coco_cls_to_name(det.cls_id)) + " " +
                             fmtConf(det.prop),
                         x1, y1, color, text_scale, thickness);
        if (label_count == 2) {
            putClassLabelBGR(frame, results.cls_labels[i], x1,
                             y1 + static_cast<int>(text_scale * 22) + 4,
                             cv::Scalar(40, 40, 200), text_scale, thickness);
        }
    }
}

// M13 动作识别：新鲜动作标签标注在对应 track 框上方（第二行，青色与一级类名区分）
void drawActionLabelsBGR(cv::Mat& frame, const ActionTaskResult& results,
                         const std::vector<std::string>& labels) {
    if (frame.empty() || results.actions.empty() || results.data.count <= 0) {
        return;
    }
    const double text_scale = std::max(0.4, frame.cols / 1600.0);
    const int thickness = std::max(1, frame.cols / 800);
    const cv::Scalar action_color(200, 180, 20);  // 青绿色
    for (const ActionItem& act : results.actions) {
        // 按 track_id 找锚点框（跟踪平滑后的位置）
        bool anchored = false;
        for (int i = 0; i < results.data.count; ++i) {
            if (results.data.results[i].track_id != act.track_id) {
                continue;
            }
            const auto& box = results.data.results[i].box;
            std::string label =
                act.action_id >= 0 && static_cast<size_t>(act.action_id) < labels.size() &&
                        !labels[act.action_id].empty()
                    ? labels[act.action_id]
                    : "action " + std::to_string(act.action_id);
            char score_buf[16];
            std::snprintf(score_buf, sizeof(score_buf), " %.2f", act.score);
            label += score_buf;
            putClassLabelBGR(frame, label, std::max(0, box.left),
                             std::max(0, box.top) + static_cast<int>(text_scale * 22) + 4,
                             action_color, text_scale, thickness);
            anchored = true;
            break;
        }
        if (!anchored) {
            // 框已消失（轨迹刚老化/缺检）：标签落到画面左侧固定行，避免静默丢弃
            putClassLabelBGR(frame, "track " + std::to_string(act.track_id), 8,
                             static_cast<int>(text_scale * 44) + 8, action_color, text_scale,
                             thickness);
        }
    }
}

// M8 人脸：绿色框 + 5 点 landmark（红点，眉眼鼻嘴顺序）+ 左上角分数
static std::vector<OverlayPrim> buildFaceOverlayPrims(const FaceTaskResult& results, int frame_cols) {
    std::vector<OverlayPrim> prims;
    prims.reserve(results.faces.size() * 7);
    const cv::Scalar green(0, 255, 0);
    const int thickness = std::max(1, frame_cols / 800);
    for (const FaceItem& face : results.faces) {
        OverlayPrim box;
        box.kind = OverlayPrim::Kind::Box;
        box.x1 = face.box.left;
        box.y1 = face.box.top;
        box.x2 = face.box.right;
        box.y2 = face.box.bottom;
        box.color = green;
        prims.push_back(std::move(box));
        for (const auto& pt : face.landmarks) {
            OverlayPrim dot;
            dot.kind = OverlayPrim::Kind::Point;
            dot.x1 = static_cast<int>(pt.x);
            dot.y1 = static_cast<int>(pt.y);
            dot.radius = std::max(2, thickness * 2);
            dot.color = cv::Scalar(0, 0, 255);
            prims.push_back(std::move(dot));
        }
        OverlayPrim label;
        label.kind = OverlayPrim::Kind::Label;
        label.anchor_x = face.box.left;
        label.anchor_y = face.box.top;
        label.text = "face " + fmtConf(face.score);
        label.color = cv::Scalar(30, 90, 30);
        prims.push_back(std::move(label));
    }
    return prims;
}

void drawFaceResultsBGR(cv::Mat& frame, const FaceTaskResult& results) {
    if (frame.empty()) {
        return;
    }
    renderOverlayPrimsBgr(frame, buildFaceOverlayPrims(results, frame.cols), 800, 1600.0);
}
