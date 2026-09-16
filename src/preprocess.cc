#include "preprocess.h"
#include <cstddef>
#include <cmath>
#include <fcntl.h>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <linux/dma-heap.h>
#include <rga/im2d.h>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <opencv2/core.hpp>

static size_t image_size_by_stride(const image_buffer_t& img)
{
    int wstride = img.width_stride > 0 ? img.width_stride : img.width;
    int hstride = img.height_stride > 0 ? img.height_stride : img.height;
    if (wstride <= 0 || hstride <= 0) {
        return 0;
    }

    switch (img.format) {
    case IMAGE_FORMAT_RGB888:
    case IMAGE_FORMAT_BGR888:
        return static_cast<size_t>(wstride) * hstride * 3;
    case IMAGE_FORMAT_RGBA8888:
        return static_cast<size_t>(wstride) * hstride * 4;
    case IMAGE_FORMAT_GRAY8:
        return static_cast<size_t>(wstride) * hstride;
    case IMAGE_FORMAT_YUV420SP_NV12:
    case IMAGE_FORMAT_YUV420SP_NV21:
        return static_cast<size_t>(wstride) * hstride * 3 / 2;
    default:
        return img.size > 0 ? static_cast<size_t>(img.size) : 0;
    }
}

static bool dma_heap_alloc_any(size_t size, int& out_fd, void*& out_addr)
{
    static const char* kHeaps[] = {
        "/dev/dma_heap/system-uncached-dma32",
        "/dev/dma_heap/system-dma32",
        "/dev/dma_heap/dma32_uncached",
        "/dev/dma_heap/dma32",
        "/dev/dma_heap/cma-uncached",
        "/dev/dma_heap/cma",
        "/dev/dma_heap/system-uncached",
        "/dev/dma_heap/system",
    };

    for (const char* heap_path : kHeaps) {
        int heap_fd = open(heap_path, O_RDONLY | O_CLOEXEC);
        if (heap_fd < 0) {
            continue;
        }

        dma_heap_allocation_data data;
        std::memset(&data, 0, sizeof(data));
        data.len = size;
        data.fd_flags = O_RDWR | O_CLOEXEC;
        data.heap_flags = 0;

        int ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data);
        close(heap_fd);
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

static bool dma_heap_alloc_dma32(size_t size, int& out_fd, void*& out_addr)
{
    static const char* kHeaps[] = {
        "/dev/dma_heap/system-uncached-dma32",
        "/dev/dma_heap/system-dma32",
        "/dev/dma_heap/dma32_uncached",
        "/dev/dma_heap/dma32",
    };

    for (const char* heap_path : kHeaps) {
        int heap_fd = open(heap_path, O_RDONLY | O_CLOEXEC);
        if (heap_fd < 0) {
            continue;
        }

        dma_heap_allocation_data data;
        std::memset(&data, 0, sizeof(data));
        data.len = size;
        data.fd_flags = O_RDWR | O_CLOEXEC;
        data.heap_flags = 0;

        int ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &data);
        close(heap_fd);
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

static void dma_heap_free(size_t size, int& fd, void*& addr)
{
    if (size > 0 && addr && addr != MAP_FAILED) {
        munmap(addr, size);
        addr = nullptr;
    }
    if (fd > 0) {
        close(fd);
        fd = -1;
    }
}

static bool get_rga_format(image_format_t format, int& rga_format)
{
    switch (format) {
    case IMAGE_FORMAT_RGB888:
        rga_format = RK_FORMAT_RGB_888;
        return true;
    case IMAGE_FORMAT_BGR888:
        rga_format = RK_FORMAT_BGR_888;
        return true;
    case IMAGE_FORMAT_RGBA8888:
        rga_format = RK_FORMAT_RGBA_8888;
        return true;
    case IMAGE_FORMAT_GRAY8:
        rga_format = RK_FORMAT_Y8;
        return true;
    case IMAGE_FORMAT_YUV420SP_NV12:
        rga_format = RK_FORMAT_YCbCr_420_SP;
        return true;
    case IMAGE_FORMAT_YUV420SP_NV21:
        rga_format = RK_FORMAT_YCrCb_420_SP;
        return true;
    default:
        return false;
    }
}

static void fill_image_cpu(image_buffer_t* dst_image, char color)
{
    if (!dst_image || !dst_image->virt_addr || dst_image->width <= 0 || dst_image->height <= 0) {
        return;
    }

    int wstride = dst_image->width_stride > 0 ? dst_image->width_stride : dst_image->width;
    int hstride = dst_image->height_stride > 0 ? dst_image->height_stride : dst_image->height;
    unsigned char* base = static_cast<unsigned char*>(dst_image->virt_addr);
    unsigned char gray = static_cast<unsigned char>(color);

    switch (dst_image->format) {
    case IMAGE_FORMAT_RGB888:
    case IMAGE_FORMAT_BGR888:
        std::memset(base, gray, static_cast<size_t>(wstride) * hstride * 3);
        break;
    case IMAGE_FORMAT_RGBA8888:
        std::memset(base, gray, static_cast<size_t>(wstride) * hstride * 4);
        break;
    case IMAGE_FORMAT_GRAY8:
        std::memset(base, gray, static_cast<size_t>(wstride) * hstride);
        break;
    case IMAGE_FORMAT_YUV420SP_NV12:
    case IMAGE_FORMAT_YUV420SP_NV21: {
        size_t y_size = static_cast<size_t>(wstride) * hstride;
        std::memset(base, gray, y_size);
        std::memset(base + y_size, 0x80, y_size / 2);
        break;
    }
    default:
        if (dst_image->size > 0) {
            std::memset(base, 0, static_cast<size_t>(dst_image->size));
        }
        break;
    }
}

static rga_buffer_t wrap_rga_fd_with_optional_import(int fd,
                                                     int width,
                                                     int height,
                                                     int format,
                                                     int wstride,
                                                     int hstride,
                                                     size_t bytes,
                                                     rga_buffer_handle_t& out_handle)
{
    out_handle = 0;
    if (bytes > 0 && bytes <= static_cast<size_t>(std::numeric_limits<int>::max())) {
        rga_buffer_handle_t handle = importbuffer_fd(fd, static_cast<int>(bytes));
        if (handle != 0) {
            out_handle = handle;
            return wrapbuffer_handle(handle, width, height, format, wstride, hstride);
        }
    }
    return wrapbuffer_fd(fd, width, height, format, wstride, hstride);
}

static int rga_resize_then_cpu_letterbox(image_buffer_t* src_image, image_buffer_t* dst_image, letterbox_t* letterbox, char color)
{
    if (!src_image || !dst_image || !letterbox) {
        return -1;
    }
    if ((src_image->virt_addr == nullptr) && src_image->fd <= 0) {
        return -1;
    }
    if ((dst_image->virt_addr == nullptr) && dst_image->fd <= 0) {
        return -1;
    }

    int src_format = RK_FORMAT_UNKNOWN;
    int dst_format = RK_FORMAT_UNKNOWN;
    if (!get_rga_format(src_image->format, src_format) || !get_rga_format(dst_image->format, dst_format)) {
        return -1;
    }

    int dst_w = dst_image->width;
    int dst_h = dst_image->height;
    if (dst_w <= 0 || dst_h <= 0 || src_image->width <= 0 || src_image->height <= 0) {
        return -1;
    }
    letterbox->crop_x = 0;
    letterbox->crop_y = 0;
    letterbox->crop_w = src_image->width;
    letterbox->crop_h = src_image->height;

    float scale_w = static_cast<float>(dst_w) / static_cast<float>(src_image->width);
    float scale_h = static_cast<float>(dst_h) / static_cast<float>(src_image->height);
    float scale = std::min(scale_w, scale_h);

    int resized_w = std::max(1, static_cast<int>(src_image->width * scale));
    int resized_h = std::max(1, static_cast<int>(src_image->height * scale));
    if (resized_w % 4 != 0) {
        resized_w -= (resized_w % 4);
        resized_w = std::max(1, resized_w);
    }
    if (resized_h % 2 != 0) {
        resized_h -= (resized_h % 2);
        resized_h = std::max(1, resized_h);
    }

    int x_pad = (dst_w - resized_w) / 2;
    int y_pad = (dst_h - resized_h) / 2;
    auto align_even_nonneg = [](int v) -> int {
        if (v < 0) {
            return 0;
        }
        return v & ~1;
    };
    if (scale_w < scale_h) {
        y_pad = align_even_nonneg(y_pad);
    } else {
        x_pad = align_even_nonneg(x_pad);
    }
    if (dst_image->format == IMAGE_FORMAT_YUV420SP_NV12 || dst_image->format == IMAGE_FORMAT_YUV420SP_NV21) {
        x_pad = align_even_nonneg(x_pad);
        y_pad = align_even_nonneg(y_pad);
    }

    letterbox->x_pad = x_pad;
    letterbox->y_pad = y_pad;
    letterbox->scale = scale;

    size_t dst_bytes = dst_image->size > 0 ? static_cast<size_t>(dst_image->size) : image_size_by_stride(*dst_image);
    unsigned char* dst_base = static_cast<unsigned char*>(dst_image->virt_addr);
    void* mapped = nullptr;
    if (!dst_base && dst_image->fd > 0 && dst_bytes > 0) {
        mapped = mmap(nullptr, dst_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, dst_image->fd, 0);
        if (mapped != MAP_FAILED) {
            dst_base = static_cast<unsigned char*>(mapped);
        }
    }
    auto unmap_dst = [&]() {
        if (mapped && mapped != MAP_FAILED) {
            munmap(mapped, dst_bytes);
        }
    };
    if (!dst_base) {
        unmap_dst();
        return -1;
    }

    image_buffer_t dst_for_cpu = *dst_image;
    dst_for_cpu.virt_addr = dst_base;
    fill_image_cpu(&dst_for_cpu, color);

    struct CachedDma {
        int fd = -1;
        void* addr = nullptr;
        size_t size = 0;
    };

    auto ensure_dma = [&](CachedDma& b, size_t bytes, bool require_dma32) -> bool {
        if (bytes == 0) {
            return false;
        }
        if (b.fd > 0 && b.addr && b.addr != MAP_FAILED && b.size >= bytes) {
            return true;
        }
        if (b.fd > 0 || (b.addr && b.addr != MAP_FAILED) || b.size > 0) {
            dma_heap_free(b.size, b.fd, b.addr);
            b.size = 0;
        }
        int fd = -1;
        void* addr = nullptr;
        if (require_dma32) {
            if (!dma_heap_alloc_dma32(bytes, fd, addr)) {
                return false;
            }
        } else {
            if (!dma_heap_alloc_any(bytes, fd, addr)) {
                return false;
            }
        }
        if (fd <= 0 || addr == nullptr || addr == MAP_FAILED) {
            return false;
        }
        b.fd = fd;
        b.addr = addr;
        b.size = bytes;
        return true;
    };

    static thread_local CachedDma tl_src_dma;
    static thread_local CachedDma tl_resized_dma;

    image_buffer_t src_for_rga = *src_image;
    if (src_for_rga.fd <= 0) {
        size_t src_bytes = image_size_by_stride(src_for_rga);
        if (!ensure_dma(tl_src_dma, src_bytes, false)) {
            unmap_dst();
            return -1;
        }
        std::memcpy(tl_src_dma.addr, src_for_rga.virt_addr, src_bytes);
        src_for_rga.fd = tl_src_dma.fd;
        src_for_rga.virt_addr = static_cast<unsigned char*>(tl_src_dma.addr);
        src_for_rga.size = static_cast<int>(src_bytes);
    }

    auto align_up = [](int v, int a) -> int {
        if (a <= 0) {
            return v;
        }
        return (v + a - 1) / a * a;
    };

    int resized_wstride = resized_w;
    int resized_hstride = resized_h;
    if (dst_image->format == IMAGE_FORMAT_RGB888 || dst_image->format == IMAGE_FORMAT_BGR888) {
        resized_wstride = align_up(resized_wstride, 16);
    } else if (dst_image->format == IMAGE_FORMAT_RGBA8888) {
        resized_wstride = align_up(resized_wstride, 8);
    } else if (dst_image->format == IMAGE_FORMAT_YUV420SP_NV12 || dst_image->format == IMAGE_FORMAT_YUV420SP_NV21) {
        resized_wstride = align_up(resized_wstride, 16);
        resized_hstride = align_up(resized_hstride, 2);
    }

    image_buffer_t resized_img = {};
    resized_img.width = resized_w;
    resized_img.height = resized_h;
    resized_img.width_stride = resized_wstride;
    resized_img.height_stride = resized_hstride;
    resized_img.format = dst_image->format;
    size_t resized_bytes = image_size_by_stride(resized_img);
    if (!ensure_dma(tl_resized_dma, resized_bytes, true)) {
        unmap_dst();
        return -1;
    }
    resized_img.fd = tl_resized_dma.fd;
    resized_img.virt_addr = static_cast<unsigned char*>(tl_resized_dma.addr);
    resized_img.size = static_cast<int>(resized_bytes);

    rga_buffer_handle_t src_handle = 0;
    rga_buffer_handle_t dst_handle = 0;
    rga_buffer_t rga_src;
    {
        int wstride = src_for_rga.width_stride > 0 ? src_for_rga.width_stride : src_for_rga.width;
        int hstride = src_for_rga.height_stride > 0 ? src_for_rga.height_stride : src_for_rga.height;
        size_t bytes = src_for_rga.size > 0 ? static_cast<size_t>(src_for_rga.size) : image_size_by_stride(src_for_rga);
        rga_src = wrap_rga_fd_with_optional_import(src_for_rga.fd, src_for_rga.width, src_for_rga.height, src_format, wstride, hstride, bytes, src_handle);
    }

    rga_buffer_t rga_dst;
    {
        int wstride = resized_img.width_stride > 0 ? resized_img.width_stride : resized_img.width;
        int hstride = resized_img.height_stride > 0 ? resized_img.height_stride : resized_img.height;
        size_t bytes = resized_bytes;
        rga_dst = wrap_rga_fd_with_optional_import(resized_img.fd, resized_img.width, resized_img.height, dst_format, wstride, hstride, bytes, dst_handle);
    }

    im_rect src_rect = {0, 0, src_for_rga.width, src_for_rga.height};
    im_rect dst_rect = {0, 0, resized_w, resized_h};
    IM_STATUS check = imcheck(rga_src, rga_dst, src_rect, dst_rect);
    auto release_handles = [&]() {
        if (src_handle != 0) {
            releasebuffer_handle(src_handle);
        }
        if (dst_handle != 0) {
            releasebuffer_handle(dst_handle);
        }
    };
    if (check != IM_STATUS_SUCCESS && check != IM_STATUS_NOERROR) {
        release_handles();
        unmap_dst();
        return -1;
    }
    IM_STATUS status = imresize(rga_src, rga_dst);
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        release_handles();
        unmap_dst();
        return -1;
    }

    int dst_stride = dst_for_cpu.width_stride > 0 ? dst_for_cpu.width_stride : dst_for_cpu.width;
    int dst_hstride = dst_for_cpu.height_stride > 0 ? dst_for_cpu.height_stride : dst_for_cpu.height;
    int src_stride = resized_img.width_stride > 0 ? resized_img.width_stride : resized_img.width;
    int src_hstride = resized_img.height_stride > 0 ? resized_img.height_stride : resized_img.height;

    switch (dst_image->format) {
    case IMAGE_FORMAT_RGB888:
    case IMAGE_FORMAT_BGR888:
        for (int yy = 0; yy < resized_h; ++yy) {
            const unsigned char* src_row = resized_img.virt_addr + static_cast<size_t>(yy) * static_cast<size_t>(src_stride) * 3;
            unsigned char* dst_row = dst_base + static_cast<size_t>(y_pad + yy) * static_cast<size_t>(dst_stride) * 3 + static_cast<size_t>(x_pad) * 3;
            std::memcpy(dst_row, src_row, static_cast<size_t>(resized_w) * 3);
        }
        break;
    case IMAGE_FORMAT_RGBA8888:
        for (int yy = 0; yy < resized_h; ++yy) {
            const unsigned char* src_row = resized_img.virt_addr + static_cast<size_t>(yy) * static_cast<size_t>(src_stride) * 4;
            unsigned char* dst_row = dst_base + static_cast<size_t>(y_pad + yy) * static_cast<size_t>(dst_stride) * 4 + static_cast<size_t>(x_pad) * 4;
            std::memcpy(dst_row, src_row, static_cast<size_t>(resized_w) * 4);
        }
        break;
    case IMAGE_FORMAT_GRAY8:
        for (int yy = 0; yy < resized_h; ++yy) {
            const unsigned char* src_row = resized_img.virt_addr + static_cast<size_t>(yy) * static_cast<size_t>(src_stride);
            unsigned char* dst_row = dst_base + static_cast<size_t>(y_pad + yy) * static_cast<size_t>(dst_stride) + static_cast<size_t>(x_pad);
            std::memcpy(dst_row, src_row, static_cast<size_t>(resized_w));
        }
        break;
    case IMAGE_FORMAT_YUV420SP_NV12:
    case IMAGE_FORMAT_YUV420SP_NV21: {
        unsigned char* y_dst = dst_base;
        unsigned char* uv_dst = dst_base + static_cast<size_t>(dst_stride) * dst_hstride;
        unsigned char* y_src = resized_img.virt_addr;
        unsigned char* uv_src = resized_img.virt_addr + static_cast<size_t>(src_stride) * src_hstride;

        for (int yy = 0; yy < resized_h; ++yy) {
            const unsigned char* src_row = y_src + static_cast<size_t>(yy) * static_cast<size_t>(src_stride);
            unsigned char* dst_row = y_dst + static_cast<size_t>(y_pad + yy) * static_cast<size_t>(dst_stride) + static_cast<size_t>(x_pad);
            std::memcpy(dst_row, src_row, static_cast<size_t>(resized_w));
        }

        int uv_h = resized_h / 2;
        int uv_y_pad = y_pad / 2;
        for (int yy = 0; yy < uv_h; ++yy) {
            const unsigned char* src_row = uv_src + static_cast<size_t>(yy) * static_cast<size_t>(src_stride);
            unsigned char* dst_row = uv_dst + static_cast<size_t>(uv_y_pad + yy) * static_cast<size_t>(dst_stride) + static_cast<size_t>(x_pad);
            std::memcpy(dst_row, src_row, static_cast<size_t>(resized_w));
        }
        break;
    }
    default:
        release_handles();
        unmap_dst();
        return -1;
    }

    release_handles();
    unmap_dst();
    return 0;
}

static std::atomic<bool> g_rga_disabled{false};

void setPreprocessRgaDisabled(bool disable) {
    g_rga_disabled.store(disable);
}

int preprocess_image(image_buffer_t* src_image, image_buffer_t* dst_image, letterbox_t* letterbox, char color)
{
    if (!src_image || !dst_image || !letterbox) {
        return -1;
    }
    int ret = -1;
    if (!g_rga_disabled.load()) {
        ret = rga_resize_then_cpu_letterbox(src_image, dst_image, letterbox, color);
    }
    if (ret == 0) {
        return 0;
    }

    static bool logged_cpu_fallback = false;
    if (!logged_cpu_fallback) {
        std::fprintf(stderr, "[Preprocess] RGA path unavailable, falling back to CPU letterbox\n");
        logged_cpu_fallback = true;
    }
    return convert_image_with_letterbox(src_image, dst_image, letterbox, color);
}

int rga_nv12_to_bgr(const image_buffer_t& src, cv::Mat& out_bgr, double scale)
{
    if (src.format != IMAGE_FORMAT_YUV420SP_NV12 && src.format != IMAGE_FORMAT_YUV420SP_NV21) {
        return -1;
    }
    if ((src.virt_addr == nullptr) && src.fd <= 0) {
        return -1;
    }
    const int src_w = src.width;
    const int src_h = src.height;
    if (src_w <= 0 || src_h <= 0) {
        return -1;
    }
    if (scale <= 0.0) {
        scale = 1.0;
    }
    if (scale > 1.0) {
        scale = 1.0;
    }

    int src_format = RK_FORMAT_UNKNOWN;
    if (!get_rga_format(src.format, src_format)) {
        return -1;
    }

    // 目标 BGR888；scale<1.0 时 RGA 同时完成缩放（Web 预览降采样）
    const int dst_w = std::max(1, static_cast<int>(std::llround(static_cast<double>(src_w) * scale)));
    const int dst_h = std::max(1, static_cast<int>(std::llround(static_cast<double>(src_h) * scale)));
    const int dst_w_aligned = (dst_w + 15) / 16 * 16;
    const size_t dst_bytes = static_cast<size_t>(dst_w_aligned) * dst_h * 3;

    // 复用线程级持久 dst dma-buf：避免每帧 open/ioctl/mmap + munmap/close
    // （CMA 大块分配在负载下可达几十 ms）。输出线程每帧拷贝后才覆盖，无别名风险。
    struct CachedDst {
        int fd = -1;
        void* addr = nullptr;
        size_t bytes = 0;
        ~CachedDst() {
            if (bytes > 0) {
                dma_heap_free(bytes, fd, addr);
            }
        }
    };
    static thread_local CachedDst dst_cache;
    if (dst_cache.bytes < dst_bytes || dst_cache.fd <= 0) {
        if (dst_cache.bytes > 0) {
            dma_heap_free(dst_cache.bytes, dst_cache.fd, dst_cache.addr);
        }
        dst_cache.bytes = 0;
        if (!dma_heap_alloc_dma32(dst_bytes, dst_cache.fd, dst_cache.addr)) {
            return -1;
        }
        dst_cache.bytes = dst_bytes;
    }
    const int dst_fd = dst_cache.fd;
    void* const dst_addr = dst_cache.addr;

    // 源若只有虚拟地址，上传到 dma 缓冲供 RGA 使用
    image_buffer_t src_for_rga = src;
    int src_fd = -1;
    void* src_addr = nullptr;
    if (src.fd <= 0) {
        const size_t src_bytes = image_size_by_stride(src);
        if (src_bytes == 0 || !dma_heap_alloc_any(src_bytes, src_fd, src_addr)) {
            return -1;
        }
        std::memcpy(src_addr, src.virt_addr, src_bytes);
        src_for_rga.fd = src_fd;
        src_for_rga.virt_addr = static_cast<unsigned char*>(src_addr);
        src_for_rga.size = static_cast<int>(src_bytes);
    }
    auto cleanup_src = [&]() {
        if (src_fd > 0 || (src_addr && src_addr != MAP_FAILED)) {
            dma_heap_free(image_size_by_stride(src), src_fd, src_addr);
        }
    };

    const int src_wstride = src.width_stride > 0 ? src.width_stride : src.width;
    const int src_hstride = src.height_stride > 0 ? src.height_stride : src.height;
    const size_t src_bytes = src_for_rga.size > 0 ? static_cast<size_t>(src_for_rga.size)
                                                  : image_size_by_stride(src_for_rga);

    rga_buffer_handle_t src_handle = 0;
    rga_buffer_handle_t dst_handle = 0;
    rga_buffer_t rga_src = wrap_rga_fd_with_optional_import(
        src_for_rga.fd, src_w, src_h, src_format, src_wstride, src_hstride, src_bytes, src_handle);
    rga_buffer_t rga_dst = wrap_rga_fd_with_optional_import(
        dst_fd, dst_w, dst_h, RK_FORMAT_BGR_888, dst_w_aligned, dst_h, dst_bytes, dst_handle);

    im_rect src_rect = {0, 0, src_w, src_h};
    im_rect dst_rect = {0, 0, dst_w, dst_h};
    // imcheck 只做配置校验，对固定尺寸转换每次调用开销不小；首次成功后缓存结果
    static thread_local bool rga_cfg_verified = false;
    IM_STATUS status = IM_STATUS_SUCCESS;
    if (!rga_cfg_verified) {
        status = imcheck(rga_src, rga_dst, src_rect, dst_rect);
        if (status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR) {
            rga_cfg_verified = true;
        }
    }
    if (rga_cfg_verified) {
        // imresize 对同尺寸也有快速路径；imcvtcolor 实测并无收益，统一走 imresize
        status = imresize(rga_src, rga_dst);
    }
    if (src_handle != 0) {
        releasebuffer_handle(src_handle);
    }
    if (dst_handle != 0) {
        releasebuffer_handle(dst_handle);
    }
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        cleanup_src();
        return -1;
    }

    // 拷贝到 cv::Mat（stride 连续时单次 memcpy）
    out_bgr = cv::Mat(dst_h, dst_w, CV_8UC3);
    const unsigned char* base = static_cast<const unsigned char*>(dst_addr);
    if (dst_w_aligned == dst_w) {
        std::memcpy(out_bgr.data, base, static_cast<size_t>(dst_w) * dst_h * 3);
    } else {
        for (int y = 0; y < dst_h; ++y) {
            std::memcpy(out_bgr.ptr(y), base + static_cast<size_t>(y) * dst_w_aligned * 3,
                        static_cast<size_t>(dst_w) * 3);
        }
    }

    cleanup_src();
    return 0;
}

int rga_nv12_crop_resize_rgb(const image_buffer_t& src,
                             int src_x, int src_y, int src_w, int src_h,
                             cv::Mat& out_rgb, int dst_x, int dst_y, int dst_w, int dst_h)
{
    if (src.format != IMAGE_FORMAT_YUV420SP_NV12 && src.format != IMAGE_FORMAT_YUV420SP_NV21) {
        return -1;
    }
    if ((src.virt_addr == nullptr) && src.fd <= 0) {
        return -1;
    }
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return -1;
    }
    if (src_x < 0 || src_y < 0 || src_x + src_w > src.width || src_y + src_h > src.height) {
        return -1;
    }
    if (out_rgb.empty() || out_rgb.type() != CV_8UC3) {
        return -1;
    }
    if (dst_x < 0 || dst_y < 0 || dst_x + dst_w > out_rgb.cols || dst_y + dst_h > out_rgb.rows) {
        return -1;
    }

    int src_format = RK_FORMAT_UNKNOWN;
    if (!get_rga_format(src.format, src_format)) {
        return -1;
    }

    // 源：优先 dma fd；纯虚拟地址时上传到 dma 缓冲（RGA 对裸 virt NV12 不可靠）
    image_buffer_t src_for_rga = src;
    int src_fd = -1;
    void* src_addr = nullptr;
    if (src.fd <= 0) {
        const size_t src_bytes = image_size_by_stride(src);
        if (src_bytes == 0 || !dma_heap_alloc_any(src_bytes, src_fd, src_addr)) {
            return -1;
        }
        std::memcpy(src_addr, src.virt_addr, src_bytes);
        src_for_rga.fd = src_fd;
        src_for_rga.virt_addr = static_cast<unsigned char*>(src_addr);
        src_for_rga.size = static_cast<int>(src_bytes);
    }
    auto cleanup_src = [&]() {
        if (src_fd > 0 || (src_addr && src_addr != MAP_FAILED)) {
            dma_heap_free(image_size_by_stride(src), src_fd, src_addr);
        }
    };

    const int src_wstride = src.width_stride > 0 ? src.width_stride : src.width;
    const int src_hstride = src.height_stride > 0 ? src.height_stride : src.height;
    const size_t src_bytes = src_for_rga.size > 0 ? static_cast<size_t>(src_for_rga.size)
                                                  : image_size_by_stride(src_for_rga);

    // 目标：RGA 输出到 16 字节对齐 stride 的线程级持久 dma-buf，再行拷贝到 out_rgb 的 (dst_x,dst_y)。
    // 复用缓存避免每帧 open/ioctl/mmap（CMA 大块分配在负载下可达几十 ms）。
    const int dst_w_aligned = (dst_w + 15) / 16 * 16;
    const size_t dst_bytes = static_cast<size_t>(dst_w_aligned) * dst_h * 3;
    struct CachedDst {
        int fd = -1;
        void* addr = nullptr;
        size_t bytes = 0;
        ~CachedDst() {
            if (bytes > 0) {
                dma_heap_free(bytes, fd, addr);
            }
        }
    };
    static thread_local CachedDst dst_cache;
    if (dst_cache.bytes < dst_bytes || dst_cache.fd <= 0) {
        if (dst_cache.bytes > 0) {
            dma_heap_free(dst_cache.bytes, dst_cache.fd, dst_cache.addr);
        }
        dst_cache.bytes = 0;
        if (!dma_heap_alloc_dma32(dst_bytes, dst_cache.fd, dst_cache.addr)) {
            cleanup_src();
            return -1;
        }
        dst_cache.bytes = dst_bytes;
    }

    rga_buffer_handle_t src_handle = 0;
    rga_buffer_handle_t dst_handle = 0;
    rga_buffer_t rga_src = wrap_rga_fd_with_optional_import(
        src_for_rga.fd, src.width, src.height, src_format, src_wstride, src_hstride, src_bytes, src_handle);
    rga_buffer_t rga_dst = wrap_rga_fd_with_optional_import(
        dst_cache.fd, dst_w, dst_h, RK_FORMAT_RGB_888, dst_w_aligned, dst_h, dst_bytes, dst_handle);

    im_rect src_rect = {src_x, src_y, src_w, src_h};
    im_rect dst_rect = {0, 0, dst_w, dst_h};
    im_rect prect = {0, 0, 0, 0};
    rga_buffer_t pat = {};

    IM_STATUS status = IM_STATUS_FAILED;
    IM_STATUS check = imcheck(rga_src, rga_dst, src_rect, dst_rect);
    if (check == IM_STATUS_SUCCESS || check == IM_STATUS_NOERROR) {
        status = improcess(rga_src, rga_dst, pat, src_rect, dst_rect, prect, -1, NULL, NULL, IM_SYNC);
    }
    if (src_handle != 0) {
        releasebuffer_handle(src_handle);
    }
    if (dst_handle != 0) {
        releasebuffer_handle(dst_handle);
    }
    if (status != IM_STATUS_SUCCESS && status != IM_STATUS_NOERROR) {
        cleanup_src();
        return -1;
    }

    // 拷贝到 out_rgb 的 (dst_x,dst_y)：对齐 stride → 连续 dst 行
    const unsigned char* base = static_cast<const unsigned char*>(dst_cache.addr);
    for (int y = 0; y < dst_h; ++y) {
        std::memcpy(out_rgb.ptr(dst_y + y) + static_cast<size_t>(dst_x) * 3,
                    base + static_cast<size_t>(y) * dst_w_aligned * 3,
                    static_cast<size_t>(dst_w) * 3);
    }

    cleanup_src();
    return 0;
}
