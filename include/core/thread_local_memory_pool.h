#ifndef THREAD_LOCAL_MEMORY_POOL_H
#define THREAD_LOCAL_MEMORY_POOL_H

#include <cstddef>
#include <cstdlib>
#include <memory>

// 线程本地内存池对外契约：仅暴露分配入口与 RAII 缓冲包装。
// 池实现（块管理、锁定策略、容量启发式）位于内部实现文件，不通过本头文件暴露。

class ThreadLocalMemoryPool;  // 前向声明：实现细节在内部编译单元

// 全局线程本地存储的内存池管理器（纯静态接口）
class ThreadLocalMemoryManager {
private:
    ThreadLocalMemoryManager() = delete;  // 纯静态类，禁止实例化
    static thread_local std::unique_ptr<ThreadLocalMemoryPool> local_pool;

public:
    static ThreadLocalMemoryPool& getPool();
    static void configure(size_t block_size, size_t max_blocks);
    static void configureForResolution(int width, int height, int threadCount);
    static void* allocate(size_t size);
    static void deallocate(void* ptr);
    static void printThreadStats();
    static void resetPool(); // 手动重置线程本地内存池，释放所有内存
};

// RAII包装器用于自动内存管理
class OptimizedImageBuffer {
private:
    void* buffer_;
    size_t size_;
    bool use_local_pool_;

public:
    OptimizedImageBuffer(size_t size, bool use_local_pool = true)
        : size_(size), use_local_pool_(use_local_pool) {

        if (use_local_pool_) {
            buffer_ = ThreadLocalMemoryManager::allocate(size_);
        } else {
            buffer_ = malloc(size_);
        }

        if (!buffer_) {
            throw std::bad_alloc();
        }
    }

    ~OptimizedImageBuffer() {
        if (buffer_) {
            if (use_local_pool_) {
                ThreadLocalMemoryManager::deallocate(buffer_);
            } else {
                free(buffer_);
            }
        }
    }

    // 禁止拷贝，允许移动
    OptimizedImageBuffer(const OptimizedImageBuffer&) = delete;
    OptimizedImageBuffer& operator=(const OptimizedImageBuffer&) = delete;

    OptimizedImageBuffer(OptimizedImageBuffer&& other) noexcept
        : buffer_(other.buffer_), size_(other.size_), use_local_pool_(other.use_local_pool_) {
        other.buffer_ = nullptr;
    }

    OptimizedImageBuffer& operator=(OptimizedImageBuffer&& other) noexcept {
        if (this != &other) {
            if (buffer_) {
                if (use_local_pool_) {
                    ThreadLocalMemoryManager::deallocate(buffer_);
                } else {
                    free(buffer_);
                }
            }

            buffer_ = other.buffer_;
            size_ = other.size_;
            use_local_pool_ = other.use_local_pool_;
            other.buffer_ = nullptr;
        }
        return *this;
    }

    void* get() const { return buffer_; }
    size_t size() const { return size_; }
};

#endif // THREAD_LOCAL_MEMORY_POOL_H
