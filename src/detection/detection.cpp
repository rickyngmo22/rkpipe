#include "detection/detection.h"
#include "core/performance.h"
#include "core/thread_local_memory_pool.h"
#include "model/yolov8.h"
#include "utils/common.h"
#include "utils/utils.h"
#include "preprocess.h"



Detection::Detection() {
    rknn_app_ctx = nullptr;
    logger = nullptr;
    performance = new PerformanceMetrics();
    modelInitialized = false;
}

Detection::~Detection() {
    delete performance;
}

bool Detection::runDetection(image_buffer_t *processedImage, letterbox_t *letter_box, object_detect_result_list &results, rknn_app_context_t *rknnAppCtx) {
    // Start total time measurement
    performance->startTotal();

    // Check basic conditions
    if (!processedImage || !letter_box) {
        printf("Processed image or letterbox is null\n");
        performance->stopTotal();
        return false;
    }

    // Check if rknn context is valid
    if (!rknnAppCtx) {
        printf("RKNN app context is null\n");
        performance->stopTotal();
        return false;
    }

    // Start inference time measurement
    performance->startInference();

    // Run detection
    object_detect_result_list rawResults;
    int ret = -1;
    try {
        ret = inference_yolov8_model(rknnAppCtx, processedImage, letter_box, &rawResults, BOX_THRESH, NMS_THRESH);
    } catch (const std::exception& e) {
        printf("Inference failed: %s\n", e.what());
        performance->stopInference();
        return false;
    }

    // Stop inference time measurement
    performance->stopInference();

    if (ret != 0) {
        printf("Inference returned error code: %d\n", ret);
        performance->stopTotal();
        return false;
    }

    // Start postprocessing time measurement
    performance->startPostprocessing();

    // Copy results
    memcpy(&results, &rawResults, sizeof(object_detect_result_list));

    // Stop postprocessing time measurement
    performance->stopPostprocessing();

    // Stop total time measurement
    performance->stopTotal();

    // Add measurement to history
    performance->addMeasurement();

    return true;
}

bool Detection::runPoseDetection(image_buffer_t *processedImage, letterbox_t *letter_box, pose_detect_result_list &results, rknn_app_context_t *rknnAppCtx) {
    // Start total time measurement
    performance->startTotal();

    // Check basic conditions
    if (!processedImage || !letter_box) {
        printf("Processed image or letterbox is null\n");
        performance->stopTotal();
        return false;
    }

    // Check if rknn context is valid
    if (!rknnAppCtx) {
        printf("RKNN app context is null\n");
        performance->stopTotal();
        return false;
    }

    // Start inference time measurement
    performance->startInference();

    // Run Pose detection
    pose_detect_result_list rawResults;
    int ret = -1;
    try {
        ret = inference_yolov8_pose_model(rknnAppCtx, processedImage, letter_box, &rawResults, BOX_THRESH, NMS_THRESH);
    } catch (const std::exception& e) {
        printf("Pose inference failed: %s\n", e.what());
        performance->stopInference();
        return false;
    }

    // Stop inference time measurement
    performance->stopInference();

    if (ret != 0) {
        printf("Pose inference returned error code: %d\n", ret);
        performance->stopTotal();
        return false;
    }

    // Start postprocessing time measurement
    performance->startPostprocessing();

    // Copy results
    memcpy(&results, &rawResults, sizeof(pose_detect_result_list));

    // Stop postprocessing time measurement
    performance->stopPostprocessing();

    // Stop total time measurement
    performance->stopTotal();

    // Add measurement to history
    performance->addMeasurement();

    return true;
}

bool Detection::preprocess(const cv::Mat &image, image_buffer_t &processedImage, letterbox_t &letter_box, int model_width, int model_height) {
    // Start preprocessing time measurement
    performance->startPreprocessing();

    // Check if image is empty
    if (image.empty()) {
        if (logger) {
#if USE_QT
            logger->logError("Image empty");
#else
            printf("Image empty\n");
#endif
        } else {
            printf("Image empty\n");
        }
        performance->stopPreprocessing();
        return false;
    }

    cv::Mat rgb;
    if (image.channels() == 3) {
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, rgb, cv::COLOR_BGRA2RGB);
    } else if (image.channels() == 1) {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    } else {
        printf("Unsupported image channels: %d\n", image.channels());
        performance->stopPreprocessing();
        return false;
    }
    if (rgb.empty()) {
        printf("RGB conversion failed\n");
        performance->stopPreprocessing();
        return false;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    src_image.width = rgb.cols;
    src_image.height = rgb.rows;
    src_image.width_stride = rgb.cols;
    src_image.height_stride = rgb.rows;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.virt_addr = rgb.data;
    src_image.size = rgb.cols * rgb.rows * 3;
    src_image.fd = 0;
    src_image.priv_data = nullptr;

    // 2. Initialize processedImage (dst_buffer)
    memset(&processedImage, 0, sizeof(image_buffer_t));
    processedImage.format = IMAGE_FORMAT_RGB888; // Model expects RGB
    processedImage.width = model_width;
    processedImage.height = model_height;
    processedImage.width_stride = model_width;
    processedImage.height_stride = model_height;
    processedImage.size = model_width * model_height * 3;
    processedImage.priv_data = nullptr;

    if (rknn_app_ctx && rknn_app_ctx->input_mem) {
        rknn_app_ctx->input_mem_synced = false;
        uint32_t w_stride = rknn_app_ctx->input_attrs ? rknn_app_ctx->input_attrs[0].w_stride : 0;
        uint32_t h_stride = rknn_app_ctx->input_attrs ? rknn_app_ctx->input_attrs[0].h_stride : 0;
        uint32_t size_with_stride = rknn_app_ctx->input_attrs ? rknn_app_ctx->input_attrs[0].size_with_stride : 0;
        if (w_stride > 0) {
            processedImage.width_stride = w_stride;
        }
        if (h_stride > 0) {
            processedImage.height_stride = h_stride;
        } else if (size_with_stride > 0 && processedImage.width_stride > 0 && rknn_app_ctx->model_channel > 0) {
            uint32_t calc_h_stride = size_with_stride / (processedImage.width_stride * rknn_app_ctx->model_channel);
            if (calc_h_stride >= static_cast<uint32_t>(processedImage.height)) {
                processedImage.height_stride = calc_h_stride;
            }
        }
        processedImage.virt_addr = static_cast<unsigned char*>(rknn_app_ctx->input_mem->virt_addr);
        processedImage.size = rknn_app_ctx->input_mem_size;
        processedImage.fd = rknn_app_ctx->input_mem->fd;
        processedImage.priv_data = rknn_app_ctx->input_mem;
    } else {
        processedImage.virt_addr = (unsigned char *)ThreadLocalMemoryManager::allocate(processedImage.size);
    }

    if (!processedImage.virt_addr) {
        processedImage.virt_addr = (unsigned char *)malloc(processedImage.size);
    }

    if (processedImage.virt_addr == NULL) {
        printf("Memory allocation failed for processed image\n");
        performance->stopPreprocessing();
        return false;
    }

    // 3. Convert using RGA (or CPU fallback)
    // Initialize letterbox structure
    memset(&letter_box, 0, sizeof(letterbox_t));

    int ret = preprocess_image(&src_image, &processedImage, &letter_box, 114);
    if (ret != 0) {
        printf("Image convert failed, ret=%d\n", ret);
        performance->stopPreprocessing();
        return false;
    }

    if (rknn_app_ctx && rknn_app_ctx->input_mem && processedImage.priv_data == rknn_app_ctx->input_mem) {
        rknn_mem_sync(rknn_app_ctx->rknn_ctx, rknn_app_ctx->input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_app_ctx->input_mem_synced = true;
    }

    // Stop preprocessing time measurement
    performance->stopPreprocessing();

    return true;
}

bool Detection::preprocess(const image_buffer_t &src_image, image_buffer_t &dst_image, letterbox_t &letter_box, int model_width, int model_height) {
    // Start preprocessing time measurement
    performance->startPreprocessing();

    // Check if source image is valid
    if (src_image.width == 0 || src_image.height == 0) {
        if (logger) {
            printf("Source image empty\n");
        } else {
            printf("Source image empty\n");
        }
        performance->stopPreprocessing();
        return false;
    }

    // 2. Initialize processedImage (dst_buffer)
    memset(&dst_image, 0, sizeof(image_buffer_t));
    dst_image.format = IMAGE_FORMAT_RGB888; // Model expects RGB
    dst_image.width = model_width;
    dst_image.height = model_height;
    dst_image.width_stride = model_width;
    dst_image.height_stride = model_height;
    dst_image.size = model_width * model_height * 3;
    dst_image.priv_data = nullptr;

    // 零拷贝直写 input_mem 要求 RGA 输出（UINT8 RGB888）与 NPU 输入张量类型一致：
    // 仅 UINT8/INT8 输入模型可直写；FP16 输入模型写 UINT8 字节会被 NPU 按 FP16
    // 解释成垃圾（应走下方普通缓冲，由 rknn_inputs_set 做类型转换）
    bool input_mem_direct = false;
    if (rknn_app_ctx && rknn_app_ctx->input_mem && rknn_app_ctx->input_attrs) {
        const rknn_tensor_type in_type = rknn_app_ctx->input_attrs[0].type;
        input_mem_direct =
            (in_type == RKNN_TENSOR_UINT8 || in_type == RKNN_TENSOR_INT8);
    }
    if (const char* dbg = std::getenv("RK_PIPE_DEBUG_FACE")) {
        if (dbg[0] != '\0' && std::strcmp(dbg, "0") != 0) {
            std::fprintf(stderr, "[rk_pipe][face][pre] input_mem_direct=%d type=%d mem=%p\n",
                         input_mem_direct ? 1 : 0,
                         rknn_app_ctx && rknn_app_ctx->input_attrs
                             ? static_cast<int>(rknn_app_ctx->input_attrs[0].type)
                             : -1,
                         rknn_app_ctx ? static_cast<void*>(rknn_app_ctx->input_mem) : nullptr);
        }
    }
    if (input_mem_direct) {
        rknn_app_ctx->input_mem_synced = false;
        uint32_t w_stride = rknn_app_ctx->input_attrs ? rknn_app_ctx->input_attrs[0].w_stride : 0;
        uint32_t h_stride = rknn_app_ctx->input_attrs ? rknn_app_ctx->input_attrs[0].h_stride : 0;
        uint32_t size_with_stride = rknn_app_ctx->input_attrs ? rknn_app_ctx->input_attrs[0].size_with_stride : 0;
        if (w_stride > 0) {
            dst_image.width_stride = w_stride;
        }
        if (h_stride > 0) {
            dst_image.height_stride = h_stride;
        } else if (size_with_stride > 0 && dst_image.width_stride > 0 && rknn_app_ctx->model_channel > 0) {
            uint32_t calc_h_stride = size_with_stride / (dst_image.width_stride * rknn_app_ctx->model_channel);
            if (calc_h_stride >= static_cast<uint32_t>(dst_image.height)) {
                dst_image.height_stride = calc_h_stride;
            }
        }
        dst_image.virt_addr = static_cast<unsigned char*>(rknn_app_ctx->input_mem->virt_addr);
        dst_image.size = rknn_app_ctx->input_mem_size;
        dst_image.fd = rknn_app_ctx->input_mem->fd;
        dst_image.priv_data = rknn_app_ctx->input_mem;
    } else {
        dst_image.virt_addr = (unsigned char *)ThreadLocalMemoryManager::allocate(dst_image.size);
    }

    if (!dst_image.virt_addr) {
        dst_image.virt_addr = (unsigned char *)malloc(dst_image.size);
    }

    if (dst_image.virt_addr == NULL) {
        printf("Memory allocation failed for processed image\n");
        performance->stopPreprocessing();
        return false;
    }

    // 3. Convert using RGA (or CPU fallback)
    // Initialize letterbox structure
    memset(&letter_box, 0, sizeof(letterbox_t));

    image_buffer_t* src_ptr = const_cast<image_buffer_t*>(&src_image);

    int ret = preprocess_image(src_ptr, &dst_image, &letter_box, 114);
    if (ret != 0) {
        printf("Image convert failed, ret=%d\n", ret);
        performance->stopPreprocessing();
        return false;
    }

    if (rknn_app_ctx && rknn_app_ctx->input_mem && dst_image.priv_data == rknn_app_ctx->input_mem) {
        rknn_mem_sync(rknn_app_ctx->rknn_ctx, rknn_app_ctx->input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
        rknn_app_ctx->input_mem_synced = true;
    }

    // Stop preprocessing time measurement
    performance->stopPreprocessing();

    return true;
}

void Detection::setModelContext(rknn_app_context_t *ctx) {
    rknn_app_ctx = ctx;
}

void Detection::setLogger(Logger *log) {
    logger = log;
}

void Detection::setModelInitialized(bool initialized) {
    modelInitialized = initialized;
}

PerformanceMetrics *Detection::getPerformanceMetrics() {
    return performance;
}

cv::Scalar Detection::getDefectColor(int cls_id) {
    // Return different colors for different object classes
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),   // Red
        cv::Scalar(0, 255, 0),   // Green
        cv::Scalar(255, 0, 0),   // Blue
        cv::Scalar(255, 255, 0), // Yellow
        cv::Scalar(0, 255, 255), // Cyan
        cv::Scalar(255, 0, 255), // Magenta
        cv::Scalar(128, 0, 0),   // Maroon
        cv::Scalar(128, 128, 0), // Olive
        cv::Scalar(0, 128, 0),   // Dark Green
        cv::Scalar(128, 0, 128), // Purple
        cv::Scalar(0, 128, 128), // Teal
        cv::Scalar(0, 0, 128)    // Navy
    };

    // Use modulo to cycle through colors if class ID exceeds color count
    int colorIndex = cls_id % colors.size();
    return colors[colorIndex];
}

void Detection::printPerformanceStats() {
    int measurement_count = performance->getTotalMeasurements();
    if (measurement_count == 0) {
        printf("No performance measurements available.\n");
        return;
    }

    double avg_preprocess = performance->getAveragePreprocessingTime();
    double avg_inference = performance->getAverageInferenceTime();
    double avg_postprocess = 0;
    double avg_total = 0;

    // 计算postprocess和total的平均值
    auto& metrics = performance->getMetricsHistory();
    if (!metrics.empty()) {
        double total_post = 0, total_time = 0;
        for (const auto& m : metrics) {
            total_post += m.postprocessingTime;
            total_time += m.totalTime;
        }
        avg_postprocess = total_post / metrics.size();
        avg_total = total_time / metrics.size();
    }

    if (avg_total <= 0) {
        avg_total = avg_preprocess + avg_inference + avg_postprocess;
    }

    printf("=== Performance Statistics (%d frames) ===\n", measurement_count);
    printf("  Preprocessing:  %.2f ms (%.1f%%)\n", avg_preprocess, avg_preprocess / avg_total * 100);
    printf("  Inference:      %.2f ms (%.1f%%)\n", avg_inference, avg_inference / avg_total * 100);
    printf("  Postprocessing: %.2f ms (%.1f%%)\n", avg_postprocess, avg_postprocess / avg_total * 100);
    printf("  ----------------------------------------\n");
    printf("  TOTAL:          %.2f ms\n", avg_total);
    printf("  FPS:            %.1f\n", 1000.0 / avg_total);
}
