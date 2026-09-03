#ifndef RKNN_CONTEXT_H
#define RKNN_CONTEXT_H

#include "rknn_api.h"
#include "common.h"

typedef struct {
    rknn_context rknn_ctx;
    rknn_context ctx;
    rknn_sdk_version rknn_version;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
    int class_num;
    // yolo26 detect 的 cls 输出语义：-1=logits（需 sigmoid），1=已 sigmoid（值域[0,1]），0=未判定。
    // 方案2（cls 入图 sigmoid）让 INT8 量化 cls 值域收窄到 [0,1] 保住真实分数；板端后处理按此跳过二次 sigmoid。
    int cls_is_sigmoided;
    rknn_tensor_mem* input_mem;
    uint32_t input_mem_size;
    bool input_io_mem_bound;
    bool input_mem_synced;
    // 输出零拷贝（仅量化模型启用）：输出 tensor 预绑定内存，省去每帧 rknn_outputs_get 拷贝
    rknn_tensor_mem** output_mem;
    uint32_t* output_mem_size;
    bool* output_io_mem_bound;
} rknn_app_context_t;

#define RKNN_APP_CONTEXT_T_DEFINED

#endif
