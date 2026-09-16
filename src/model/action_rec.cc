#include "model/action_rec.h"

#include <cmath>
#include <cstring>

#include "core/rknn_model.h"
#include "postprocess/postprocess_common.h"

namespace {

float readValue(const rknn_output& output, const rknn_tensor_attr& attr, size_t index) {
    switch (attr.type) {
        case RKNN_TENSOR_FLOAT32:
            return static_cast<const float*>(output.buf)[index];
        case RKNN_TENSOR_FLOAT16:
            return fp16_to_float(static_cast<const uint16_t*>(output.buf)[index]);
        case RKNN_TENSOR_INT8:
            return deqnt_affine_to_f32(static_cast<const int8_t*>(output.buf)[index], attr.zp, attr.scale);
        case RKNN_TENSOR_UINT8:
            return (static_cast<float>(static_cast<const uint8_t*>(output.buf)[index]) -
                    static_cast<float>(attr.zp)) *
                   attr.scale;
        default:
            return 0.0f;
    }
}

}  // namespace

int init_action_rec_model(const char* model_path, rknn_app_context_t* app_ctx) {
    return init_rknn_model(model_path, app_ctx);
}

int release_action_rec_model(rknn_app_context_t* app_ctx) {
    return release_rknn_model(app_ctx);
}

int inference_action_rec_model(rknn_app_context_t* app_ctx,
                               const float* window,
                               const std::vector<uint32_t>& dims,
                               ClsTop1* out) {
    if (!app_ctx || !app_ctx->rknn_ctx || !window || dims.size() < 2 || !out) {
        return -1;
    }
    *out = ClsTop1{};

    size_t elems = 1;
    for (uint32_t d : dims) {
        elems *= d;
    }
    if (elems == 0) {
        return -1;
    }

    // 骨架序列非图像：归一化已在组装侧完成，FLOAT32 按模型原生布局（NCHW）直接写入
    rknn_input rin{};
    rin.index = 0;
    rin.type = RKNN_TENSOR_FLOAT32;
    rin.fmt = RKNN_TENSOR_NCHW;
    rin.size = static_cast<uint32_t>(elems * sizeof(float));
    rin.buf = const_cast<float*>(window);
    if (rknn_inputs_set(app_ctx->rknn_ctx, 1, &rin) < 0) {
        return -1;
    }
    if (rknn_run(app_ctx->rknn_ctx, nullptr) < 0) {
        return -1;
    }

    rknn_output rout{};
    rout.index = 0;
    rout.want_float = 0;  // 自行按 attr 反量化（fp16/int8/u8/f32 全覆盖）
    if (rknn_outputs_get(app_ctx->rknn_ctx, 1, &rout, nullptr) < 0) {
        return -1;
    }

    const rknn_tensor_attr& attr = app_ctx->output_attrs[0];
    if (attr.n_dims < 1) {
        rknn_outputs_release(app_ctx->rknn_ctx, 1, &rout);
        return -1;
    }
    const int num_actions = static_cast<int>(attr.dims[attr.n_dims - 1]);
    const size_t type_size = (attr.type == RKNN_TENSOR_FLOAT32) ? 4
                             : (attr.type == RKNN_TENSOR_FLOAT16) ? 2
                             : 1;
    const size_t out_elems = rout.size / type_size;
    if (num_actions <= 0 || out_elems < static_cast<size_t>(num_actions)) {
        rknn_outputs_release(app_ctx->rknn_ctx, 1, &rout);
        return -1;
    }

    // batch=1：取末 num_actions 个元素
    std::vector<float> values(num_actions);
    const size_t base = out_elems - static_cast<size_t>(num_actions);
    for (int c = 0; c < num_actions; ++c) {
        values[c] = readValue(rout, attr, base + c);
    }
    rknn_outputs_release(app_ctx->rknn_ctx, 1, &rout);

    // 已 softmax（行和≈1）→ 分数取原值；raw logits → softmax
    double sum = 0.0;
    for (int c = 0; c < num_actions; ++c) {
        sum += values[c];
    }
    *out = clsTop1(values.data(), num_actions, std::fabs(sum - 1.0) < 0.05);
    return 0;
}
