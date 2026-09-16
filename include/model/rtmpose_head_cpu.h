#ifndef _RKNN_DEMO_RTMPOSE_HEAD_CPU_H_
#define _RKNN_DEMO_RTMPOSE_HEAD_CPU_H_

#include <stdint.h>
#include <string>
#include <vector>

// RTMPose SimCC head 的 CPU 复刻（替代有问题的 RKNN head 转换）。
//
// 背景：rtmpose head 子图含 MatMul + RMSNorm(ReduceL2) + 高斯注意力 + SimCC 投影，
// rknn-toolkit2 2.3.2 FP16 转换后大部分关键点通道输出 0（转换损坏）。
// 本实现从 head ONNX 逐算子复刻，权重由 export_head_weights.py 导出为二进制文件，
// 已在板端用 onnxruntime 对拍验证 cos_sim=1.0、17/17 关键点位置一致。
//
// 权重目录（export_head_weights.py 产物）：
//   onnx__MatMul_684.f32     [48,256]     mlp
//   onnx__MatMul_687.f32     [256,1152]   uv
//   onnx__MatMul_711.f32     [512,256]    o
//   onnx__MatMul_712.f32     [256,384]    cls_x
//   onnx__MatMul_713.f32     [256,512]    cls_y
//   model.head.gau.res_scale.scale.f32     [256]
//   _head_gau_Expand.f32 / _head_gau_Expand_1.f32   [1,17,2,128] 位置编码
//   以及各标量（1/sqrt(D)、g、sqrt128 等）
//   ref_feat.f32 / ref_simcc_x.f32 / ref_simcc_y.f32  对拍参考（运行时不需要）

typedef struct {
    std::vector<float> w_mlp;    // [48,256]
    std::vector<float> w_uv;     // [256,1152]
    std::vector<float> w_o;      // [512,256]
    std::vector<float> w_clsx;   // [256,384]
    std::vector<float> w_clsy;   // [256,512]
    std::vector<float> res_scale;  // [256]
    std::vector<float> pos_coeff;  // [17,2,128]
    std::vector<float> pos_bias;   // [17,2,128]
    float inv_sqrt_48 = 0.0f;   // 1/sqrt(48)
    float inv_sqrt_256 = 0.0f;  // 1/sqrt(256)
    float sqrt_128 = 0.0f;      // sqrt(128)
    float g_mlp = 0.0f;
    float g_gau = 0.0f;
    bool loaded = false;
    // 预分配计算工作区（init 时一次性分配，避免每帧 400KB malloc/free 抖动）；
    // mutable：compute 侧（const head）也允许写入这块临时缓冲
    mutable std::vector<float> ws;
} rtmpose_head_cpu_t;

// 从权重目录加载（export_head_weights.py 输出目录）
int init_rtmpose_head_cpu(rtmpose_head_cpu_t* head, const std::string& weight_dir);

void release_rtmpose_head_cpu(rtmpose_head_cpu_t* head);

// 计算 head：输入 feat（NCHW [C,H,W]，C=17,H=8,W=6 逻辑布局，float32）
// 输出 simcc_x [C, 2*W]、simcc_y [C, 2*H]（float32 logits）
int compute_rtmpose_head_cpu(const rtmpose_head_cpu_t* head, const float* feat,
                             int c, int h, int w,
                             float* simcc_x, int xbins,
                             float* simcc_y, int ybins);

#endif  // _RKNN_DEMO_RTMPOSE_HEAD_CPU_H_
