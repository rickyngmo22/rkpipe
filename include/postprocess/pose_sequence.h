#pragma once

// M13 动作识别（pose→ST-GCN 时序级联）：按 track_id 键控的时序骨架窗口缓冲。
// 与 M0/M6 的"帧"级联不同，本组件按"轨迹 × 时间窗"级联：pose 主任务输出的
// 跟踪结果（results[i].track_id 已由 trackPoseResults 填充）逐帧喂入，攒满 T 帧后
// 组装 ST-GCN 输入张量 [C=3, T, V]（x / y / visibility 三通道）。
//
// 时间轴语义（实现定稿，对应 docs/action_rec.md §2）：
//   - 窗口按全局流水线帧序（update 的 frame_index）推进，轨迹诞生帧为窗口起点；
//   - 轨迹本帧缺席 → 插入全零槽（visibility=0，ST-GCN 标准 zero-padding，不插值）；
//   - 中断 gap >= window_t → 旧窗作废清空重来（全是零槽的窗口无动作证据）；
//   - 轨迹连续 stale_frames 帧未更新 → 整条老化清除（防 track_id 复用后残留窗口污染）；
//   - 归一化口径（norm_mode，须与训练侧一致，见 docs/action_rec.md §2-3）：
//       box   （默认）逐帧用该帧 box：x_n=(kx-cx)/scale, scale=max(w,h)，自训模型用；
//       image 原帧分辨率：x_n=(kx-w/2)/(w/2)（mmaction2 PreNormalize2D 口径，
//             NTU60 coco17 官方权重即此口径）；w/h 缺失时沿用最近一次有效值。
//
// 纯逻辑、无硬件依赖（rknn 维度向量只是 uint32 数组），CI 单测可覆盖。

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "postprocess/postprocess.h"  // pose_detect_result_list / KEYPOINT_NUM

// 组装出的 ST-GCN 输入张量：[C, T, V] float 行主序，偏移 = c*T*V + t*V + v
struct PoseWindowTensor {
    int C = 3;
    int T = 0;
    int V = 0;
    std::vector<float> data;
    // rknn 固定形状输入维度向量（NCHW 语义），action_rec 推理直接使用
    std::vector<uint32_t> rknn_dims;
};

class PoseSequenceBuffer {
public:
    // window_t: 窗口帧数（AppConfig action_window_t，默认 30；须与模型输入 T 一致）
    // num_joints: 关键点数（COCO 17）
    // stale_frames: 轨迹老化帧数，-1 = 取 window_t（一个窗口周期未出现即清）
    // norm_mode: "box"（默认，逐帧 box 中心/尺度）| "image"（原帧分辨率，PreNormalize2D 口径）
    explicit PoseSequenceBuffer(int window_t = 30, int num_joints = KEYPOINT_NUM,
                                int stale_frames = -1,
                                const std::string& norm_mode = "box");

    // 每帧调用（跟踪后）：对 list 中 track_id>0 的结果入缓冲（同帧重复取第一条），
    // 其余已跟踪轨迹补零槽，随后做 stale 老化清除。
    // frame_width/height：image 模式的归一化基准（原帧尺寸）；box 模式忽略。
    void update(const pose_detect_result_list& poses, long long frame_index,
                int frame_width = 0, int frame_height = 0);

    // 该轨迹窗口是否已攒满（可组装/推理）
    bool ready(int track_id) const;

    // 组装该轨迹的归一化窗口张量；未攒满返回 false
    bool buildWindow(int track_id, PoseWindowTensor* out) const;

    // 推理节流（action_interval：每窗口仅推一次）：推理完成后清空该轨迹重新攒窗
    void resetTrack(int track_id);
    void reset();

    int windowT() const { return window_t_; }
    int numJoints() const { return num_joints_; }
    size_t trackCount() const { return tracks_.size(); }

private:
    struct Slot {
        long long frame_index = 0;
        bool valid = false;          // false = 缺帧补零槽
        int frame_w = 0;             // 入槽时原帧尺寸（image 归一化口径基准）
        int frame_h = 0;
        float box[4] = {0, 0, 0, 0}; // left, top, right, bottom（归一化基准）
        std::vector<float> kpt;      // [V*3] x, y, conf
    };
    struct TrackBuf {
        std::deque<Slot> slots;
        long long last_frame = -1;  // 最后一条槽的帧号（含零槽，算 gap/防重复插零用）
        long long last_seen = -1;   // 最后一次真实检测的帧号（stale 老化判定用）
    };

    Slot makeSlot(const pose_detect_result& r, long long frame_index, int frame_w,
                  int frame_h) const;
    Slot makeZeroSlot(long long frame_index) const;

    int window_t_;
    int num_joints_;
    int stale_frames_;
    bool norm_image_ = false;        // true = image 模式（PreNormalize2D 口径）
    int last_frame_w_ = 0;           // image 模式最近一次有效帧尺寸（缺失时沿用）
    int last_frame_h_ = 0;
    std::unordered_map<int, TrackBuf> tracks_;
};
