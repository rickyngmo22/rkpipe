#include "postprocess/pose_sequence.h"

#include <algorithm>

PoseSequenceBuffer::PoseSequenceBuffer(int window_t, int num_joints, int stale_frames,
                                       const std::string& norm_mode)
    : window_t_(window_t > 0 ? window_t : 30),
      num_joints_(num_joints > 0 ? num_joints : KEYPOINT_NUM),
      stale_frames_(stale_frames > 0 ? stale_frames : window_t_),
      norm_image_(norm_mode == "image") {}

PoseSequenceBuffer::Slot PoseSequenceBuffer::makeSlot(const pose_detect_result& r,
                                                      long long frame_index, int frame_w,
                                                      int frame_h) const {
    Slot s;
    s.frame_index = frame_index;
    s.valid = true;
    s.frame_w = frame_w;
    s.frame_h = frame_h;
    s.box[0] = static_cast<float>(r.box.left);
    s.box[1] = static_cast<float>(r.box.top);
    s.box[2] = static_cast<float>(r.box.right);
    s.box[3] = static_cast<float>(r.box.bottom);
    s.kpt.resize(static_cast<size_t>(num_joints_) * 3);
    for (int v = 0; v < num_joints_; ++v) {
        const pose_keypoint& kp = r.keypoints[v];
        s.kpt[v * 3 + 0] = kp.x;
        s.kpt[v * 3 + 1] = kp.y;
        s.kpt[v * 3 + 2] = kp.conf;
    }
    return s;
}

PoseSequenceBuffer::Slot PoseSequenceBuffer::makeZeroSlot(long long frame_index) const {
    Slot s;
    s.frame_index = frame_index;
    s.valid = false;
    s.kpt.assign(static_cast<size_t>(num_joints_) * 3, 0.0f);
    return s;
}

void PoseSequenceBuffer::update(const pose_detect_result_list& poses, long long frame_index,
                                int frame_width, int frame_height) {
    // image 模式：记录最近一次有效帧尺寸（调用方未传时沿用，缺省首帧前 box 口径兜底）
    if (norm_image_ && frame_width > 0 && frame_height > 0) {
        last_frame_w_ = frame_width;
        last_frame_h_ = frame_height;
    }
    // stale 老化：连续 stale_frames_ 帧无真实检测的轨迹整条清除
    // （零槽推进不续命；防消失轨迹占用内存、防 track_id 复用后残留窗口污染）
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (it->second.last_seen >= 0 && frame_index - it->second.last_seen > stale_frames_) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }

    std::unordered_map<int, bool> seen_this_frame;
    for (int i = 0; i < poses.count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        const pose_detect_result& r = poses.results[i];
        if (r.track_id <= 0) {
            continue;  // 0=未跟踪，不入时序缓冲
        }
        if (!seen_this_frame.emplace(r.track_id, true).second) {
            continue;  // 同帧同轨迹重复结果取第一条
        }
        TrackBuf& buf = tracks_[r.track_id];
        if (buf.last_frame >= 0 && frame_index > buf.last_frame + 1) {
            const long long gap = frame_index - buf.last_frame - 1;
            if (gap >= window_t_) {
                buf.slots.clear();  // 旧窗作废：全零窗无动作证据，重攒
            } else {
                for (long long g = 0; g < gap; ++g) {
                    buf.slots.push_back(makeZeroSlot(buf.last_frame + 1 + g));
                }
            }
        }
        buf.slots.push_back(makeSlot(r, frame_index, last_frame_w_, last_frame_h_));
        buf.last_frame = frame_index;
        buf.last_seen = frame_index;
        while (static_cast<int>(buf.slots.size()) > window_t_) {
            buf.slots.pop_front();
        }
    }

    // 本帧缺席的已跟踪轨迹补零槽（保持时间轴连续，visibility=0 掩码）
    for (auto& kv : tracks_) {
        if (seen_this_frame.count(kv.first)) {
            continue;
        }
        TrackBuf& buf = kv.second;
        if (buf.last_frame < 0) {
            continue;
        }
        buf.slots.push_back(makeZeroSlot(frame_index));
        buf.last_frame = frame_index;
        while (static_cast<int>(buf.slots.size()) > window_t_) {
            buf.slots.pop_front();
        }
    }
}

bool PoseSequenceBuffer::ready(int track_id) const {
    auto it = tracks_.find(track_id);
    return it != tracks_.end() && static_cast<int>(it->second.slots.size()) >= window_t_;
}

bool PoseSequenceBuffer::buildWindow(int track_id, PoseWindowTensor* out) const {
    if (!out) {
        return false;
    }
    auto it = tracks_.find(track_id);
    if (it == tracks_.end() || static_cast<int>(it->second.slots.size()) < window_t_) {
        return false;
    }
    const TrackBuf& buf = it->second;
    const int T = window_t_;
    const int V = num_joints_;
    const size_t plane = static_cast<size_t>(T) * V;
    out->C = 3;
    out->T = T;
    out->V = V;
    out->data.assign(plane * 3, 0.0f);
    out->rknn_dims = {1, 3, static_cast<uint32_t>(T), static_cast<uint32_t>(V)};

    int t = 0;
    for (const Slot& slot : buf.slots) {
        if (slot.valid) {
            if (norm_image_) {
                // image 口径（mmaction2 PreNormalize2D）：x_n=(kx-w/2)/(w/2)，原帧尺寸基准
                float hw = slot.frame_w > 0 ? slot.frame_w * 0.5f : 1.0f;
                float hh = slot.frame_h > 0 ? slot.frame_h * 0.5f : 1.0f;
                for (int v = 0; v < V; ++v) {
                    out->data[0 * plane + static_cast<size_t>(t) * V + v] =
                        (slot.kpt[v * 3 + 0] - hw) / hw;
                    out->data[1 * plane + static_cast<size_t>(t) * V + v] =
                        (slot.kpt[v * 3 + 1] - hh) / hh;
                    out->data[2 * plane + static_cast<size_t>(t) * V + v] = slot.kpt[v * 3 + 2];
                }
            } else {
                // box 口径：逐帧 box 中心/尺度归一化（自训模型需同口径）
                const float cx = (slot.box[0] + slot.box[2]) * 0.5f;
                const float cy = (slot.box[1] + slot.box[3]) * 0.5f;
                float scale = std::max(slot.box[2] - slot.box[0], slot.box[3] - slot.box[1]);
                if (scale < 1.0f) {
                    scale = 1.0f;
                }
                for (int v = 0; v < V; ++v) {
                    out->data[0 * plane + static_cast<size_t>(t) * V + v] =
                        (slot.kpt[v * 3 + 0] - cx) / scale;
                    out->data[1 * plane + static_cast<size_t>(t) * V + v] =
                        (slot.kpt[v * 3 + 1] - cy) / scale;
                    out->data[2 * plane + static_cast<size_t>(t) * V + v] = slot.kpt[v * 3 + 2];
                }
            }
        }
        // 零槽保持全 0（visibility=0 掩码，ST-GCN zero-padding）
        ++t;
    }
    return true;
}

void PoseSequenceBuffer::resetTrack(int track_id) {
    auto it = tracks_.find(track_id);
    if (it != tracks_.end()) {
        tracks_.erase(it);
    }
}

void PoseSequenceBuffer::reset() {
    tracks_.clear();
}
