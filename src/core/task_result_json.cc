#include "core/task_result_json.h"

#include <cmath>
#include <sstream>

#include "postprocess/postprocess.h"

namespace {

// 浮点 → 合法 JSON 数字:非有限值(NaN/Inf,病态模型输出)钳为 0,避免产出 nan/Infinity
std::string jsonNum(float v) {
    if (!std::isfinite(v)) {
        return "0";
    }
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

void writeBoxXYWH(std::ostringstream& oss, const image_rect_t& box) {
    oss << "[" << box.left << "," << box.top << "," << (box.right - box.left) << ","
        << (box.bottom - box.top) << "]";
}

const char* classLabel(int cls_id) {
    const char* name = coco_cls_to_name(cls_id);
    return name ? name : "null";
}

}  // namespace

std::string jsonEscape(const std::string& value) {
    static const char* kHex = "0123456789abcdef";
    std::string out;
    out.reserve(value.size() + 16);
    for (char raw : value) {
        const unsigned char ch = static_cast<unsigned char>(raw);
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (ch < 0x20) {
                    out += "\\u00";
                    out += kHex[(ch >> 4) & 0xF];
                    out += kHex[ch & 0xF];
                } else {
                    out.push_back(raw);
                }
                break;
        }
    }
    return out;
}

std::string encodeBoxMaskRle(const cv::Mat& binary_mask) {
    if (binary_mask.empty() || binary_mask.type() != CV_8UC1 ||
        binary_mask.cols <= 0 || binary_mask.rows <= 0) {
        return "[]";
    }
    const cv::Mat contiguous = binary_mask.isContinuous() ? binary_mask : binary_mask.clone();
    const auto* data = contiguous.ptr<uint8_t>();
    const int total = contiguous.rows * contiguous.cols;

    std::ostringstream oss;
    oss << "[";
    uint8_t cur = data[0] > 0 ? 1 : 0;
    if (cur == 1) {
        oss << "0,";  // 计数数组从 0 值游程开始
    }
    int run = 1;
    for (int i = 1; i < total; ++i) {
        const uint8_t v = data[i] > 0 ? 1 : 0;
        if (v == cur) {
            ++run;
            continue;
        }
        oss << run << ",";
        run = 1;
        cur = v;
    }
    oss << run << "]";
    return oss.str();
}

std::string buildFrameResultJson(const PipelineFrame& frame,
                                 const std::vector<int>* detect_track_ids) {
    std::ostringstream oss;
    const int width = !frame.matFrame.empty() ? frame.matFrame.cols : frame.bufferFrame.width;
    const int height = !frame.matFrame.empty() ? frame.matFrame.rows : frame.bufferFrame.height;
    oss << "{\"schema_version\":" << kResultPayloadSchemaVersion
        << ",\"frame\":" << frame.index
        << ",\"width\":" << width
        << ",\"height\":" << height
        << ",\"source\":\"" << jsonEscape(frame.sourceName) << "\"";

    if (!frame.hasResult) {
        oss << ",\"type\":\"none\"";
        oss << "}";
        return oss.str();
    }

    if (const auto* det = std::get_if<DetectTaskResult>(&frame.result)) {
        oss << ",\"type\":\"detect\",\"dets\":[";
        const int count = det->data.count < OBJ_NUMB_MAX_SIZE ? det->data.count : OBJ_NUMB_MAX_SIZE;
        for (int i = 0; i < count; ++i) {
            const object_detect_result& o = det->data.results[i];
            if (i) {
                oss << ",";
            }
            oss << "{\"bbox\":";
            writeBoxXYWH(oss, o.box);
            oss << ",\"score\":" << jsonNum(o.prop)
                << ",\"cls\":" << o.cls_id
                << ",\"label\":\"" << jsonEscape(classLabel(o.cls_id)) << "\"";
            if (detect_track_ids && i < static_cast<int>(detect_track_ids->size())) {
                oss << ",\"track_id\":" << (*detect_track_ids)[i];
            }
            oss << "}";
        }
        oss << "]";
    } else if (const auto* pose = std::get_if<PoseTaskResult>(&frame.result)) {
        oss << ",\"type\":\"pose\",\"poses\":[";
        const int count = pose->data.count < OBJ_NUMB_MAX_SIZE ? pose->data.count : OBJ_NUMB_MAX_SIZE;
        for (int i = 0; i < count; ++i) {
            const pose_detect_result& p = pose->data.results[i];
            if (i) {
                oss << ",";
            }
            oss << "{\"bbox\":";
            writeBoxXYWH(oss, p.box);
            oss << ",\"score\":" << jsonNum(p.box_conf)
                << ",\"cls\":" << p.cls_id
                << ",\"track_id\":" << p.track_id << ",\"kpts\":[";
            for (int k = 0; k < KEYPOINT_NUM; ++k) {
                if (k) {
                    oss << ",";
                }
                oss << "[" << jsonNum(p.keypoints[k].x) << "," << jsonNum(p.keypoints[k].y)
                    << "," << jsonNum(p.keypoints[k].conf) << "]";
            }
            oss << "]}";
        }
        oss << "]";
    } else if (const auto* obb = std::get_if<OBBTaskResult>(&frame.result)) {
        oss << ",\"type\":\"obb\",\"obbs\":[";
        const int count = obb->data.count < OBJ_NUMB_MAX_SIZE ? obb->data.count : OBJ_NUMB_MAX_SIZE;
        for (int i = 0; i < count; ++i) {
            const obb_detect_result& o = obb->data.results[i];
            if (i) {
                oss << ",";
            }
            oss << "{\"box\":[" << o.box.x << "," << o.box.y << "," << o.box.w << "," << o.box.h
                << "," << jsonNum(o.box.angle) << "]"
                << ",\"score\":" << jsonNum(o.prop)
                << ",\"cls\":" << o.cls_id
                << ",\"track_id\":" << o.track_id << "}";
        }
        oss << "]";
    } else if (const auto* seg = std::get_if<SegTaskResult>(&frame.result)) {
        const seg_detect_result_list& segs = seg->data;
        oss << ",\"type\":\"seg\",\"segs\":[";
        const size_t n = segs.boxes.size();
        for (size_t i = 0; i < n; ++i) {
            const cv::Rect& r = segs.boxes[i];
            if (i) {
                oss << ",";
            }
            oss << "{\"bbox\":[" << r.x << "," << r.y << "," << r.width << "," << r.height << "]";
            if (i < segs.scores.size()) {
                oss << ",\"score\":" << jsonNum(segs.scores[i]);
            }
            if (i < segs.class_ids.size()) {
                oss << ",\"cls\":" << segs.class_ids[i];
            }
            if (i < seg->track_ids.size()) {
                oss << ",\"track_id\":" << seg->track_ids[i];
            }
            const bool has_mask = i < segs.masks.size() && !segs.masks[i].empty();
            oss << ",\"mask\":" << (has_mask ? 1 : 0);
            if (has_mask) {
                oss << ",\"rle\":" << encodeBoxMaskRle(segs.masks[i]);
            }
            oss << "}";
        }
        oss << "]";
    } else if (const auto* depth = std::get_if<DepthTaskResult>(&frame.result)) {
        // 深度栅格不入 payload(逐帧 640 级栅格经事件/文件外发体积不可控),仅携带
        // 元数据:roi 为低分辨率深度图在原帧中的位置,range 为 8bit 反相图归一化范围(米)
        oss << ",\"type\":\"depth\",\"depth\":{\"roi\":[" << depth->roi.x << "," << depth->roi.y
            << "," << depth->roi.width << "," << depth->roi.height << "],\"range\":["
            << jsonNum(depth->depth_lo) << "," << jsonNum(depth->depth_hi) << "]}";
    } else if (const auto* cc = std::get_if<CompositeClsTaskResult>(&frame.result)) {
        // M0 两阶级联:一级检测框 + 二级 top-1(sub 字段,与 cls_ids/cls_labels 一一对应)
        oss << ",\"type\":\"composite_cls\",\"dets\":[";
        const int count = cc->data.count < OBJ_NUMB_MAX_SIZE ? cc->data.count : OBJ_NUMB_MAX_SIZE;
        for (int i = 0; i < count; ++i) {
            const object_detect_result& o = cc->data.results[i];
            if (i) {
                oss << ",";
            }
            oss << "{\"bbox\":";
            writeBoxXYWH(oss, o.box);
            oss << ",\"score\":" << jsonNum(o.prop)
                << ",\"cls\":" << o.cls_id
                << ",\"label\":\"" << jsonEscape(classLabel(o.cls_id)) << "\"";
            if (i < static_cast<int>(cc->cls_ids.size()) && cc->cls_ids[i] >= 0) {
                const std::string lbl =
                    i < static_cast<int>(cc->cls_labels.size()) ? cc->cls_labels[i] : std::string();
                oss << ",\"sub\":{\"cls\":" << cc->cls_ids[i]
                    << ",\"score\":" << jsonNum(cc->cls_scores[i])
                    << ",\"label\":\"" << jsonEscape(lbl) << "\"}";
            }
            oss << "}";
        }
        oss << "]";
    } else if (const auto* face = std::get_if<FaceTaskResult>(&frame.result)) {
        // M8 人脸检测:框 + 5 点 landmark
        oss << ",\"type\":\"face\",\"faces\":[";
        for (size_t i = 0; i < face->faces.size(); ++i) {
            const FaceItem& f = face->faces[i];
            if (i) {
                oss << ",";
            }
            oss << "{\"bbox\":";
            writeBoxXYWH(oss, f.box);
            oss << ",\"score\":" << jsonNum(f.score) << ",\"kpts\":[";
            for (int k = 0; k < 5; ++k) {
                if (k) {
                    oss << ",";
                }
                oss << "[" << jsonNum(f.landmarks[k].x) << "," << jsonNum(f.landmarks[k].y) << "]";
            }
            oss << "]}";
        }
        oss << "]";
    } else if (const auto* act = std::get_if<ActionTaskResult>(&frame.result)) {
        // M13 动作识别:pose 主结果原样保留 + 本帧新鲜动作项
        oss << ",\"type\":\"action\",\"poses\":[";
        const int count = act->data.count < OBJ_NUMB_MAX_SIZE ? act->data.count : OBJ_NUMB_MAX_SIZE;
        for (int i = 0; i < count; ++i) {
            const pose_detect_result& p = act->data.results[i];
            if (i) {
                oss << ",";
            }
            oss << "{\"bbox\":";
            writeBoxXYWH(oss, p.box);
            oss << ",\"score\":" << jsonNum(p.box_conf)
                << ",\"cls\":" << p.cls_id
                << ",\"track_id\":" << p.track_id << "}";
        }
        oss << "],\"actions\":[";
        for (size_t i = 0; i < act->actions.size(); ++i) {
            const ActionItem& a = act->actions[i];
            if (i) {
                oss << ",";
            }
            oss << "{\"track_id\":" << a.track_id
                << ",\"action\":" << a.action_id
                << ",\"score\":" << jsonNum(a.score) << "}";
        }
        oss << "]";
    } else {
        oss << ",\"type\":\"none\"";
        oss << "}";
        return oss.str();
    }

    oss << "}";
    return oss.str();
}
