#include "llm/detection_json.h"

#include <cmath>

#include <nlohmann/json.hpp>

#include "postprocess/postprocess.h"

namespace llm {

namespace {

// float→double 直接 dump 会带二进制尾数（0.42f → 0.41999…），统一取整到 4 位小数
inline double roundConf(float v) {
    return std::round(static_cast<double>(v) * 10000.0) / 10000.0;
}

std::string className(int cls_id, const std::vector<std::string>& class_names) {
    if (cls_id >= 0 && static_cast<std::size_t>(cls_id) < class_names.size()) {
        return class_names[static_cast<std::size_t>(cls_id)];
    }
    char* name = coco_cls_to_name(cls_id);
    if (name && *name && std::string(name) != "null") {
        return name;
    }
    return "class_" + std::to_string(cls_id);
}

nlohmann::json detectToJson(const object_detect_result_list& list,
                            const std::vector<std::string>& class_names) {
    nlohmann::json items = nlohmann::json::array();
    for (int i = 0; i < list.count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        const object_detect_result& r = list.results[i];
        items.push_back({
            {"cls_id", r.cls_id},
            {"name", className(r.cls_id, class_names)},
            {"conf", roundConf(r.prop)},
            {"box", {r.box.left, r.box.top, r.box.right, r.box.bottom}},
        });
    }
    return {
        {"type", "detect"},
        {"count", items.size()},
        {"items", items},
    };
}

nlohmann::json poseToJson(const pose_detect_result_list& list,
                          const std::vector<std::string>& class_names) {
    nlohmann::json items = nlohmann::json::array();
    for (int i = 0; i < list.count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        const pose_detect_result& r = list.results[i];
        nlohmann::json keypoints = nlohmann::json::array();
        for (const pose_keypoint& kpt : r.keypoints) {
            keypoints.push_back({kpt.x, kpt.y, kpt.conf});
        }
        items.push_back({
            {"cls_id", r.cls_id},
            {"name", className(r.cls_id, class_names)},
            {"conf", roundConf(r.box_conf)},
            {"box", {r.box.left, r.box.top, r.box.right, r.box.bottom}},
            {"track_id", r.track_id},
            {"keypoints", keypoints},
        });
    }
    return {
        {"type", "pose"},
        {"count", items.size()},
        {"items", items},
    };
}

nlohmann::json obbToJson(const obb_detect_result_list& list,
                         const std::vector<std::string>& class_names) {
    nlohmann::json items = nlohmann::json::array();
    for (int i = 0; i < list.count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        const obb_detect_result& r = list.results[i];
        items.push_back({
            {"cls_id", r.cls_id},
            {"name", className(r.cls_id, class_names)},
            {"conf", roundConf(r.prop)},
            {"box", {{"x", r.box.x}, {"y", r.box.y}, {"w", r.box.w}, {"h", r.box.h}, {"angle", r.box.angle}}},
            {"track_id", r.track_id},
        });
    }
    return {
        {"type", "obb"},
        {"count", items.size()},
        {"items", items},
    };
}

nlohmann::json segToJson(const SegTaskResult& result,
                         const std::vector<std::string>& class_names) {
    nlohmann::json items = nlohmann::json::array();
    const std::size_t count = result.data.boxes.size();
    for (std::size_t i = 0; i < count; ++i) {
        nlohmann::json item = {
            {"cls_id", result.data.class_ids[i]},
            {"name", className(result.data.class_ids[i], class_names)},
            {"conf", roundConf(result.data.scores[i])},
            {"box", {result.data.boxes[i].x, result.data.boxes[i].y,
                     result.data.boxes[i].width, result.data.boxes[i].height}},
        };
        if (i < result.track_ids.size()) {
            item["track_id"] = result.track_ids[i];
        }
        items.push_back(std::move(item));
    }
    return {
        {"type", "seg"},
        {"count", items.size()},
        {"items", items},
    };
}

// M0 两阶级联：一级检测框 + 二级 top-1（sub 字段，未命中二级时省略）
nlohmann::json compositeClsToJson(const CompositeClsTaskResult& result,
                                  const std::vector<std::string>& class_names) {
    nlohmann::json items = nlohmann::json::array();
    for (int i = 0; i < result.data.count && i < OBJ_NUMB_MAX_SIZE; ++i) {
        const object_detect_result& r = result.data.results[i];
        nlohmann::json item = {
            {"cls_id", r.cls_id},
            {"name", className(r.cls_id, class_names)},
            {"conf", roundConf(r.prop)},
            {"box", {r.box.left, r.box.top, r.box.right, r.box.bottom}},
        };
        if (i < static_cast<int>(result.cls_ids.size()) && result.cls_ids[i] >= 0) {
            item["sub"] = {
                {"cls_id", result.cls_ids[i]},
                {"score", roundConf(result.cls_scores[i])},
                {"label", i < static_cast<int>(result.cls_labels.size())
                              ? result.cls_labels[i]
                              : std::string()},
            };
        }
        items.push_back(std::move(item));
    }
    return {
        {"type", "composite_cls"},
        {"count", items.size()},
        {"items", items},
    };
}

// M8 人脸检测：框 + 5 点 landmark
nlohmann::json faceToJson(const FaceTaskResult& result) {
    nlohmann::json items = nlohmann::json::array();
    for (const FaceItem& f : result.faces) {
        nlohmann::json landmarks = nlohmann::json::array();
        for (const cv::Point2f& p : f.landmarks) {
            landmarks.push_back({p.x, p.y});
        }
        items.push_back({
            {"conf", roundConf(f.score)},
            {"box", {f.box.left, f.box.top, f.box.right, f.box.bottom}},
            {"landmarks", landmarks},
        });
    }
    return {
        {"type", "face"},
        {"count", items.size()},
        {"items", items},
    };
}

// M13 动作识别：pose 主结果原样保留（type=action），附加本帧新鲜动作项
nlohmann::json actionToJson(const ActionTaskResult& result,
                            const std::vector<std::string>& class_names) {
    nlohmann::json json = poseToJson(result.data, class_names);
    nlohmann::json actions = nlohmann::json::array();
    for (const ActionItem& a : result.actions) {
        actions.push_back({
            {"track_id", a.track_id},
            {"action_id", a.action_id},
            {"score", roundConf(a.score)},
        });
    }
    json["type"] = "action";
    json["actions"] = actions;
    return json;
}

}  // namespace

std::string taskResultToJson(const TaskResult& result, const std::vector<std::string>& class_names) {
    nlohmann::json json = {{"type", "none"}};

    if (const DetectTaskResult* d = std::get_if<DetectTaskResult>(&result)) {
        json = detectToJson(d->data, class_names);
    } else if (const PoseTaskResult* p = std::get_if<PoseTaskResult>(&result)) {
        json = poseToJson(p->data, class_names);
    } else if (const OBBTaskResult* o = std::get_if<OBBTaskResult>(&result)) {
        json = obbToJson(o->data, class_names);
    } else if (const SegTaskResult* s = std::get_if<SegTaskResult>(&result)) {
        json = segToJson(*s, class_names);
    } else if (const DepthTaskResult* dep = std::get_if<DepthTaskResult>(&result)) {
        json = {{"type", "depth"}, {"summary", "depth map only"}};
        (void)dep;
    } else if (const SemTaskResult* sem = std::get_if<SemTaskResult>(&result)) {
        json = {{"type", "sem"}, {"class_num", sem->class_num}};
    } else if (const OCRDetectTaskResult* od = std::get_if<OCRDetectTaskResult>(&result)) {
        json = {{"type", "ocr_det"}, {"count", od->polygons.size()}};
    } else if (const OCRTaskResult* ot = std::get_if<OCRTaskResult>(&result)) {
        nlohmann::json lines = nlohmann::json::array();
        for (const OCRTextLine& line : ot->lines) {
            lines.push_back({{"text", line.text}, {"score", line.score}});
        }
        json = {{"type", "ocr"}, {"count", lines.size()}, {"items", lines}};
    } else if (const Detect3DTaskResult* d3 = std::get_if<Detect3DTaskResult>(&result)) {
        json = {{"type", "detect3d"}, {"count", d3->items.size()}};
    } else if (const CompositeClsTaskResult* cc = std::get_if<CompositeClsTaskResult>(&result)) {
        json = compositeClsToJson(*cc, class_names);
    } else if (const FaceTaskResult* f = std::get_if<FaceTaskResult>(&result)) {
        json = faceToJson(*f);
    } else if (const ActionTaskResult* act = std::get_if<ActionTaskResult>(&result)) {
        json = actionToJson(*act, class_names);
    }

    return json.dump();
}

}  // namespace llm
