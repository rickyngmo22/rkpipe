#pragma once

// 3D 线框投影纯函数（D3 路线 B 共用）：Detect3DItem + P2 → 8 角点图像坐标。
// 数据来源不绑定 detect3d 任务：KITTI 3D 头回归（旧）或 aux depth 接地采样（现）均可填。
// 支持两种布局：
//   原版 [1,300,14]：[x1,y1,x2,y2, conf, cls, cx3d,cy3d,depth,sin,cos,h3,w3,l3]（已解码）
//   板端 [1,N,38]：原始量，解码全部在本头文件完成（图内只留真机验证过的算子）：
//     [0:4) 框 [4:7) 类别分数 [7:9) 中心偏移 [9] log深度 [10:13) 尺寸残差 [13] q3d
//     [14:26) 朝向 bin 分数 [26:38) bin 残差
// 纯函数无硬件依赖，CI 单测覆盖。

#include <algorithm>
#include <cmath>

#include "utils/common.h"

struct Detect3DItem {
    image_rect_t box = {};   // 2D 框（映射回原帧后）
    float conf = 0.0f;
    int cls_id = -1;
    float center_u = 0.0f;   // 3D 投影中心（原帧像素）
    float center_v = 0.0f;
    float depth_m = 0.0f;
    float sin_alpha = 0.0f;
    float cos_alpha = 0.0f;
    float h3 = 0.0f;         // 3D 尺寸（米）：高/宽/长
    float w3 = 0.0f;
    float l3 = 0.0f;
};

// 3D 线框投影：由检测项 + P2（3x4 行主序，12 值）计算包围盒 8 角点的图像坐标。
// 约定：相机系 x 右 / y 下 / z 前；center_u/v 为 3D 包围盒**几何中心（半高）**的投影像素
// （真机截图实测：按底面中心画会整体抬高约 h/2，见 2026-08-29 线框标定记录）；
// 朝向约定（2026-09-02 真机修正）：模型输出的 sin/cos 即最终航向 ry（车长方向相对
// 相机 x 轴），直接 atan2 还原，不叠加视角修正 theta——原 ry = alpha + atan2(Z,X) 会把
// 正前方车辆转横约 90°（车长投影到侧向，线框横向摊开且越偏离画面中心越歪）。
// 备选式（若模型预测的是 KITTI 观测角 α）：ry = alpha + atan2(X, Z)；画面中心附近与
// 直读几乎等价，若修后两侧远处车辆线框仍不贴合再切换验证。
// 角点序：0-3 底面（沿 +x 起），4-7 顶面；12 条边 = 底 4 + 顶 4 + 立柱 4。
// 返回 false：P2 无效 / 深度非正 / 尺寸非正 / 角点落到相机后方。
inline bool computeDetect3DCorners2D(const Detect3DItem& it, const float* p2, float out[8][2]) {
    if (p2 == nullptr || it.depth_m <= 0.0f) {
        return false;
    }
    const float fx = p2[0], fy = p2[5], cx = p2[2], cy = p2[6];
    if (!(fx > 0.0f && fy > 0.0f)) {
        return false;
    }
    const float Z = it.depth_m;
    const float X = (it.center_u - cx) * Z / fx;
    const float Y = (it.center_v - cy) * Z / fy;
    const float ry = std::atan2(it.sin_alpha, it.cos_alpha);
    const float l = it.l3, h = it.h3, w = it.w3;
    if (!(l > 0.0f && h > 0.0f && w > 0.0f)) {
        return false;
    }
    const float kXc[8] = {l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2};
    const float kYc[8] = {h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2};
    const float kZc[8] = {w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2};
    const float c = std::cos(ry), s = std::sin(ry);
    for (int i = 0; i < 8; ++i) {
        const float X3 = c * kXc[i] + s * kZc[i] + X;
        const float Y3 = kYc[i] + Y;
        const float Z3 = -s * kXc[i] + c * kZc[i] + Z;
        const float pw = p2[8] * X3 + p2[9] * Y3 + p2[10] * Z3 + p2[11];
        if (!(pw > 1e-3f)) {
            return false;  // 角点在相机后方，线框不绘制
        }
        out[i][0] = (p2[0] * X3 + p2[1] * Y3 + p2[2] * Z3 + p2[3]) / pw;
        out[i][1] = (p2[4] * X3 + p2[5] * Y3 + p2[6] * Z3 + p2[7]) / pw;
    }
    return true;
}
