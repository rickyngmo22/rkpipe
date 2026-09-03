// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "postprocess/postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <iostream>
#include <cmath>
#include <algorithm>

#include <set>
#include <vector>
#include <chrono>
static thread_local double g_last_obb_post_time_ms = 0.0;
static thread_local double g_last_obb_post_decode_time_ms = 0.0;
static thread_local double g_last_obb_post_sort_time_ms = 0.0;
static thread_local double g_last_obb_post_nms_time_ms = 0.0;
static thread_local double g_last_obb_post_pack_time_ms = 0.0;
static thread_local int g_last_obb_candidate_count = 0;
static thread_local int g_last_obb_class_count = 0;
static thread_local long long g_last_obb_nms_comparisons = 0;

const int anchor[3][6] = {{10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326}
};

inline static int clamp(float val, int min, int max) {
    return val > min ? (val < max ? val : max) : min;
}

[[maybe_unused]] static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}


std::vector<float> rbbox_to_corners(const std::vector<float> &rbbox) {
    float cx = rbbox[0] + rbbox[2] / 2;
    float cy = rbbox[1] + rbbox[3] / 2;
    float x_d = rbbox[2];
    float y_d = rbbox[3];
    float angle = rbbox[4];
    float a_cos = std::cos(angle);
    float a_sin = std::sin(angle);
    std::vector<float> corners(8, 0.0);
    float corners_x[4] = {-x_d / 2, -x_d / 2, x_d / 2, x_d / 2};
    float corners_y[4] = {-y_d / 2, y_d / 2, y_d / 2, -y_d / 2};
    for (int i = 0; i < 4; ++i) {
        corners[2 * i] = a_cos * corners_x[i] - a_sin * corners_y[i] + cx;
        corners[2 * i + 1] = a_sin * corners_x[i] + a_cos * corners_y[i] + cy;
    }
    return corners;
}

bool point_in_quadrilateral(float pt_x, float pt_y, const std::vector<float> &corners) {
    float ab0 = corners[2] - corners[0];
    float ab1 = corners[3] - corners[1];

    float ad0 = corners[6] - corners[0];
    float ad1 = corners[7] - corners[1];

    float ap0 = pt_x - corners[0];
    float ap1 = pt_y - corners[1];

    float abab = ab0 * ab0 + ab1 * ab1;
    float abap = ab0 * ap0 + ab1 * ap1;
    float adad = ad0 * ad0 + ad1 * ad1;
    float adap = ad0 * ap0 + ad1 * ap1;

    return abab >= abap && abap >= 0 && adad >= adap && adap >= 0;
}


int line_segment_intersection(const std::vector<float> &pts1, const std::vector<float> &pts2, int i, int j
                              , bool &ret1, float &point_x, float &point_y) {
    std::vector<float> A(2), B(2), C(2), D(2), ret(2);
    A[0] = pts1[2 * i];
    A[1] = pts1[2 * i + 1];

    B[0] = pts1[2 * ((i + 1) % 4)];
    B[1] = pts1[2 * ((i + 1) % 4) + 1];

    C[0] = pts2[2 * j];
    C[1] = pts2[2 * j + 1];

    D[0] = pts2[2 * ((j + 1) % 4)];
    D[1] = pts2[2 * ((j + 1) % 4) + 1];

    float BA0 = B[0] - A[0];
    float BA1 = B[1] - A[1];
    float DA0 = D[0] - A[0];
    float CA0 = C[0] - A[0];
    float DA1 = D[1] - A[1];
    float CA1 = C[1] - A[1];

    bool acd = DA1 * CA0 > CA1 * DA0;
    bool bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0]);

    if (acd != bcd) {
        bool abc = CA1 * BA0 > BA1 * CA0;
        bool abd = DA1 * BA0 > BA1 * DA0;

        if (abc != abd) {
            float DC0 = D[0] - C[0];
            float DC1 = D[1] - C[1];
            float ABBA = A[0] * B[1] - B[0] * A[1];
            float CDDC = C[0] * D[1] - D[0] * C[1];
            float DH = BA1 * DC0 - BA0 * DC1;
            float Dx = ABBA * DC0 - BA0 * CDDC;
            float Dy = ABBA * DC1 - BA1 * CDDC;
            ret[0] = Dx / DH;
            ret[1] = Dy / DH;
            ret1 = true;
            point_x = ret[0] ;
            point_y = ret[1] ;
            return 0;
        }
    }
    ret1 = false;
    point_x = ret[0] ;
    point_y = ret[1] ;
    return 0;
}


bool compare_points(const std::vector<float> &pt1, const std::vector<float> &pt2, const std::vector<float> &center) {
    float vx1 = pt1[0] - center[0];
    float vy1 = pt1[1] - center[1];
    float vx2 = pt2[0] - center[0];
    float vy2 = pt2[1] - center[1];
    float d1 = std::sqrt(vx1 * vx1 + vy1 * vy1);
    float d2 = std::sqrt(vx2 * vx2 + vy2 * vy2);
    vx1 /= d1;
    vy1 /= d1;
    vx2 /= d2;
    vy2 /= d2;
    if (vy1 < 0) {
        vx1 = -2 - vx1;
    }
    if (vy2 < 0) {
        vx2 = -2 - vx2;
    }
    return vx1 < vx2;
}

void sort_vertex_in_convex_polygon(std::vector<std::vector<float>> &int_pts, int num_of_inter) {
    if (num_of_inter > 0) {
        std::vector<float> center(2, 0);
        for (int i = 0; i < num_of_inter; ++i) {
            center[0] += int_pts[i][0];
            center[1] += int_pts[i][1];
        }
        center[0] /= num_of_inter;
        center[1] /= num_of_inter;
        std::sort(int_pts.begin(), int_pts.end(), [&center](const std::vector<float> &pt1, const std::vector<float> &pt2) {
            return compare_points(pt1, pt2, center);
        });
    }
}

float triangle_area(const std::vector<float> &a, const std::vector<float> &b, const std::vector<float> &c) {
    return std::abs((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0;
}

float polygon_area(const std::vector<std::vector<float>> &int_pts, int num_of_inter) {
    float area_val = 0.0;
    for (int i = 1; i < num_of_inter - 1; ++i) {
        area_val += triangle_area(int_pts[0], int_pts[i], int_pts[i + 1]);
    }
    return area_val;
}

float Cal_IOU(float x1, float y1, float w1, float h1, float angle1, float x2, float y2, float w2, float h2, float angle2) {
    std::vector<float> rbbox1 = {x1, y1, w1, h1, angle1};
    std::vector<float> rbbox2 = {x2, y2, w2, h2, angle2};

    std::vector<float> corners1 = rbbox_to_corners(rbbox1);
    std::vector<float> corners2 = rbbox_to_corners(rbbox2);

    std::vector<std::vector<float>> pts;
    int num_pts = 0;
    for (int i = 0; i < 4; ++i) {
        float point_x = corners1[2 * i];
        float point_y = corners1[2 * i + 1];
        if (point_in_quadrilateral(point_x, point_y, corners2)) {
            num_pts++;
            pts.push_back({point_x, point_y});
        }
    }
    for (int i = 0; i < 4; ++i) {
        float point_x = corners2[2 * i];
        float point_y = corners2[2 * i + 1];
        if (point_in_quadrilateral(point_x, point_y, corners1)) {
            num_pts++;
            pts.push_back({point_x, point_y});
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float point_x, point_y;
            bool ret;
            line_segment_intersection(corners1, corners2, i, j, ret, point_x, point_y);
            if (ret) {
                num_pts++;
                pts.push_back({point_x, point_y});
            }
        }
    }

    sort_vertex_in_convex_polygon(pts, num_pts);

    float polygon_area_val = polygon_area(pts, num_pts);

    float area_union = rbbox1[2] * rbbox1[3] + rbbox2[2] * rbbox2[3] - polygon_area_val;
    return polygon_area_val / area_union;
}

static float Cal_IOU_cached(const std::vector<float> &corners1, const std::vector<float> &corners2, float w1, float h1, float w2, float h2) {
    std::vector<std::vector<float>> pts;
    int num_pts = 0;
    for (int i = 0; i < 4; ++i) {
        float point_x = corners1[2 * i];
        float point_y = corners1[2 * i + 1];
        if (point_in_quadrilateral(point_x, point_y, corners2)) {
            num_pts++;
            pts.push_back({point_x, point_y});
        }
    }
    for (int i = 0; i < 4; ++i) {
        float point_x = corners2[2 * i];
        float point_y = corners2[2 * i + 1];
        if (point_in_quadrilateral(point_x, point_y, corners1)) {
            num_pts++;
            pts.push_back({point_x, point_y});
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float point_x, point_y;
            bool ret;
            line_segment_intersection(corners1, corners2, i, j, ret, point_x, point_y);
            if (ret) {
                num_pts++;
                pts.push_back({point_x, point_y});
            }
        }
    }

    sort_vertex_in_convex_polygon(pts, num_pts);

    float polygon_area_val = polygon_area(pts, num_pts);
    float area_union = w1 * h1 + w2 * h2 - polygon_area_val;
    return polygon_area_val / area_union;
}

static inline float AabbIoU(float x0_min, float y0_min, float x0_max, float y0_max,
                            float x1_min, float y1_min, float x1_max, float y1_max) {
    float inter_w = std::max(0.0f, std::min(x0_max, x1_max) - std::max(x0_min, x1_min));
    float inter_h = std::max(0.0f, std::min(y0_max, y1_max) - std::max(y0_min, y1_min));
    float inter = inter_w * inter_h;
    float area0 = std::max(0.0f, x0_max - x0_min) * std::max(0.0f, y0_max - y0_min);
    float area1 = std::max(0.0f, x1_max - x1_min) * std::max(0.0f, y1_max - y1_min);
    float union_area = area0 + area1 - inter;
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    return inter / union_area;
}


static int nms(int validCount, const std::vector<float> &outputLocations, const std::vector<int> &classIds, std::vector<int> &order,
               int filterId, float threshold) {
    std::vector<std::vector<float>> corners_cache(validCount);
    std::vector<float> centers(validCount * 2);
    std::vector<float> radii(validCount);
    std::vector<float> aabbs(validCount * 4);
    for (int i = 0; i < validCount; ++i) {
        float x = outputLocations[i * 5 + 0];
        float y = outputLocations[i * 5 + 1];
        float w = outputLocations[i * 5 + 2];
        float h = outputLocations[i * 5 + 3];
        float angle = outputLocations[i * 5 + 4];
        corners_cache[i] = rbbox_to_corners({x, y, w, h, angle});
        centers[i * 2 + 0] = x + w * 0.5f;
        centers[i * 2 + 1] = y + h * 0.5f;
        radii[i] = 0.5f * std::sqrt(w * w + h * h);
        aabbs[i * 4 + 0] = x;
        aabbs[i * 4 + 1] = y;
        aabbs[i * 4 + 2] = x + w;
        aabbs[i * 4 + 3] = y + h;
    }
    for (int i = 0; i < validCount; ++i) {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId) {
            continue;
        }
        float w0 = outputLocations[n * 5 + 2];
        float h0 = outputLocations[n * 5 + 3];
        float cx0 = centers[n * 2 + 0];
        float cy0 = centers[n * 2 + 1];
        float r0 = radii[n];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId) {
                continue;
            }
            float w1 = outputLocations[m * 5 + 2];
            float h1 = outputLocations[m * 5 + 3];
            float cx1 = centers[m * 2 + 0];
            float cy1 = centers[m * 2 + 1];
            float r1 = radii[m];
            float dx = cx0 - cx1;
            float dy = cy0 - cy1;
            float rsum = r0 + r1;
            if (dx * dx + dy * dy > rsum * rsum) {
                continue;
            }

            float aabb_iou = AabbIoU(aabbs[n * 4 + 0], aabbs[n * 4 + 1], aabbs[n * 4 + 2], aabbs[n * 4 + 3],
                                     aabbs[m * 4 + 0], aabbs[m * 4 + 1], aabbs[m * 4 + 2], aabbs[m * 4 + 3]);
            if (aabb_iou <= threshold) {
                continue;
            }

            float iou = Cal_IOU_cached(corners_cache[n], corners_cache[m], w0, h0, w1, h1);
            g_last_obb_nms_comparisons++;
            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

static float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y) {
    return -1.0 * logf((1.0 / y) - 1.0);
}

inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}
static float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}


void softmax(float *input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_exp += expf(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
        input[i] = expf(input[i] - max_val) / sum_exp;
    }
}

static int process_i8(int8_t *input, int8_t *angle_feature, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int32_t angle_feature_zp, float angle_feature_scale, int index, int class_num) {
    int input_loc_len = 64;
    int validCount = 0;

    int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < class_num; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_i8) {
                    float box_conf_f32 = sigmoid(deqnt_affine_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ],
                                                 zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }

                    float xywh_add[2], xywh_sub[2];
                    xywh_add[0] = xywh_[0] + xywh_[2];
                    xywh_add[1] = xywh_[1] + xywh_[3];
                    xywh_sub[0] = (xywh_[2] - xywh_[0]) / 2;
                    xywh_sub[1] = (xywh_[3] - xywh_[1]) / 2;
                    float angle_feature_ = deqnt_affine_to_f32(angle_feature[index + (h * grid_w) + w], angle_feature_zp, angle_feature_scale);
                    angle_feature_ = (angle_feature_ - 0.25) * 3.1415927410125732;
                    float angle_feature_cos = cos(angle_feature_);
                    float angle_feature_sin = sin(angle_feature_);
                    float xy_mul1 = xywh_sub[0] * angle_feature_cos;
                    float xy_mul2 = xywh_sub[1] * angle_feature_sin;
                    float xy_mul3 = xywh_sub[0] * angle_feature_sin;
                    float xy_mul4 = xywh_sub[1] * angle_feature_cos;
                    xywh_[0] = ((xy_mul1 - xy_mul2) + w + 0.5) * stride;
                    xywh_[1] = ((xy_mul3 + xy_mul4) + h + 0.5) * stride;
                    xywh_[2] = xywh_add[0] * stride;
                    xywh_[3] = xywh_add[1] * stride;
                    xywh[0] = (xywh_[0] - xywh_[2] / 2);
                    xywh[1] = (xywh_[1] - xywh_[3] / 2);
                    xywh[2] = xywh_[2];
                    xywh[3] = xywh_[3];
                    boxes.push_back(xywh[0]);
                    boxes.push_back(xywh[1]);
                    boxes.push_back(xywh[2]);
                    boxes.push_back(xywh[3]);
                    boxes.push_back(angle_feature_);
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}


[[maybe_unused]] static int process_u8(uint8_t *input, uint8_t *angle_feature, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int32_t angle_feature_zp, float angle_feature_scale, int index, int class_num) {
    int input_loc_len = 64;
    int validCount = 0;

    uint8_t thres_i8 = qnt_f32_to_affine_u8(unsigmoid(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < class_num; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_i8) {
                    float box_conf_f32 = sigmoid(deqnt_affine_u8_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ],
                                                 zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_u8_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }

                    float xywh_add[2], xywh_sub[2];
                    xywh_add[0] = xywh_[0] + xywh_[2];
                    xywh_add[1] = xywh_[1] + xywh_[3];
                    xywh_sub[0] = (xywh_[2] - xywh_[0]) / 2;
                    xywh_sub[1] = (xywh_[3] - xywh_[1]) / 2;
                    float angle_feature_ = deqnt_affine_u8_to_f32(angle_feature[index + (h * grid_w) + w], angle_feature_zp, angle_feature_scale);
                    angle_feature_ = (angle_feature_ - 0.25) * 3.1415927410125732;
                    float angle_feature_cos = cos(angle_feature_);
                    float angle_feature_sin = sin(angle_feature_);
                    float xy_mul1 = xywh_sub[0] * angle_feature_cos;
                    float xy_mul2 = xywh_sub[1] * angle_feature_sin;
                    float xy_mul3 = xywh_sub[0] * angle_feature_sin;
                    float xy_mul4 = xywh_sub[1] * angle_feature_cos;
                    xywh_[0] = ((xy_mul1 - xy_mul2) + w + 0.5) * stride;
                    xywh_[1] = ((xy_mul3 + xy_mul4) + h + 0.5) * stride;
                    xywh_[2] = xywh_add[0] * stride;
                    xywh_[3] = xywh_add[1] * stride;
                    xywh[0] = (xywh_[0] - xywh_[2] / 2);
                    xywh[1] = (xywh_[1] - xywh_[3] / 2);
                    xywh[2] = xywh_[2];
                    xywh[3] = xywh_[3];
                    boxes.push_back(xywh[0]);
                    boxes.push_back(xywh[1]);
                    boxes.push_back(xywh[2]);
                    boxes.push_back(xywh[3]);
                    boxes.push_back(angle_feature_);
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}


int post_process_obb(rknn_app_context_t *app_ctx, void *outputs, letterbox_t *letter_box, float conf_threshold, float nms_threshold, obb_detect_result_list *od_results) {
    auto t0 = std::chrono::high_resolution_clock::now();
    rknn_output *_outputs = (rknn_output *)outputs;
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = app_ctx->model_width;
    int model_in_h = app_ctx->model_height;
    memset(od_results, 0, sizeof(obb_detect_result_list));
    int index = 0;
    int class_num = app_ctx->class_num > 0 ? app_ctx->class_num : get_obj_class_num();

    for (int i = 0; i < 3; i++) {
        grid_h = app_ctx->output_attrs[i].dims[2];
        grid_w = app_ctx->output_attrs[i].dims[3];
        stride = model_in_h / grid_h;
        if (app_ctx->is_quant) {
            validCount += process_i8((int8_t *)_outputs[i].buf, (int8_t *)_outputs[3].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, app_ctx->output_attrs[i].zp, app_ctx->output_attrs[i].scale,
                                     app_ctx->output_attrs[3].zp, app_ctx->output_attrs[3].scale, index, class_num);
        }
        index += grid_h * grid_w;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    g_last_obb_post_decode_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (validCount <= 0) {
        g_last_obb_post_sort_time_ms = 0.0;
        g_last_obb_post_nms_time_ms = 0.0;
        g_last_obb_post_pack_time_ms = 0.0;
        g_last_obb_post_time_ms = g_last_obb_post_decode_time_ms;
        return 0;
    }
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
    auto t2 = std::chrono::high_resolution_clock::now();
    g_last_obb_post_sort_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::set<int> class_set(std::begin(classId), std::end(classId));
    g_last_obb_candidate_count = validCount;
    g_last_obb_class_count = (int)class_set.size();
    g_last_obb_nms_comparisons = 0;

    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    g_last_obb_post_nms_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    int last_count = 0;
    od_results->count = 0;

    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 5 + 0] - letter_box->x_pad;
        float y1 = filterBoxes[n * 5 + 1] - letter_box->y_pad;
        float w = filterBoxes[n * 5 + 2];
        float h = filterBoxes[n * 5 + 3];
        float angle = filterBoxes[n * 5 + 4];

        int id = classId[n];
        float obj_conf = objProbs[i];

        od_results->results[last_count].box.x = (int)(clamp(x1, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.y = (int)(clamp(y1, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.w = (int)(clamp(w, 0, model_in_w) / letter_box->scale);
        od_results->results[last_count].box.h = (int)(clamp(h, 0, model_in_h) / letter_box->scale);
        od_results->results[last_count].box.angle = angle;
        od_results->results[last_count].prop = obj_conf;
        od_results->results[last_count].cls_id = id;
        last_count++;
    }
    od_results->count = last_count;
    auto t4 = std::chrono::high_resolution_clock::now();
    g_last_obb_post_pack_time_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    g_last_obb_post_time_ms = std::chrono::duration<double, std::milli>(t4 - t0).count();
    return 0;
}

double get_last_obb_postprocess_time_ms() {
    return g_last_obb_post_time_ms;
}

double get_last_obb_postprocess_decode_time_ms() {
    return g_last_obb_post_decode_time_ms;
}

double get_last_obb_postprocess_sort_time_ms() {
    return g_last_obb_post_sort_time_ms;
}

double get_last_obb_postprocess_nms_time_ms() {
    return g_last_obb_post_nms_time_ms;
}

double get_last_obb_postprocess_pack_time_ms() {
    return g_last_obb_post_pack_time_ms;
}

int get_last_obb_candidate_count() { return g_last_obb_candidate_count; }
int get_last_obb_class_count() { return g_last_obb_class_count; }
long long get_last_obb_nms_comparisons() { return g_last_obb_nms_comparisons; }
