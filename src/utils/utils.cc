#include "utils.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>

static bool to_rgb_mat(const image_buffer_t* image, cv::Mat& rgb)
{
    if (!image || !image->virt_addr || image->width <= 0 || image->height <= 0) {
        return false;
    }

    if (image->format == IMAGE_FORMAT_RGB888) {
        int step = (image->width_stride > 0 ? image->width_stride : image->width) * 3;
        rgb = cv::Mat(image->height, image->width, CV_8UC3, image->virt_addr, step);
        return true;
    }

    if (image->format == IMAGE_FORMAT_BGR888) {
        int step = (image->width_stride > 0 ? image->width_stride : image->width) * 3;
        cv::Mat bgr(image->height, image->width, CV_8UC3, image->virt_addr, step);
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        return true;
    }

    if (image->format == IMAGE_FORMAT_RGBA8888) {
        int step = (image->width_stride > 0 ? image->width_stride : image->width) * 4;
        cv::Mat rgba(image->height, image->width, CV_8UC4, image->virt_addr, step);
        cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);
        return true;
    }

    if (image->format == IMAGE_FORMAT_GRAY8) {
        int step = (image->width_stride > 0 ? image->width_stride : image->width);
        cv::Mat gray(image->height, image->width, CV_8UC1, image->virt_addr, step);
        cv::cvtColor(gray, rgb, cv::COLOR_GRAY2RGB);
        return true;
    }

    if (image->format == IMAGE_FORMAT_YUV420SP_NV12 || image->format == IMAGE_FORMAT_YUV420SP_NV21) {
        int step = (image->width_stride > 0 ? image->width_stride : image->width);
        int h = image->height_stride > 0 ? image->height_stride : image->height;
        cv::Mat yuv(h * 3 / 2, image->width, CV_8UC1, image->virt_addr, step);
        int code = image->format == IMAGE_FORMAT_YUV420SP_NV21 ? cv::COLOR_YUV2RGB_NV21 : cv::COLOR_YUV2RGB_NV12;
        cv::cvtColor(yuv, rgb, code);
        if (h > image->height) {
            rgb = rgb(cv::Rect(0, 0, image->width, image->height)).clone();
        }
        return true;
    }

    return false;
}

static bool write_dst_mat(const cv::Mat& src_rgb, image_buffer_t* dst_image)
{
    if (!dst_image || !dst_image->virt_addr || dst_image->width <= 0 || dst_image->height <= 0) {
        return false;
    }

    int width_stride = dst_image->width_stride > 0 ? dst_image->width_stride : dst_image->width;
    int height = dst_image->height;

    if (dst_image->format == IMAGE_FORMAT_RGB888) {
        int step = width_stride * 3;
        cv::Mat dst(height, dst_image->width, CV_8UC3, dst_image->virt_addr, step);
        src_rgb.copyTo(dst);
        return true;
    }

    if (dst_image->format == IMAGE_FORMAT_BGR888) {
        int step = width_stride * 3;
        cv::Mat dst(height, dst_image->width, CV_8UC3, dst_image->virt_addr, step);
        cv::Mat bgr;
        cv::cvtColor(src_rgb, bgr, cv::COLOR_RGB2BGR);
        bgr.copyTo(dst);
        return true;
    }

    if (dst_image->format == IMAGE_FORMAT_RGBA8888) {
        int step = width_stride * 4;
        cv::Mat dst(height, dst_image->width, CV_8UC4, dst_image->virt_addr, step);
        cv::Mat rgba;
        cv::cvtColor(src_rgb, rgba, cv::COLOR_RGB2RGBA);
        rgba.copyTo(dst);
        return true;
    }

    if (dst_image->format == IMAGE_FORMAT_GRAY8) {
        int step = width_stride;
        cv::Mat dst(height, dst_image->width, CV_8UC1, dst_image->virt_addr, step);
        cv::Mat gray;
        cv::cvtColor(src_rgb, gray, cv::COLOR_RGB2GRAY);
        gray.copyTo(dst);
        return true;
    }

    return false;
}

int read_image(const char* path, image_buffer_t* image)
{
    if (!path || !image) {
        return -1;
    }
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        return -1;
    }

    int size = bgr.cols * bgr.rows * 3;
    unsigned char* buffer = static_cast<unsigned char*>(malloc(size));
    if (!buffer) {
        return -1;
    }
    std::memcpy(buffer, bgr.data, size);

    image->width = bgr.cols;
    image->height = bgr.rows;
    image->width_stride = bgr.cols;
    image->height_stride = bgr.rows;
    image->format = IMAGE_FORMAT_BGR888;
    image->virt_addr = buffer;
    image->size = size;
    image->fd = 0;
    image->priv_data = nullptr;

    return 0;
}

int write_image(const char* path, const image_buffer_t* image)
{
    if (!path || !image || !image->virt_addr || image->width <= 0 || image->height <= 0) {
        return -1;
    }

    int width_stride = image->width_stride > 0 ? image->width_stride : image->width;
    if (image->format == IMAGE_FORMAT_BGR888) {
        int step = width_stride * 3;
        cv::Mat bgr(image->height, image->width, CV_8UC3, image->virt_addr, step);
        return cv::imwrite(path, bgr) ? 0 : -1;
    }
    if (image->format == IMAGE_FORMAT_RGB888) {
        int step = width_stride * 3;
        cv::Mat rgb(image->height, image->width, CV_8UC3, image->virt_addr, step);
        cv::Mat bgr;
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
        return cv::imwrite(path, bgr) ? 0 : -1;
    }
    if (image->format == IMAGE_FORMAT_RGBA8888) {
        int step = width_stride * 4;
        cv::Mat rgba(image->height, image->width, CV_8UC4, image->virt_addr, step);
        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        return cv::imwrite(path, bgr) ? 0 : -1;
    }
    if (image->format == IMAGE_FORMAT_GRAY8) {
        int step = width_stride;
        cv::Mat gray(image->height, image->width, CV_8UC1, image->virt_addr, step);
        return cv::imwrite(path, gray) ? 0 : -1;
    }
    return -1;
}

int convert_image(image_buffer_t* src_image, image_buffer_t* dst_image, image_rect_t* src_box, image_rect_t* dst_box, char color)
{
    if (!src_image || !dst_image || !dst_image->virt_addr) {
        return -1;
    }

    cv::Mat src_rgb;
    if (!to_rgb_mat(src_image, src_rgb)) {
        return -1;
    }

    int src_left = src_box ? src_box->left : 0;
    int src_top = src_box ? src_box->top : 0;
    int src_right = src_box ? src_box->right : src_image->width;
    int src_bottom = src_box ? src_box->bottom : src_image->height;

    src_left = std::max(0, std::min(src_left, src_image->width - 1));
    src_top = std::max(0, std::min(src_top, src_image->height - 1));
    src_right = std::max(src_left + 1, std::min(src_right, src_image->width));
    src_bottom = std::max(src_top + 1, std::min(src_bottom, src_image->height));

    cv::Mat src_crop = src_rgb(cv::Rect(src_left, src_top, src_right - src_left, src_bottom - src_top));

    int dst_left = dst_box ? dst_box->left : 0;
    int dst_top = dst_box ? dst_box->top : 0;
    int dst_right = dst_box ? dst_box->right : dst_image->width;
    int dst_bottom = dst_box ? dst_box->bottom : dst_image->height;

    dst_left = std::max(0, std::min(dst_left, dst_image->width - 1));
    dst_top = std::max(0, std::min(dst_top, dst_image->height - 1));
    dst_right = std::max(dst_left + 1, std::min(dst_right, dst_image->width));
    dst_bottom = std::max(dst_top + 1, std::min(dst_bottom, dst_image->height));

    int dst_w = dst_right - dst_left;
    int dst_h = dst_bottom - dst_top;

    cv::Mat resized;
    cv::resize(src_crop, resized, cv::Size(dst_w, dst_h));

    cv::Mat dst_rgb(dst_image->height, dst_image->width, CV_8UC3, cv::Scalar((unsigned char)color, (unsigned char)color, (unsigned char)color));
    resized.copyTo(dst_rgb(cv::Rect(dst_left, dst_top, dst_w, dst_h)));

    return write_dst_mat(dst_rgb, dst_image) ? 0 : -1;
}

int convert_image_with_letterbox(image_buffer_t* src_image, image_buffer_t* dst_image, letterbox_t* letterbox, char color)
{
    if (!src_image || !dst_image || !dst_image->virt_addr || !letterbox) {
        return -1;
    }

    cv::Mat src_rgb;
    if (!to_rgb_mat(src_image, src_rgb)) {
        return -1;
    }

    int dst_w = dst_image->width;
    int dst_h = dst_image->height;
    if (dst_w <= 0 || dst_h <= 0) {
        return -1;
    }

    letterbox->crop_x = 0;
    letterbox->crop_y = 0;
    letterbox->crop_w = src_rgb.cols;
    letterbox->crop_h = src_rgb.rows;

    int src_w = src_rgb.cols;
    int src_h = src_rgb.rows;

    int resized_w = dst_w;
    int resized_h = dst_h;
    float scale = 1.0f;

    float scale_w = static_cast<float>(dst_w) / static_cast<float>(src_w);
    float scale_h = static_cast<float>(dst_h) / static_cast<float>(src_h);
    if (scale_w < scale_h) {
        scale = scale_w;
        resized_h = static_cast<int>(src_h * scale);
    } else {
        scale = scale_h;
        resized_w = static_cast<int>(src_w * scale);
    }

    if (resized_w % 4 != 0) {
        resized_w -= (resized_w % 4);
    }
    if (resized_h % 2 != 0) {
        resized_h -= (resized_h % 2);
    }
    resized_w = std::max(1, resized_w);
    resized_h = std::max(1, resized_h);

    int padding_w = dst_w - resized_w;
    int padding_h = dst_h - resized_h;
    int x_pad = 0;
    int y_pad = 0;
    if (scale_w < scale_h) {
        y_pad = padding_h / 2;
        if (y_pad % 2 != 0) {
            y_pad -= (y_pad % 2);
            if (y_pad < 0) {
                y_pad = 0;
            }
        }
    } else {
        x_pad = padding_w / 2;
        if (x_pad % 2 != 0) {
            x_pad -= (x_pad % 2);
            if (x_pad < 0) {
                x_pad = 0;
            }
        }
    }

    letterbox->x_pad = x_pad;
    letterbox->y_pad = y_pad;
    letterbox->scale = scale;

    cv::Mat resized;
    cv::resize(src_rgb, resized, cv::Size(resized_w, resized_h));

    cv::Mat dst_rgb(dst_h, dst_w, CV_8UC3, cv::Scalar((unsigned char)color, (unsigned char)color, (unsigned char)color));
    resized.copyTo(dst_rgb(cv::Rect(x_pad, y_pad, resized_w, resized_h)));

    return write_dst_mat(dst_rgb, dst_image) ? 0 : -1;
}

int get_image_size(image_buffer_t* image)
{
    if (!image) {
        return 0;
    }
    int channels = 0;
    if (image->format == IMAGE_FORMAT_GRAY8) {
        channels = 1;
    } else if (image->format == IMAGE_FORMAT_RGB888 || image->format == IMAGE_FORMAT_BGR888) {
        channels = 3;
    } else if (image->format == IMAGE_FORMAT_RGBA8888) {
        channels = 4;
    } else if (image->format == IMAGE_FORMAT_YUV420SP_NV12 || image->format == IMAGE_FORMAT_YUV420SP_NV21) {
        return image->width * image->height * 3 / 2;
    }
    return image->width * image->height * channels;
}

static int count_lines(FILE* file)
{
    int count = 0;
    char ch;

    while (!feof(file))
    {
        ch = fgetc(file);
        if (ch == '\n')
        {
            count++;
        }
    }
    count += 1;

    rewind(file);
    return count;
}

extern "C" {

int read_data_from_file(const char *path, char **out_data)
{
    *out_data = NULL;
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != (int)fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}

int write_data_to_file(const char *path, const char *data, unsigned int size)
{
    FILE *fp;

    fp = fopen(path, "w");
    if(fp == NULL) {
        printf("open error: %s\n", path);
        return -1;
    }

    fwrite(data, 1, size, fp);
    fflush(fp);

    fclose(fp);
    return 0;
}

char** read_lines_from_file(const char* filename, int* line_count)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return NULL;
    }

    int num_lines = count_lines(file);
    printf("num_lines=%d\n", num_lines);
    char** lines = (char**)malloc(num_lines * sizeof(char*));
    memset(lines, 0, num_lines * sizeof(char*));

    char buffer[1024];
    int line_index = 0;

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        buffer[strcspn(buffer, "\n")] = '\0';

        lines[line_index] = (char*)malloc(strlen(buffer) + 1);
        strcpy(lines[line_index], buffer);

        line_index++;
    }

    fclose(file);

    *line_count = num_lines;
    return lines;
}

void free_lines(char** lines, int line_count)
{
    for (int i = 0; i < line_count; i++) {
        if (lines[i] != NULL) {
            free(lines[i]);
        }
    }
    free(lines);
}

}
