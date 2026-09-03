#include "postprocess/postprocess.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *labels[MAX_CLASS_NUM];
static int obj_class_num = 1;

int get_obj_class_num()
{
    return obj_class_num;
}

void set_obj_class_num(int class_num)
{
    if (class_num <= 0) {
        class_num = 1;
    }
    if (class_num > MAX_CLASS_NUM) {
        class_num = MAX_CLASS_NUM;
    }
    obj_class_num = class_num;
}

static char *readLine(FILE *fp, char *buffer, int *len)
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL;

    while ((ch = fgetc(fp)) != '\n' && ch != EOF)
    {
        buff_len++;
        buffer = (char *)realloc(buffer, buff_len + 1);
        if (!buffer)
            return NULL;

        buffer[i++] = (char)ch;
    }
    buffer[i] = '\0';

    *len = buff_len;

    if (ch == EOF && (i == 0 || buff_len == 0))
    {
        free(buffer);
        return NULL;
    }
    return buffer;
}

static int readLines(const char *fileName, char *lines[], int max_line)
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;

    if (file == NULL)
    {
        printf("Open %s fail!\n", fileName);
        return -1;
    }

    while ((s = readLine(file, s, &n)) != NULL)
    {
        lines[i++] = s;
        if (i >= max_line)
            break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char *locationFilename, char *label[])
{
    printf("load lable %s\n", locationFilename);
    int ret = readLines(locationFilename, label, obj_class_num);
    if (ret < 0) {
        return ret;
    }
    return ret;
}

int init_post_process(const char* label_file_path)
{
    int ret = 0;
    ret = loadLabelName(label_file_path, labels);
    if (ret < 0)
    {
        printf("Load %s failed!\n", label_file_path);
        return -1;
    }
    printf("Load %d Classes:\n", ret);
    for (int i = 0; i < ret; i++) {
        if (labels[i]) {
            printf("Class %d: %s\n", i, labels[i]);
        }
    }
    return 0;
}

char *coco_cls_to_name(int cls_id)
{

    if (cls_id < 0 || cls_id >= get_obj_class_num())
    {
        return const_cast<char*>("null");
    }

    if (labels[cls_id])
    {
        return labels[cls_id];
    }

    return const_cast<char*>("null");
}

void deinit_post_process()
{
    for (int i = 0; i < MAX_CLASS_NUM; i++)
    {
        if (labels[i] != nullptr)
        {
            free(labels[i]);
            labels[i] = nullptr;
        }
    }
}
