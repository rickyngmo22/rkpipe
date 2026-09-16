#include <mutex>
#include "postprocess/postprocess.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- 标签表实例化（专项#4-P2）----
// 全局表：单实例/console 路径与"当前 init"的表（labels_mutex 保护 swap）
// 线程绑定表：多实例下各 handle 的 worker/output 线程绑定自己的表快照
// （shared_ptr 持有保命），coco_cls_to_name/get_obj_class_num 优先读绑定表。
static std::shared_ptr<const LabelTable> g_label_table;
static std::mutex labels_mutex;
static thread_local std::shared_ptr<const LabelTable> t_thread_table;

std::shared_ptr<const LabelTable> capture_label_table() {
    std::lock_guard<std::mutex> lock(labels_mutex);
    return g_label_table;
}

void bind_thread_label_table(std::shared_ptr<const LabelTable> table) {
    t_thread_table = std::move(table);
}

// 解析当前线程应使用的表：绑定表优先，回退全局表
static const LabelTable* currentLabelTable() {
    if (t_thread_table) {
        return t_thread_table.get();
    }
    std::lock_guard<std::mutex> lock(labels_mutex);
    return g_label_table.get();
}

int get_obj_class_num()
{
    const LabelTable* tbl = currentLabelTable();
    return tbl ? tbl->obj_class_num : 0;
}

void set_obj_class_num(int class_num)
{
    // 写当前全局表的类别数（init 之后由 configureClassAndLabels 调用）
    std::lock_guard<std::mutex> lock(labels_mutex);
    if (!g_label_table) {
        g_label_table = std::make_shared<LabelTable>();
    }
    auto table = std::const_pointer_cast<LabelTable>(g_label_table);
    if (class_num > 0) {
        table->obj_class_num = class_num;
    }
}





int init_post_process(const char* label_file_path)
{
    // 幂等化：SDK 多 handle 场景可能重复 init；每次构造新表原子替换全局表
    //（旧表由已绑定线程的 shared_ptr 续命，无悬垂）
    auto table = std::make_shared<LabelTable>();
    FILE* fp = fopen(label_file_path, "rb");
    if (!fp) {
        printf("Load %s failed!\n", label_file_path);
        return -1;
    }
    std::vector<char> buf(4096);
    std::string line;
    while (fgets(buf.data(), static_cast<int>(buf.size()), fp)) {
        line.assign(buf.data());
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (!line.empty()) {
            table->names.push_back(line);
        }
    }
    fclose(fp);
    table->obj_class_num = static_cast<int>(table->names.size());
    {
        std::lock_guard<std::mutex> lock(labels_mutex);
        printf("Load %d Classes:\n", table->obj_class_num);
        for (int i = 0; i < table->obj_class_num; ++i) {
            printf("Class %d: %s\n", i, table->names[i].c_str());
        }
        g_label_table = std::move(table);
    }
    return 0;
}

char *coco_cls_to_name(int cls_id)
{
    const LabelTable* tbl = currentLabelTable();
    if (!tbl || cls_id < 0 || cls_id >= tbl->obj_class_num || cls_id >= static_cast<int>(tbl->names.size())) {
        return const_cast<char*>("null");
    }
    // 表由 shared_ptr 续命（绑定线程持有/全局表在生命周期内），返回内部指针
    return const_cast<char*>(tbl->names[cls_id].c_str());
}

void deinit_post_process()
{
    std::lock_guard<std::mutex> lock(labels_mutex);
    g_label_table.reset();  // 已绑定线程的表由 shared_ptr 续命，无悬垂
}
