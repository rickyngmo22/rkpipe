# 逐帧结构化结果 payload:schema v1

本文是逐帧结果 JSON 的**权威 schema 文档**。两条传输通道使用完全相同的格式:

| 通道 | 可用性 | 说明 |
|---|---|---|
| JSONL 文件汇(环境变量 `RK_PIPE_RESULT_JSONL=<文件路径>`) | **当前核心库即可用**(仅 pipeline 模式) | 每帧一行追加写入,实现见 `io/result_sink.h` |
| `RK_PIPE_EVENT_RESULT` 事件 payload | 随核心库 Release 提供 | 门面逐帧调用序列化器(`core/task_result_json.h`)后经 `rkpipe_event_cb` 下发;契约注释见 `include/rkpipe/rkpipe.h` |

> 说明:YAML 配置键 `result_jsonl_path` 将随配套核心库 Release 与事件下发一同启用——

序列化实现:`include/core/task_result_json.h` + `src/core/task_result_json.cc`(纯逻辑,CI 单测覆盖)。

## 信封(所有 type 共有)

```json
{
  "schema_version": 1,
  "frame": 42,
  "width": 1920,
  "height": 1080,
  "source": "video/demo.mp4",
  "type": "detect"
}
```

| 字段 | 类型 | 语义 |
|---|---|---|
| `schema_version` | int | 本文为 **1**;v1 内字段只增不改,破坏性变更升 2 并同步本文 |
| `frame` | int | 帧序号(`PipelineFrame.index`,单调递增) |
| `width` / `height` | int | 原始帧尺寸(像素) |
| `source` | string | 本帧源名(图片目录输入 = 文件路径;视频/流输入可能为空) |
| `type` | string | 结果类型,决定后续载荷字段,见下表 |

信封**不带时间戳**:`PipelineFrame` 无采集时刻字段;需要墙钟请按接收时刻打戳(v1.x 视核心支持再议)。

| type | 载荷字段 | 任务 |
|---|---|---|
| `detect` | `dets` | 目标检测(YOLO26/v8/v5) |
| `pose` | `poses` | 人体关键点 |
| `obb` | `obbs` | 旋转框 |
| `seg` | `segs` | 实例分割 |
| `depth` | `depth` | 单目深度(仅元数据) |
| `none` | (无) | 本帧无结果(仅事件通道可能出现;JSONL 汇不落无结果帧) |

## 通用约定

- **坐标系**:所有框/点/角度均已 letterbox 逆映射回**原帧像素**坐标(后处理完成映射后才有结果)。
- **bbox 编码**:`[x, y, w, h]`(整型,左上角 + 宽高),与 `--dump-detections` JSONL、`tools/eval/eval_coco.py` 口径一致。
- **浮点**:流默认精度(约 6 位有效数字);非有限值(NaN/Inf,病态模型输出)钳为 `0` 保证 JSON 合法。
- **类别**:`cls` 为 label 文件行号;`label` 为经 `coco_cls_to_name` 解析的名称(labels 未加载/越界时为 `"null"`)。生产集成建议以 `cls` + label 文件自解析,`label` 仅为便利字段。
- **跟踪 ID**:`track_id` 为 `0` 表示未跟踪,正数从 1 起。detect 结果**当前不带 `track_id`**(`object_detect_result` 无此字段且布局不可变;跟踪 ID 只存在于跟踪器返回值,待 RESULT 事件缝由核心传入);pose/obb/seg 的 ID 已在结果结构体内,不受影响。

## 各 type 载荷

### detect

```json
"dets": [
  {"bbox": [100, 200, 220, 380], "score": 0.871, "cls": 0, "label": "person"}
]
```

### pose

```json
"poses": [
  {"bbox": [100, 40, 180, 420], "score": 0.9, "cls": 0, "label": "person", "track_id": 3,
   "kpts": [[213.5, 96.2, 0.95], "...共 17 组 [x, y, conf]"]}
]
```

COCO-17 关键点,`conf` 为 sigmoid 置信度;`conf == 0` 视为缺失(含跟踪 coasted 预测帧——其关键点全 0)。显示/评测阈值参考 `KEYPOINT_THRESH`(0.2)。

### obb

```json
"obbs": [
  {"box": [512, 300, 180, 80, 1.5708], "score": 0.8, "cls": 1, "label": "...", "track_id": 0}
]
```

`box` 为 `[x, y, w, h, angle]`:`x,y` 是旋转框**外接水平框**左上角,`angle` 为**弧度**(跟踪开启时 x,y 为平滑后中心换算值,w/h/angle 不变)。

### seg

```json
"segs": [
  {"bbox": [200, 150, 160, 300], "score": 0.9, "cls": 0, "track_id": 2,
   "mask": 1, "rle": [0, 2, 8, 1]}
]
```

`mask: 0` 表示本目标无掩膜(配置 `seg_mask: 0` 或超过 `mask_top_k` 上限时),此时无 `rle` 字段。`rle` 是**框内裁剪掩膜**的行主序游程长度编码:计数数组从 **0 值游程**开始(首像素为前景时首项为 0),逐值 0/1 交替;重建到整帧时按信封 `width/height` 与 `bbox` 摆放,与 `tools/eval/eval_coco.py` 的 `decode_box_rle` 闭合:

```python
def decode_box_rle(rle, box):           # 与 eval_coco.py 一致
    mask = np.zeros(box[3] * box[2], dtype=np.uint8)
    idx, val = 0, 0
    for count in rle:
        if val:
            mask[idx:idx + count] = 1
        idx += count
        val = 1 - val
    return mask.reshape(box[3], box[2])
```


### depth(仅元数据)

```json
"depth": {"roi": [8, 8, 632, 352], "range": [1.5, 72.25]}
```

深度结果是**低分辨率有效区栅格**(不放大到原帧),逐帧栅格不经事件/文件外发;`roi` 为其在原帧中的摆放位置,`range` 为深度 8bit 反相图的归一化范围(米,近=亮)。需要栅格本体请走视频输出或后续版本的拉取接口。


```json
```

`points` 为四边形四顶点(顺序:四点多边形),文本经 JSON 逃逸(UTF-8 原样透传)。

## 使用示例

```bash
RK_PIPE_RESULT_JSONL=results/detect_results.jsonl ./build/console_detector examples/configs/detect_video.yaml
# 每帧一行:
# {"schema_version":1,"frame":0,"width":1920,"height":1080,"source":"video/demo.mp4","type":"detect","dets":[...]}
```

目录需已存在;文件每次运行按 truncate 重新打开。

## 版本与兼容策略

- v1 内:**只增不改**——新增字段不升版本;消费方按"未知字段忽略"解析。
- 破坏性变更(字段语义/编码/坐标系变化)→ `schema_version: 2`,本文同步给出 v1→v2 迁移说明。
- 序列化器与本文档同仓库演进,CI 单测(`result_json` / `result_sink` 用例)锁定字段名与编码口径。
