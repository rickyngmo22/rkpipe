#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rknn_eval_web —— 评测控制台（零依赖：仅 Python 标准库）。

部署形态：**板端开启服务，PC 浏览器直接操作**（无需 PC 端 ssh/rsync 环境）。
  板端:  tools/eval/eval_web_service.sh start     （或 python3 tools/eval/rknn_eval_web.py --port 8081）
  PC:    浏览器打开 http://<板IP>:8081
         ① 上传数据集 zip（images/ + COCO JSON）或填板上路径
         ② 选模型（板上 model/ 或上传 .rknn）→ 提交评测
         ③ /jobs 看进度，详情页看实时日志与报告，失败可重试

页面：
  /                提交页（板上直评：模型/数据/参数）
  /jobs            任务队列 + 全量历史（重启后自动还原，自动刷新）
  /job/<id>        任务详情：实时日志 + 预览快照 + 完成后报告链接 + 失败重试
  /compare         历史对比（web_runs/ + eval_runs/ 条形图）
API：
  POST /create                 表单提交 → 302 /job/<id>
  POST /retry                  失败任务人工重试（同一命令，新 out_dir）→ 302 /job/<new>
  POST /api/upload?name=       原始字节上传：数据集 zip（自动解压探测）/ .rknn 模型
  GET  /api/job/<id>           状态 JSON（前端轮询）
  GET  /job/<id>/log.txt       日志尾部
  GET  /job/<id>/preview.jpg   板端 MJPEG 抓帧（每 5s 刷新一次缓存）

任务历史（设计文档 §2：状态落库、重启可恢复展示、不自动重跑）：
  启动时扫描 web_runs/*/web_job.json 原样恢复；上次中断（queued/running）标为
  error 供人工重试；eval_runs/、runs/ 下 CLI 产物从 summary_all.json 合成
  历史条目（/jobs 与详情页均可看报告）。
"""

import argparse
import glob
import hashlib
import html
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import yaml
except ImportError:
    yaml = None
try:
    import cv2
except ImportError:
    cv2 = None

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root():
    """从本文件向上找含 CMakeLists.txt 的目录（仓库根）；找不到退回上两级。

    不写死层数，这样 tools/eval 的布局变化或把脚本挪位置都不会静默算错。
    """
    d = HERE
    for _ in range(4):
        if os.path.exists(os.path.join(d, "CMakeLists.txt")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return os.path.dirname(os.path.dirname(HERE))


REPO_ROOT = _repo_root()
# 用绝对路径（旧实现是相对路径 "build/rknn_eval"，判定结果依赖进程 CWD）
# build*/ 在 .gitignore 里，删掉后需重建：
#   cmake -B build -S . -DOPENCV_ROOT=/userdata/opencv-4.11.0-install
#   cmake --build build --target rknn_eval -j 8
EVAL_BIN_CANDIDATES = [
    os.path.join(REPO_ROOT, "build/rknn_eval"),
    os.path.join(REPO_ROOT, "build-ci/rknn_eval"),
]

JOBS = {}  # id -> dict(state, cmd, log_path, out_dir, tag, created, ...)
JOB_ORDER = []
LOCK = threading.Lock()
QUEUE = []  # 待执行 job id
RUNNING = {}  # board_key -> jid（同板串行）
PREVIEW_CACHE = {}  # jid -> (ts, jpeg_bytes)：板端 MJPEG 抓帧缓存（2s）
PREVIEW_IDX = {}    # jid -> 下一个快照槽位（out_dir/preview/shot_<i%6>.jpg 轮转）
SHOT_SLOTS = 6      # 运行页底部展示的快照张数（每 ~3s 抽 1 帧，跳帧刷新）
DATASET_DIRS = ["datasets", os.path.join(HERE, "datasets")]
# PC 数据集拉取的板上缓存根目录（按数据集 key 稳定存放 → rsync 增量，二次评测≈0 传输）
RK_EVAL_DATA = os.environ.get("RK_EVAL_DATA", "/userdata/rk_eval_data")

DEFAULTS = dict(
    task="detect",
    model="",
    images="test_images",
    ann="",
    label="assets/labels/coco_80_labels_list.txt",
    obj_num="80",
    conf="0.25",
    threads="4",
    batch="500",
    model_dir="model",
)

PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>rknn_eval 控制台</title>
<style>
:root{--bg:#f4f6f9;--card:#fff;--line:#e3e8ef;--txt:#1c2333;--sub:#6b7487;
--acc:#2f6fed;--ok:#0a7d32;--err:#b3261e;--run:#8a6d00;--dark:#10141c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);
font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif}
header{background:#fff;border-bottom:1px solid var(--line);padding:10px 24px;
display:flex;align-items:center;gap:18px;position:sticky;top:0;z-index:9}
header h1{font-size:16px;margin:0}
header nav a{color:var(--sub);text-decoration:none;margin-right:14px;font-size:13px}
header nav a:hover{color:var(--acc)}
main{max-width:1080px;margin:24px auto;padding:0 20px}
h2{font-size:17px;margin:0 0 14px}
h3{font-size:13px;margin:16px 0 6px;color:var(--sub);font-weight:600;
text-transform:uppercase;letter-spacing:.04em}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin-bottom:18px;box-shadow:0 1px 2px rgba(16,24,40,.04)}
table{border-collapse:collapse;width:100%%}
td,th{border-bottom:1px solid var(--line);padding:8px 10px;font-size:13px;text-align:left}
th{color:var(--sub);font-weight:600;background:#fafbfd}
tr:hover td{background:#f7f9fc}
label{display:inline-block;width:130px;font-size:13px;color:var(--sub)}
input,select{padding:6px 8px;margin:3px 0;width:400px;border:1px solid var(--line);
border-radius:6px;background:#fff;font:inherit;color:var(--txt)}
input[type=checkbox]{width:auto;margin-right:4px;vertical-align:middle}
input[type=file]{width:auto;border:none;padding:2px 0;color:var(--sub)}
input:focus,select:focus{outline:2px solid #cdddfb;border-color:var(--acc)}
.btn{padding:8px 22px;width:auto;cursor:pointer;background:var(--acc);color:#fff;
border:none;border-radius:6px;font:inherit}
.btn:hover{background:#2559c4}
.btn.sub{background:#fff;color:var(--acc);border:1px solid var(--acc)}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;
vertical-align:middle}
.badge.ok{background:#e6f4ea;color:var(--ok)}
.badge.err{background:#fce8e6;color:var(--err)}
.badge.run{background:#fff4d6;color:var(--run)}
.badge.q{background:#eef1f6;color:var(--sub)}
.badge.canceled{background:#eef1f6;color:var(--sub);text-decoration:line-through}
.small{font-size:12px;color:var(--sub)}
pre{background:var(--dark);color:#cdd6e4;padding:12px;font-size:12px;line-height:1.5;
overflow:auto;max-height:380px;border-radius:8px;margin:8px 0}
.shots{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.shots img{width:100%%;border-radius:8px;border:1px solid var(--line);background:#000;
min-height:110px;object-fit:contain}
.shots img.empty{visibility:hidden}
.pcmp{display:flex;gap:12px;flex-wrap:wrap}
.pcol{flex:1;min-width:190px}
.pcol img{width:100%;border-radius:8px;border:1px solid var(--line);background:#000;
display:block;margin-top:4px;min-height:90px;object-fit:contain}
.visgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}
.visgrid img{width:100%%;border-radius:8px;border:1px solid var(--line)}
details.cmd{margin:8px 0}
details.cmd summary{cursor:pointer;color:var(--sub);font-size:12px}
</style></head><body>
<header><h1>rknn_eval 控制台</h1>
<nav><a href="/">提交评测</a><a href="/jobs">队列 / 历史</a><a href="/compare">历史对比</a></nav>
<span class="small">RK3588 板端精度 / 性能</span></header>
<main>
<!--BODY-->
</main></body></html>"""


def render_page(body):
    """占位符替换而非 % 格式化：body（日志/报告/命令）可能含任意 % 字符。"""
    return PAGE.replace("<!--BODY-->", body)

FORM = """
<div class="card">
<h2>提交新评测</h2>
<form method="post" action="/create" onsubmit="return beforeSubmit(event)">
<h3>模型</h3>
<label>选择模型</label><select name="model" id="model_sel">%(model_options)s</select><br>
<label>或上传模型</label><input type="file" id="model_file" accept=".rknn" onchange="uploadModel(this)">
<span id="model_up_stat" class="small">.rknn 上传到板上 uploads/models/（大文件请直接拷到 model/）</span><br>
<label>任务</label><select name="task" onchange="onTaskChange(this)">%(task_options)s</select>
<span class="small">detect / pose / seg / obb</span><br>
<h3>数据集</h3>
<label>数据来源</label><select name="ds_source" onchange="dsMode(this.value)">
<option value="sync">PC 文件夹同步（增量上传）</option>
<option value="path">板上路径</option></select><br>
<span id="ds_sync">
<label>图片文件夹</label><input type="file" id="sync_dir" webkitdirectory onchange="syncFolder(this)"><br>
<label>标注文件</label><input type="file" id="sync_ann" onchange="syncAnnFile(this)">
<span id="sync_stat" class="small">图片与标注分开选、目录随意、无需打包；标注格式随任务（detect/seg/pose 选 COCO JSON，obb 选切片级 gt jsonl），
文件名与扩展名不限，类别数自动回填。板端剩余空间：%(free_gb)s GB</span><br>
<div id="up_prog" style="display:none;margin:6px 0 2px">
  <progress id="up_bar" value="0" max="100" style="width:100%%;height:14px"></progress>
  <span id="up_text" class="small"></span>
</div>
</span>
<span id="ds_path" style="display:none">
<label>图片目录</label><input name="images" value="%(images)s"><br>
<label>标注 JSON</label><input name="ann" value="%(ann)s"><br>
</span>
<input type="hidden" name="uploaded_ds" value="">
<input type="hidden" name="sync_key" value="">
<h3>参数</h3>
<label>label 文件</label><input name="label" value="%(label)s" placeholder="可选：模型类别表（顺序校验）"><br>
<label>类别数</label><input name="obj_num" value="%(obj_num)s" size="6"><br>
<label>conf / threads</label><input name="conf" value="%(conf)s" size="6">
<input name="threads" value="%(threads)s" size="4"><br>
<label>数据缓存</label><input type="checkbox" name="keep_ds"> 评测完成后保留上传的数据
（默认<b>用完即删</b>防爆板端存储；勾选后同一文件夹二次评测只传差异图片）<br>
<label>对比模型</label><span id="cmp_rows"><select name="cmp" id="cmp_main" onchange="onCmpChange()"><option value="">不对比（单模型）</option>%(model_options)s</select></span>
<input class="btn sub" type="button" id="cmp_add" value="＋添加" onclick="addCmpRow()" style="width:auto;padding:4px 10px" disabled><span class="small">可加多个：N 模型并行对比，线程均分</span><br>
<h3>输出</h3>
<label>可视化</label><input type="checkbox" name="vis" checked> 渲染检测结果图（完成后展示）<br>
<label>实时画面</label><input type="checkbox" name="preview" checked> 推理时板端抽帧快照
（运行页底部展示，约 3s 一帧，FPS 略降 ~5%%；端口
<input name="preview_port" value="8090" size="5">）<br>
<br><input class="btn" type="submit" id="submit_btn" value="开始评测">
</form>
</div>
<script>
var f = document.forms[0];
var SYNC = {key: '', map: {}, uploaded: 0, total: 0, imgOk: false, annOk: false, uploading: false};
var IMG_EXT = /\.(jpg|jpeg|png|bmp|tif|tiff)$/i;
function syncStat(t) { document.getElementById('sync_stat').textContent = t; }
function dsMode(v) {
  document.getElementById('ds_sync').style.display = v == 'sync' ? '' : 'none';
  document.getElementById('ds_path').style.display = v == 'path' ? '' : 'none';
}
function filterModels(task) {
  document.querySelectorAll('select[name=model], select[name=cmp]').forEach(function(sel) {
    var first = null;
    for (var i = 0; i < sel.options.length; i++) {
      var o = sel.options[i];
      var dt = o.getAttribute('data-task') || '';
      var ok = !o.value || !dt || dt.split(' ').indexOf(task) >= 0;
      o.style.display = ok ? '' : 'none';
      if (ok && o.value && first === null) first = o;
    }
    var cur = sel.options[sel.selectedIndex];
    if (!cur || cur.style.display === 'none')
      sel.value = (sel.name === 'model' && first) ? first.value : '';
  });
}
function onTaskChange(sel) { filterModels(sel.value); }
function onCmpChange() {
  var main = document.getElementById('cmp_main');
  var add = document.getElementById('cmp_add');
  add.disabled = !main.value;
  if (!main.value) {
    var rows = document.getElementById('cmp_rows');
    var sels = rows.querySelectorAll('select[name=cmp]');
    for (var i = 1; i < sels.length; i++) rows.removeChild(sels[i].parentNode);
  }
}
function addCmpRow() {
  var rows = document.getElementById('cmp_rows');
  var sel = document.createElement('select');
  sel.name = 'cmp';
  var blank = document.createElement('option');
  blank.value = ''; blank.text = '（选择对比模型）';
  sel.appendChild(blank);
  for (var i = 0; i < f.model_sel.options.length; i++)
    if (f.model_sel.options[i].style.display !== 'none')
      sel.appendChild(f.model_sel.options[i].cloneNode(true));
  var div = document.createElement('div');
  div.style.margin = '3px 0';
  var rm = document.createElement('input');
  rm.type = 'button'; rm.value = '✕ 移除'; rm.className = 'btn sub';
  rm.style.cssText = 'width:auto;padding:2px 8px;margin-left:6px';
  rm.onclick = function() { rows.removeChild(div); };
  div.appendChild(sel); div.appendChild(rm);
  rows.appendChild(div);
}
filterModels(document.forms[0].task.value);
dsMode(document.forms[0].ds_source.value);
function dsKey(top) {
  return (top.replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 40) || 'ds') + '_' + Date.now().toString(36);
}
/* 上传期间阻止 PC 休眠（浏览器不支持则静默跳过）——原实现缺失导致 ReferenceError */
var _wakeLock = null;
function requestWake() {
  try {
    if (navigator.wakeLock && !_wakeLock)
      navigator.wakeLock.request('screen').then(function(l) { _wakeLock = l; }, function() {});
  } catch (e) {}
}
function releaseWake() {
  try { if (_wakeLock) { _wakeLock.release(); _wakeLock = null; } } catch (e) {}
}
/* 选完图片文件夹: 先与板上缓存比对清单, 只传差异(4 并发, 完成即补位), 完成即就绪 */
function syncFolder(input) {
  var files = input.files;
  if (!files || !files.length) return;
  var top = files[0].webkitRelativePath.split('/')[0];
  var list = [], manifest = [];
  SYNC.map = {};
  for (var i = 0; i < files.length; i++) {
    var rp = files[i].webkitRelativePath || files[i].name;
    var parts = rp.split('/');
    if (parts.length > 1) parts = parts.slice(1);
    rp = parts.join('/');
    if (!rp || !IMG_EXT.test(rp) || files[i].name.startsWith('.')) continue;
    SYNC.map[rp] = files[i];
    manifest.push([rp, files[i].size]);
  }
  if (!manifest.length) { syncStat('✗ 所选文件夹里没有图片（jpg/jpeg/png/bmp/tif/tiff）'); return; }
  var prevKey = f.sync_key.value;
  SYNC.key = ''; SYNC.uploading = true; SYNC.imgOk = false;
  f.sync_key.value = '';
  requestWake();
  var prog = document.getElementById('up_prog');
  var bar  = document.getElementById('up_bar');
  var txt  = document.getElementById('up_text');
  if (prog) prog.style.display = '';
  if (bar) bar.value = 0;
  syncStat('清单比对中（' + manifest.length + ' 张图片）...');
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/sync/manifest');
  xhr.onload = function() {
    var r = null;
    try { r = JSON.parse(xhr.responseText); } catch (e) {}
    if (!r || !r.ok) {
      syncStat('✗ ' + ((r && r.error) || '清单比对失败'));
      SYNC.uploading = false; releaseWake(); return;
    }
    SYNC.key = r.key; f.sync_key.value = r.key;
    if (prevKey && prevKey !== r.key) { SYNC.annOk = false; f.ann.value = ''; }  // 换了数据集, 旧标注作废
    var need = r.need || [];
    if (!need.length) {
      if (prog) prog.style.display = 'none';
      finishImages(manifest.length, r.have, 0);
      return;
    }
    syncStat('需上传 ' + need.length + ' / ' + manifest.length + ' 张（板上已有 ' + r.have + '）...');
    var totalBytes = 0;
    for (var z = 0; z < need.length; z++) totalBytes += (SYNC.map[need[z]] || {}).size || 0;
    var sentBytes = 0, done = 0, fail = 0, i = 0, act = 0, lastPaint = 0;
    var slotLoadedMap = {};   // 每连接已发字节(全局累加用)
    var paint = function(force) {
      var now = Date.now();
      if (!force && now - lastPaint < 100) return;   // 限频 100ms, 避免频繁重排
      lastPaint = now;
      var pct = totalBytes ? Math.min(100, Math.round(sentBytes / totalBytes * 100)) : 0;
      if (bar) bar.value = pct;
      if (txt) txt.textContent = pct + '%%（' + (sentBytes / 1048576).toFixed(1) +
          ' / ' + (totalBytes / 1048576).toFixed(1) + ' MB）';
      syncStat('上传图片 ' + done + ' / ' + need.length + (fail ? '（失败 ' + fail + '）' : '') + ' ...');
    };
    var onFileDone = function(ok) {
      act--; done++;
      if (!ok) fail++;
      paint(false);
      if (i < need.length) { pump(); return; }   // 关键: 每完成一个就补位, 否则只传前 4 张
      if (act > 0) return;                       // 仍有在途请求, 等它们回来
      paint(true);
      if (fail) {
        syncStat('✗ ' + fail + ' / ' + need.length + ' 张上传失败，请重新选择文件夹');
        SYNC.uploading = false; releaseWake(); return;
      }
      finishImages(manifest.length, r.have, need.length);
    };
    var pump = function() {
      while (act < 4 && i < need.length) {
        var nm = need[i++]; act++;
        uploadOne(nm, SYNC.map[nm], function(ok) { onFileDone(ok); },
                  function(loaded) {                     // 单连接字节进度 → 全局累加
          sentBytes += loaded - (slotLoadedMap[nm] || 0);
          slotLoadedMap[nm] = loaded;
          paint(false);
        });
      }
    };
    pump();
  };
  xhr.onerror = function() {
    syncStat('✗ 清单比对失败（网络中断）'); SYNC.uploading = false; releaseWake();
  };
  xhr.send(JSON.stringify({name: top, key: '', files: manifest}));
}
/* 图片就绪: 取板上权威路径(兼容所选文件夹内含 images/ 子目录) 并回填类别数 */
function finishImages(total, have, sent) {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', '/api/sync/done?key=' + encodeURIComponent(SYNC.key));
  xhr.onload = function() {
    var r = null;
    try { r = JSON.parse(xhr.responseText); } catch (e) {}
    if (r && r.ok && r.images) {
      f.images.value = r.images;
      if (r.obj_num && f.obj_num) f.obj_num.value = r.obj_num;
    } else {
      f.images.value = '/userdata/rk_eval_data/' + SYNC.key;
    }
    SYNC.imgOk = true; SYNC.uploading = false;
    var prog = document.getElementById('up_prog');
    var txt  = document.getElementById('up_text');
    if (prog) prog.style.display = 'none';
    if (txt) txt.textContent = '';
    syncStat('✓ 图片已就绪（共 ' + total + ' 张，本次新传 ' + sent + '，复用 ' + have + '）。' +
             (SYNC.annOk ? '标注已就绪，可提交' : '请选择标注文件'));
    releaseWake();
  };
  xhr.onerror = function() {
    f.images.value = '/userdata/rk_eval_data/' + SYNC.key;
    SYNC.imgOk = true; SYNC.uploading = false;
    syncStat('✓ 图片已上传（共 ' + total + ' 张）。' +
             (SYNC.annOk ? '标注已就绪，可提交' : '请选择标注文件'));
    releaseWake();
  };
  xhr.send();
}
function uploadOne(rel, file, cb, onProgress) {
  var x = new XMLHttpRequest();
  x.open('POST', '/api/sync/file?key=' + encodeURIComponent(SYNC.key) + '&p=' + encodeURIComponent(rel));
  if (onProgress && x.upload) x.upload.onprogress = function(e) {
    if (e.lengthComputable && onProgress) onProgress(e.loaded, e.total);
  };
  x.onload = x.onerror = function() { cb(x.status == 200); };
  x.send(file);
}
/* 标注: 选完立即上传到缓存根 */
function syncAnnFile(input) {
  var file = input.files[0];
  if (!file) return;
  if (!SYNC.key) { syncStat('✗ 请先选择图片文件夹'); input.value = ''; return; }
  syncStat('上传标注 ' + file.name + ' ...');
  uploadOne(file.name, file, function(ok) {
    if (ok) {
      SYNC.annOk = true;
      f.ann.value = '/userdata/rk_eval_data/' + SYNC.key + '/' + file.name;
      syncStat('✓ 标注已上传');
    } else syncStat('✗ 标注上传失败，请重选');
  });
}
/* 提交: 等图片+标注都就绪才允许(上传状态由 syncFolder/syncAnnFile 维护) */
function beforeSubmit(ev) {
  if (f.ds_source.value !== 'sync') return true;
  if (SYNC.uploading) { alert('图片还在上传中，请稍候'); ev.preventDefault(); return false; }
  if (!SYNC.imgOk) { alert('图片尚未上传，请重新选择图片文件夹'); ev.preventDefault(); return false; }
  if (!f.ann.value) { alert('请先选择标注文件'); ev.preventDefault(); return false; }
  return true;
}
function uploadDatasetZip(input) {
  var file = input.files[0];
  if (!file) return;
  var stat = document.getElementById('zip_stat');
  stat.textContent = '上传中 ' + file.name + ' (' + Math.round(file.size / 1048576) + 'MB)...';
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload?name=' + encodeURIComponent(file.name));
  xhr.onload = function() {
    try {
      var r = JSON.parse(xhr.responseText);
      if (!r.ok) { stat.textContent = '✗ ' + (r.error || '上传失败'); return; }
      if (r.images) {
        f.images.value = r.images;
        f.ann.value = r.ann || '';
        if (r.obj_num) f.obj_num.value = r.obj_num;
        if (r.label && f.elements.label) f.elements.label.value = r.label;
        f.ds_source.value = 'path';
        if (typeof dsMode === 'function') dsMode('path');
        var extra = r.ann ? '' : '（zip 内未找到单文件标注，请在下方补充标注）';
        stat.textContent = '✓ 已就绪：' + r.images + extra;
        syncStat('数据集已就绪，可直接提交评测');
      } else { stat.textContent = '✗ 上传返回异常'; }
    } catch (e) { stat.textContent = '✗ 上传失败'; }
  };
  xhr.onerror = function() { stat.textContent = '✗ 上传失败(网络)'; };
  xhr.send(file);
}
function uploadModel(input) {
  var file = input.files[0];
  if (!file) return;
  var stat = document.getElementById('model_up_stat');
  stat.textContent = '上传中 ' + file.name + ' (' + Math.round(file.size/1048576) + 'MB)...';
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload?name=' + encodeURIComponent(file.name));
  xhr.onload = function() {
    try {
      var r = JSON.parse(xhr.responseText);
      if (r.ok && r.kind == 'model') {
        var o = document.createElement('option');
        o.value = r.path; o.text = file.name; o.selected = true;
        o.setAttribute('data-task', '');
        f.model.appendChild(o);
        stat.textContent = '✓ 已就绪: ' + r.path;
      } else { stat.textContent = '✗ ' + (r.error || '上传失败'); }
    } catch (e) { stat.textContent = '✗ 上传失败'; }
  };
  xhr.send(file);
}
</script>

<p class="small">脚本化评测仍可用 CLI：本地直评 build/rknn_eval，或 PC 端 tools/eval/run_eval.py 走 ssh</p>
"""

TASKS = ["detect", "pose", "seg", "obb"]  # rtmpose 暂不开放（需要时加回）


def scan_datasets():
    """扫描已注册数据集（datasets/*.yaml，与 convert_dataset.py 同 schema）。"""
    out = []
    seen = set()
    for d in DATASET_DIRS:
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            ap = os.path.abspath(p)
            if ap in seen:
                continue
            seen.add(ap)
            meta = {"spec": ap, "name": os.path.splitext(os.path.basename(p))[0],
                    "task": "detect", "images": "", "ann": "", "label": "", "obj_num": ""}
            if yaml:
                try:
                    y = yaml.safe_load(open(p, encoding="utf-8"))
                    meta["task"] = y.get("type", "detect")
                    meta["images"] = y.get("images", "")
                    lf = y.get("labels", "")
                    # 转换产物优先：name.coco.json（水平框）
                    cand = os.path.splitext(ap)[0] + ".coco.json"
                    if os.path.exists(cand):
                        meta["ann"] = cand
                    elif lf and lf.endswith(".json") and os.path.exists(lf):
                        meta["ann"] = lf
                    if y.get("label_file"):
                        meta["label"] = y["label_file"]
                    if isinstance(y.get("names"), list):
                        meta["obj_num"] = str(len(y["names"]))
                except Exception:
                    pass
            out.append(meta)
    return out


def _grab_jpeg(port, cache_key=None):
    """从板端 MJPEG 预览的 /snapshot.jpg 抓一帧（2s 缓存），返回 JPEG 字节。"""
    ent = PREVIEW_CACHE.get(cache_key)
    if ent and time.time() - ent[0] < 1:
        return ent[1]
    if cv2 is None:
        return None
    cap = cv2.VideoCapture("http://127.0.0.1:%s/snapshot.jpg" % port)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return ent[1] if ent else None
    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok2:
        return ent[1] if ent else None
    data = buf.tobytes()
    if cache_key:
        PREVIEW_CACHE[cache_key] = (time.time(), data)
    return data


PREVIEW_LAST = {}  # 快照目录 -> 上一帧 md5（同帧不重复落盘、不推进轮转位）


def _rotating_write(d, idx, data):
    """快照轮转落盘：目录内保留最近 SHOT_SLOTS 张；与上一帧相同则跳过。"""
    h = hashlib.md5(data).hexdigest()
    if PREVIEW_LAST.get(d) == h:
        return False
    PREVIEW_LAST[d] = h
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "shot_%d.jpg" % (idx % SHOT_SLOTS)), "wb") as f:
        f.write(data)
    return True


def grab_preview(job):
    """单模型任务的板端抽帧（同时落盘到 out_dir/preview/ 供存档展示）。"""
    jid = job["id"]
    m = re.search(r"--preview-port (\d+)", job.get("cmd", ""))
    if not m and "run_eval.py" not in job.get("cmd", "") \
            and "--preview" not in job.get("cmd", ""):
        return None
    data = _grab_jpeg(m.group(1) if m else "8090", cache_key=jid)
    if data is None:
        return None
    try:
        if _rotating_write(os.path.join(job["out_dir"], "preview"),
                           PREVIEW_IDX.get(jid, 0), data):
            PREVIEW_IDX[jid] = PREVIEW_IDX.get(jid, 0) + 1
    except OSError:
        pass
    return data


def sample_previews(job):
    """并行对比任务的板端抽帧：每个模型实例一个端口，
    快照按模型分目录落盘（out_dir/preview/<tag>/），任务页同帧分列对比。"""
    for i, (tag, port) in enumerate(job.get("preview_ports") or []):
        try:
            data = _grab_jpeg(str(port), cache_key="%s_%s" % (job["id"], tag))
            if data is not None:
                key = "%s_%s" % (job["id"], tag)
                if _rotating_write(os.path.join(job["out_dir"], "preview", tag),
                                   PREVIEW_IDX.get(key, 0), data):
                    PREVIEW_IDX[key] = PREVIEW_IDX.get(key, 0) + 1
        except OSError:
            pass


def find_eval_bin():
    """返回评测二进制的绝对路径；找不到返回 ""，由调用方明确报错。

    旧实现在找不到时 fallback 成裸名 "rknn_eval"，会静默生成一条
    必然 `/bin/sh: rknn_eval: not found`（rc=127）的哑任务，页面上只显示
    日志报错、看不出是"二进制没构建"——改成显式失败。
    """
    for c in EVAL_BIN_CANDIDATES:
        if os.path.exists(c):
            return c
    return shutil.which("rknn_eval") or ""


def run_job(job):
    """执行单个任务（每个 board_key 同时最多一个 run_job 在跑）。"""
    job["state"] = "running"
    job["t_start"] = time.time()
    # 开了实时预览的任务：后台采样线程每 2s 抽帧轮转落盘——单模型写到
    # preview/，并行对比任务按模型写到 preview/<tag>/（任务页同帧分列对比）。
    stop = threading.Event()
    if job.get("preview_ports") and cv2 is not None:
        def sampler():
            while not stop.is_set() and job["state"] == "running":
                try:
                    sample_previews(job)
                except Exception:
                    pass
                stop.wait(1)  # 并行批次较短，加密采样避免错过批次窗口
        threading.Thread(target=sampler, daemon=True).start()
    elif "--preview" in job["cmd"] and cv2 is not None:
        def sampler():
            fail = 0
            while not stop.is_set() and job["state"] == "running":
                try:
                    ok = grab_preview(job) is not None
                except Exception:
                    ok = False
                # 批次间隙(模型加载/等补传)预览服务不在, 连续失败退避到 5s 再试,
                # 服务恢复后自动回到 1s 采样——避免空转的同时保证画面及时续上
                fail = 0 if ok else fail + 1
                stop.wait(5 if fail > 4 else 1)
        threading.Thread(target=sampler, daemon=True).start()
        threading.Thread(target=sampler, daemon=True).start()
    # 新进程组：终止时 killpg 连子进程（rsync/rknn_eval 等）一起带走
    logf = open(job["log_path"], "a")
    p = subprocess.Popen(job["cmd"], shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         start_new_session=True)
    with LOCK:
        job["proc"] = p
    for line in p.stdout:
        logf.write(line)
        logf.flush()
    p.wait()
    logf.close()
    stop.set()
    job["rc"] = p.returncode
    if job.get("cancel_requested"):
        job["state"] = "canceled"
    else:
        job["state"] = "done" if p.returncode == 0 else "error"
    job["t_end"] = time.time()
    save_job(job)
    # 数据缓存清理（省空间 vs 可重试的折中）：
    #   - 成功结束 → 清理缓存（若没有其他排队/运行任务共享同一缓存）；
    #   - 失败/取消 → 保留缓存，便于重试走增量复用；任务卡注说明保留原因。
    # 共享检查必须做：同一数据集 key 会被重试任务与并行任务引用，
    # 否则一个任务终止会把另一任务的批次文件连坐删除（已踩坑）。
    cd_ = job.get("cleanup_dir") or ""
    if job["state"] == "done" and cd_ and os.path.isdir(cd_):
        shared = any(
            j.get("id") != job["id"] and j.get("state") in ("queued", "running")
            and (j.get("cleanup_dir") == cd_
                 or (j.get("images") or "").startswith(cd_ + os.sep))
            for j in list(JOBS.values()))
        if shared:
            job["note"] = (job.get("note") + " | " if job.get("note") else "") + \
                "数据缓存被其他任务使用，未清理"
            save_job(job)
        else:
            shutil.rmtree(cd_, ignore_errors=True)
            if not os.path.isdir(cd_):
                job["note"] = (job.get("note") + " | " if job.get("note") else "") + \
                    "已清理上传数据缓存（勾选\"保留数据缓存\"可增量复评）"
                save_job(job)
    elif job["state"] in ("error", "canceled") and cd_:
        job["note"] = (job.get("note") + " | " if job.get("note") else "") + \
            "上传数据已保留，重试可直接复用"
    with LOCK:
        job.pop("proc", None)
        if RUNNING.get(job["board_key"]) == job["id"]:
            del RUNNING[job["board_key"]]


def cancel_job(jid):
    """人工终止：排队中的直接移除；运行中的 kill 整个进程组（含子进程）。
    注意 save_job 不能在 LOCK 内调用（其内部也拿同一把非重入锁，会死锁）。"""
    queued = False
    with LOCK:
        j = JOBS.get(jid)
        if not j:
            return None, "job 不存在"
        state = j.get("state")
        if state == "queued":
            if jid in QUEUE:
                QUEUE.remove(jid)
            j["state"] = "canceled"
            j["t_end"] = time.time()
            queued = True
        elif state == "running":
            j["cancel_requested"] = True
        else:
            return None, "任务已结束（%s），无需终止" % state
        p = j.get("proc")
    if queued:
        save_job(j)
        return jid, None
    if state == "running" and p is not None and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                p.terminate()
            except ProcessLookupError:
                pass
    return jid, None


def scheduler_loop():
    """调度：同一板串行，不同板并行。"""
    while True:
        with LOCK:
            pick = None
            for jid in QUEUE:
                job = JOBS.get(jid)
                if job and job["board_key"] not in RUNNING:
                    pick = jid
                    break
            if pick:
                QUEUE.remove(pick)
                RUNNING[JOBS[pick]["board_key"]] = pick
        if pick is None:
            time.sleep(0.3)
            continue
        threading.Thread(target=run_job, args=(JOBS[pick],), daemon=True).start()


def save_job(job):
    with LOCK:
        path = os.path.join(job["out_dir"], "web_job.json")
        os.makedirs(job["out_dir"], exist_ok=True)
        json.dump({k: job.get(k) for k in
                   ("id", "state", "cmd", "rc", "tag", "task", "created",
                    "t_start", "t_end", "log_path", "out_dir", "board_key",
                    "source", "note", "pc_out", "remote_result_dir",
                    "preview_ports")},
                  open(path, "w"), indent=2, ensure_ascii=False)


def load_model_registry():
    """model/models.json：文件名 -> 任务（字符串或数组）。
    未登记的板上模型不进网页下拉；uploads/models 的上传模型不受限。"""
    try:
        with open(os.path.join(DEFAULTS["model_dir"], "models.json"),
                  encoding="utf-8") as f:
            reg = json.load(f)
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in reg.items() if not k.startswith("_")}


def _dir_mtime(d):
    try:
        return os.path.getmtime(d)
    except OSError:
        return 0.0


def _board_key_from_cmd(cmd):
    m = re.search(r"--board (\S+)", cmd or "")
    return m.group(1) if m else "local"


def load_history():
    """重启后还原整个任务历史（不自动重跑，中断任务人工重试）。

    web_runs/、runs/ 下的 web_job.json → 原样恢复（queued/running 说明上次
    进程被杀，标为 error + 备注）；eval_runs/、runs/ 下 CLI 产物（只有
    summary_all.json）→ 合成 done 条目，详情页直接渲染其 report.md。"""
    restored = []
    for base in ("web_runs", "runs"):
        for jf in sorted(glob.glob(os.path.join(base, "*", "web_job.json"))):
            try:
                rec = json.load(open(jf, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(rec, dict):
                continue
            d = os.path.dirname(jf)
            jid = rec.get("id") or os.path.basename(d)
            note = rec.get("note", "")
            state = rec.get("state", "unknown")
            if state in ("queued", "running"):
                state, note = "error", (note or "服务重启时中断，可人工重试")
            restored.append(dict(
                id=jid, state=state, cmd=rec.get("cmd", ""), rc=rec.get("rc"),
                tag=rec.get("tag", ""), task=rec.get("task", "?"),
                created=rec.get("created") or _dir_mtime(d),
                t_start=rec.get("t_start"), t_end=rec.get("t_end"),
                log_path=rec.get("log_path") or os.path.join(d, "run.log"),
                out_dir=rec.get("out_dir") or d,
                board_key=rec.get("board_key") or _board_key_from_cmd(rec.get("cmd", "")),
                source=rec.get("source") or
                ("remote" if "--board" in (rec.get("cmd") or "") else "board"),
                note=rec.get("note", ""),
                pc_out=rec.get("pc_out", ""),
                remote_result_dir=rec.get("remote_result_dir", ""),
                preview_ports=rec.get("preview_ports")
                or _parse_preview_ports_from_script(dict(out_dir=d)),
                cleanup_dir=rec.get("cleanup_dir", ""),
                form=rec.get("form") or {}))
    for base in ("eval_runs", "runs"):
        for d in sorted(glob.glob(os.path.join(base, "*"))):
            if not os.path.isdir(d) or os.path.exists(os.path.join(d, "web_job.json")):
                continue
            if not os.path.exists(os.path.join(d, "summary_all.json")):
                continue
            jid = "cli_" + os.path.basename(d)
            task, tag, note = "?", "", "CLI 产物"
            try:
                sa = json.load(open(os.path.join(d, "summary_all.json"), encoding="utf-8"))
                task = sa.get("task", task)
                runs = sa.get("runs") or []
                if runs:
                    tag = os.path.splitext(os.path.basename(runs[0].get("model", "")))[0]
                cfg = sa.get("config") or {}
                note = "CLI 产物（历史还原）: %s" % (cfg.get("images") or os.path.basename(d))
            except (OSError, ValueError):
                pass
            restored.append(dict(
                id=jid, state="done", cmd="", rc=0, tag=tag, task=task,
                created=_dir_mtime(d), t_start=None, t_end=None,
                log_path=os.path.join(d, "run.log"), out_dir=d,
                board_key="restored", source="cli", note=note))
    restored.sort(key=lambda j: j["created"])
    with LOCK:
        for j in restored:
            if j["id"] not in JOBS:
                JOBS[j["id"]] = j
                JOB_ORDER.append(j["id"])
    return len(restored)


def _pc_ssh_opts(jid):
    """板→PC 方向的 ssh 选项：连接复用 + 保活 + 首连自动收 host key。"""
    return ("-o ControlMaster=auto "
            "-o ControlPath=/tmp/rkweb_%s_%%r@%%h:%%p "
            "-o ControlPersist=600 "
            "-o ServerAliveInterval=15 -o ServerAliveCountMax=4 "
            "-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10" % jid)


def _model_tag(path, taken):
    """模型文件名 → 唯一 tag（脚本/转储文件名安全字符集）。"""
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(path))[0]) or "m"
    t, i = tag, 2
    while t in taken:
        t = "%s_%d" % (tag, i)
        i += 1
    taken.add(t)
    return t


def _write_job_script(out_dir, task, models, images, ann_arg, ann, label, obj_num,
                      conf, threads, vis, preview, pv_base, eval_bin):
    """生成多模型并行评测脚本（bash）：全部图片一次推理，N 个模型实例并行
    （线程数 = CPU 核数 ÷ 模型数），各自独立预览端口；dump/metrics 按模型
    分目录落盘，最后一次汇总评测出对比报告。返回 (脚本路径, preview_ports)。"""
    n = len(models)
    per_th = threads if n == 1 else max(1, (os.cpu_count() or 4) // n)
    q = shlex.quote
    out_dir = os.path.abspath(out_dir)
    eval_bin = os.path.abspath(eval_bin)
    images = os.path.abspath(images)

    L = ["#!/bin/bash", "set -e",
         "OUT=%s; IMG=%s; EVAL=%s" % (q(out_dir), q(images), q(eval_bin)),
         'mkdir -p "$OUT/mparts"']
    wait_lines = []
    for i, (tag, mpath) in enumerate(models):
        a = [q(eval_bin), "--task", task, "--model", q(mpath),
             '--images "$IMG"', "--conf", str(conf), "--threads", str(per_th),
             "--dump-only", '--out-dir "$OUT/%s"' % tag,
             "--name", tag]
        if label:
            a += ["--label", q(label)]
        if obj_num:
            a += ["--obj-num", str(obj_num)]
        if preview:
            a += ["--preview", "--preview-port", str(pv_base + i)]
        if vis:
            a += ['--vis-dir "$OUT/vis/%s"' % tag, "--vis-sample", "20"]
        L.append(" ".join(a) + " & p%d=$!" % i)
        wait_lines.append("  wait $p%d || rc=1" % i)
    L = ["#!/bin/bash", "set -e",
         "OUT=%s; IMG=%s; EVAL=%s" % (q(out_dir), q(images), q(eval_bin)),
         'mkdir -p "$OUT/mparts"',
         'rc=0'] + \
        ["  " + " ".join(a) + " & p%d=$!" % i for i, (a, tag) in enumerate([])]  # placeholder
    # 重新按正确结构组织
    L = ["#!/bin/bash", "set -e",
         "OUT=%s; IMG=%s; EVAL=%s" % (q(out_dir), q(images), q(eval_bin)),
         'mkdir -p "$OUT/mparts"',
         'rc=0']
    for i, (tag, mpath) in enumerate(models):
        a = [q(eval_bin), "--task", task, "--model", q(mpath),
             '--images "$IMG"', "--conf", str(conf), "--threads", str(per_th),
             "--dump-only", '--out-dir "$OUT/%s"' % tag,
             "--name", tag]
        if label:
            a += ["--label", q(label)]
        if obj_num:
            a += ["--obj-num", str(obj_num)]
        if preview:
            a += ["--preview", "--preview-port", str(pv_base + i)]
        if vis:
            a += ['--vis-dir "$OUT/vis/%s"' % tag, "--vis-sample", "20"]
        L.append(" ".join(a) + " & p%d=$!" % i)
        wait_lines.append("  wait $p%d || rc=1" % i)
    L += wait_lines
    tags = " ".join(t for t, _ in models)
    L += ['  for t in %s; do' % tags,
          '    cat "$OUT/$t/dump_$t.jsonl" >> "$OUT/dump_$t.jsonl"',
          '    mkdir -p "$OUT/mparts/$t"',
          '    cp "$OUT/$t/metrics_$t.json" "$OUT/mparts/$t/m_$t.json" || true',
          '  done']
    # 合并 metrics
    L += ["python3 - \"$OUT\" <<'PYEOF'",
          'import json, glob, os, sys',
          'out = sys.argv[1]',
          'for d in sorted(glob.glob(os.path.join(out, "mparts", "*"))):',
          '    tag = os.path.basename(d)',
          '    fr = el = 0.0',
          '    for p in sorted(glob.glob(os.path.join(d, "*.json"))):',
          '        try:',
          '            m = json.load(open(p))["rk_pipe"]["pipeline"]',
          '            fr += m.get("processed_frames", 0)',
          '            el += m.get("elapsed_seconds", 0)',
          '        except Exception:',
          '            pass',
          '    if fr and el:',
          '        json.dump({"rk_pipe": {"pipeline": {"avg_fps": fr / el,',
          '            "processed_frames": int(fr), "elapsed_seconds": el}},',
          '            "note": "多模型并行汇总"},',
          '            open(os.path.join(out, "metrics_%s.json" % tag), "w"), indent=1)',
          'PYEOF',
          ]
    # 汇总评测
    tag0, _ = models[0]
    a = [q(eval_bin), "--task", task, "--reuse-dump", '"$OUT/dump_%s.jsonl"' % tag0,
         "--name", tag0, '--out-dir "$OUT"', "--conf", str(conf)]
    a += [ann_arg, q(ann)]
    if label:
        a += ["--label", q(label)]
    if obj_num:
        a += ["--obj-num", str(obj_num)]
    for tag, _ in models[1:]:
        a += ["--compare", "%s=dump:\"$OUT/dump_%s.jsonl\"" % (tag, tag)]
    L += ["echo '@stage evaluating 汇总评测出报告'",
          "  ".join(a),
          'echo "@stage done"']
    path = os.path.join(out_dir, "run_job.sh")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    os.chmod(path, 0o755)
    preview_ports = [[t, pv_base + i] for i, (t, _) in enumerate(models)] if preview else []
    return path, preview_ports


def start_job(form):
    """板上直评：推理 + 评测一条命令闭环（PC 浏览器提交）。

    数据来源：path 板上路径 / pc Linux PC ssh 拉取 / sync PC 文件夹同步 / upload zip。
    多模型对比（逗号分隔）或设置每批张数时，生成流水线脚本：
      图片分批搬入 work → N 模型并行 dump（线程均分）→ 跑完即释放 → 合并汇总对比报告。
    """
    jid = uuid.uuid4().hex[:8]
    out_dir = os.path.join("web_runs", jid)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")

    task = form.get("task", ["detect"])[0]
    model = form.get("model", [""])[0].strip()
    ds_source = form.get("ds_source", ["path"])[0]
    if not model or not os.path.exists(model):
        return None, "模型不存在: %s（可先在上方上传 .rknn）" % model

    else:
        images = form.get("images", [""])[0].strip()
        ann = form.get("ann", [""])[0].strip()
        if not images or not os.path.isdir(images):
            return None, "图片目录不存在: %s" % images
        if task != "obb" and (not ann or not os.path.exists(ann)):
            return None, "标注不存在: %s" % ann

    # 数据缓存保留策略：默认"用完即删"防爆板端存储；勾选保留后同一文件夹
    # 二次评测走增量只传差异。仅作用于服务管理的根（同步缓存 / zip 解包目录）。
    keep_ds = form.get("keep_ds") == ["on"]
    ds_cache_root = _cleanup_dir_for(images)      # 服务管理的数据缓存根(共享,可能被多任务引用)
    cleanup_dir = "" if (keep_ds or not ds_cache_root) else ds_cache_root
    conf = form.get("conf", ["0.001"])[0]
    threads = int(form.get("threads", ["4"])[0] or 4)
    label = form.get("label", [""])[0].strip()
    obj_num = form.get("obj_num", [""])[0].strip()
    # obj_num 未填且标注是 COCO JSON → 自动从 categories 数回填（网页易用性）
    if not obj_num and ann and ann.lower().endswith(".json") and os.path.isfile(ann):
        try:
            cj = json.load(open(ann, encoding="utf-8"))
            cats = cj.get("categories") or []
            if cats:
                obj_num = str(len(cats))
        except (OSError, ValueError):
            pass
    vis = form.get("vis") == ["on"]
    preview = form.get("preview") == ["on"]
    pv_base = int(form.get("preview_port", ["8090"])[0].strip() or 8090)
    ann_arg = "--gt" if task == "obb" else "--ann"

    # 多模型对比：主模型 + 对比模型（逗号分隔）
    taken = set()
    models = [(_model_tag(model, taken), model)]
    main_abs = os.path.abspath(model)
    seen_paths = {main_abs}
    for c in form.get("cmp", []):
        c = c.strip()
        if c and os.path.exists(c) and os.path.abspath(c) not in seen_paths:
            seen_paths.add(os.path.abspath(c))
            models.append((_model_tag(c, taken), c))
    try:
        batch = int(form.get("batch", ["0"])[0] or 0)
    except ValueError:
        batch = 0
    use_script = len(models) > 1
    note = "" 

    note = ""
    preview_ports = []
    eval_bin = find_eval_bin()
    if not eval_bin:
        return None, ("未找到评测二进制 build/rknn_eval —— 板端需先构建："
                      "cmake -B build -S . -DOPENCV_ROOT=/userdata/opencv-4.11.0-install "
                      "&& cmake --build build --target rknn_eval -j 8")
    if use_script:
        script, preview_ports = _write_job_script(
            out_dir, task, models, images, ann_arg, ann, label, obj_num,
            conf, threads, vis, preview, pv_base, eval_bin)
        cmd_s = "bash %s" % shlex.quote(script)
        bits = []
        if len(models) > 1:
            bits.append("%d 模型并行对比 × 每模型 %d 线程"
                        % (len(models), max(1, (os.cpu_count() or 4) // len(models))))
        note = " | ".join(bits)
    else:
        cmd = [eval_bin, "--task", task, "--model", model,
               "--images", images, ann_arg, ann,
               "--conf", conf, "--threads", str(threads),
               "--out-dir", out_dir, "--name", models[0][0]]
        if label:
            cmd += ["--label", label]
        if obj_num:
            cmd += ["--obj-num", obj_num]
        if len(models) > 1:
            for tag, mpath in models[1:]:
                cmd += ["--compare", "%s=%s" % (tag, mpath)]
        if vis:
            cmd += ["--vis-dir", os.path.join(out_dir, "vis"), "--vis-sample", "20"]
        if preview:
            cmd += ["--preview", "--preview-port", str(pv_base)]
        cmd_s = " ".join("'%s'" % c if " " in c else c for c in cmd)

    with LOCK:
        # form 快照: 任务页"重新提交"跳回提交页时按此预填(模型/任务/参数/数据来源)
        form_snap = dict(task=task, model=model, images=images, ann=ann,
                         label=label, obj_num=obj_num, conf=str(conf),
                         threads=str(threads), batch=str(batch),
                         ds_source=ds_source,
                                                  vis="on" if vis else "", preview="on" if preview else "",
                         keep_ds="on" if keep_ds else "",
                         preview_port=str(pv_base))
        job = dict(id=jid, state="queued", cmd=cmd_s, rc=None,
                   tag=models[0][0], task=task, created=time.time(),
                   log_path=log_path, out_dir=out_dir,
                   board_key="local", source="board", note=note,
                   preview_ports=preview_ports, cleanup_dir=cleanup_dir,
                   form=form_snap)
        JOBS[jid] = job
        JOB_ORDER.append(jid)
        QUEUE.append(jid)
    save_job(job)
    return jid, None


def delete_job(jid):
    """删除单个历史任务：内存条目 + 全部产物目录（进行中的任务不可删）。"""
    with LOCK:
        j = JOBS.get(jid)
        if not j:
            return None, "job 不存在"
        if j.get("state") in ("queued", "running"):
            return None, "任务进行中，请先终止再删除"
        JOBS.pop(jid, None)
        if jid in JOB_ORDER:
            JOB_ORDER.remove(jid)
        if jid in QUEUE:
            QUEUE.remove(jid)
        out_dir = j.get("out_dir")
    if out_dir and os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    return jid, None


def clear_history():
    """清空全部历史（含产物）；排队/运行中的任务保留。返回 (清除数, 错误列表)。"""
    with LOCK:
        targets = [jid for jid, j in JOBS.items()
                   if j.get("state") not in ("queued", "running")]
    errs = []
    for jid in targets:
        _, err = delete_job(jid)
        if err:
            errs.append("%s: %s" % (jid, err))
    return len(targets) - len(errs), errs


def job_stage(j):
    """运行中任务的子阶段：优先 run_eval --state-file 的 state.json（CLI 驱动），
    否则解析 run.log 里的 @stage 标记（PC 拉取/评测/回传流水线）。"""
    if j.get("state") != "running" or not j.get("out_dir"):
        return ""
    try:
        s = json.load(open(os.path.join(j["out_dir"], "state.json"), encoding="utf-8"))
    except (OSError, ValueError):
        s = None
    if s:
        st = s.get("state", "")
        keys = ("note", "pct", "batch", "batches", "model", "model_idx", "models", "error")
        info = " ".join("%s=%s" % (k, s[k]) for k in keys if s.get(k) not in (None, ""))
        return ("%s %s" % (st, info)).strip()
    try:
        with open(j["log_path"], errors="replace") as f:
            lines = f.readlines()[-300:]
        for line in reversed(lines):
            if line.startswith("@stage "):
                return line.strip()[7:]
    except OSError:
        pass
    return ""


def retry_job(jid):
    """人工重试（设计文档 §2：不自动重跑）：同一命令重新入队，换新 out_dir
    保留上次产物；PC 数据集模式下拉取为 rsync 增量，重试只补差异。"""
    old = JOBS.get(jid)
    if not old:
        return None, "job 不存在"
    if not old.get("cmd"):
        return None, "该任务无命令记录（CLI 历史条目不可重试）"
    # 分批消费任务(数据在 PC, 缓存按批消费): 重试=原目录断点续跑——batch_state
    # 记录已消费批号, 脚本从下一批继续; 已合并的 dump/metrics 保留不重跑。
    # 若后续批数据缺失(取消时未传完), 任务会等待批目录, 需 PC 端"补传数据"。
    if _is_batch_consume_job(old):
        if old.get("state") in ("queued", "running"):
            return None, "任务正在进行中"
        err = _requeue_resume(old)
        return (old["id"], None) if err is None else (None, err)
    new_id = uuid.uuid4().hex[:8]
    out_dir = os.path.join("web_runs", new_id)
    os.makedirs(out_dir, exist_ok=True)
    cmd = old["cmd"].replace(old["out_dir"], out_dir)
    # 脚本型任务（分批/并行）：脚本内硬编码了旧 out_dir，复制并改写路径
    old_sh = os.path.join(old["out_dir"], "run_job.sh")
    if os.path.exists(old_sh):
        new_sh = os.path.join(out_dir, "run_job.sh")
        with open(old_sh, encoding="utf-8") as f:
            txt = f.read()
        with open(new_sh, "w", encoding="utf-8") as f:
            f.write(txt.replace(old["out_dir"], out_dir))
        os.chmod(new_sh, 0o755)
    rrd = old.get("remote_result_dir") or ""
    if rrd:  # PC 结果目录里的旧任务号 → 新任务号
        new_rrd = "%s/%s" % (rrd.rsplit("/", 1)[0], new_id)
        cmd = cmd.replace(rrd, new_rrd)
        rrd = new_rrd
    job = dict(id=new_id, state="queued", cmd=cmd, rc=None,
               tag=old.get("tag", ""), task=old.get("task", "?"),
               created=time.time(), log_path=os.path.join(out_dir, "run.log"),
               out_dir=out_dir, board_key=old.get("board_key", "local"),
               source=old.get("source", "board"),
               note=old.get("note", "").replace(jid, new_id),
               pc_out=old.get("pc_out", ""), remote_result_dir=rrd,
               preview_ports=old.get("preview_ports", []))
    with LOCK:
        JOBS[new_id] = job
        JOB_ORDER.append(new_id)
        QUEUE.append(new_id)
    save_job(job)
    return new_id, None


UPLOAD_DIR = "uploads"


def safe_name(name):
    return os.path.basename(name.replace("\\", "/")).strip() or "upload.bin"


def probe_dataset(d):
    """解压/同步后的数据集目录探测：images/ 与标注 JSON（COCO/labelme 产物）。"""
    images, ann, names_f = None, None, None
    for root, dirs, files in os.walk(d):
        if images is None and "images" in dirs:
            images = os.path.join(root, "images")
        for f in files:
            lf = f.lower()
            if ann is None and lf.endswith(".json") and (
                    "instance" in lf or "coco" in lf or lf.startswith("ann")
                    or lf.endswith(".coco.json")):
                ann = os.path.join(root, f)
            if names_f is None and lf in ("data.yaml", "names.txt", "labels_list.txt"):
                names_f = os.path.join(root, f)
    if images is None:  # 容错：找第一个含图片的目录
        for root, dirs, files in os.walk(d):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                images = root
                break
    if ann is None:  # 兜底：能解析出 categories 的 JSON；仅有一个 JSON 时直取
        cands = []
        for root, _, files in os.walk(d):
            cands += [os.path.join(root, f) for f in files if f.lower().endswith(".json")]
        for c in cands:
            try:
                cj = json.load(open(c, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(cj, dict) and cj.get("categories"):
                ann = c
                break
        if ann is None and len(cands) == 1:
            ann = cands[0]
    return images, ann, names_f


# ---- PC 文件夹同步（浏览器 webkitdirectory 增量上传；Windows 零环境）----

def _valid_key(key):
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key or ""))


def _valid_relpath(p):
    p = (p or "").replace("\\", "/")
    return bool(p) and ".." not in p.split("/") and not p.startswith("/") and ":" not in p


def _free_bytes(path):
    """目录所在文件系统的剩余字节；探测失败返回 -1（不拦截）。"""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return -1


def _parse_preview_ports_from_script(job):
    """服务重启 restore 时 preview_ports 可能未持久化——从 run_job.sh 解析
    每个模型实例的 --name/--preview-port，保证对比任务重启后仍能逐列抓帧。"""
    script = os.path.join(job.get("out_dir") or "", "run_job.sh")
    ports = []
    try:
        for line in open(script, encoding="utf-8"):
            m = re.search(r"--name\s+(\S+).*?--preview-port\s+(\d+)", line)
            if m and (m.group(1), int(m.group(2))) not in ports:
                ports.append((m.group(1), int(m.group(2))))
    except OSError:
        pass
    return ports


def _shots_total_of(job):
    """任务当前已落盘的快照帧总数（单模型=preview/*.jpg；对比=preview/<tag>/*.jpg）。"""
    pv = os.path.join(job.get("out_dir") or "", "preview")
    n = 0
    try:
        for name in os.listdir(pv):
            p = os.path.join(pv, name)
            if os.path.isfile(p) and name.startswith("shot_"):
                n += 1
            elif os.path.isdir(p):
                n += len(glob.glob(os.path.join(p, "shot_*.jpg")))
    except OSError:
        pass
    return n


def _is_batch_consume_job(job):
    """分批消费任务判定: 脚本含 ROOT=/TOTAL= 即为 PC 分批上传模式。
    不依赖 cleanup_dir——服务重启 restore 的旧任务可能缺失该字段。"""
    if not job:
        return False
    try:
        with open(os.path.join(job.get("out_dir") or "", "run_job.sh"),
                  encoding="utf-8") as f:
            head = f.read(2000)
        return "TOTAL=" in head and "ROOT=" in head
    except OSError:
        return False


def _requeue_resume(job):
    """分批任务断点续跑入队: 保留 batch_state/已合并产物, 从下一批继续。"""
    if job.get("state") in ("queued", "running"):
        return "任务正在进行中"
    job["state"] = "queued"
    job["rc"] = None
    job["t_start"] = job["t_end"] = None
    job["cancel_requested"] = False   # 清取消标志, 否则完成时仍记 canceled
    job["note"] = "断点续跑（按需补传缺失批次）"
    save_job(job)
    with LOCK:
        if job["id"] not in QUEUE:
            QUEUE.append(job["id"])
    return None


def _cleanup_dir_from_script(script):
    """从 run_job.sh 的 ROOT= 提取数据缓存根(restore 旧任务 cleanup_dir 缺失时)。"""
    try:
        with open(script, encoding="utf-8") as f:
            m = re.search(r'ROOT=([^\s;]+)', f.read(4000))
        if m and os.path.isdir(m.group(1)):
            return m.group(1)
    except OSError:
        pass
    return ""


def _batch_total_of(job):
    try:
        with open(os.path.join(job.get("out_dir") or "", "run_job.sh"),
                  encoding="utf-8") as f:
            m = re.search(r"TOTAL=(\d+)", f.read(4000))
        return int(m.group(1)) if m else 0
    except (OSError, ValueError):
        return 0


def _batch_done_of(job):
    """分批消费任务当前已消费批号（读脚本写出的 batch_state；无则 -1）。"""
    try:
        with open(os.path.join(job.get("out_dir") or "", "batch_state"),
                  encoding="utf-8") as f:
            return int(f.read().strip() or -1)
    except (OSError, ValueError):
        return -1


def _cleanup_dir_for(images):
    """由 images 路径推断可清理的数据缓存根目录（仅限服务管理的两个根）。

    - 同步缓存: RK_EVAL_DATA/<key>/...  → RK_EVAL_DATA/<key>
    - zip 解包:  uploads/ds_xxx/...      → uploads/ds_xxx
    - 板上自有路径（如 model/、test_images/）返回 ""，绝不清理。
    """
    images = os.path.abspath(images or "")
    sync_root = os.path.abspath(RK_EVAL_DATA) + os.sep
    up_root = os.path.abspath(UPLOAD_DIR) + os.sep
    for root in (sync_root, up_root):
        if images.startswith(root):
            rel = images[len(root):].replace("\\", "/").split("/")[0]
            return os.path.join(root, rel) if rel else ""
    return ""


def handle_sync_manifest(body):
    """比对 PC 文件夹清单与板上缓存（按大小），返回差异清单（增量上传）。"""
    try:
        req = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return {"ok": False, "error": "manifest 解析失败"}
    files = req.get("files") or []
    if not files:
        return {"ok": False, "error": "manifest 为空"}
    key = req.get("key")
    if not _valid_key(key):
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", safe_name(req.get("name", "ds")))[:40]
        key = "%s_%s" % (name, hashlib.md5(
            json.dumps(files, sort_keys=True).encode()).hexdigest()[:6])
    root = os.path.join(RK_EVAL_DATA, key)
    need, need_bytes = [], 0
    for idx, ent in enumerate(files):
        if not isinstance(ent, (list, tuple)) or len(ent) < 2:
            continue
        p, sz = ent[0], ent[1]
        if not _valid_relpath(p):
            continue
        try:
            if os.path.getsize(os.path.join(root, p)) == sz:
                continue  # 板上已有同大小文件 → 复用
        except OSError:
            pass
        need.append(p)
        need_bytes += int(sz or 0)
    # 空间预检：需要上传的字节 + 1GB 余量，超过剩余空间直接拒绝（防爆板端存储）
    free = _free_bytes(RK_EVAL_DATA)
    if free >= 0 and free < need_bytes + (1 << 30):
        return {"ok": False, "error": "板端剩余空间不足：需上传 %.2fGB，仅剩 %.2fGB（保留 1GB 余量）。"
                 "可删除板上旧缓存后重试，或改用小数据集" % (need_bytes / 2**30, free / 2**30)}
    return {"ok": True, "key": key, "need": need, "have": len(files) - len(need),
            "free_bytes": free}


def handle_sync_file(key, rel, body):
    if not _valid_key(key) or not _valid_relpath(rel):
        return {"ok": False, "error": "非法路径"}
    fp = os.path.join(RK_EVAL_DATA, key, rel.replace("\\", "/"))
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "wb") as f:
        f.write(body)
    return {"ok": True}


def handle_sync_reset(key):
    """清空指定 key 的板上缓存（分批重传前调用，防止旧批残留干扰）。"""
    if not _valid_key(key):
        return {"ok": False, "error": "非法 key"}
    root = os.path.join(RK_EVAL_DATA, key)
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    return {"ok": True}


def handle_sync_done(key, ann_rel=None):
    if not _valid_key(key):
        return {"ok": False, "error": "非法 key"}
    root = os.path.join(RK_EVAL_DATA, key)
    if not os.path.isdir(root):
        return {"ok": False, "error": "该缓存不存在（请重新选文件夹）"}
    # 分批上传模式：图片落在 root/b0.. 子目录，标注在 root/ann.json
    b0 = os.path.join(root, "b0")
    if os.path.isdir(b0) and any(
            f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
            for f in os.listdir(b0)):
        images, ann, names_f = b0, "", ""
        if os.path.isfile(os.path.join(root, "ann.json")):
            ann = os.path.join(root, "ann.json")
    else:
        images, ann, names_f = probe_dataset(root)
    if not images:
        return {"ok": False, "error": "未找到图片（需含 images/ 目录或任意含图片的子目录）"}
    if ann_rel:  # 浏览器分选流程：标注是单独上传的固定文件
        if not _valid_relpath(ann_rel):
            return {"ok": False, "error": "非法标注路径"}
        cand = os.path.join(root, ann_rel)
        if os.path.isfile(cand):
            ann = cand
        else:
            return {"ok": False, "error": "标注尚未上传: %s" % ann_rel}
    r = {"ok": True, "key": key, "path": root, "images": images,
         "ann": ann or "", "label_hint": names_f or ""}
    if ann:
        try:
            cj = json.load(open(ann))
            if isinstance(cj, dict) and cj.get("categories"):
                r["obj_num"] = str(len(cj["categories"]))
        except (OSError, ValueError):
            pass
    return r


def handle_upload(query, body):
    name = safe_name(query.get("name", ["upload.bin"])[0])
    # 空间预检：zip + 解包约需 3 倍 zip 体积 + 1GB 余量（防爆板端存储）
    free = _free_bytes(UPLOAD_DIR)
    if free >= 0 and free < len(body) * 3 + (1 << 30):
        return {"ok": False, "error": "板端剩余空间不足：zip 上传+解包约需 %.2fGB，仅剩 %.2fGB"
                 % (len(body) * 3 / 2**30, free / 2**30)}
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, name)
    with open(dest, "wb") as f:
        f.write(body)
    if name.lower().endswith(".zip"):
        import zipfile
        out = os.path.join(UPLOAD_DIR, "ds_" + os.path.splitext(name)[0])
        os.makedirs(out, exist_ok=True)
        try:
            with zipfile.ZipFile(dest) as z:
                base = os.path.abspath(out)
                for m in z.namelist():
                    target = os.path.abspath(os.path.join(out, m))
                    if not target.startswith(base + os.sep):
                        return {"ok": False, "error": "zip 内路径非法"}
                z.extractall(out)
        except zipfile.BadZipFile:
            return {"ok": False, "error": "非法 zip"}
        images, ann, names_f = probe_dataset(out)
        if not images:
            return {"ok": False, "error": "zip 中未找到图片目录（需含 images/）"}
        r = {"ok": True, "path": out, "images": images,
             "ann": ann or "", "label_hint": names_f or ""}
        if ann:
            try:
                cj = json.load(open(ann))
                if isinstance(cj, dict) and cj.get("categories"):
                    r["obj_num"] = str(len(cj["categories"]))
            except Exception:
                pass
        return r
    if name.lower().endswith(".rknn"):
        # 模型归位 uploads/models/，提交页下拉即可扫到
        mdir = os.path.join(UPLOAD_DIR, "models")
        os.makedirs(mdir, exist_ok=True)
        final = os.path.join(mdir, name)
        if os.path.abspath(final) != os.path.abspath(dest):
            os.replace(dest, final)
        return {"ok": True, "path": final, "kind": "model"}
    return {"ok": True, "path": dest}


def tail(path, n=40):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def compare_page(qs):
    """跨任务对比页：扫 web_runs/ + eval_runs/ 的 summary，CSS 条形图对比。"""
    from compare_runs import load_runs
    dirs = sorted(glob.glob("web_runs/*/") + glob.glob("eval_runs/*/"))
    rows = [r for r in load_runs([d.rstrip("/") for d in dirs])
            if qs.get("kind", [r["kind"]])[0] == r["kind"]]
    if not rows:
        return ("<h2>历史对比</h2><p class=small>暂无已完成任务。"
                "支持目录: web_runs/*/ 与 eval_runs/*/</p>"
                "<p><a href='/jobs'>← 队列</a></p>")
    kinds = sorted({r["kind"] for r in load_runs([d.rstrip("/") for d in dirs])})
    kind = qs.get("kind", [kinds[0]])[0]
    nav = " | ".join("<a href='/compare?kind=%s'%s>%s</a>" %
                     (k, " style='font-weight:bold'" if k == kind else "", k) for k in kinds)
    mkeys = ["AP", "AP50", "AP75", "AP_medium", "AP_large", "AR_20", "AR_max", "mAP50", "mAP50_95"]
    shown = [k for k in mkeys if any(k in r["s"] for r in rows)]
    maxes = {k: max((float(r["s"][k]) for r in rows if k in r["s"]), default=0) or 1 for k in shown}
    maxfps = max((r["fps"] for r in rows), default=1) or 1
    html_rows = []
    for r in rows:
        cells = ["<td>%s/%s</td>" % (html.escape(r["dir"]), html.escape(r["tag"]))]
        for k in shown:
            if k not in r["s"]:  # 不同任务类型指标集合不同（如 kpts 无 AP_small）
                cells.append("<td>-</td>")
                continue
            v = float(r["s"][k]) * 100
            bar = max(4, int(v / (maxes[k] * 100) * 120))
            cells.append("<td>%.1f<div style='background:#4a7de0;height:6px;width:%dpx'></div></td>"
                         % (v, bar))
        cells.append("<td>%.1f<div style='background:#0a9d52;height:6px;width:%dpx'></div></td>"
                     % (r["fps"], max(4, int(r["fps"] / maxfps * 120))))
        html_rows.append("<tr>" + "".join(cells) + "</tr>")
    return ("<h2>历史对比 — %s</h2><meta http-equiv=refresh content=30>"
            "<p>%s</p>"
            "<table><tr><th>任务/模型</th>%s<th>FPS</th></tr>%s</table>"
            "<p class=small>条形长度 = 相对本页最大值。精度条蓝、速度条绿。</p>"
            "<p><a href='/jobs'>← 队列</a> | <a href='/'>+ 新评测</a></p>"
            % (html.escape(kind), nav,
               "".join("<th>%s</th>" % k for k in shown),
               "".join(html_rows)))


def mdish(text):
    """极简 markdown 渲染：表格/标题/代码块/加粗。"""
    out, in_code, in_table = [], False, False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("|"):
            if not in_table:
                out.append('<table>'); in_table = True
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "th" if not out or "<table>" in out[-1] else "td"
            out.append("<tr>" + "".join("<%s>%s</%s>" % (tag, html.escape(c), tag) for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>"); in_table = False
        m = re.match(r"^(#{1,4}) (.*)", line)
        if m:
            out.append("<h%d>%s</h%d>" % (min(4, len(m.group(1)) + 1), html.escape(m.group(2)), min(4, len(m.group(1)) + 1)))
            continue
        out.append(html.escape(line) + "<br>")
    if in_table:
        out.append("</table>")
    return "\n".join(out)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if ctype.startswith("text/html") or "javascript" in ctype:
            self.send_header("Cache-Control", "no-store")  # 页面/JS 迭代频繁,禁缓存防旧脚本
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            opts = "".join("<option%s>%s</option>" % (" selected" if t == DEFAULTS["task"] else "", t) for t in TASKS)
            m_opts, seen = [], set()
            reg = load_model_registry()
            for md, limited in [(DEFAULTS["model_dir"], True),
                                (os.path.join(UPLOAD_DIR, "models"), False)]:
                if not os.path.isdir(md):
                    continue
                for f in sorted(os.listdir(md)):
                    if not f.endswith(".rknn") or f in seen:
                        continue
                    seen.add(f)
                    tasks = reg.get(f, [])
                    if isinstance(tasks, str):
                        tasks = [tasks]
                    if limited and not tasks:
                        continue  # 未登记的板上模型不进下拉
                    dt = "" if not limited else " ".join(tasks)
                    sel = " selected" if not m_opts else ""
                    m_opts.append("<option value=\"%s\" data-task=\"%s\"%s>%s</option>" %
                                  (html.escape(os.path.join(md, f), quote=True),
                                   html.escape(dt, quote=True), sel, html.escape(f)))
            free_b = _free_bytes(DEFAULTS["model_dir"])
            page = FORM % {**DEFAULTS, "task_options": opts,
                           "model_options": "".join(m_opts),
                           "free_gb": "%.1f" % (free_b / 2**30) if free_b >= 0 else "?"}
            self._send(200, render_page(page))
        elif u.path == "/jobs":
            rows = []
            for jid in reversed(JOB_ORDER[-200:]):
                j = JOBS[jid]
                cls = {"done": "ok", "error": "err", "running": "run",
                       "canceled": "canceled"}.get(j["state"], "q")
                dur = (j.get("t_end") or time.time()) - (j.get("t_start") or j["created"])
                created = time.strftime("%m-%d %H:%M", time.localtime(j["created"]))
                act = ""
                if j["state"] in ("queued", "running"):
                    act = ("<form method=post action=/cancel style='display:inline'>"
                           "<input type=hidden name=jid value=%s>"
                           "<input class='btn sub' type=submit value=终止></form>" % jid)
                elif j["state"] in ("error", "canceled") and j.get("cmd") and not _is_batch_consume_job(j):
                    act = ("<form method=post action=/retry style='display:inline'>"
                           "<input type=hidden name=jid value=%s>"
                           "<input class='btn sub' type=submit value=重试></form>" % jid)
                if j["state"] not in ("queued", "running"):
                    act += ("<form method=post action=/delete style='display:inline' "
                            "onsubmit=\"return confirm('删除该任务的全部产物？')\">"
                            "<input type=hidden name=jid value=%s>"
                            "<input class='btn sub' type=submit value=删除></form>" % jid)
                rows.append("<tr><td><a href='/job/%s'>%s</a></td><td class=small>%s</td>"
                            "<td>%s <code>%s</code></td>"
                            "<td><span class='badge %s'>%s</span></td><td class=small>%s</td>"
                            "<td>%.0fs</td><td class=small>%s</td><td>%s</td></tr>"
                            % (jid, jid, html.escape(j.get("source", "")),
                               html.escape(j.get("task", "?")), html.escape(j.get("tag", "")),
                               cls, j["state"], html.escape(job_stage(j) or j.get("note", "")),
                               dur, created, act))
            body = render_page("<div class=card><h2>任务队列 / 全量历史</h2>"
                           "<meta http-equiv=refresh content=5>"
                           "<p class=small>重启后自动从 web_runs/（web_job.json）与 "
                           "eval_runs/、runs/（CLI 产物）还原；中断任务标红，可人工重试。</p>"
                           "<table><tr><th>job</th><th>来源</th><th>任务/模型</th><th>状态</th>"
                           "<th>阶段/备注</th><th>耗时</th><th>创建</th><th></th></tr>"
                           + "".join(rows) + "</table></div>"
                           "<p><a class=btn href='/'>+ 新评测</a> "
                           "<form method=post action=/clear style='display:inline' "
                           "onsubmit=\"return confirm('清空全部历史及产物？进行中的任务不受影响')\">"
                           "<input class='btn sub' type=submit value=清空历史></form></p>")
            self._send(200, body)
        elif u.path.startswith("/api/job/"):
            jid = u.path.split("/")[3]
            j = JOBS.get(jid)
            if not j:
                self._send(404, json.dumps({"error": "no such job"}), "application/json")
                return
            r = {"id": jid, "state": j.get("state"), "task": j.get("task"),
                 "tag": j.get("tag"), "stage": job_stage(j),
                 "elapsed": round((j.get("t_end") or time.time())
                                  - (j.get("t_start") or j["created"]), 1),
                 "report_ready": os.path.exists(os.path.join(j["out_dir"], "report.md")),
                 "note": j.get("note", ""),
                 "batch_done": _batch_done_of(j),
                 "shots_total": _shots_total_of(j),
                 "form": j.get("form") or {}}
            self._send(200, json.dumps(r, ensure_ascii=False), "application/json")
        elif u.path == "/api/sync/refill-done":
            q = parse_qs(u.query)
            rj = q.get("resume", [""])[0]
            r = {"ok": False}
            job = JOBS.get(rj)
            if job and _is_batch_consume_job(job):
                err = None
                if job.get("state") in ("error", "canceled"):
                    err = _requeue_resume(job)
                if err:
                    r = {"ok": False, "error": err}
                else:
                    r = {"ok": True, "state": job.get("state")}
            elif job:
                r = {"ok": True, "state": job.get("state")}
            else:
                r = {"ok": False, "error": "任务不存在"}
            self._send(200, json.dumps(r), "application/json")
        elif u.path == "/api/sync/reset":
            q = parse_qs(u.query)
            r = handle_sync_reset(q.get("key", [""])[0])
            self._send(200, json.dumps(r), "application/json")
        elif u.path == "/api/sync/done":
            q = parse_qs(u.query)
            r = handle_sync_done(q.get("key", [""])[0], q.get("ann", [""])[0])
            self._send(200, json.dumps(r, ensure_ascii=False), "application/json")
        elif u.path == "/compare":
            self._send(200, render_page(compare_page(parse_qs(u.query))))
        elif u.path.endswith("/preview.jpg"):
            jid = u.path.split("/")[2]
            j = JOBS.get(jid)
            data = grab_preview(j) if j else None
            if data:
                self._send(200, data, "image/jpeg")
            else:
                self._send(404, "no preview", "text/plain")
        elif u.path.startswith("/vis/"):
            # 任务可视化：/vis/<jid>/<file> 或并行对比的 /vis/<jid>/<tag>/<file>
            parts = u.path.split("/")
            if len(parts) in (4, 5) and parts[1] == "vis" and JOBS.get(parts[2]):
                rel = "/".join(parts[3:])
                f = os.path.join(JOBS[parts[2]]["out_dir"], "vis",
                                 os.path.basename(parts[3]) if len(parts) == 4
                                 else parts[3] + "/" + os.path.basename(parts[4]))
                if os.path.isfile(f):
                    self._send(200, open(f, "rb").read(), "image/jpeg")
                    return
            self._send(404, "no image", "text/plain")
        elif u.path.startswith("/files/"):
            # 任务产物通用访问：/files/<jid>/<相对路径>（快照流/报告附件等，支持多级）
            parts = u.path.split("/")
            if len(parts) >= 4 and JOBS.get(parts[2]):
                rel = "/".join(parts[3:])
                base = os.path.abspath(JOBS[parts[2]]["out_dir"])
                fp = os.path.abspath(os.path.join(base, rel))
                if fp.startswith(base + os.sep) and os.path.isfile(fp):
                    ctype = ("image/jpeg" if fp.endswith((".jpg", ".jpeg")) else
                             "image/png" if fp.endswith(".png") else
                             "text/plain; charset=utf-8")
                    self._send(200, open(fp, "rb").read(), ctype)
                    return
            self._send(404, "no file", "text/plain")
        elif u.path.endswith("/log.txt"):
            jid = u.path.split("/")[2]
            j = JOBS.get(jid)
            if not j:
                self._send(404, "no job", "text/plain"); return
            self._send(200, tail(j["log_path"]).encode(), "text/plain; charset=utf-8")
        elif u.path.startswith("/job/"):
            jid = u.path.split("/")[2]
            j = JOBS.get(jid)
            if not j:
                self._send(404, render_page("job 不存在")); return
            extra = ""
            if j["state"] in ("done", "error"):
                rep = os.path.join(j["out_dir"], "report.md")
                if os.path.exists(rep):
                    extra = ("<div class=card><h2>评测报告</h2>"
                              + mdish(open(rep, errors="replace").read()) + "</div>")
                    vis_dir = os.path.join(j["out_dir"], "vis")
                    if os.path.isdir(vis_dir):
                        vis_subs = sorted(d for d in os.listdir(vis_dir)
                                          if os.path.isdir(os.path.join(vis_dir, d)))
                        if vis_subs:
                            # 并行对比：每个模型一列检测结果图（同数据不同模型）
                            cols = ""
                            for t in vis_subs:
                                imgs = sorted(glob.glob(os.path.join(vis_dir, t, "*_vis.jpg")))[:6]
                                if imgs:
                                    cell = "".join(
                                        "<a href='/vis/%s/%s/%s'><img src='/vis/%s/%s/%s'></a>"
                                        % (jid, t, html.escape(os.path.basename(i)),
                                           jid, t, html.escape(os.path.basename(i)))
                                        for i in imgs)
                                    cols += ("<div class=pcol><b class=small>%s</b>%s</div>"
                                             % (html.escape(t), cell))
                            if cols:
                                extra += ("<div class=card><h2>检测结果对比（同数据不同模型，每模型前 6 张）</h2>"
                                          "<div class=pcmp>%s</div></div>" % cols)
                        else:
                            imgs = sorted(glob.glob(os.path.join(vis_dir, "*_vis.jpg")))[:8]
                            if imgs:
                                cell = "".join(
                                    "<a href='/vis/%s/%s'><img src='/vis/%s/%s'></a>"
                                    % (j["id"], html.escape(os.path.basename(i)),
                                       j["id"], html.escape(os.path.basename(i)))
                                    for i in imgs)
                                extra += ("<div class=card><h2>检测可视化（前 %d 张，共 %d 张）</h2>"
                                          "<div class=visgrid>%s</div></div>"
                                          % (len(imgs),
                                             len(glob.glob(os.path.join(vis_dir, "*_vis.jpg"))), cell))
                else:
                    extra = ("<div class=card><h2>结果</h2><pre>"
                             + html.escape(tail(j["log_path"], 30)) + "</pre></div>")
            shots = ""
            pv_dir = os.path.join(j["out_dir"], "preview")
            n_shots = len(glob.glob(os.path.join(pv_dir, "shot_*.jpg")))
            ports = j.get("preview_ports") or []
            sub_tags = sorted(d for d in os.listdir(pv_dir)
                              if os.path.isdir(os.path.join(pv_dir, d))) \
                if os.path.isdir(pv_dir) else []
            running_pv = j["state"] == "running" and (
                bool(ports) or "--preview" in j.get("cmd", ""))
            if running_pv and ports:
                # 并行对比：每个模型一列，同帧数据分列对比。
                # 只渲染已落盘的帧——空槽不再渲染成黑框；首批画面未出时给占位提示
                cols = ""
                for tag, _port in ports:
                    pvt = os.path.join(pv_dir, tag)
                    imgs = "".join(
                        "<img src='/files/%s/preview/%s/shot_%d.jpg'>"
                        % (jid, tag, i) for i in range(SHOT_SLOTS)
                        if os.path.exists(os.path.join(pvt, "shot_%d.jpg" % i)))
                    if not imgs:
                        imgs = ("<div class=small style='padding:8px;color:#888'>"
                                "等待首批画面…（批数据同步/模型加载中）</div>")
                    cols += ("<div class=pcol><b class=small>%s</b>%s</div>"
                             % (html.escape(tag), imgs))
                shots = ("<div class=card><h2>推理画面（多模型同帧对比）"
                         "<span class=small>（每模型一列，帧随推理实时追加）</span></h2>"
                         "<div class=pcmp id=shots>%s</div></div>" % cols)
            elif running_pv:
                slots = "".join(
                    "<img src='/files/%s/preview/shot_%d.jpg'>"
                    % (jid, i) for i in range(SHOT_SLOTS)
                    if os.path.exists(os.path.join(pv_dir, "shot_%d.jpg" % i)))
                if not slots:
                    slots = ("<div class=small style='padding:8px;color:#888'>"
                             "等待首批画面…（批数据同步/模型加载中）</div>")
                shots = ("<div class=card><h2>推理画面"
                         "<span class=small>（板端抽帧快照，帧随推理实时追加）</span></h2>"
                         "<div class=shots id=shots>%s</div></div>" % slots)
            elif j["state"] in ("done", "error", "canceled") and sub_tags:
                # 运行结束也保留快照流（否则页面刷新后画面消失，像"没拍过"）
                cols = ""
                for t in sub_tags:
                    imgs = "".join(
                        "<img src='/files/%s/preview/%s/shot_%d.jpg'>"
                        % (jid, t, i) for i in range(SHOT_SLOTS)
                        if os.path.exists(os.path.join(pv_dir, t, "shot_%d.jpg" % i)))
                    if imgs:
                        cols += ("<div class=pcol><b class=small>%s</b>%s</div>"
                                 % (html.escape(t), imgs))
                if cols:
                    shots = ("<div class=card><h2>推理画面（多模型同帧对比）"
                             "<span class=small>（运行期间抽帧存档）</span></h2>"
                             "<div class=pcmp>%s</div></div>" % cols)
            elif n_shots and j["state"] in ("done", "error", "canceled"):
                imgs = "".join(
                    "<img src='/files/%s/preview/shot_%d.jpg'>"
                    % (jid, i) for i in range(SHOT_SLOTS)
                    if os.path.exists(os.path.join(pv_dir, "shot_%d.jpg" % i)))
                shots = ("<div class=card><h2>推理画面"
                         "<span class=small>（运行期间抽帧存档 %d 张）</span></h2>"
                         "<div class=shots>%s</div></div>" % (n_shots, imgs))
            retry_btn = ""
            if j["state"] in ("error", "canceled") and j.get("cmd"):
                if _is_batch_consume_job(j):
                    # 分批任务: 不整单重试, 补传缺失批次后自动续跑
                    retry_btn = ("<a class='btn sub' href='/?resume=%s' "
                                 "style='text-decoration:none'>补传缺失批次并续跑</a>" % jid)
                else:
                    retry_btn = ("<form method=post action=/retry style='display:inline'>"
                                 "<input type=hidden name=jid value=%s>"
                                 "<input class='btn sub' type=submit value='重试此任务'></form>" % jid)
            cancel_btn = ""
            if j["state"] in ("queued", "running"):
                cancel_btn = ("<form method=post action=/cancel style='display:inline'>"
                              "<input type=hidden name=jid value=%s>"
                              "<input class='btn sub' type=submit value='终止任务'></form>" % jid)
            badge_cls = {"done": "ok", "error": "err", "running": "run",
                         "canceled": "canceled"}.get(j["state"], "q")
            # 分批消费任务正在等批(驱动随页面关闭/刷新丢失): 提供补传入口
            refill = ""
            if j["state"] in ("queued", "running") and _is_batch_consume_job(j):
                bd = _batch_done_of(j)
                tot = _batch_total_of(j)
                if tot > 0 and bd + 1 < tot:
                    refill = ("<p class=small style='color:#b06000'>任务正在等待批次数据"
                              "（已消费 %d/%d 批）。若上传页面已关闭，请"
                              "<a href='/?resume=%s'>重新选择数据集文件夹补传缺失批次</a>"
                              "（只传缺失部分，任务自动继续）</p>" % (bd + 1, tot, jid))
            stage_line = job_stage(j)
            head = (
                "<div class=card><h2>任务 <code>%s</code> "
                "<span class='badge %s' id=state>%s</span> "
                "<span class=small>[%s]</span></h2>"
                "<p class=small id=stage>%s</p>"
                "<p class=small>耗时 <span id=elapsed>%.0f</span> s</p>%s"
                "<details class=cmd><summary>命令</summary>"
                "<pre>%s</pre></details>%s</div>"
                % (jid, badge_cls, j["state"], html.escape(j.get("source", "")),
                   html.escape("阶段: " + stage_line) if stage_line else "",
                   (j.get("t_end") or time.time()) - (j.get("t_start") or j["created"]),
                   "<p class='small' style='color:var(--err)'>%s</p>" % html.escape(j["note"])
                   if j.get("note") else "",
                   html.escape(j.get("cmd", "")), cancel_btn + " " + retry_btn))
            # 局部刷新：状态/阶段/日志/快照流每 3s 拉一次，完成/失败时整页刷新出报告
            script = """
<script>
var jid = '%s';
var lastShots = -1;
var timer = setInterval(function() {
  fetch('/api/job/' + jid).then(function(r) { return r.json(); }).then(function(j) {
    var st = document.getElementById('state');
    if (st) { st.textContent = j.state; st.className = 'badge ' +
      ({done:'ok', error:'err', running:'run', canceled:'canceled'})[j.state] || 'q'; }
    var sg = document.getElementById('stage');
    if (sg && j.stage) sg.textContent = '阶段: ' + j.stage;
    var el = document.getElementById('elapsed');
    if (el) el.textContent = j.elapsed;
    if (j.state == 'running') {
      fetch('/job/' + jid + '/log.txt').then(function(r) { return r.text(); })
        .then(function(t) { var e = document.getElementById('log');
                            if (e) e.textContent = t; });
      // 新帧落盘(shots_total 增长)时局部刷新画面区: 补上新增的槽, 旧帧加时间戳破缓存
      if (typeof j.shots_total === 'number' && j.shots_total !== lastShots) {
        lastShots = j.shots_total;
        fetch('/job/' + jid).then(function(r) { return r.text(); })
          .then(function(html2) {
            var doc = new DOMParser().parseFromString(html2, 'text/html');
            var sh = doc.querySelector('#shots');
            if (sh) {
              var cur = document.querySelector('#shots');
              if (cur) cur.outerHTML = sh.outerHTML;
            }
          });
      } else {
        var t = Date.now();
        document.querySelectorAll('#shots img').forEach(function(im) {
          im.classList.remove('empty');
          im.src = im.src.split('?')[0] + '?t=' + t;
        });
      }
    } else if (j.state == 'done' || j.state == 'error' || j.state == 'canceled') {
      clearInterval(timer);
      fetch('/job/' + jid).then(function(r) { return r.text(); })
        .then(function(html) {
          var doc = new DOMParser().parseFromString(html, 'text/html');
          var m = doc.body.querySelector('main');
          if (m) document.body.querySelector('main').innerHTML = m.innerHTML;
        });
    }
  }).catch(function() {});
}, 2000);
</script>""" % jid
            log_card = ("<div class=card><h2>日志</h2><pre id=log>%s</pre></div>"
                        % html.escape(tail(j["log_path"])))
            body = render_page(head + shots + log_card + extra + script +
                           "<p><a href='/jobs'>← 队列 / 历史</a></p>")
            self._send(200, body)
        else:
            self._send(404, render_page("404"))

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/create":
            n = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(n).decode())
            jid, err = start_job(form)
            if err:
                self._send(400, render_page("提交失败: %s<p><a href='/'>返回</a></p>" % html.escape(err)))
                return
            self.send_response(302)
            self.send_header("Location", "/job/" + jid)
            self.end_headers()
        elif u.path == "/retry":
            n = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(n).decode())
            new_id, err = retry_job(form.get("jid", [""])[0])
            if err:
                self._send(400, render_page("重试失败: %s<p><a href='/jobs'>返回</a></p>" % html.escape(err)))
                return
            self.send_response(302)
            self.send_header("Location", "/job/" + new_id)
            self.end_headers()
        elif u.path == "/cancel":
            n = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(n).decode())
            jid, err = cancel_job(form.get("jid", [""])[0])
            if err:
                self._send(400, render_page("终止失败: %s<p><a href='/jobs'>返回</a></p>" % html.escape(err)))
                return
            self.send_response(302)
            self.send_header("Location", "/job/" + jid)
            self.end_headers()
        elif u.path == "/delete":
            n = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(n).decode())
            jid, err = delete_job(form.get("jid", [""])[0])
            if err:
                self._send(400, render_page("删除失败: %s<p><a href='/jobs'>返回</a></p>" % html.escape(err)))
                return
            self.send_response(302)
            self.send_header("Location", "/jobs")
            self.end_headers()
        elif u.path == "/clear":
            n = int(self.headers.get("Content-Length", 0))
            clear_history()
            self.send_response(302)
            self.send_header("Location", "/jobs")
            self.end_headers()
        elif u.path == "/api/upload":
            q = parse_qs(u.query)
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n < 512 * 1048576 else b""
            if not body and n:
                self._send(413, json.dumps({"ok": False, "error": "文件过大(>512MB)"}), "application/json")
                return
            r = handle_upload(q, body)
            self._send(200, json.dumps(r), "application/json")
        elif u.path == "/api/sync/manifest":
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n < 16 * 1048576 else b""
            self._send(200, json.dumps(handle_sync_manifest(body)), "application/json")
        elif u.path == "/api/sync/file":
            q = parse_qs(u.query)
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n < 512 * 1048576 else b""
            if not body and n:
                self._send(413, json.dumps({"ok": False, "error": "文件过大(>512MB)"}), "application/json")
                return
            r = handle_sync_file(q.get("key", [""])[0], q.get("p", [""])[0], body)
            self._send(200, json.dumps(r), "application/json")
        elif u.path == "/api/sync/reset":
            q = parse_qs(u.query)
            r = handle_sync_reset(q.get("key", [""])[0])
            self._send(200, json.dumps(r), "application/json")
        elif u.path == "/api/sync/done":
            q = parse_qs(u.query)
            r = handle_sync_done(q.get("key", [""])[0], q.get("ann", [""])[0])
            self._send(200, json.dumps(r, ensure_ascii=False), "application/json")
        else:
            self._send(404, render_page("404"))


def board_urls(port):
    """本机所有 IPv4（供 PC 浏览器访问的地址提示）。"""
    urls = []
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True,
                             text=True, timeout=2).stdout.split()
        urls = ["http://%s:%d" % (ip, port) for ip in out]
    except Exception:
        pass
    return urls or ["http://127.0.0.1:%d" % port]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()
    n_restored = load_history()
    t = threading.Thread(target=scheduler_loop, daemon=True)
    t.start()
    print("rknn_eval_web 评测控制台（板端服务）")
    print("  PC 浏览器打开: %s" % "  或  ".join(board_urls(args.port)))
    print("  评测二进制: %s | 注册数据集: datasets/*.yaml | 历史任务: %d"
          % (find_eval_bin() or "未构建(缺失 build/rknn_eval)", n_restored))
    sys.stdout.flush()
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
