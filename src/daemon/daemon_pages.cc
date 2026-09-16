// 监控页与事件时间线页（自 daemon_main.cc 拆出）：纯 HTML 字符串，无全局状态依赖。
#include <string>

#include "daemon/daemon_internal.h"

// 事件时间线页（/timeline，D4/M10）：轮询 /api/events，按任务/类型/复核结论/时间窗过滤，
// 顶部统计条 + 新到旧事件列表（快照/片段路径、复核徽章、原始 JSON 可展开）
std::string timelineHtml() {
    return R"HTML(<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rk_pipe 事件时间线</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: "SF Mono", Menlo, Consolas, monospace; background: #11141a; color: #d7dde8; margin: 0; padding: 16px; }
  h1 { font-size: 18px; margin: 0 0 12px; color: #8ab4ff; }
  a { color: #8ab4ff; }
  .toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  input, select, button { background: #1c2230; color: #d7dde8; border: 1px solid #2c3547; border-radius: 6px; padding: 6px 10px; font-size: 13px; }
  .hint { color: #5a6478; font-size: 12px; }
  .chip { padding: 2px 10px; border-radius: 11px; font-size: 12px; background: #232b3c; color: #8a93a6; }
  .chip b { color: #d7dde8; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
  .evcard { background: #161b26; border: 1px solid #262f42; border-radius: 8px; padding: 8px 12px; margin-bottom: 6px; font-size: 13px; }
  .evcard.false { border-color: #7a3a3a; }
  .evrow { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
  .evt { color: #5a6478; font-size: 12px; width: 82px; flex-shrink: 0; }
  .evtask { color: #8ab4ff; flex-shrink: 0; }
  .badge { padding: 1px 8px; border-radius: 9px; font-size: 11px; flex-shrink: 0; }
  .badge.ok { background: #1f4d2f; color: #7ce38b; }
  .badge.bad { background: #4d1f1f; color: #ff8a8a; }
  .badge.warn { background: #4d3a1f; color: #ffce7a; }
  .badge.plain { background: #232b3c; color: #8a93a6; }
  #ruleModal { position: fixed; inset: 0; background: rgba(0,0,0,.6); display: none; align-items: center; justify-content: center; z-index: 60; }
  #ruleBox { width: min(880px, 94vw); background: #161b26; border: 1px solid #2c3547; border-radius: 10px; }
  #ruleHead { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-bottom: 1px solid #262f42; }
  #ruleBody { padding: 10px 12px; }
  #ruleCanvasWrap { position: relative; width: 100%; }
  #ruleImg { width: 100%; display: block; background: #000; border-radius: 6px; }
  #ruleCanvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; cursor: crosshair; }
  #ruleCtl { display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }
  .path { color: #5a6478; font-size: 11px; word-break: break-all; }
  details { margin-top: 4px; }
  details pre { margin: 4px 0 0; font-size: 11px; color: #b8c2d4; white-space: pre-wrap; word-break: break-all; background: #0d1016; padding: 6px 8px; border-radius: 6px; }
  #loadmore { display: block; margin: 10px auto; }
</style>
</head>
<body>
<h1>rk_pipe 事件时间线 <span class="hint">（<a href="/">返回监控面板</a>）</span></h1>
<div class="toolbar">
  <select id="window"><option value="3600000">近 1 小时</option><option value="21600000">近 6 小时</option><option value="86400000" selected>近 24 小时</option><option value="0">全部</option></select>
  <select id="verdict"><option value="">全部结论</option><option value="true">可信</option><option value="false">误报</option><option value="uncertain">存疑</option><option value="none">未复核</option></select>
  <select id="type"><option value="">全部类型</option><option value="line_cross">line_cross</option><option value="intrusion">intrusion</option><option value="dwell">dwell</option><option value="absence">absence</option><option value="crowd">crowd</option><option value="abandoned">abandoned</option><option value="fall">fall</option><option value="near_distance">near_distance</option><option value="count">count</option></select>
  <input id="task" placeholder="任务名（留空=全部）" size="14">
  <select id="limit"><option>200</option><option>500</option><option value="1000">1000（最大）</option></select>
  <label class="hint"><input type="checkbox" id="auto" checked style="vertical-align:-2px"> 5s 自动刷新</label>
  <button onclick="load()">刷新</button>
</div>
<div class="chips" id="chips"><span class="hint">加载中…</span></div>
<div id="list"></div>
<button id="loadmore" onclick="more()">加载更早事件</button>
<div class="hint" style="text-align:center">数据源 GET /api/events（内存环形缓冲 + JSONL 落盘，重启回放恢复）</div>
<script>
let fetchLimit = 200;
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = ts => { const d = new Date(Number(ts)); const p = n => String(n).padStart(2,'0'); return `${p(d.getMonth()+1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`; };
function evType(b) { return b.rule_type || b.event || 'event'; }
function verdictBadge(it) {
  if (!it.review) return '<span class="badge plain">未复核</span>';
  const v = it.review.verdict;
  const cls = v === 'true' ? 'ok' : v === 'false' ? 'bad' : 'warn';
  const label = v === 'true' ? '可信' : v === 'false' ? '误报' : v === 'uncertain' ? '存疑' : v;
  const src = it.review.source === 'llm' ? 'LLM' : '人工';
  return `<span class="badge ${cls}">${src}·${esc(label)}</span>`;
}
function evDetail(b) {
  if (Array.isArray(b.events) && b.events.length) {
    return b.events.map(e => esc(e.type || '') + (e.rule ? '@' + esc(e.rule) : '') + (e.detail ? '(' + esc(e.detail) + ')' : '')).join('、');
  }
  if (b.total !== undefined && Array.isArray(b.classes)) {
    return b.classes.map(c => `cls${c.class}×${c.count}`).join(' ');
  }
  if (b.count !== undefined) return `count=${esc(b.count)}`;
  return '';
}
function chip(label, n, cls) { return `<span class="chip ${cls || ''}">${esc(label)} <b>${n}</b></span>`; }
function render(items) {
  const win = Number(document.getElementById('window').value);
  const vd = document.getElementById('verdict').value;
  const ty = document.getElementById('type').value;
  const tk = document.getElementById('task').value.trim();
  const now = Date.now();
  const rows = [];
  const byType = {}, byVerdict = {true:0, false:0, uncertain:0, none:0};
  for (const it of items) {
    if (win > 0 && now - Number(it.recv_ts) > win) continue;
    const b = it.body || {};
    const t = evType(b);
    byType[t] = (byType[t] || 0) + 1;
    const v = it.review ? (it.review.verdict || 'uncertain') : 'none';
    byVerdict[v] = (byVerdict[v] || 0) + 1;
    if (vd && vd !== v) continue;
    if (ty && ty !== t) continue;
    if (tk && (b.task || '') !== tk) continue;
    rows.push(it);
  }
  // 顶部统计（按时间窗统计，不受其他过滤影响）
  let html = chip('窗口内事件', rows.length ? items.filter(it => win === 0 || now - Number(it.recv_ts) <= win).length : 0);
  html += Object.entries(byType).sort((a,b) => b[1]-a[1]).map(([t,n]) => chip(t, n)).join('');
  html += chip('可信', byVerdict.true, 'ok') + chip('误报', byVerdict.false, 'bad') +
          chip('存疑', byVerdict.uncertain, 'warn') + chip('未复核', byVerdict.none);
  document.getElementById('chips').innerHTML = html;
  if (!rows.length) { document.getElementById('list').innerHTML = '<div class="hint">窗口内无匹配事件</div>'; return; }
  document.getElementById('list').innerHTML = rows.map(it => {
    const b = it.body || {};
    const paths = [b.snapshot, b.clip].filter(Boolean).map(p => `<div class="path">📎 ${esc(p)}</div>`).join('');
    return `<div class="evcard ${it.review && it.review.verdict === 'false' ? 'false' : ''}">
      <div class="evrow">
        <span class="evt">${fmt(it.recv_ts)}</span>
        <span class="evtask">${esc(b.task || '?')}</span>
        <span class="badge plain">${esc(evType(b))}</span>
        <span>${evDetail(b)}</span>
        ${verdictBadge(it)}
        <span class="hint">#${it.id}</span>
      </div>
      ${paths}
      <details><summary class="hint">原始 JSON</summary><pre>${esc(JSON.stringify(b, null, 1))}</pre></details>
    </div>`;
  }).join('');
}
async function load() {
  try {
    const r = await fetch('/api/events?limit=' + fetchLimit);
    const data = await r.json();
    render((data.events || []).slice().reverse());  // 新到旧
  } catch (e) { document.getElementById('chips').innerHTML = '<span class="hint">拉取失败: ' + esc(e) + '</span>'; }
}
function more() { fetchLimit = Math.min(fetchLimit + 300, 1000); load(); }
setInterval(() => { if (document.getElementById('auto').checked) load(); }, 5000);
load();
</script>
</body>
</html>)HTML";
}

// 监控网页（/ 与 /monitor）：轮询 REST API + 内嵌 MJPEG 实时预览 + 任务控制
std::string monitorHtml() {
    return R"HTML(<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rk_pipe 监控面板</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: "SF Mono", Menlo, Consolas, monospace; background: #11141a; color: #d7dde8; margin: 0; padding: 16px; }
  h1 { font-size: 18px; margin: 0 0 12px; color: #8ab4ff; }
  .toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
  input, select, button { background: #1c2230; color: #d7dde8; border: 1px solid #2c3547; border-radius: 6px; padding: 6px 10px; font-size: 13px; }
  button { cursor: pointer; }
  button:hover:not(:disabled) { border-color: #8ab4ff; }
  button:disabled { opacity: .5; cursor: wait; }
  .boardbar { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; background: #161b26; border: 1px solid #262f42; border-radius: 10px; padding: 10px 14px; margin-bottom: 14px; font-size: 13px; }
  .bitem { display: inline-flex; gap: 6px; align-items: center; }
  .bar { display: inline-block; width: 90px; height: 10px; background: #232b3c; border-radius: 5px; overflow: hidden; vertical-align: middle; }
  .bar i { display: block; height: 100%; width: 0; background: #4f9d5f; border-radius: 5px; transition: width .4s; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }
  .card { background: #161b26; border: 1px solid #262f42; border-radius: 10px; padding: 12px; }
  .card.ok { border-color: #2f6b3f; }
  .card.down { border-color: #7a3a3a; }
  .head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .title { font-weight: bold; color: #e8eef8; }
  .res { font-size: 12px; padding: 2px 8px; border-radius: 10px; background: #1f4d2f; color: #7ce38b; margin-left: auto; margin-right: 8px; font-family: monospace; }
  .status { font-size: 12px; padding: 2px 8px; border-radius: 10px; }
  .status.running { background: #1f4d2f; color: #7ce38b; }
  .status.stopped { background: #4d1f1f; color: #ff8a8a; }
  .meta { font-size: 12px; color: #8a93a6; margin: 2px 0; word-break: break-all; }
  .stats { font-size: 12px; color: #8ab4ff; margin: 6px 0; }
  .previewWrap { min-height: 0; }
  .preview { width: 100%; border-radius: 6px; background: #000; aspect-ratio: 16/9; object-fit: cover; display: block; }
  .btns { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }
  .hint { color: #5a6478; font-size: 12px; }
  .toast { position: fixed; right: 16px; bottom: 16px; background: #1f4d2f; color: #7ce38b; border: 1px solid #2f6b3f; padding: 10px 16px; border-radius: 8px; font-size: 13px; z-index: 99; transition: opacity .3s; }
  .toast.err { background: #4d1f1f; color: #ff8a8a; border-color: #7a3a3a; }
  .toast.hide { opacity: 0; }
  #logModal { position: fixed; inset: 0; background: rgba(0,0,0,.6); display: none; align-items: center; justify-content: center; z-index: 50; }
  #logBox { width: min(900px, 92vw); height: 70vh; background: #0d1016; border: 1px solid #2c3547; border-radius: 10px; display: flex; flex-direction: column; }
  #logHead { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-bottom: 1px solid #262f42; }
  #logText { flex: 1; overflow: auto; padding: 10px 12px; margin: 0; font-size: 12px; white-space: pre-wrap; word-break: break-all; color: #b8c2d4; }
  #editModal { position: fixed; inset: 0; background: rgba(0,0,0,.6); display: none; align-items: center; justify-content: center; z-index: 55; }
  #editBox { width: min(560px, 92vw); background: #161b26; border: 1px solid #2c3547; border-radius: 10px; }
  #editHead { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-bottom: 1px solid #262f42; }
  #editForm { display: flex; flex-direction: column; gap: 8px; padding: 12px; }
  #editForm .row { display: flex; align-items: center; gap: 8px; }
  #editForm label { width: 90px; color: #8a93a6; font-size: 12px; flex-shrink: 0; }
  #editForm input, #editForm select { flex: 1; min-width: 0; }
  #editForm textarea { flex: 1; min-width: 0; height: 76px; background: #1c2230; color: #d7dde8; border: 1px solid #2c3547; border-radius: 6px; padding: 6px 10px; font-size: 12px; font-family: inherit; resize: vertical; }
  .events { background: #161b26; border: 1px solid #262f42; border-radius: 10px; padding: 10px 14px; margin-bottom: 14px; font-size: 12px; }
  .events .evhead { color: #8a93a6; margin-bottom: 6px; }
  .evrow { display: flex; gap: 10px; align-items: center; padding: 2px 0; }
  .evrow .evt { color: #5a6478; width: 64px; flex-shrink: 0; }
  .evrow .evtask { color: #8ab4ff; width: 70px; flex-shrink: 0; }
  .evrow .evtype { color: #d7dde8; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badge { padding: 1px 8px; border-radius: 9px; font-size: 11px; flex-shrink: 0; }
  .badge.ok { background: #1f4d2f; color: #7ce38b; }
  .badge.bad { background: #4d1f1f; color: #ff8a8a; }
  .badge.warn { background: #4d3a1f; color: #ffce7a; }
  .badge.plain { background: #232b3c; color: #8a93a6; }
</style>
</head>
<body>
<h1>rk_pipe 监控面板 <!--UI-SHIELD(时间线入口) 屏蔽前: <span class="hint">（<a href="/timeline">事件时间线 →</a>）</span>--></h1>
<div class="boardbar">
  <span class="bitem"><span id="cpuLabel">CPU</span> <span class="bar"><i id="cpu"></i></span></span>
  <span class="bitem">MEM <span class="bar"><i id="mem"></i></span><span id="memv" class="hint"></span></span>
  <span class="bitem">NPU <span class="bar"><i id="npu0"></i></span><span class="bar"><i id="npu1"></i></span><span class="bar"><i id="npu2"></i></span><span id="npuv" class="hint"></span></span>
  <span class="bitem">DISK <span class="bar"><i id="disk"></i></span><span id="diskv" class="hint"></span></span>
  <span class="bitem" id="temp"></span>
  <span class="bitem" id="load"></span>
  <span id="statusline" class="hint">在线</span>
</div>
<div class="toolbar">
  <button onclick="openAdd()">新增任务</button>
  <input id="cfgfile" placeholder="JSON 路径(留空=--streams-file)" size="34">
  <button onclick="reloadConfig()">热加载 JSON</button>
  <button onclick="refresh()">刷新</button>
  <span class="hint">每 2s 自动刷新</span>
</div>
<!--UI-SHIELD(最近事件)--><div class="events" id="events" style="display:none">
  <div class="evhead">最近事件（LLM/人工复核结论，每 5s 刷新，详情 GET /api/events）</div>
  <div id="evlist"><span class="hint">加载中…</span></div>
</div>
<div class="grid" id="grid"><span class="hint">加载中…</span></div>

<div id="logModal"><div id="logBox">
  <div id="logHead"><span class="title" id="logTitle">日志</span><div><button onclick="loadLog()">刷新</button> <button onclick="document.getElementById('logModal').style.display='none'">关闭</button></div></div>
  <pre id="logText"></pre>
</div></div>

<div id="ruleModal"><div id="ruleBox">
  <div id="ruleHead"><span class="title" id="ruleTitle">画规则</span><div><button onclick="closeRule()">关闭</button></div></div>
  <div id="ruleBody">
    <div id="ruleCanvasWrap"><img id="ruleImg" alt="preview"><canvas id="ruleCanvas"></canvas></div>
    <div id="ruleCtl">
      <label>类型</label>
      <select id="ruleType" onchange="ruleTypeChanged()"><option value="intrusion">区域入侵</option><option value="dwell">区域滞留</option><option value="line_cross">绊线</option></select>
      <label>方向</label>
      <select id="ruleDir"><option value="both">双向</option><option value="A2B">仅 A-&gt;B</option><option value="B2A">仅 B-&gt;A</option></select>
      <label>类别</label><input id="ruleCls" size="6" value="0">
      <button onclick="undoPoint()">撤销点</button>
      <button onclick="clearPts()">清空</button>
      <button id="ruleSaveBtn" onclick="saveRule()">保存规则并重启</button>
      <span class="hint" id="ruleHintTxt"></span>
    </div>
  </div>
</div></div>

<div id="editModal"><div id="editBox">
  <div id="editHead"><span class="title" id="editTitle">修改任务</span><div><button onclick="closeEdit()">关闭</button></div></div>
  <form id="editForm">
    <div class="row"><label>任务类型</label>
      <select name="task"><option value="detect">detect</option><option value="pose">pose</option><option value="obb">obb</option><option value="seg">seg</option><option value="sem">sem</option></select>
    </div>
    <div class="row"><label>输入源</label><input name="input" size="40"></div>
    <div class="row"><label>模型</label><input name="model" size="40" placeholder="留空=按任务类型默认"></div>
    <div class="row"><label>线程数</label><input name="threads" size="6" value="3"></div>
    <div class="row"><label>端口</label><input name="port" size="6" placeholder="留空=自动分配"></div>
    <div class="row"><label>NPU 核掩码</label><input name="npu_core_mask" size="10" placeholder="可选"></div>
    <div class="row"><label>NPU 起始核</label><input name="npu_core_start" size="6" placeholder="0-2 留空自动"></div>
    <div class="row"><label>播放帧率</label>
      <select name="pace"><option value="0">最快速度</option><option value="1">原帧率播放</option></select>
      <span class="hint">仅本地视频生效</span>
    </div>
    <div class="row"><label>推流地址</label><input name="output" size="40" placeholder="留空=仅网页预览；rtmp/rtsp/udp/srt"></div>
    <div class="row"><label>编码</label>
      <select name="codec"><option value="">默认 H.264</option><option value="hevc">H.265 (HEVC)</option></select>
      <span class="hint">推流编码，需播放器支持</span>
    </div>
    <div class="row"><label>ID 跟踪</label>
      <select name="tracking"><option value="0">关</option><option value="1">开 (OC-SORT)</option></select>
      <span class="hint">所有任务默认关，含 detect</span>
    </div>
    <div class="row"><label>辅助模型</label><input name="aux_model" size="40" placeholder="可选：多任务组合，如 depth/pose"></div>
    <div class="row"><label>辅助任务</label>
      <select name="aux_task"><option value="detect">detect</option><option value="pose">pose</option><option value="obb">obb</option><option value="seg">seg</option><option value="depth">depth</option><option value="sem">sem</option></select>
    </div>
    <div class="row"><label>事件/告警 YAML</label>
      <textarea name="extra_yaml" placeholder="可选：原样追加到任务 yaml。事件规则示例：&#10;event_line: &quot;450,740,1250,680&quot;&#10;event_line_cross: 1&#10;event_region: &quot;820,480,1230,500,1300,810,750,810&quot;&#10;event_intrusion: 1&#10;alert_enabled: 1&#10;alert_webhook_url: &quot;http://192.0.2.10:9000/alert&quot;"></textarea>
    </div>
    <button type="submit" style="margin-top:6px;" id="editSubmit">保存并重启</button>
  </form>
</div></div>

<script>
var host = location.hostname;
// 写操作鉴权 token：支持 URL ?token= 传入（记住后自动附加到写请求；GET 无需 token）
var API_TOKEN = new URLSearchParams(location.search).get('token') || localStorage.getItem('rkpipe_token') || '';
if (API_TOKEN) { try{ localStorage.setItem('rkpipe_token', API_TOKEN); }catch(e){} }
function withToken(u){ return API_TOKEN ? u+(u.indexOf('?')>=0?'&':'?')+'token='+encodeURIComponent(API_TOKEN) : u; }
var cards = {};   // id -> {el, img, imgSrc}
var lastTasks = {};  // id -> 最近一次任务数据（供修改弹窗预填）
var editId = null;
var logId = null;
var grid = document.getElementById('grid');

function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];});}
function basename(p){var i=String(p||'').lastIndexOf('/');return i>=0?p.slice(i+1):p;}
function fmtUptime(s){
  if(!s) return '0s';
  var h=Math.floor(s/3600), m=Math.floor(s%3600/60), sec=s%60, out='';
  if(h) out+=h+'h '; if(m||h) out+=m+'m '; out+=sec+'s'; return out;
}
function toast(msg, isErr){
  var d=document.createElement('div'); d.className='toast'+(isErr?' err':'');
  d.textContent=msg; document.body.appendChild(d);
  setTimeout(function(){ d.classList.add('hide'); setTimeout(function(){ d.remove(); }, 300); }, 2600);
}

function cardEl(t){
  var c=document.createElement('div'); c.className='card';
  c.innerHTML =
    '<div class="head"><span class="title"></span><span class="res"></span><span class="status"></span></div>'+
    '<div class="meta in"></div>'+
    '<div class="meta info"></div>'+
    '<div class="previewWrap"></div>'+
    '<div class="stats"></div>'+
    '<div class="btns">'+
      '<button data-act="edit" data-id="'+t.id+'">修改</button>'+
      '<button data-act="preview" data-id="'+t.id+'">预览</button>'+
      '<button data-act="log" data-id="'+t.id+'">日志</button>'+
      /*UI-SHIELD(画规则)*/ ''+
      '<button data-act="restart" data-id="'+t.id+'">重启</button>'+
      '<button data-act="stop" data-id="'+t.id+'">停止</button>'+
    '</div>';
  return c;
}

// 增量更新：卡片/预览元素持久复用，只改文本 → MJPEG 流不断连重连。
// 有推流（webrtc 非空）时用 iframe 内嵌 MediaMTX WebRTC 播放器，否则 MJPEG <img>。
function makeImg(c, t){
  var pw=c.el.querySelector('.previewWrap');
  if (t.webrtc){
    var fr=document.createElement('iframe'); fr.className='preview'; fr.alt='webrtc preview';
    fr.allow='autoplay; encrypted-media'; fr.allowFullscreen=true;
    fr.style.cssText='width:100%;aspect-ratio:16/9;border:0;background:#000;border-radius:8px;display:block;';
    fr.src=t.webrtc; c.img=fr; c.imgSrc=t.webrtc;
    pw.appendChild(fr);
    return;
  }
  var im=document.createElement('img'); im.className='preview'; im.alt='preview';
  var src='http://'+host+':'+t.port+'/stream.mjpg?t='+Date.now();  // 时间戳强制新连接
  im.src=src; c.img=im; c.imgSrc=src;
  pw.appendChild(im);
  // 加载失败（端口未就绪/进程重启中）→ 2s 后自动重连，无需手动刷新
  im.onerror=function(){
    if(cards[t.id]===c && c.img===im){
      c.img=null; c.imgSrc='';
      setTimeout(function(){
        var cur=lastTasks[t.id];
        if(cards[t.id]===c && cur && cur.status==='running'){ updateCard(cur); }
      }, 2000);
    }
  };
}
function updateCard(t){
  var c = cards[t.id];
  c.el.className = 'card ' + (t.status==='running' ? 'ok' : 'down');
  c.el.querySelector('.title').textContent = '#'+t.id+' '+t.task;
  var st=c.el.querySelector('.status'); st.className='status '+t.status; st.textContent=t.status;
  // 分辨率显示：来源为子进程 /status.json 的 frame_width x frame_height（如 1280x720）
  var res=c.el.querySelector('.res');
  var ch=t.child;
  if(ch && (ch.frame_width||0)>0 && (ch.frame_height||0)>0){
    res.textContent=ch.frame_width+'x'+ch.frame_height;
  }else{ res.textContent=''; }
  c.el.querySelector('.in').textContent = 'input: '+t.input;
  var codecTag = t.codec ? ' · codec '+t.codec : '';
  c.el.querySelector('.info').textContent =
    'model: '+basename(t.model)+' · port '+t.port+' · pid '+t.pid+' · uptime '+fmtUptime(t.uptime_s)+' · restarts '+t.restarts + codecTag;
  var pw = c.el.querySelector('.previewWrap');
  // 仅当子进程确认已发布帧（published_frames>0）才建立预览连接，
  // 否则连上 /stream.mjpg 后服务端无帧可发会一直挂起，画面空白需手动刷新才能出现
  var ready = (t.status==='running') && t.child && t.child.running && (t.child.published_frames||0) > 0;
  // 子进程重新出帧（false→true）时重建连接（重启/崩溃拉起/改配置）；
  // 其余情况保留旧 <img>，推理/播放结束后画面定格在最后一帧，而不是被移除成空白
  if (ready && !c.prevReady && c.img){ c.img.remove(); c.img=null; c.imgSrc=''; }
  c.prevReady = ready;
  if (ready && !c.img){ makeImg(c, t); }
  var stats = '<span class="hint">—</span>';
  if (t.child){
    var s=t.child;
    stats = 'publish '+(s.publish_fps||0).toFixed(1)+' fps · encode '+(s.encode_ms||0)+' ms · clients '+(s.clients||0)+
            ' · pub '+(s.published_frames||0)+' · drop '+(s.dropped_frames||0);
  }
  c.el.querySelector('.stats').innerHTML = stats;
}

function renderTasks(tasks){
  // 移除残留的非卡片节点（如初始“加载中…”占位，否则它占住 grid 第一格）
  while (grid.firstChild && !grid.firstChild.classList.contains('card')){
    grid.removeChild(grid.firstChild);
  }
  var seen={};
  for (var i=0;i<tasks.length;i++){
    var t=tasks[i]; seen[t.id]=1; lastTasks[t.id]=t;
    if (!cards[t.id]){ var c={el:cardEl(t), img:null, imgSrc:'', prevReady:false}; cards[t.id]=c; grid.appendChild(c.el); }
    updateCard(t);
  }
  for (var id in cards){ if(!seen[id]){ cards[id].el.remove(); delete cards[id]; delete lastTasks[id]; } }
}

function renderBoard(b){
  if(!b) return;
  var cu=b.cpu_usage||0, mu=b.mem_used_pct||0;
  document.getElementById('cpuLabel').textContent='CPU '+Math.round(cu)+'%';
  document.getElementById('cpu').style.width=Math.min(100,cu)+'%';
  document.getElementById('mem').style.width=Math.min(100,mu)+'%';
  document.getElementById('memv').textContent=Math.round((b.mem_total_mb||0)-(b.mem_avail_mb||0))+'/'+(b.mem_total_mb||0)+' MB ('+Math.round(mu)+'%)';
  var npu=b.npu_load||[], nid=['npu0','npu1','npu2'];
  var npuv=document.getElementById('npuv');
  if(npu.length>=3){
    for(var i=0;i<3;i++){ document.getElementById(nid[i]).style.width=Math.min(100,npu[i])+'%'; }
    npuv.textContent=Math.round(npu[0])+'/'+Math.round(npu[1])+'/'+Math.round(npu[2])+'%';
  }else{
    for(var i=0;i<3;i++){ document.getElementById(nid[i]).style.width='0'; }
    npuv.textContent='N/A';
  }
  var du=b.disk_used_pct||0;
  document.getElementById('disk').style.width=Math.min(100,du)+'%';
  document.getElementById('diskv').textContent=Math.round(b.disk_free_mb||0)+'MB free / '+(du?Math.round(du)+'%':'—');
  document.getElementById('temp').textContent='温度 '+(b.temp_c||0).toFixed(1)+'°C';
  document.getElementById('load').textContent='load '+(b.load[0]||0).toFixed(2)+' / '+(b.load[1]||0).toFixed(2)+' / '+(b.load[2]||0).toFixed(2);
}

var offline=false;
async function refresh(){
  if(document.hidden) return;
  var d;
  try{
    var r=await fetch('/api/summary');
    if(!r.ok) throw 0;
    d=await r.json();
  }catch(e){
    var sl=document.getElementById('statusline');
    if(!offline){ sl.textContent='离线'; sl.style.color='#ff8a8a'; offline=true; }
    return;
  }
  if(offline){ var sl=document.getElementById('statusline'); sl.textContent='在线'; sl.style.color='#5a6478'; offline=false; }
  renderBoard(d.board);
  renderTasks(d.tasks||[]);
}

// 操作反馈：confirm + 按钮禁用 + toast
async function doAction(id, method, url, confirmMsg){
  if(confirmMsg && !confirm(confirmMsg)) return;
  var c=cards[id]; if(!c) return;
  var btns=c.el.querySelectorAll('button[data-act]');
  btns.forEach(function(b){ b.disabled=true; });
  try{
    var r=await fetch(withToken(url),{method:method});
    toast(r.ok ? ('任务 #'+id+' 操作成功') : ('任务 #'+id+' 失败 '+r.status), !r.ok);
  }catch(e){ toast('请求失败：'+e, true); }
  btns.forEach(function(b){ b.disabled=false; });
  refresh();
}

// 热加载 JSON 配置文件（读文件 + applyConfig 热切换）。输入框填路径则加载指定文件，留空用 daemon 的 --streams-file
async function reloadConfig(){
  var b=document.querySelector('.toolbar button[onclick="reloadConfig()"]');
  var inp=document.getElementById('cfgfile');
  var f=(inp?inp.value.trim():'');
  if(b) b.disabled=true;
  try{
    var r=await fetch(withToken('/api/reload'+(f?'?file='+encodeURIComponent(f):'')),{method:'POST'});
    var res=null; if(r.ok){ try{ res=await r.json(); }catch(e){} }
    toast(res&&res.ok ? ('配置已热加载，共 '+(res.configured_tasks||0)+' 路'+(res.file?' ('+res.file+')':'')) : ('热加载失败 '+r.status), !(res&&res.ok));
  }catch(e){ toast('请求失败：'+e, true); }
  if(b) b.disabled=false;
  refresh();
}

async function refreshEvents(){
  try{
    var r=await fetch('/api/events?limit=8');
    if(!r.ok) throw 0;
    var d=await r.json();
    var el=document.getElementById('evlist');
    if(!d.events || !d.events.length){ el.innerHTML='<span class="hint">暂无事件</span>'; return; }
    el.innerHTML=d.events.map(function(e){
      var b=e.body||{};
      var v=e.review?e.review.verdict:'';
      var src=e.review?e.review.source:'';
      var cls=v==='true'?'ok':(v==='false'?'bad':(v==='uncertain'?'warn':(v?'warn':'plain')));
      var tag=v?'<span class="badge '+cls+'">'+(src==='llm'?'LLM·':(src==='manual'?'人工·':''))+(v==='true'?'可信':(v==='false'?'误报':(v==='uncertain'?'存疑':'错误')))+'</span>':'';
      var t=new Date(e.recv_ts); function p2(x){return (x<10?'0':'')+x;}
      var ts=p2(t.getHours())+':'+p2(t.getMinutes())+':'+p2(t.getSeconds());
      var detail=b.rule_type||b.event||('count×'+(b.total||'?'));
      return '<div class="evrow"><span class="evt">'+ts+'</span><span class="evtask">#'+esc(b.task||'?')+'</span><span class="evtype" title="'+esc(e.review?e.review.reason:'')+'">'+esc(detail)+'</span>'+tag+'</div>';
    }).join('');
  }catch(e){}
}
setInterval(refreshEvents,5000);

async function loadLog(){
  if(logId==null) return;
  var pre=document.getElementById('logText');
  pre.textContent='加载中…';
  try{
    var r=await fetch('/api/tasks/'+logId+'/log?tail=300');
    if(!r.ok){ pre.textContent='日志不可用（HTTP '+r.status+'）'; return; }
    pre.textContent=await r.text();
    pre.scrollTop=pre.scrollHeight;
  }catch(e){ pre.textContent='日志加载失败：'+e; }
}
function showLog(id){
  logId=id;
  document.getElementById('logTitle').textContent='任务 #'+id+' 日志';
  document.getElementById('logModal').style.display='flex';
  loadLog();
}

// ---------------- 画规则编辑器：MJPEG 上画布取点 → 区域/绊线 → POST rules ----------------
var ruleTaskId=null, rulePts=[], ruleFrameW=1280, ruleFrameH=720, ruleExisting=[];
function openRule(id){
  var t=lastTasks[id]; if(!t) return;
  ruleTaskId=id; rulePts=[]; ruleExisting=[];
  ruleFrameW=(t.child&&t.child.frame_width)||1280;
  ruleFrameH=(t.child&&t.child.frame_height)||720;
  document.getElementById('ruleTitle').textContent='画规则 - 任务 #'+id+' ('+ruleFrameW+'x'+ruleFrameH+')';
  document.getElementById('ruleModal').style.display='flex';
  var img=document.getElementById('ruleImg');
  img.onload=function(){
    var cv=document.getElementById('ruleCanvas');
    cv.width=img.clientWidth; cv.height=img.clientHeight;
    drawRule();
  };
  img.src='http://'+host+':'+t.port+'/stream.mjpg?t='+Date.now();
  fetch('/api/tasks/'+id+'/rules').then(function(r){return r.json();}).then(function(d){
    ruleExisting=d.rules||[]; updateRuleHint(); drawRule();
  }).catch(function(){});
  drawRule();
}
function closeRule(){ document.getElementById('ruleModal').style.display='none'; }
function ruleTypeChanged(){ clearPts(); }
function clearPts(){ rulePts=[]; drawRule(); }
function undoPoint(){ rulePts.pop(); drawRule(); }
function updateRuleHint(){
  document.getElementById('ruleHintTxt').textContent =
    '已有规则 '+ruleExisting.length+' 条；'+(document.getElementById('ruleType').value==='line_cross'?'绊线点击 2 点':'区域点击 ≥3 点');
}
function drawRule(){
  var cv=document.getElementById('ruleCanvas'); if(!cv.width) return;
  var ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  var sx=cv.width/ruleFrameW, sy=cv.height/ruleFrameH;
  function toPts(geo){ var a=(geo||'').split(',').map(Number), ps=[]; for(var i=0;i+1<a.length;i+=2) ps.push([a[i]*sx,a[i+1]*sy]); return ps; }
  // 已有规则（淡青）：绊线/区域
  ctx.strokeStyle='rgba(0,255,255,.6)'; ctx.lineWidth=2;
  ruleExisting.forEach(function(r){
    var ps=toPts(r.line||r.region); if(ps.length<2) return;
    ctx.beginPath(); ps.forEach(function(p,i){ i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); });
    if(r.region) ctx.closePath();
    ctx.stroke();
  });
  // 正在画的点集（黄）
  var type=document.getElementById('ruleType').value;
  var pts=rulePts.map(function(p){ return [p[0]*sx,p[1]*sy]; });
  ctx.strokeStyle='#ffff00'; ctx.fillStyle='#ffff00'; ctx.lineWidth=2;
  pts.forEach(function(p){ ctx.beginPath(); ctx.arc(p[0],p[1],4,0,Math.PI*2); ctx.fill(); });
  if(type==='line_cross' && pts.length===2){
    ctx.beginPath(); ctx.moveTo(pts[0][0],pts[0][1]); ctx.lineTo(pts[1][0],pts[1][1]); ctx.stroke();
  } else if(type!=='line_cross' && pts.length>=3){
    ctx.beginPath(); pts.forEach(function(p,i){ i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]); }); ctx.closePath(); ctx.stroke();
  }
}
document.getElementById('ruleCanvas').addEventListener('click', function(e){
  var rect=this.getBoundingClientRect();
  var x=Math.round((e.clientX-rect.left)*ruleFrameW/this.width);
  var y=Math.round((e.clientY-rect.top)*ruleFrameH/this.height);
  var max=document.getElementById('ruleType').value==='line_cross'?2:16;
  if(rulePts.length<max){ rulePts.push([x,y]); drawRule(); }
});
async function saveRule(){
  if(ruleTaskId==null) return;
  var type=document.getElementById('ruleType').value;
  var rule={id:'r'+(Date.now()%100000), type:type};
  if(type==='line_cross'){
    if(rulePts.length!==2){ toast('绊线需要恰好 2 个点', true); return; }
    rule.line=rulePts.map(function(p){return p.join(',');}).join(',');
    rule.direction=document.getElementById('ruleDir').value;
  } else {
    if(rulePts.length<3){ toast('区域至少 3 个点', true); return; }
    rule.region=rulePts.map(function(p){return p.join(',');}).join(',');
    if(type==='dwell'){ rule.dwell_seconds='10'; }
  }
  var cls=document.getElementById('ruleCls').value.trim();
  if(cls!=='') rule.classes=cls;
  var all=ruleExisting.concat([rule]);
  var btn=document.getElementById('ruleSaveBtn'); btn.disabled=true;
  try{
    var r=await fetch(withToken('/api/tasks/'+ruleTaskId+'/rules'),{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({rules:all})
    });
    toast(r.ok?('规则已保存（'+all.length+' 条），任务重启中'):('保存失败 '+r.status), !r.ok);
    if(r.ok){ closeRule(); }
  }catch(e){ toast('请求失败：'+e, true); }
  btn.disabled=false;
}

function openEdit(id){
  var t=lastTasks[id]; if(!t) return;
  editId=id;
  var f=document.getElementById('editForm');
  f.querySelector('[name=task]').value=t.task||'detect';
  f.querySelector('[name=input]').value=t.input||'';
  f.querySelector('[name=model]').value=t.model||'';
  f.querySelector('[name=threads]').value=t.threads||'3';
  f.querySelector('[name=port]').value=(t.port&&t.port>0)?t.port:'';
  f.querySelector('[name=npu_core_mask]').value=t.npu_core_mask||'';
  f.querySelector('[name=npu_core_start]').value=t.npu_core_start||'';
  f.querySelector('[name=pace]').value=(t.pace==='1')?'1':'0';
  f.querySelector('[name=output]').value=t.output||'';
  f.querySelector('[name=codec]').value=t.codec||'';
  f.querySelector('[name=tracking]').value=(t.tracking==='1')?'1':'0';
  f.querySelector('[name=aux_model]').value=t.aux_model||'';
  f.querySelector('[name=aux_task]').value=t.aux_task||'detect';
  f.querySelector('[name=extra_yaml]').value=t.extra_yaml||'';
  document.getElementById('editTitle').textContent='修改任务 #'+id;
  document.getElementById('editSubmit').textContent='保存并重启';
  document.getElementById('editModal').style.display='flex';
}
function closeEdit(){ document.getElementById('editModal').style.display='none'; }

// 新增任务：复用同一个编辑弹窗卡片，字段置默认值
function openAdd(){
  editId=null;
  var f=document.getElementById('editForm');
  f.querySelector('[name=task]').value='detect';
  f.querySelector('[name=input]').value='';
  f.querySelector('[name=model]').value='';
  f.querySelector('[name=threads]').value='3';
  f.querySelector('[name=port]').value='';
  f.querySelector('[name=npu_core_mask]').value='';
  f.querySelector('[name=npu_core_start]').value='';
  f.querySelector('[name=pace]').value='0';
  f.querySelector('[name=output]').value='';
  f.querySelector('[name=codec]').value='';
  f.querySelector('[name=tracking]').value='0';
  f.querySelector('[name=aux_model]').value='';
  f.querySelector('[name=aux_task]').value='detect';
  f.querySelector('[name=extra_yaml]').value='';
  document.getElementById('editTitle').textContent='新增任务';
  document.getElementById('editSubmit').textContent='新增并启动';
  document.getElementById('editModal').style.display='flex';
  setTimeout(function(){ f.querySelector('[name=input]').focus(); }, 50);
}

document.getElementById('editForm').addEventListener('submit', async function(ev){
  ev.preventDefault();
  var id=editId, f=this;
  var p=new URLSearchParams();
  ['task','input','model','threads','port','npu_core_mask','npu_core_start','pace','output','codec','tracking','aux_model','aux_task'].forEach(function(k){
    var v=f.querySelector('[name='+k+']').value.trim();
    if(v!=='') p.set(k,v);
  });
  var ey=f.querySelector('[name=extra_yaml]').value.trim();
  if(ey!=='') p.set('extra_yaml', ey);
  else if(id!=null) p.set('extra_yaml', '');   // 更新时清空=删除透传段
  if(id==null){
    // 新增任务：POST /api/tasks
    try{
      var r=await fetch(withToken('/api/tasks?'+p.toString()),{method:'POST'});
      var res=null;
      if(r.ok){ try{ res=await r.json(); }catch(e){} }
      if(res && res.ok){ lastTasks[res.task.id]=res.task; toast('已新增任务 #'+res.task.id); }
      else{ toast('新增失败 '+(res&&res.error?('：'+res.error):('HTTP '+r.status)), true); }
    }catch(e){ toast('请求失败：'+e, true); }
    closeEdit();
    refresh();
    return;
  }
  try{
    var r=await fetch(withToken('/api/tasks/'+id+'/update?'+p.toString()),{method:'POST'});
    if(r.ok){
      try{
        var res=await r.json();
        if(res && res.task){ lastTasks[id]=res.task; updateCard(res.task); }  // 立即反映新配置（如 pace 原帧率）
      }catch(e){}
      if(cards[id]) cards[id].needReconnect=true;
      toast('任务 #'+id+' 已更新并重启');
    }else{
      toast('更新失败 '+r.status, true);
    }
  }catch(e){ toast('请求失败：'+e, true); }
  closeEdit();
  refresh();
});

grid.addEventListener('click', function(ev){
  var b=ev.target.closest ? ev.target.closest('button[data-act]') : null;
  if(!b) return;
  var id=+b.dataset.id;
  var act=b.dataset.act;
  if(act==='edit'){ openEdit(id); }
  else if(act==='rules'){ openRule(id); }
  else if(act==='preview'){ window.open('http://'+host+':'+ (cards[id]? cards[id].el.querySelector('.info').textContent.split('· port ')[1].split(' ')[0] : '') +'/','_blank'); }
  else if(act==='log'){ showLog(id); }
  else if(act==='restart'){ if(cards[id]) cards[id].needReconnect=true; doAction(id,'POST','/api/tasks/'+id+'/restart'); }
  else if(act==='stop'){ doAction(id,'DELETE','/api/tasks/'+id); }
});

document.addEventListener('visibilitychange', function(){ if(!document.hidden) refresh(); });

refresh();
refreshEvents();
setInterval(refresh, 2000);
</script>
</body>
</html>
)HTML";
}
