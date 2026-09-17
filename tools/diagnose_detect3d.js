#!/usr/bin/env node
/**
 * detect3d 线框诊断工具（路线 B）——从板端 dump 的真实检测框评估线框质量。
 *
 * 回答三个问题（2026-09-17 用本工具得出「距离门限」结论）：
 *   1. 框稳不稳？（追踪同一目标，看帧间框尺寸抖动 —— 判断「框松」是否成立）
 *   2. 框自洽吗？（车宽法 Z vs 车高法 Z —— 两条独立标尺应给同一距离）
 *   3. 线框会溢出吗？（投影包围盒 vs 2D 框 —— 按距离分桶统计，定门限）
 *
 * 用法：
 *   # 板端先导出
 *   ./build/console_detector -c configs/run_yolo26_detect_depth3d.yaml \
 *       --video test_videos/x.mp4 --dump-detections /tmp/dets.jsonl
 *   # 本地分析
 *   node tools/diagnose_detect3d.js tools/evidence/d3_real_baseline17_mid30s.jsonl
 *
 * 可选：--cam-height <m> 接地法反推距离用的相机高度（默认 1.5）
 *       --fx/--fy/--cx/--cy 覆盖内参（默认取配置的 P2 值）
 */
const fs = require('fs');

const args = process.argv.slice(2);
const getNum = (k, d) => { const i = args.indexOf(k); return i >= 0 ? parseFloat(args[i + 1]) : d; };
const file = args.find(a => !a.startsWith('--') && !/^-?\d/.test(a)) || 'tools/evidence/d3_real_baseline17_mid30s.jsonl';
const CAM_H = getNum('--cam-height', 1.5);
const fx = getNum('--fx', 599.8746), fy = getNum('--fy', 599.8746);
const cx = getNum('--cx', 990.0), cy = getNum('--cy', 634.4805);
const P2 = [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0];
const kWidthRatio = 0.4158;

const L = fs.readFileSync(file, 'utf8').trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
const W = L[0].width, H = L[0].height;
const q = (a, p) => { a = a.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * (a.length - 1)))]; };
const med = a => q(a, 0.5);

const clsName = { 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck' };
const TARGET = parseInt(args.find(a => a.startsWith('--cls='))?.split('=')[1] ?? '2', 10);

console.log(`素材帧数 ${L.length}  分辨率 ${W}x${H}  目标类别 ${clsName[TARGET] || TARGET}`);
console.log(`内参 fx=${fx} fy=${fy} cx=${cx} cy=${cy}  接地法相机高=${CAM_H}m\n`);

// ---------- 投影 ----------
function project(it) {
  const Z = it.depth_m;
  const X = (it.center_u - cx) * Z / fx, Y = (it.center_v - cy) * Z / fy;
  const ry = Math.atan2(it.sin_alpha, it.cos_alpha);
  const l = it.l3, h = it.h3, w = it.w3;
  const kXc = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
  const kYc = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
  const kZc = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
  const c = Math.cos(ry), s = Math.sin(ry), o = [];
  for (let i = 0; i < 8; i++) {
    const X3 = c*kXc[i] + s*kZc[i] + X, Y3 = kYc[i] + Y, Z3 = -s*kXc[i] + c*kZc[i] + Z;
    const pw = P2[8]*X3 + P2[9]*Y3 + P2[10]*Z3 + P2[11];
    if (!(pw > 1e-3)) return null;
    o.push([(P2[0]*X3+P2[1]*Y3+P2[2]*Z3+P2[3])/pw, (P2[4]*X3+P2[5]*Y3+P2[6]*Z3+P2[7])/pw]);
  }
  return o;
}
const bbOf = p => ({ x1: Math.min(...p.map(v=>v[0])), x2: Math.max(...p.map(v=>v[0])), y1: Math.min(...p.map(v=>v[1])), y2: Math.max(...p.map(v=>v[1])) });

// ---------- 追踪（IoU 贪心） ----------
function iou(a, b) {
  const [ax,ay,aw,ah]=a,[bx,by,bw,bh]=b;
  const x1=Math.max(ax,bx),y1=Math.max(ay,by),x2=Math.min(ax+aw,bx+bw),y2=Math.min(ay+ah,by+bh);
  if(x2<=x1||y2<=y1) return 0;
  const inter=(x2-x1)*(y2-y1);
  return inter/(aw*ah+bw*bh-inter);
}
function trackAll(clsF, minHits) {
  let next = 1; const tr = [], done = [];
  for (let fi = 0; fi < L.length; fi++) {
    const dets = L[fi].dets.filter(d => clsF.includes(d.cls)).map(d => ({ bbox: d.bbox, fi }));
    const used = new Array(dets.length).fill(false);
    for (const t of tr) {
      let best=-1, bi=-1;
      for (let i=0;i<dets.length;i++){ if(used[i])continue; const v=iou(t.last,dets[i].bbox); if(v>best){best=v;bi=i;} }
      if (best>=0.4){ used[bi]=true; t.pts.push(dets[bi]); t.last=dets[bi].bbox; t.lastFi=fi; }
    }
    for (let i=0;i<dets.length;i++) if(!used[i]) tr.push({id:next++,pts:[dets[i]],last:dets[i].bbox,lastFi:fi});
    for (let i=tr.length-1;i>=0;i--) if(fi-tr[i].lastFi>10) done.push(tr.splice(i,1)[0]);
  }
  done.push(...tr);
  return done.filter(t => t.pts.length >= minHits).sort((a,b)=>b.pts.length-a.pts.length);
}

// ===== 1. 框稳定性 =====
console.log('=== 1. 框稳定性（追踪同一目标的帧间抖动）===');
const tracks = trackAll([TARGET], 30);
console.log('id   n    框高中位  帧间|Δh|中位  帧间|Δy2|中位  连续性');
let allDh = [];
for (const t of tracks.slice(0, 12)) {
  const p = t.pts;
  let gaps=0; for(let i=1;i<p.length;i++) if(p[i].fi-p[i-1].fi!==1) gaps++;
  const cont = 1 - gaps/(p.length-1);
  if (cont < 0.85) continue;
  const dh=[], dy=[];
  for(let i=1;i<p.length;i++){
    dh.push(Math.abs(p[i].bbox[3]-p[i-1].bbox[3]));
    dy.push(Math.abs((p[i].bbox[1]+p[i].bbox[3])-(p[i-1].bbox[1]+p[i-1].bbox[3])));
  }
  allDh.push(...dh);
  console.log(`${String(t.id).padEnd(4)} ${String(p.length).padEnd(5)} ${String(med(p.map(x=>x.bbox[3]))).padEnd(9)} ${med(dh).toFixed(1).padEnd(14)} ${med(dy).toFixed(1).padEnd(15)} ${cont.toFixed(2)}`);
}
if (allDh.length) {
  console.log(`\n全局帧间 |Δ框高|：中位 ${med(allDh).toFixed(1)} px，p90 ${q(allDh,.9).toFixed(1)} px`);
  console.log('→ 中位数接近 0 说明框很紧很稳，「框松」假设不成立。');
}

// ===== 2. 几何自洽性 =====
console.log('\n=== 2. 几何自洽性（车宽法 Z vs 车高法 Z）===');
const SPEC = { 0:{w:0.5,h:1.7}, 2:{w:1.85,h:1.5}, 5:{w:2.55,h:3.2}, 7:{w:2.5,h:3.0} }[TARGET] || {w:1.85,h:1.5};
const bigBoxes = [];
for (const l of L) for (const d of l.dets) if (d.cls === TARGET && d.bbox[3] > 200) bigBoxes.push(d.bbox);
console.log(`大框(h>200px) n=${bigBoxes.length}，按真实尺寸 车宽${SPEC.w}m/车高${SPEC.h}m 反推：`);
if (bigBoxes.length) {
  const zw = bigBoxes.map(b => SPEC.w * fx / b[2]);
  const zh = bigBoxes.map(b => SPEC.h * fy / b[3]);
  const ratio = zw.map((v,i) => zh[i]/v);
  console.log(`  车宽法 Z: 中位 ${med(zw).toFixed(2)}m`);
  console.log(`  车高法 Z: 中位 ${med(zh).toFixed(2)}m`);
  console.log(`  两者比值: 中位 ${med(ratio).toFixed(2)}x  （≈1.0 表示框的宽高比与真实车身一致 → 框贴合）`);
}

// ===== 3. 线框溢出 vs 距离 =====
console.log('\n=== 3. 线框溢出 vs 距离（定门限的依据）===');
const samples = [];
for (const l of L) for (const d of l.dets) {
  if (d.cls !== TARGET) continue;
  const [x,y,w,h] = d.bbox;
  if (h < 25 || h > 400) continue;
  const Z = fy * 1.5 / ((y+h) - cy);
  if (!(Z > 1 && Z < 200)) continue;
  samples.push({ box: d.bbox, Z });
}
console.log(`样本 ${samples.length} 个\n`);
console.log('Z区间       n     比值中位   溢出%');
const bins = [[0,5],[5,8],[8,12],[12,18],[18,30],[30,999]];
for (const [lo,hi] of bins) {
  const ss = samples.filter(s => s.Z >= lo && s.Z < hi);
  if (!ss.length) continue;
  const rs = []; let ovf = 0;
  for (const s of ss) {
    const [x,y,w,h] = s.box, Z = s.Z;
    // 按当前实现（侧视公式）
    const it = { box:s.box, depth_m:Z, center_u:x+w/2, center_v:y+h/2,
                 h3:h*Z/fy, l3:w*Z/fx, w3:w*Z/fx*kWidthRatio, sin_alpha:0, cos_alpha:1 };
    const p = project(it);
    if (!p) { ovf++; continue; }
    const bb = bbOf(p);
    rs.push((bb.x2-bb.x1)/w);
    if (bb.x2 > W || bb.x1 < 0) ovf++;
  }
  console.log(`${String(lo+'-'+hi).padEnd(11)} ${String(ss.length).padEnd(6)} ${med(rs).toFixed(2).padEnd(10)} ${(ovf/ss.length*100).toFixed(0)}%`);
}
console.log('\n→ 选「比值≈1.0 且溢出 0%」的最小距离作为 detect3d_min_depth_m。');
