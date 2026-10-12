'use strict';
const $ = id => document.getElementById(id);
const canvas = $('plot'), ctx = canvas.getContext('2d');
const featureCache = new Map();
let atlas, points = [], byId, selected = null, hovered = null;
let width = 1, height = 1, scale = 1, panX = 0, panY = 0, drag = null, bounds;
let currentExamples = [], visibleCount = 0, requestId = 0, hoverTimer;
const SVG = 'http://www.w3.org/2000/svg';
const imageURL = (id, thumb = false) => `${document.body.dataset.imageBase || ''}${thumb ? 'thumbs' : 'images'}/${String(id).padStart(4, '0')}.webp`;

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`数据加载失败（${response.status}）`);
  return response.json();
}
function position(p) {
  return [(25 + (p.x - bounds.minX) / (bounds.maxX - bounds.minX || 1) * (width - 50)) * scale + panX,
          (25 + (p.y - bounds.minY) / (bounds.maxY - bounds.minY || 1) * (height - 50)) * scale + panY];
}
function draw() {
  ctx.clearRect(0, 0, width, height);
  for (const p of points) {
    const [x, y] = position(p);
    if (x < -10 || y < -10 || x > width + 10 || y > height + 10) continue;
    const focus = p.feature === selected || p.feature === hovered;
    ctx.beginPath(); ctx.arc(x, y, focus ? 6 : 3.2, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${210 - p.coherence * 75} 43% ${37 + p.coherence * 15}%)`; ctx.fill();
    if (focus) { ctx.strokeStyle = p.feature === hovered ? '#f08032' : '#163e36'; ctx.lineWidth = 2; ctx.stroke(); }
  }
  const p = byId?.get(hovered ?? selected);
  if (p?.x !== null && p) {
    const [x, y] = position(p); ctx.fillStyle = '#173d34'; ctx.font = 'bold 12px system-ui';
    ctx.fillText(`f${p.feature}`, Math.min(width - 55, x + 9), Math.max(15, y - 7));
  }
}
function resize() {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  width = r.width; height = r.height; canvas.width = width * dpr; canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); if (bounds) draw();
}
function nearest(x, y) {
  let best = null, distance = 12;
  for (const p of points) {
    const [px, py] = position(p), d = Math.hypot(x - px, y - py);
    if (d < distance) { best = p; distance = d; }
  }
  return best;
}
function coordinates(event) {
  const r = canvas.getBoundingClientRect(); return [event.clientX - r.left, event.clientY - r.top];
}
function hideHover() {
  clearTimeout(hoverTimer); hovered = null; $('hover-preview').hidden = true; if (bounds) draw();
}
function moveTooltip(x, y) {
  const tooltip = $('hover-preview');
  const w = tooltip.offsetWidth, h = tooltip.offsetHeight;
  let left = x + 18;
  if (left + w > window.innerWidth - 10) left = x - w - 18;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, Math.min(y + 12, window.innerHeight - h - 8))}px`;
}
function preview(p, x, y) {
  $('hover-title').textContent = `维度 f${p.feature} · 高激活原图`;
  $('hover-stats').textContent = `${p.count} 个实例 · embedding 一致性 ${p.coherence.toFixed(3)}`;
  $('hover-images').replaceChildren();
  for (const ex of p.preview) {
    const info = atlas.images[ex.image], figure = document.createElement('figure');
    const img = document.createElement('img'); img.src = imageURL(ex.image, true);
    img.alt = `${info.city} ${info.heading}° 原始街景，第 ${ex.rank} 名`;
    const caption = document.createElement('figcaption'); caption.textContent = `#${ex.rank} · ${info.city} · ${info.heading}°`;
    figure.append(img, caption); $('hover-images').append(figure);
  }
  $('hover-preview').hidden = false; moveTooltip(x, y);
}
function original(ex, thumb = false) {
  const container = document.createElement('div'); container.className = 'original';
  const img = document.createElement('img'), info = atlas.images[ex.image];
  img.src = imageURL(ex.image, thumb); img.alt = `${info.city} ${info.heading}° 原始街景`;
  img.loading = 'lazy';
  const box = document.createElement('span'); box.className = 'bbox';
  const [x0, y0, x1, y1] = ex.bbox;
  Object.assign(box.style, {left: `${x0 * 100}%`, top: `${y0 * 100}%`, width: `${(x1 - x0) * 100}%`, height: `${(y1 - y0) * 100}%`});
  container.append(img, box); return container;
}
function crop(ex) {
  const svg = document.createElementNS(SVG, 'svg'), img = document.createElementNS(SVG, 'image');
  const [x0, y0, x1, y1] = ex.bbox;
  const [w, h] = atlas.images[ex.image].size;
  svg.classList.add('crop'); svg.setAttribute('viewBox', `${x0 * w} ${y0 * h} ${(x1 - x0) * w} ${(y1 - y0) * h}`);
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `维度 ${selected} 的激活区域`);
  img.setAttribute('href', imageURL(ex.image)); img.setAttribute('width', String(w)); img.setAttribute('height', String(h));
  img.setAttribute('preserveAspectRatio', 'none'); svg.append(img); return svg;
}
function showImage(ex, isCrop = false) {
  const info = atlas.images[ex.image];
  $('image-title').textContent = `f${selected} · #${ex.rank} · ${info.city} · ${info.heading}°`;
  $('image-content').replaceChildren(isCrop ? crop(ex) : original(ex));
  $('image-caption').textContent = `激活 ${ex.activation.toFixed(4)} · ${info.source} · pano ${info.pano}`;
  $('image-link').href = imageURL(ex.image); $('image-dialog').showModal(); hideHover();
}
function appendExamples() {
  const next = Math.min(visibleCount + 20, currentExamples.length);
  for (const ex of currentExamples.slice(visibleCount, next)) {
    const info = atlas.images[ex.image], card = document.createElement('article'); card.className = 'card';
    const head = document.createElement('div'); head.className = 'card-head';
    head.textContent = `#${ex.rank} · ${info.city} · ${info.heading}°`;
    const base = original(ex); base.onclick = () => showImage(ex);
    const region = crop(ex); region.onclick = () => showImage(ex, true);
    const a = document.createElement('span'); a.className = 'image-label original-label'; a.textContent = '原图 · 橙框为激活区域';
    const b = document.createElement('span'); b.className = 'image-label crop-label'; b.textContent = '激活区域';
    const foot = document.createElement('div'); foot.className = 'card-foot';
    foot.textContent = `激活 ${ex.activation.toFixed(4)} · ${info.source} · ${info.pano}`;
    card.append(head, base, a, region, b, foot); $('examples').append(card);
  }
  visibleCount = next; $('load-more').hidden = next >= currentExamples.length;
  $('examples-status').textContent = `已展示 ${next} / ${currentExamples.length} 个实例；点击图像可放大。`;
}
async function select(id, changeHash = true) {
  const p = byId.get(id);
  if (!p) { $('examples-status').textContent = '请输入 0–4095 的整数维度编号。'; return; }
  selected = id; const token = ++requestId; currentExamples = []; visibleCount = 0;
  $('feature-input').value = id; $('detail-title').textContent = `维度 f${id}`;
  $('detail-stats').textContent = `${p.count} 个实例 · embedding 一致性 ${p.coherence.toFixed(3)}${p.x === null ? ' · 实例不足，未进入 UMAP' : ''}`;
  $('examples').replaceChildren(); $('load-more').hidden = true;
  $('examples-status').textContent = '正在加载实例…'; hideHover(); draw();
  if (changeHash) history.replaceState(null, '', `#f${id}`);
  try {
    if (!featureCache.has(id)) featureCache.set(id, json(`features/${id}.json`));
    const data = await featureCache.get(id);
    if (token !== requestId) return;
    currentExamples = data.examples; appendExamples();
    if (!data.examples.length) $('examples-status').textContent = '这个维度在当前样本中没有正激活实例。';
  } catch (error) {
    featureCache.delete(id); if (token === requestId) $('examples-status').textContent = error.message;
  }
}
canvas.addEventListener('pointerdown', e => {
  if (!atlas) return; canvas.setPointerCapture(e.pointerId); const [x, y] = coordinates(e);
  drag = {x, y, px: panX, py: panY, moved: false}; hideHover();
});
canvas.addEventListener('pointermove', e => {
  if (!atlas) return;
  const [x, y] = coordinates(e);
  if (drag) {
    const dx = x - drag.x, dy = y - drag.y;
    if (Math.hypot(dx, dy) > 4) drag.moved = true;
    panX = drag.px + dx; panY = drag.py + dy; draw(); return;
  }
  const p = nearest(x, y);
  canvas.style.cursor = p ? 'pointer' : 'grab';
  if (!p) { hideHover(); return; }
  if (hovered === p.feature) { if (!$('hover-preview').hidden) moveTooltip(e.clientX, e.clientY); return; }
  clearTimeout(hoverTimer); $('hover-preview').hidden = true; hovered = p.feature; draw();
  hoverTimer = setTimeout(() => preview(p, e.clientX, e.clientY), 90);
});
canvas.addEventListener('pointerup', e => {
  if (drag && !drag.moved) { const p = nearest(...coordinates(e)); if (p) select(p.feature); }
  drag = null;
});
canvas.addEventListener('pointercancel', () => { drag = null; hideHover(); });
canvas.addEventListener('pointerleave', hideHover);
canvas.addEventListener('wheel', e => {
  e.preventDefault(); if (!atlas) return; hideHover();
  const [x, y] = coordinates(e), next = Math.max(.5, Math.min(20, scale * Math.exp(-e.deltaY * .001)));
  const ratio = next / scale; panX = x - (x - panX) * ratio; panY = y - (y - panY) * ratio; scale = next; draw();
}, {passive: false});
$('jump').onclick = () => select(Number($('feature-input').value));
$('feature-input').addEventListener('keydown', e => { if (e.key === 'Enter') select(Number(e.target.value)); });
$('reset').onclick = () => { scale = 1; panX = panY = 0; hideHover(); draw(); };
$('view-mode').onchange = e => { document.body.dataset.view = e.target.value; };
$('show-box').onchange = e => { document.body.dataset.box = String(e.target.checked); };
$('load-more').onclick = appendExamples;
$('close-image').onclick = () => $('image-dialog').close();
window.addEventListener('resize', resize);
window.addEventListener('scroll', hideHover, {passive: true});
window.addEventListener('hashchange', () => { const id = /^#f(\d+)$/.exec(location.hash); if (id && byId) select(Number(id[1]), false); });
async function init() {
  try {
    atlas = await json('atlas.json'); points = atlas.points.filter(p => p.x !== null);
    byId = new Map(atlas.points.map(p => [p.feature, p]));
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    bounds = {minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys)};
    $('model-label').textContent = `SAE DIMENSION ATLAS · W${atlas.width} / k${atlas.k}`;
    $('model-spec').textContent = `${atlas.width} 维字典、k=${atlas.k}`;
    $('feature-input').max = atlas.width - 1;
    $('nav-' + atlas.width)?.setAttribute('aria-current', 'page');
    $('projection-note').textContent = `坐标相近表示实例语义相近，颜色不是分类准确率。${atlas.width - points.length} 个实例不足的维度不投影，仍可用编号查看。`;
    $('summary').textContent = `${points.length} / ${atlas.width} 个维度进入投影 · ${atlas.cities} 城市 · ${atlas.panos} 全景点位 · ${atlas.crop_count.toLocaleString()} 个激活区域`;
    resize(); const id = /^#f(\d+)$/.exec(location.hash); await select(id ? Number(id[1]) : 2073, false);
  } catch (error) { $('summary').textContent = `${error.message}，请刷新重试。`; }
}
init();
