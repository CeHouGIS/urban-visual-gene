/* Current-model gallery: one template -> several independently mapped views. */
(() => {
  const root = document.getElementById('prototype-atlas');
  if (!root) return;
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pid = p => `P${String(p.prototype).padStart(3, '0')}`;
  const modes = {overlay:'贡献热图', original:'街景原图', crop:'局部裁剪', masked:'保留主要贡献区域'};
  let all = [], filtered = [], current, mode = 'overlay';
  root.innerHTML = `<div class="pg-toolbar">
    <label>模板尺度<select id="pg-scale"><option value="all">全部尺度</option><option value="3">3 × 3 patches</option><option value="7">7 × 7 patches</option><option value="15">15 × 15 patches</option></select></label>
    <label>筛选原型<input id="pg-search" type="search" placeholder="编号 P012 / f123 / 城市" aria-label="按原型编号、组成维度或城市筛选"></label>
    <label class="pg-template-select">选择原型<select id="pg-select"></select></label>
  </div><p class="pg-count" id="pg-count" role="status" aria-live="polite"></p>
  <div id="pg-detail"></div><details id="pg-catalog-panel"><summary>展开全部原型目录</summary><div class="pg-catalog" id="pg-catalog"></div></details>
  <dialog class="pg-dialog" id="pg-dialog" aria-labelledby="pg-dialog-title"></dialog>`;
  const select = root.querySelector('#pg-select'), detail = root.querySelector('#pg-detail');
  const catalog = root.querySelector('#pg-catalog'), dialog = root.querySelector('#pg-dialog');
  const smallNumber = value => Number(value).toLocaleString('en', {maximumFractionDigits:3});
  function readHash() { const match = location.hash.match(/^#p(\d+)$/i); return match ? Number(match[1]) : null; }
  function show(p, updateHash = true) {
    if (!p) return;
    current = p; select.value = String(p.prototype);
    if (updateHash) history.replaceState(null, '', `#p${p.prototype}`);
    catalog.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.id) === p.prototype)));
    const index = filtered.indexOf(p), cities = new Set(p.instances.map(x => x.city)).size;
    detail.innerHTML = `<div class="pg-heading"><h3>${pid(p)} <span class="tag">${p.scale} × ${p.scale} patches</span></h3>
      <div class="controls"><button id="pg-prev" ${index <= 0 ? 'disabled' : ''}>← 上一个</button><button id="pg-next" ${index >= filtered.length - 1 ? 'disabled' : ''}>下一个 →</button></div></div>
      <p>${p.instances.length}个不同点位 · ${cities}座城市 · 模板组成中权重最高的SAE维度：</p>
      <div class="pg-dimensions">${p.top_dimensions.slice(0,12).map(d => `<a href="../semantic/feature.html?f=${d.dimension}" target="_blank" rel="noopener" title="查看SAE维度f${d.dimension}的解释；模板权重${smallNumber(d.weight)}">f${d.dimension}</a>`).join('')}</div>
      <details><summary>查看模板空间布局与组成权重</summary><img class="pg-template-figure" src="${esc(p.template)}" loading="lazy" alt="${pid(p)}的相对空间布局与主要SAE维度权重"></details>
      ${p.instances.length ? `<div class="pg-modes" aria-label="实例显示方式">${Object.entries(modes).map(([key,label]) => `<button data-mode="${key}" aria-pressed="${key === mode}">${label}</button>`).join('')}</div>
      <p class="pg-count">点击任一实例，可同时对照原图、贡献热图、裁剪和保留区域。热图颜色在每个实例内部归一化。</p>
      <div class="pg-examples">${p.instances.map((r,i) => `<article class="pg-example"><button class="pg-image-button" data-instance="${i}" aria-label="放大${pid(p)}实例${i+1}，${esc(r.city)}"><img src="${esc(r[mode])}" loading="lazy" alt="${pid(p)}实例${i+1}：${esc(r.city)}，方向${r.heading}°，${modes[mode]}"></button><h4>${String(i+1).padStart(2,'0')} · ${esc(r.city)} · ${r.heading}°</h4><p>响应 ${smallNumber(r.coefficient_fp32)} · 移除后误差增加 ${smallNumber(r.removing_response_delta_sse)}</p><p>点位 ${esc(r.panoid)}</p></article>`).join('')}</div>` : `<div class="pg-empty">${p.status === 'no_positive_response' ? '该模板在完整开发集没有正响应，暂无可展示实例。' : '候选未通过正响应、正重建贡献及原图可读性核验，暂无可展示实例。'}模板槽位保留，不补入其他模型的图片。</div>`}`;
    detail.querySelector('#pg-prev').onclick = () => show(filtered[index-1]);
    detail.querySelector('#pg-next').onclick = () => show(filtered[index+1]);
    detail.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => { mode = b.dataset.mode; show(p, false); });
    detail.querySelectorAll('[data-instance]').forEach(b => b.onclick = () => openInstance(p, Number(b.dataset.instance)));
  }
  function openInstance(p, index) {
    const r = p.instances[index];
    dialog.innerHTML = `<div class="pg-dialog-head"><h2 id="pg-dialog-title">${pid(p)} · 实例${index+1} · ${esc(r.city)} · ${r.heading}°</h2><button id="pg-close" aria-label="关闭实例对照">关闭 ×</button></div>
      <p>模板覆盖${p.scale}×${p.scale} patches；实际边缘裁剪范围 [${r.box_patch.join(', ')}]。灰色遮罩按模板贡献峰值的25%截取，不是物体分割。</p>
      <div class="pg-dialog-images">${Object.entries(modes).sort(([a],[b]) => (a==='original' ? -1 : b==='original' ? 1 : 0)).map(([key,label]) => `<figure><a href="${esc(r[key])}" target="_blank" rel="noopener"><img src="${esc(r[key])}" alt="${pid(p)}实例${index+1} ${label}"></a><figcaption>${label} · 点击查看大图</figcaption></figure>`).join('')}</div>
      <p>点位：${esc(r.panoid)} · 移除该局部响应后 SSE 增加 ${smallNumber(r.removing_response_delta_sse)}。正贡献说明其参与重建，语义一致性仍需逐实例核查。</p>`;
    dialog.querySelector('#pg-close').onclick = () => dialog.close();
    dialog.showModal();
  }
  dialog.addEventListener('click', e => { if (e.target === dialog) { const b = dialog.getBoundingClientRect(); if (e.clientX < b.left || e.clientX > b.right || e.clientY < b.top || e.clientY > b.bottom) dialog.close(); } });
  function filter(preferred = current?.prototype, updateHash = true) {
    const scale = root.querySelector('#pg-scale').value;
    const text = root.querySelector('#pg-search').value.trim().toLowerCase();
    filtered = all.filter(p => (scale === 'all' || p.scale === Number(scale)) && (!text ||
      pid(p).toLowerCase().includes(text) || String(p.prototype) === text ||
      p.top_dimensions.some(d => `f${d.dimension}` === text) || p.instances.some(r => r.city.toLowerCase().includes(text))));
    root.querySelector('#pg-count').textContent = `显示 ${filtered.length} / ${all.length} 个模板；每个模板可切换查看最多8个点位。`;
    select.innerHTML = filtered.map(p => `<option value="${p.prototype}">${pid(p)} · ${p.scale}×${p.scale} · ${p.instances.length}个实例</option>`).join('');
    catalog.innerHTML = filtered.map(p => `<button data-id="${p.prototype}" aria-pressed="false" class="${p.instances.length ? '' : 'pg-inactive'}"><strong>${pid(p)} · ${p.scale}×${p.scale}</strong><span class="pg-preview">${p.instances.slice(0,3).map(r => `<img src="${esc(r.crop)}" loading="lazy" alt="${pid(p)}局部实例，${esc(r.city)}">`).join('')}</span><small>${p.instances.length ? `${p.instances.length}个点位 · ${new Set(p.instances.map(r=>r.city)).size}座城市` : '暂无核验通过的实例'}</small></button>`).join('');
    catalog.querySelectorAll('button').forEach(b => b.onclick = () => { show(all.find(p => p.prototype === Number(b.dataset.id))); detail.scrollIntoView({behavior:'smooth', block:'start'}); });
    if (!filtered.length) { detail.innerHTML = '<p class="pg-empty">没有匹配的原型，请调整筛选条件。</p>'; current = null; return; }
    show(filtered.find(p => p.prototype === preferred) || filtered[0], updateHash);
  }
  select.onchange = () => show(all.find(p => p.prototype === Number(select.value)));
  root.querySelector('#pg-scale').onchange = () => filter();
  root.querySelector('#pg-search').oninput = () => filter();
  window.addEventListener('hashchange', () => { const id = readHash(); if (id === null) return; root.querySelector('#pg-scale').value = 'all'; root.querySelector('#pg-search').value = ''; filter(id, false); });
  fetch('../data/prototype-gallery.json?v=1').then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }).then(data => {
    all = data.prototypes; const linked = readHash(); filter(linked ?? all.find(p => p.instances.length)?.prototype, false); root.setAttribute('aria-busy', 'false');
    if (linked !== null) detail.scrollIntoView({block:'start'});
  }).catch(error => { root.innerHTML = `<p role="alert">实例数据未能载入：${esc(error.message)}。请刷新重试，或下载下方完整索引。</p>`; root.setAttribute('aria-busy','false'); });
})();
