(() => {
  const list = document.getElementById('atlas-list');
  const status = document.getElementById('atlas-status');
  const pageLabel = document.getElementById('atlas-page');
  const previous = document.getElementById('atlas-prev');
  const next = document.getElementById('atlas-next');
  const search = document.getElementById('atlas-search');
  const sort = document.getElementById('atlas-sort');
  const opacity = document.getElementById('atlas-opacity');
  const dialog = document.getElementById('atlas-dialog');
  const perPage = 12;
  const names = { stable:'仅稳定', split:'发生分流', unresolved:'关系未确定', inactive:'测试集未激活', orphan:'无近似旧对应' };
  let atlas;
  let category = 'all';
  let page = 0;
  const expanded = new Set();
  const focused = new Map();

  function escape(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
  }

  function keyFor(row) { return `${row.kind === 'orphan' ? 'c' : 'p'}${row.id}`; }

  function records() {
    if (!atlas) return [];
    const source = category === 'orphan' ? atlas.unassigned_children : atlas.parents;
    const query = search.value.trim().toLowerCase();
    const numeric = /^(?:(1024):)?f?(\d+)$/.exec(query);
    return source.filter(row => {
      if (category !== 'all' && category !== 'orphan' && row.kind !== category) return false;
      if (!query) return true;
      if (numeric) {
        const id = Number(numeric[2]);
        if (category === 'orphan') return row.id === id;
        if (!numeric[1] && id < 512) return row.id === id;
        return row.stable_child === id || row.edges?.some(edge => edge.child === id);
      }
      return row.examples.some(example => example.city.toLowerCase().includes(query));
    }).sort((a,b) => sort.value === 'id' ? a.id - b.id : b.support - a.support || a.id - b.id);
  }

  function tile(example, width, feature, label) {
    if (!example) return '';
    const image = escape(example.image);
    const mask = feature === null ? '' : escape(example.maps[`${width}:${feature}`] || '');
    const title = escape(`${example.city} · ${example.heading}° · ${label}`);
    return `<button type="button" class="atlas-tile" data-action="zoom" data-image="${image}" data-mask="${mask}" data-title="${title}" aria-label="放大 ${title}"><img loading="lazy" src="${image}" alt="${escape(example.city)} 街景">${mask ? `<img loading="lazy" class="atlas-mask" src="${mask}" alt="">` : ''}<span>${escape(label)}</span></button>`;
  }

  function selectedChild(row) {
    const key = keyFor(row);
    if (focused.has(key)) return focused.get(key);
    if (row.kind === 'inactive') return null;
    if (row.kind === 'orphan') return row.id;
    return row.edges?.[0]?.child ?? null;
  }

  function selectedExample(row, child) {
    return row.examples.find(example => example.focus_child === child) || row.examples[0];
  }

  function metrics(row, child) {
    if (row.kind === 'orphan') return `最近旧特征 f${row.nearest_parent} · 最大 decoder ${row.max_decoder_cosine} · 最大激活 ${row.max_activation_cosine}`;
    const edge = row.edges?.find(item => item.child === child);
    if (row.kind === 'inactive') return '仅为预激活候选；该特征在 151,040 个测试 patch 中没有稀疏激活';
    if (!edge) return row.split_coverage !== null ? `两个子特征合计覆盖父激活 ${Math.round(row.split_coverage * 100)}%` : '点击子特征查看对应指标';
    const overlap = edge.support_jaccard !== undefined ? ` · 支持集 Jaccard ${edge.support_jaccard}` : '';
    return `${edge.kind === 'reference' ? '最近特征，仅供比较 · ' : ''}decoder ${edge.decoder_cosine} · 激活 ${edge.activation_cosine}${overlap}`;
  }

  function flow(row, child) {
    if (row.kind === 'orphan') {
      return `<button type="button" class="atlas-parent" data-action="focus" data-key="${keyFor(row)}" data-child="parent" aria-pressed="${child === null}">W512 f${row.nearest_parent}?</button><span class="atlas-arrow">⇢</span><button type="button" class="atlas-child reference" data-action="focus" data-key="${keyFor(row)}" data-child="${row.id}" aria-pressed="${child === row.id}">W1024 f${row.id}</button>`;
    }
    const parent = `<button type="button" class="atlas-parent" data-action="focus" data-key="${keyFor(row)}" data-child="parent" aria-pressed="${child === null}">W512 f${row.id}</button>`;
    if (!row.edges?.length) return `${parent}<span class="atlas-flow-empty">未找到可靠后继</span>`;
    return `${parent}<span class="atlas-arrow">${row.kind === 'unresolved' || row.kind === 'inactive' ? '⇢' : '→'}</span>${row.edges.map(edge => `<button type="button" class="atlas-child ${edge.kind === 'reference' ? 'reference' : edge.kind.includes('split') ? 'split' : ''}" data-action="focus" data-key="${keyFor(row)}" data-child="${edge.child}" aria-pressed="${child === edge.child}" title="${edge.kind === 'reference' ? '未通过稳定或分流阈值；仅供比较' : '点击查看该后继的街景响应'}">W1024 f${edge.child}</button>`).join('')}`;
  }

  function exampleStrip(row, example) {
    if (!example) return '';
    const parent = row.kind === 'orphan' ? row.nearest_parent : row.id;
    const children = row.kind === 'orphan' ? [row.id] : [...new Set((row.edges || []).map(edge => edge.child))];
    return `<div class="atlas-sample"><h4>${escape(example.city)} · ${example.heading}°${row.kind === 'split' ? ` · 子特征 f${example.focus_child} 的独占激活实例` : ''}</h4><div class="atlas-sample-images">${tile(example, 512, null, '街景原图')}${tile(example, 512, parent, `W512 f${parent}`)}${children.map(id => tile(example, 1024, id, `W1024 f${id}`)).join('')}</div></div>`;
  }

  function rowMarkup(row) {
    const child = selectedChild(row);
    const example = selectedExample(row, child);
    const parent = row.kind === 'orphan' ? row.nearest_parent : row.id;
    const childTile = child === null ? '' : tile(example, 1024, child, `W1024 f${child}`);
    const caption = example ? `${escape(example.city)} · ${example.heading}°${row.kind === 'inactive' ? ' · 未被 TopK 选中' : ''}` : '当前没有可核验的街景实例';
    const key = keyFor(row);
    return `<article class="atlas-row" id="feature-${key}"><div class="atlas-top"><div class="atlas-identity"><span class="atlas-id">${row.kind === 'orphan' ? 'W1024' : 'W512'} f${row.id}</span><span class="atlas-kind ${row.kind}">${names[row.kind]}</span><span class="atlas-support">稀疏激活 patch：${row.support.toLocaleString()}</span></div><div><div class="atlas-flow">${flow(row, child)}</div><div class="atlas-metrics">${metrics(row, child)}</div></div><div><div class="atlas-preview">${example ? `${tile(example, 512, null, '原图')}${tile(example, 512, parent, `W512 f${parent}`)}${childTile}` : '<span class="atlas-preview-empty">没有可核验图片</span>'}</div><div class="atlas-meta"><span>${caption}</span><button type="button" class="atlas-expand" data-action="expand" data-key="${key}" aria-expanded="${expanded.has(key)}">${expanded.has(key) ? '收起证据 ↑' : '展开全部证据 ↓'}</button></div></div></div>${expanded.has(key) ? `<div class="atlas-extra"><p>${row.kind === 'split' ? '两张街景分别突出不同子特征；一个父特征可同时拥有稳定后继。' : row.kind === 'inactive' ? '这张街景仅用于检查稠密预激活；该旧特征在测试集的稀疏输出中从未被选中。' : row.kind === 'orphan' ? '左侧最近旧特征仅作参照；它未达到可靠单一对应标准。' : '在相同街景上对照父子特征的正预激活，关系统计仍以稀疏输出为准。'}</p>${row.examples.map(example => exampleStrip(row, example)).join('') || '<p>没有可核验图片。</p>'}</div>` : ''}</article>`;
  }

  function render() {
    if (!atlas) return;
    const rows = records();
    const pages = Math.max(1, Math.ceil(rows.length / perPage));
    page = Math.min(page, pages - 1);
    const visible = rows.slice(page * perPage, (page + 1) * perPage);
    list.innerHTML = visible.map(rowMarkup).join('') || '<p class="atlas-status">没有符合筛选条件的特征。</p>';
    status.textContent = category === 'orphan' ? `W1024 无近似旧对应候选：${rows.length} 个` : `W512 旧特征：${rows.length} / 512 个`;
    pageLabel.textContent = `${page + 1} / ${pages}`;
    previous.disabled = page === 0;
    next.disabled = page >= pages - 1;
  }

  document.getElementById('atlas-buckets').addEventListener('click', event => {
    const button = event.target.closest('button[data-kind]');
    if (!button) return;
    category = button.dataset.kind;
    page = 0;
    document.querySelectorAll('#atlas-buckets button').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    render();
  });
  search.addEventListener('input', () => { page = 0; render(); });
  sort.addEventListener('change', () => { page = 0; render(); });
  opacity.addEventListener('input', () => {
    const strength = Number(opacity.value) / 100;
    list.style.setProperty('--heat-opacity', strength);
    dialog.style.setProperty('--heat-opacity', strength);
    list.style.setProperty('--base-brightness', 1 - .28 * strength);
    dialog.style.setProperty('--base-brightness', 1 - .28 * strength);
  });
  previous.addEventListener('click', () => { if (page > 0) { page--; render(); document.getElementById('atlas').scrollIntoView(); } });
  next.addEventListener('click', () => { page++; render(); document.getElementById('atlas').scrollIntoView(); });
  list.addEventListener('click', event => {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    const action = button.dataset.action;
    if (action === 'zoom') {
      document.getElementById('atlas-dialog-base').src = button.dataset.image;
      if (button.dataset.mask) document.getElementById('atlas-dialog-mask').src = button.dataset.mask;
      else document.getElementById('atlas-dialog-mask').removeAttribute('src');
      document.getElementById('atlas-dialog-mask').hidden = !button.dataset.mask;
      document.getElementById('atlas-dialog-title').textContent = button.dataset.title;
      dialog.showModal();
      return;
    }
    if (action === 'expand') {
      expanded.has(button.dataset.key) ? expanded.delete(button.dataset.key) : expanded.add(button.dataset.key);
      render();
      return;
    }
    if (action === 'focus') {
      focused.set(button.dataset.key, button.dataset.child === 'parent' ? null : Number(button.dataset.child));
      render();
    }
  });
  document.getElementById('atlas-dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });

  fetch('atlas_512_1024.json').then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }).then(data => {
    if (data.parents.length !== 512 || data.unassigned_children.length !== 30) throw new Error('实例清单数量不完整');
    atlas = data;
    render();
  }).catch(error => { status.textContent = `实例清单加载失败：${error.message}`; });
})();
