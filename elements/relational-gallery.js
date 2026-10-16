(() => {
  const select = document.getElementById('relation-slot');
  const detail = document.getElementById('relation-detail');
  const count = document.getElementById('relation-slot-count');
  if (!select || !detail) return;
  const node = (tag, className, value) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (value !== undefined) element.textContent = value;
    return element;
  };
  fetch('prototype-examples.json').then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }).then(data => {
    for (const slot of data.slots) {
      const option = node('option', '', `槽位 ${String(slot.slot).padStart(2, '0')} · ${slot.development_regions} 个区域`);
      option.value = slot.slot;
      select.append(option);
    }
    const render = () => {
      const slot = data.slots[Number(select.value)];
      detail.replaceChildren();
      count.textContent = `${slot.development_regions} 个开发区域 · ${slot.cities} 城市`;
      const summary = node('p', 'relation-layout-summary');
      summary.append(node('strong', '', '学习到的布局：'));
      summary.append(document.createTextNode(slot.layout.leaders.slice(0, 5).map(item => `${item.label_zh}偏${item.position_zh}`).join('；')));
      detail.append(summary);
      const legend = node('div', 'relation-legend');
      for (const item of slot.layout.leaders) {
        const entry = node('span', 'relation-legend-item');
        const swatch = node('i', 'relation-swatch');
        swatch.style.backgroundColor = item.color;
        entry.append(swatch, document.createTextNode(`${item.label_zh} · ${item.position_zh}`));
        legend.append(entry);
      }
      detail.append(legend);
      const grid = node('div', 'relation-grid');
      for (const item of slot.examples) {
        const figure = node('figure', 'relation-example');
        const link = node('a');
        link.href = `examples/${item.image}`;
        link.target = '_blank';
        link.rel = 'noopener';
        const image = node('img');
        image.loading = 'lazy';
        image.src = link.href;
        image.alt = `${item.city}，${item.heading}度方向的${item.region_size}乘${item.region_size} patch区域，四方向原图、学习布局mask与裁剪`;
        link.append(image);
        figure.append(link);
        figure.append(node('figcaption', '', `${item.city} · ${item.heading}° · ${item.region_size}×${item.region_size} patch · 开发MSE ${item.mse.toFixed(3)}`));
        grid.append(figure);
      }
      detail.append(grid);
      history.replaceState(null, '', `#relation-slot-${slot.slot}`);
    };
    const match = location.hash.match(/^#relation-slot-(\d+)$/);
    if (match && Number(match[1]) < data.slots.length) select.value = Number(match[1]);
    select.addEventListener('change', render);
    render();
  }).catch(error => { count.textContent = `实例载入失败：${error.message}`; });
})();
