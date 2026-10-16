(() => {
  const select = document.getElementById('relation-slot');
  const detail = document.getElementById('relation-detail');
  const count = document.getElementById('relation-slot-count');
  if (!select || !detail) return;
  const familyNames = {building:'建筑及附属部件',tree:'乔木及其部件',low_vegetation:'草灌及低矮植被',road:'道路及车行路面',pedestrian:'步行空间',barrier:'围合与防护设施',vehicle:'车辆及其部件',person:'行人',utility:'市政杆线设施',sign:'标识与交通控制',furniture:'街道家具',construction:'施工设施'};
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
      const leaders = node('p', 'relation-families');
      leaders.append(node('strong', '', '响应较高的候选要素：'));
      leaders.append(document.createTextNode(slot.mean_element_strengths.map(item => familyNames[item.family] || item.family).join(' · ')));
      detail.append(leaders);
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
        image.alt = `${item.city}，${item.heading}度方向的${item.region_size}乘${item.region_size} patch区域，四方向原图与裁剪`;
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
