(() => {
  'use strict';
  const root = document.querySelector('#semantic-hierarchy-map');
  const summary = document.querySelector('#map-summary');
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      drawActivation(entry.target).catch(() => {
        entry.target.replaceWith(document.createTextNode('图像载入失败'));
      });
    }
  }, {rootMargin: '180px'});

  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text !== undefined) element.textContent = text;
    if (className) element.className = className;
    return element;
  }
  function image(url) {
    return new Promise((resolve, reject) => {
      const item = new Image();
      item.onload = () => resolve(item);
      item.onerror = reject;
      item.src = url;
    });
  }
  async function drawActivation(canvas) {
    const [original, heat] = await Promise.all([
      image(canvas.dataset.image), image(`data:image/png;base64,${canvas.dataset.heat}`)
    ]);
    canvas.width = 320;
    canvas.height = 320;
    const context = canvas.getContext('2d');
    context.drawImage(original, 0, 0, 320, 320);
    const layer = document.createElement('canvas');
    layer.width = heat.naturalWidth;
    layer.height = heat.naturalHeight;
    const paint = layer.getContext('2d', {willReadFrequently: true});
    paint.drawImage(heat, 0, 0);
    const pixels = paint.getImageData(0, 0, layer.width, layer.height);
    for (let i = 0; i < pixels.data.length; i += 4) {
      const value = pixels.data[i] / 255;
      pixels.data[i] = 255;
      pixels.data[i + 1] = Math.round(210 * Math.min(1, value * 1.5));
      pixels.data[i + 2] = 30;
      pixels.data[i + 3] = Math.round(225 * Math.max(0, (value - .08) / .92));
    }
    paint.putImageData(pixels, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(layer, 0, 0, 320, 320);
    delete canvas.dataset.heat;
  }
  function renderFamily(family) {
    const row = node('section', undefined, 'map-row');
    row.id = `family-${family.id}`;
    row.dataset.role = family.role;
    const parent = node('div', undefined, 'map-parent');
    parent.append(node('h2', family.label_en),
      node('small', `${family.label} · ${family.count.toLocaleString()} 维 · ${family.children.length} 子类`));
    const childColumn = node('div', undefined, 'map-child-column');
    const children = node('div', undefined, 'map-children');
    const detail = node('div', '点击编号查看子类名称与维度数。', 'map-child-detail');
    family.children.forEach((child, index) => {
      const button = node('button', String(index + 1).padStart(2, '0'), 'map-child');
      button.type = 'button';
      button.title = `${child.label} · ${child.count} 维`;
      button.setAttribute('aria-label', `${family.label}，子类 ${index + 1}：${child.label}，${child.count} 维`);
      button.addEventListener('click', () => {
        children.querySelectorAll('button').forEach(item => item.setAttribute('aria-pressed', 'false'));
        button.setAttribute('aria-pressed', 'true');
        detail.replaceChildren(node('strong', `${String(index + 1).padStart(2, '0')} · ${child.label}`),
          node('span', `${child.count.toLocaleString()} 维`));
      });
      children.append(button);
    });
    childColumn.append(children, detail);
    const gallery = node('div', undefined, 'map-gallery');
    family.examples.forEach(example => {
      const link = node('a', undefined, 'map-thumb');
      link.href = `../elements/#f${example.feature}`;
      link.title = `${example.label} · f${example.feature} · ${example.city} · ${example.heading}°`;
      const canvas = node('canvas');
      canvas.dataset.image = example.image;
      canvas.dataset.heat = example.heat;
      canvas.setAttribute('role', 'img');
      canvas.setAttribute('aria-label', `${example.label}，f${example.feature} 激活高亮`);
      link.append(canvas, node('small', `f${example.feature} · ${example.label}`));
      gallery.append(link);
      observer.observe(canvas);
    });
    row.append(parent, childColumn, gallery);
    return row;
  }
  fetch('map.json').then(response => {
    if (!response.ok) throw Error('语义图谱载入失败');
    return response.json();
  }).then(data => {
    if (data.family_count !== data.families.length || data.leaf_count !== 4096) throw Error('父类或叶节点数量与图谱不符');
    for (const family of data.families) root.append(renderFamily(family));
    summary.textContent = `${data.family_count} 个父类 · ${data.total.toLocaleString()} 个 SAE 维度 · 证据不足 ${data.weak_count.toLocaleString()} 维 · 无正激活 ${data.zero_count.toLocaleString()} 维`;
  }).catch(error => { summary.textContent = error.message; });
})();
