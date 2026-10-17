(() => {
  'use strict';
  const $ = selector => document.querySelector(selector);
  let data, mode = 'hierarchy', selection = 0;

  function labelLink(family, label, count) {
    const link = document.createElement('a');
    const query = new URLSearchParams({family, label});
    link.href = `../elements/?${query}#qwen-taxonomy`;
    link.textContent = `${label} · ${count} 维`;
    link.title = `查看“${label}”的 SAE 维度与全景激活`;
    return link;
  }

  function currentRows() {
    const family = $('#structure-family').value;
    const query = $('#structure-search').value.trim().toLowerCase();
    const rows = data[mode];
    return rows.filter(row => {
      if (family && row.family !== family) return false;
      const words = mode === 'unlinked' ? [row.label, row.family_label] :
        [row.name, row.parent?.label, row.family_label,
         ...(mode === 'hierarchy' ? row.children : row.labels).map(child => child.label)];
      return !query || words.some(word => String(word || '').toLowerCase().includes(query));
    });
  }

  function renderDetail(row) {
    const detail = $('#structure-detail');
    detail.replaceChildren();
    if (!row) {
      const text = document.createElement('p');
      text.textContent = '没有匹配的标签关系。';
      detail.append(text);
      return;
    }
    const heading = document.createElement('h2');
    heading.textContent = mode === 'hierarchy' ? row.parent.label :
      mode === 'parallel' ? row.name : row.label;
    detail.append(heading);
    const note = document.createElement('p');
    note.textContent = mode === 'hierarchy' ?
      row.relation === 'is_a' ? `以下标签是“${row.parent.label}”的具体类型。` :
        `以下标签是“${row.parent.label}”的部件，不是它的子类型。` :
      mode === 'parallel' ? row.basis === 'shared_parent' ?
        `这些标签同属“${row.parent}”，且关系类型相同；同层并列不表示它们互斥。` :
        `这些标签属于同一宽泛类别；不据此推断彼此的上下级。` :
      '尚未凭现有 Qwen 标签指定可靠的同层或上下级关系。';
    detail.append(note);
    const lead = document.createElement('div');
    lead.className = 'structure-lead';
    lead.textContent = row.family_label + (mode === 'hierarchy' ?
      ` · ${row.relation === 'is_a' ? '属于某类' : '整体与部件'}` : '');
    detail.append(lead);
    if (mode === 'hierarchy') {
      const parent = document.createElement('div');
      parent.className = 'structure-parent';
      parent.append(labelLink(row.family, row.parent.label, row.parent.count));
      detail.append(parent);
    }
    const members = mode === 'hierarchy' ? row.children : mode === 'parallel' ? row.labels : [row];
    const list = document.createElement('div');
    list.className = 'structure-members';
    for (const member of members) list.append(labelLink(row.family, member.label, member.count));
    detail.append(list);
    const footnote = document.createElement('p');
    footnote.className = 'muted';
    footnote.textContent = '点击标签可查看对应维度的四方向原图、激活高亮和逐维解释。';
    detail.append(footnote);
  }

  function render() {
    const rows = currentRows();
    const list = $('#structure-list');
    list.replaceChildren();
    $('#structure-count').textContent = `${mode === 'hierarchy' ? '上下级' : mode === 'parallel' ? '同层并列' : '未判定'} ${rows.length} 组 · ${data.counts.linked_labels} / ${data.counts.eligible_labels} 个合格标签进入待核查关系`;
    if (selection >= rows.length) selection = 0;
    for (const [index, row] of rows.entries()) {
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('aria-pressed', String(index === selection));
      const title = document.createElement('strong');
      title.textContent = mode === 'hierarchy' ? row.parent.label :
        mode === 'parallel' ? row.name : row.label;
      const meta = document.createElement('small');
      meta.textContent = row.family_label + ' · ' + (mode === 'hierarchy' ?
        `${row.children.length} 个${row.relation === 'is_a' ? '类型' : '部件'}` :
        mode === 'parallel' ? `${row.labels.length} 个同层标签` : `${row.count} 维`);
      button.append(title, meta);
      button.onclick = () => {selection = index; render();};
      list.append(button);
    }
    renderDetail(rows[selection]);
  }

  async function main() {
    const response = await fetch('structure.json');
    if (!response.ok) throw Error('标签结构数据载入失败');
    data = await response.json();
    for (const family of data.families) {
      const option = document.createElement('option');
      option.value = family.id;
      option.textContent = family.label;
      $('#structure-family').append(option);
    }
    for (const next of ['hierarchy', 'parallel', 'unlinked']) {
      $('#' + next + '-tab').onclick = () => {
        mode = next;
        selection = 0;
        for (const tab of ['hierarchy', 'parallel', 'unlinked'])
          $('#' + tab + '-tab').setAttribute('aria-pressed', String(tab === next));
        render();
      };
    }
    $('#structure-family').onchange = () => {selection = 0; render();};
    $('#structure-search').oninput = () => {selection = 0; render();};
    render();
  }
  main().catch(error => {$('#structure-count').textContent = error.message;});
})();
