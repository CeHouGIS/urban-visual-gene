(() => {
  'use strict';
  const script = document.currentScript;
  const dataURL = new URL('../data/project-progress.json', script.src);
  const siteURL = new URL('../', script.src);
  const panel = document.querySelector('#project-progress');
  const body = document.querySelector('#project-progress-rows');
  if (!panel || !body) return;
  const narrow = matchMedia('(max-width: 1100px)');
  panel.open = !narrow.matches;
  narrow.addEventListener('change', event => { panel.open = !event.matches; });
  const labels = {complete: '已完成', partial: '待补齐', running: '进行中', waiting: '等待资源', pending: '待开始', paused: '已暂停', error: '需处理', unknown: '待核实'};
  const icons = {complete: '✓', running: '●', waiting: '◷', pending: '○', paused: 'Ⅱ', error: '!', unknown: '?'};
  let lastSnapshot = null;
  let loading = false;
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function meter(parent, value, label) {
    if (!value || !value.total) return;
    const percent = Math.min(100, Math.max(0, value.current / value.total * 100));
    const caption = el('div', 'project-progress-meter-label', `${percent.toFixed(1)}% · ${value.current.toLocaleString()} / ${value.total.toLocaleString()} ${value.unit}`);
    const bar = el('progress');
    bar.max = value.total;
    bar.value = value.current;
    bar.setAttribute('aria-label', `${label}：${caption.textContent}`);
    parent.append(caption, bar);
  }
  function render(data) {
    if (data.schema_version !== 1 || !Array.isArray(data.rows)) throw new Error('Unknown progress format');
    const research = data.research;
    document.querySelector('#project-research-goal').textContent = research.goal;
    document.querySelector('#project-research-focus').textContent = research.focus;
    document.querySelector('#project-next-step').textContent = research.next_step;
    document.querySelector('#project-milestone-count').textContent = `研究里程碑 · ${research.completed} / ${research.milestones.length} 阶段已完成`;
    const milestones = document.querySelector('#project-milestones');
    const openMilestones = new Set([...milestones.querySelectorAll('details[open]')].map(d => d.dataset.id));
    const firstRender = !lastSnapshot;
    milestones.replaceChildren();
    research.milestones.forEach((item, index) => {
      const detail = el('details', 'project-milestone');
      detail.dataset.id = item.id;
      detail.open = openMilestones.has(item.id) || (firstRender && item.id === 'patterns');
      const summary = el('summary');
      summary.append(el('span', 'project-milestone-number', String(index + 1).padStart(2, '0')), el('span', 'project-milestone-name', item.label), el('span', `project-progress-badge is-${item.status}`, labels[item.status] || '待核实'));
      const content = el('div', 'project-milestone-content');
      content.append(el('p', 'project-milestone-question', item.question), el('p', '', item.evidence));
      if (item.remaining) content.append(el('p', 'project-milestone-remaining', '待完成：' + item.remaining));
      content.append(el('p', 'project-milestone-criterion', '完成判据：' + item.criterion));
      detail.append(summary, content);
      milestones.append(detail);
    });
    const blockers = document.querySelector('#project-blockers');
    blockers.replaceChildren();
    if (research.blockers.length) {
      blockers.append(el('strong', '', '当前阻碍'));
      research.blockers.forEach(text => blockers.append(el('p', '', text)));
    }
    const scope = document.querySelector('#project-research-scope');
    scope.replaceChildren(el('p', '', research.scope_note));
    research.caveats.forEach(text => scope.append(el('p', '', text)));
    scope.append(el('p', '', `写作与核查计划复核：${research.reviewed_on}；运行进度来自后台状态。`));
    document.querySelector('#project-runtime-count').textContent = `${data.rows.filter(r => r.status === 'complete').length}/${data.rows.length}`;
    body.replaceChildren();
    for (const row of data.rows) {
      const tr = el('tr');
      tr.dataset.stage = row.id;
      const cell = el('td');
      const title = el(row.link ? 'a' : 'span', 'project-progress-title', row.label);
      if (row.link) title.href = new URL(row.link, siteURL);
      cell.append(title, el('div', 'project-progress-detail', row.detail));
      meter(cell, row.progress, row.label);
      if (row.runs) {
        const runs = el('ul', 'project-progress-runs');
        for (const run of row.runs) {
          const item = el('li');
          item.append(el('span', `project-progress-run-icon is-${run.status}`, icons[run.status] || '?'), el('span', '', run.label));
          item.append(el('div', 'project-progress-detail', `${labels[run.status] || '待核实'} · ${run.detail}`));
          meter(item, run.progress, run.label);
          runs.append(item);
        }
        cell.append(runs);
      }
      const status = el('td');
      status.append(el('span', `project-progress-badge is-${row.status}`, labels[row.status] || '待核实'));
      tr.append(cell, status);
      body.append(tr);
    }
    document.querySelector('#project-progress-headline').textContent = data.headline;
    document.querySelector('#project-progress-note').textContent = data.note;
    const updated = new Date(data.updated_utc);
    const time = document.querySelector('#project-progress-time');
    time.dateTime = data.updated_utc;
    time.textContent = new Intl.DateTimeFormat('zh-CN', {timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false}).format(updated) + ' 北京时间';
    lastSnapshot = data;
  }
  function freshness() {
    if (!lastSnapshot) return;
    const mins = Math.max(0, Math.floor((Date.now() - new Date(lastSnapshot.updated_utc).getTime()) / 60000));
    const finished = lastSnapshot.rows.every(r => r.status === 'complete');
    document.querySelector('#project-progress-freshness').textContent = finished ? '计算任务已完成 · 研究结论仍按里程碑核查' : `发布快照 · ${mins < 1 ? '刚刚' : mins + ' 分钟前'}更新 · 每分钟检查`;
  }
  async function refresh() {
    if (loading) return;
    loading = true;
    const button = document.querySelector('#project-progress-refresh');
    button.disabled = true;
    try {
      const url = new URL(dataURL);
      url.searchParams.set('t', Math.floor(Date.now() / 60000));
      const response = await fetch(url, {cache: 'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (data.updated_utc !== lastSnapshot?.updated_utc) render(data);
      freshness();
    } catch (error) {
      document.querySelector('#project-progress-freshness').textContent = lastSnapshot ? '刷新失败，保留上次快照；可稍后重试。' : '进度暂时无法读取，请点击刷新重试。';
    } finally {
      loading = false;
      button.disabled = false;
    }
  }
  document.querySelector('#project-progress-refresh').addEventListener('click', refresh);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refresh(); });
  setInterval(() => { if (!document.hidden) refresh(); }, 60000);
  refresh();
})();
