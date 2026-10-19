(function(){
  'use strict';
  const rows=[...document.querySelectorAll('#road-table tbody tr')];
  const search=document.getElementById('road-search');
  const filter=document.getElementById('road-filter');
  const count=document.getElementById('road-count');
  function update(){
    const term=search.value.trim().toLowerCase(),kind=filter.value;
    let visible=0;
    for(const row of rows){
      const hit=row.dataset.label.toLowerCase().includes(term)&&(!kind||
        (kind==='same'&&['same','both'].includes(row.dataset.class))||
        (kind==='cross'&&['cross','both'].includes(row.dataset.class))||
        (kind==='unknown'&&row.dataset.class==='unknown'));
      row.hidden=!hit;
      visible+=hit;
    }
    count.textContent=`显示 ${visible} / ${rows.length} 个可审阅组合`;
  }
  search.addEventListener('input',update);
  filter.addEventListener('change',update);
  update();
})();
