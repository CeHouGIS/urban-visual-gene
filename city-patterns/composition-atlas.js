/* Vanilla canvas atlas: exact sampled cell counts, no interpolation. */
(async function(){
  'use strict';
  const byId=id=>document.getElementById(id);
  const get=async url=>{const r=await fetch(url);if(!r.ok)throw new Error(`${url}: ${r.status}`);return r.json();};
  try{
    const [data,cellData,encoded,examples]=await Promise.all([
      get('data/compositions.json'),get('data/composition_cells.json'),
      get('data/composition_cell_counts.json'),get('composition-examples.json')]);
    let currentPage=0, selected=null, revision=0;
    const pageSize=18, patterns=data.patterns, cities=data.cities;
    const pct=x=>(100*x).toFixed(1)+'%';
    const name=c=>c.split('/').pop();
    const cityCells=cities.map((_,i)=>cellData.cells.map((v,j)=>({v,j})).filter(r=>r.v[0]===i));
    function filtered(){
      const term=byId('pattern-search').value.trim().toLowerCase(),order=byId('pattern-order').value;
      const evidence=byId('pattern-evidence').value;
      return patterns.filter(p=>(!term||p.label.toLowerCase().includes(term))&&(!order||p.order===Number(order))&&(!evidence||p.example_count>0)).sort((a,b)=>
        (byId('pattern-sort').value==='complexity'? b.order-a.order:0)||
        Math.min(b.discovery_rate,b.evaluation_rate)-Math.min(a.discovery_rate,a.evaluation_rate));
    }
    function cardList(){
      const pool=filtered(),pages=Math.max(1,Math.ceil(pool.length/pageSize));
      currentPage=Math.min(currentPage,pages-1);
      byId('pattern-count').textContent=`${pool.length} 个组合 · 2 类 ${patterns.filter(p=>p.order===2).length} / 3 类 ${patterns.filter(p=>p.order===3).length} / 4 类 ${patterns.filter(p=>p.order===4).length}`;
      byId('pattern-page').textContent=`${currentPage+1} / ${pages}`;
      byId('prev-patterns').disabled=currentPage===0;byId('next-patterns').disabled=currentPage===pages-1;
      byId('pattern-cards').replaceChildren();
      for(const p of pool.slice(currentPage*pageSize,(currentPage+1)*pageSize)){
        const button=document.createElement('button');button.type='button';button.className='pattern-card'+(p===selected?' selected':'');
        const title=document.createElement('strong');title.textContent=p.label;button.append(title);
        if(p.audit_flags.length){const flag=document.createElement('small');flag.textContent='⚠ 抽查存在语义疑点';button.append(flag);}
        const sub=document.createElement('small');sub.textContent=`${p.order} 类 · 发现 ${pct(p.discovery_rate)} / 复查 ${pct(p.evaluation_rate)} · ${p.positive_cities} 城、${p.positive_cells.toLocaleString()} 格网${p.example_count?' · '+p.example_count+' 组四方向实例':''}`;button.append(sub);
        button.addEventListener('click',()=>select(p,true).catch(showError));byId('pattern-cards').append(button);
      }
    }
    function color(v){
      if(v===0)return 'rgb(213,220,216)';
      const stops=[[229,239,216],[96,170,147],[18,109,100]],x=Math.min(1,v)*2,k=Math.min(1,Math.floor(x)),t=x-k;
      return `rgb(${stops[k].map((a,i)=>Math.round(a+(stops[k+1][i]-a)*t)).join(',')})`;
    }
    function maps(counts){
      const root=byId('city-maps');root.replaceChildren();
      for(let i=0;i<cities.length;i++){
        const points=cityCells[i],panel=document.createElement('article');panel.className='city-map';
        const title=document.createElement('h4');title.textContent=`${name(cities[i])} · ${pct(selected.rates[i].rate)}`;
        const canvas=document.createElement('canvas');canvas.width=600;canvas.height=420;
        canvas.setAttribute('aria-label',`${cities[i]} 观测格网匹配比例`);
        const hint=document.createElement('p');hint.textContent=`${selected.rates[i].cells.toLocaleString()} / ${points.length.toLocaleString()} 格网有匹配；${selected.rates[i].positive.toLocaleString()} / ${selected.rates[i].panoramas.toLocaleString()} 全景。`;
        panel.append(title,canvas,hint);root.append(panel);
        const ctx=canvas.getContext('2d'),xs=points.map(p=>p.v[1]),ys=points.map(p=>p.v[2]);
        const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
        const scale=Math.min(540/Math.max(1,maxX-minX+1),350/Math.max(1,maxY-minY+1));
        const ox=(600-(maxX-minX+1)*scale)/2,oy=(390-(maxY-minY+1)*scale)/2;
        const lookup=new Map();
        for(const {v,j} of points){
          ctx.fillStyle=color(counts[j]/v[3]);ctx.fillRect(ox+(v[1]-minX)*scale,oy+(maxY-v[2])*scale,Math.max(1.3,scale*.92),Math.max(1.3,scale*.92));
          lookup.set(v[1]+','+v[2],{v,j});
        }
        ctx.fillStyle='#63716b';ctx.font='18px sans-serif';ctx.fillText(`E ${minX}–${maxX} km · N ${minY}–${maxY} km`,12,411);
        canvas.addEventListener('mousemove',e=>{
          const b=canvas.getBoundingClientRect(),x=(e.clientX-b.left)*600/b.width,y=(e.clientY-b.top)*420/b.height;
          const gx=Math.floor((x-ox)/scale)+minX,gy=maxY-Math.floor((y-oy)/scale),p=lookup.get(gx+','+gy);
          hint.textContent=p?`格网 (${gx}, ${gy})：${counts[p.j]} / ${p.v[3]} 全景匹配（${pct(counts[p.j]/p.v[3])}）。`:`格网 (${gx}, ${gy}) 未采样。`;
        });
      }
    }
    async function decode(value){
      const binary=Uint8Array.from(atob(value),c=>c.charCodeAt(0));
      const stream=new Blob([binary]).stream().pipeThrough(new DecompressionStream('deflate'));
      const counts=new Uint8Array(await new Response(stream).arrayBuffer());
      if(counts.length!==cellData.cells.length)throw new Error('格网记录长度不匹配');
      return counts;
    }
    async function select(p,scroll){
      selected=p;const token=++revision;location.hash='p'+p.pattern_id;cardList();
      byId('selected-title').textContent=p.label;
      byId('selected-detail').textContent=`${p.order} 类组合 · ${p.positive_panoramas.toLocaleString()} 个全景匹配 · ${p.positive_cells.toLocaleString()} 个格网 / ${p.positive_cities} 城有匹配。空间块边际零假设 lift ${p.block_lift.toFixed(2)}（描述值，未作本轮显著性确认）。`;
      if(p.audit_flags.length)byId('selected-detail').textContent+=' 抽查存在语义疑点：'+[...new Set(p.audit_flags)].join(' ');
      byId('city-rates').replaceChildren();
      for(let i=0;i<cities.length;i++){
        const line=document.createElement('div');line.className='city-rate';
        const label=document.createElement('span');label.textContent=name(cities[i]);
        const value=document.createElement('span');value.textContent=pct(p.rates[i].rate);
        const bar=document.createElement('span');bar.className='bar';const fill=document.createElement('i');fill.style.width=pct(p.rates[i].rate);bar.append(fill);
        line.append(label,value,bar);byId('city-rates').append(line);
      }
      byId('composition-evidence').replaceChildren();const evidence=examples.filter(e=>e.pattern_id===p.pattern_id);
      if(!evidence.length)byId('composition-evidence').textContent='本组合尚未补算四方向 patch 归因；可先核查支持度及城市分布。';
      for(const item of evidence){
        const detail=document.createElement('details'),summary=document.createElement('summary');summary.textContent=`${name(item.city)} · 四方向原图与高亮`;detail.append(summary);
        const desc=document.createElement('p');desc.textContent='panoid: '+item.panoid;detail.append(desc);
        if(item.review_note){const note=document.createElement('p');note.className='note';note.textContent='图像抽查：'+item.review_note;detail.append(note);}
        item.labels.forEach((label,i)=>{const key=document.createElement('span');key.className='mask-key';key.style.background=item.colors[i];key.textContent=label;detail.append(key);});
        for(const kind of ['original','overlay']){const a=document.createElement('a');a.href=item[kind];const img=document.createElement('img');img.src=item[kind];img.loading='lazy';img.alt=kind==='original'?'0／90／180／270 度完整四方向原图':'多个响应单元的激活高亮';a.append(img);detail.append(a);}
        byId('composition-evidence').append(detail);
      }
      byId('map-status').textContent='正在绘制格网分布…';
      const counts=await decode(encoded[p.pattern_id]);if(token!==revision)return;maps(counts);
      byId('map-status').textContent=`30 城 ${counts.length.toLocaleString()} 个观测格网；采用同一组冻结阈值。`;
      if(scroll)byId('distribution').scrollIntoView({behavior:'smooth',block:'start'});
    }
    function showError(error){byId('map-status').textContent='地图读取失败：'+error.message;console.error(error);}
    for(const id of ['pattern-search','pattern-order','pattern-sort','pattern-evidence'])byId(id).addEventListener('input',()=>{currentPage=0;cardList();});
    byId('prev-patterns').addEventListener('click',()=>{currentPage--;cardList();});
    byId('next-patterns').addEventListener('click',()=>{currentPage++;cardList();});
    const hash=location.hash.slice(2),initial=patterns.find(p=>p.pattern_id===hash)||[...patterns].sort((a,b)=>Math.min(b.discovery_rate,b.evaluation_rate)-Math.min(a.discovery_rate,a.evaluation_rate)).find(p=>p.order===3&&p.example_count>0)||patterns[0];
    if(initial)await select(initial,false);else byId('pattern-count').textContent='当前支持度规则下没有重复组合。';
  }catch(error){byId('pattern-count').textContent='图谱读取失败：'+error.message;console.error(error);}
})();
