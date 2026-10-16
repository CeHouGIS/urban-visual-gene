(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  const states = {candidate_ready:'暂定要素候选',needs_review:'待核查',insufficient_evidence:'证据不足',unobserved:'无正激活证据',context_or_material:'背景 / 材料 / 上下文'};
  const flags = {no_interpretation:'尚无解释',fewer_than_five_examples:'少于5个实例',mixed_or_unclear:'共同语义混合或不明',low_self_reported_confidence:'模型自述信心低',ambiguous_or_unmapped_label:'名称含糊或尚未归类',non_element_or_unclear_response:'非要素或不明确回答',embedding_mixture_flag:'embedding证据提示混合',masked_vs_unmasked_family_disagreement:'遮罩与未遮罩解释的父类不一致'};
  let data, selected, limit = 100;
  function el(tag, text, className) {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
  function filtered() {
    const query=$('#element-search').value.trim().toLowerCase(), family=$('#element-family').value, status=$('#element-status').value, conflict=$('#element-conflicts').checked;
    return data.dimensions.filter(r=>(!query||[String(r.feature),'f'+r.feature,r.label_zh,r.family_zh,r.reference_label_zh].some(v=>v.toLowerCase().includes(query)))&&(!family||r.family===family)&&(!status||r.status===status)&&(!conflict||r.family_disagreement)).sort((a,b)=>a.review_priority-b.review_priority||a.feature-b.feature);
  }
  function listing() {
    const rows=filtered(), list=$('#element-list');list.replaceChildren();
    for(const row of rows.slice(0,limit)) {
      const button=el('button',`f${row.feature} · ${row.label_zh||'未解释'}`);
      button.type='button';button.setAttribute('aria-pressed',String(row.feature===selected));
      button.append(el('small',`${row.family_zh} · ${states[row.status]}`));
      button.addEventListener('click',()=>{location.hash='f'+row.feature;});list.append(button);
    }
    $('#element-count').textContent=`匹配 ${rows.length.toLocaleString()} / 4,096 维 · 当前显示 ${Math.min(rows.length,limit)}`;
    $('#element-more').hidden=rows.length<=limit;
  }
  function select() {
    const match=location.hash.match(/^#f(\d+)$/);if(!match)return;
    const row=data.dimensions.find(r=>r.feature===Number(match[1]));if(!row)return;selected=row.feature;
    if(!filtered().some(r=>r.feature===selected)){
      $('#element-search').value='';$('#element-family').value='';$('#element-status').value='';$('#element-conflicts').checked=false;limit=100;
    }
    const detail=$('#element-detail');detail.replaceChildren(el('h3',`f${row.feature} · ${row.label_zh||'无解释证据'}`));
    detail.append(el('p',`细标签：${row.fine_label_zh||'—'} → 父类：${row.family_zh}`),el('p',`${states[row.status]} · ${row.crop_count}个实例 · ${row.source==='vlm_dimensions_top10_expanded'?'扩大样本补充':'固定验证样本'} · 人工独立验证未完成`));
    if(row.quality_flags.length)detail.append(el('p',row.quality_flags.map(f=>flags[f]||f).join('；'),'element-flags'));
    detail.append(el('p','共同依据：'+(row.cues_zh.join('；')||'—')),el('p','局限：'+(row.caveats_zh.join('；')||'—')));
    detail.append(el('p',row.provisional_graph_candidate?'可用于探索性关系输入；候选资格不等于语义已验证。':'当前不作为独立景观要素参与组合多样性计数；原始激活保留。'));
    if(row.main_evidence_url){const link=el('a','查看原图、激活与四方向全景');link.href=row.main_evidence_url;detail.append(link);}
    function figure(url, caption){const fig=el('figure'),a=el('a'),img=el('img');a.href=url;a.target='_blank';a.rel='noopener';img.src=url;img.alt=caption;img.loading='lazy';a.append(img);fig.append(a,el('figcaption',caption));detail.append(fig);}
    if(row.sheet)figure(row.sheet,'实际masked Top10输入：只保留高激活区域；灰色是人工遮罩。点击放大。');
    if(row.family_disagreement){detail.append(el('p',`未遮罩Top100参考解释：${row.reference_label_zh}；当前masked解释：${row.label_zh}。两者名称冲突须结合图像判断，不能投票定真值。`));if(row.reference_sheet)figure(row.reference_sheet,'未遮罩Top100输入的最多前10例预览（展示压缩）；完整原图证据可从上方逐维链接查看。');}
    listing();
  }
  async function main(){
    const response=await fetch('hierarchy.json');if(!response.ok)throw Error('要素表载入失败');data=await response.json();
    for(const family of data.families){const option=el('option',family.label_zh);option.value=family.id;$('#element-family').append(option);}
    for(const [value,label] of Object.entries(states)){const option=el('option',label);option.value=value;$('#element-status').append(option);}
    for(const id of ['#element-search','#element-family','#element-status','#element-conflicts'])$(id).addEventListener(id==='#element-search'?'input':'change',()=>{
      limit=100;const rows=filtered();
      if(rows.length&&!rows.some(r=>r.feature===selected)){location.hash='f'+rows[0].feature;select();}
      else if(!rows.length){selected=undefined;$('#element-detail').replaceChildren(el('p','没有匹配的维度，请调整筛选条件。'));listing();}
      else listing();
    });
    $('#element-more').addEventListener('click',()=>{limit+=100;listing();});addEventListener('hashchange',select);
    listing();if(!location.hash)location.hash='f'+filtered()[0].feature;select();
  }
  main().catch(error=>{$('#element-count').textContent=error.message;});
})();
