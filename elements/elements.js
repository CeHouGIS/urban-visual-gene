(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  const states = {candidate_ready:'暂定要素候选',needs_review:'待核查',insufficient_evidence:'证据不足',unobserved:'无正激活证据',context_or_material:'背景 / 材料 / 上下文'};
  const flags = {no_interpretation:'尚无解释',fewer_than_five_examples:'少于5个实例',mixed_or_unclear:'共同语义混合或不明',low_self_reported_confidence:'模型自述信心低',ambiguous_or_unmapped_label:'名称含糊或尚未归类',non_element_or_unclear_response:'非要素或不明确回答',embedding_mixture_flag:'embedding证据提示混合',masked_vs_unmasked_family_disagreement:'遮罩与未遮罩解释的父类不一致',support_below_screening_minimum:'正激活全景不足20个，暂停命名'};
  let data, categoryGraph, selected, limit = 100, activeFamily, activeLabel;
  function el(tag, text, className) {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
  function categoryButton(label, count, maximum, selected) {
    const button=el('button',undefined,'category-item');
    button.type='button';button.setAttribute('aria-pressed',String(selected));
    const title=el('span',undefined,'category-item-title');
    title.append(el('span',label),el('strong',count.toLocaleString()));
    const track=el('span',undefined,'category-track');
    const fill=el('span',undefined,'category-fill');
    fill.style.width=`${100*count/maximum}%`;
    track.append(fill);button.append(title,track);
    return button;
  }
  function renderCategoryDetail(family, label) {
    const detail=$('#category-feature-detail');detail.replaceChildren();
    if(!label){
      detail.append(el('p',`${family.label}包含 ${family.label_count} 个细标签；选择细标签查看其 SAE 维度。`));
      return;
    }
    detail.append(el('h3',`${family.label} › ${label.label} · ${label.count.toLocaleString()} 维`));
    const galleryLink=el('a',`查看 ${label.examples.length} 个全景激活实例 ↓`);
    galleryLink.href='#label-activations';detail.append(galleryLink);
    const intro=el('p','点击维度编号查看高激活图像与 Qwen 原始解释。');
    detail.append(intro);
    const links=el('div',undefined,'category-feature-links');
    function show(count) {
      links.replaceChildren();
      for(const feature of label.features.slice(0,count)){
        const link=el('a',`f${feature}`);link.href=`#f${feature}`;
        link.addEventListener('click',()=>{
          if(location.hash===link.getAttribute('href'))select();
          requestAnimationFrame(()=>$('#element-detail').scrollIntoView({behavior:'smooth',block:'start'}));
        });
        links.append(link);
      }
    }
    show(Math.min(label.count,24));detail.append(links);
    if(label.count>24){
      const more=el('button',`显示全部 ${label.count.toLocaleString()} 维`,'category-feature-more');
      more.type='button';more.addEventListener('click',()=>{show(label.count);more.remove();});detail.append(more);
    }
  }
  function sourceImage(path){return new Promise((resolve,reject)=>{
    const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(Error(`原图载入失败：${path}`));image.src=path;
  });}
  async function drawActivation(canvas,path,heat){
    const [image,map]=await Promise.all([sourceImage(path),sourceImage(`data:image/png;base64,${heat}`)]);
    canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
    const context=canvas.getContext('2d');context.drawImage(image,0,0);
    const layer=document.createElement('canvas');layer.width=map.naturalWidth;layer.height=map.naturalHeight;
    const paint=layer.getContext('2d',{willReadFrequently:true});paint.drawImage(map,0,0);
    const pixels=paint.getImageData(0,0,layer.width,layer.height);
    for(let i=0;i<pixels.data.length;i+=4){
      const value=pixels.data[i]/255;
      pixels.data[i]=255;pixels.data[i+1]=Math.round(210*Math.min(1,value*1.5));
      pixels.data[i+2]=30;pixels.data[i+3]=Math.round(225*Math.max(0,(value-.08)/.92));
    }
    paint.putImageData(pixels,0,0);context.imageSmoothingEnabled=false;
    context.drawImage(layer,0,0,canvas.width,canvas.height);
  }
  function panoramaRow(example,withHeat){
    const row=el('div',undefined,'label-panorama');
    for(let index=0;index<4;index++){
      const view=el('div',undefined,'label-view');
      const heading=index*90;
      if(withHeat&&heading===example.heading){
        const canvas=el('canvas');canvas.setAttribute('aria-label',`${heading}°方向 SAE 激活叠加图`);
        view.append(canvas);
        drawActivation(canvas,example.views[index],example.heat).catch(error=>view.append(el('p',error.message)));
      }else{
        const image=el('img');image.src=example.views[index];image.loading='lazy';
        image.alt=`${example.city}，${heading}°方向原图`;view.append(image);
      }
      view.append(el('small',`${heading}°${withHeat&&heading===example.heading?' · 高亮':''}`));
      row.append(view);
    }
    return row;
  }
  function openPanorama(example,label){
    const dialog=$('#label-panorama-dialog'),body=$('#label-dialog-body');body.replaceChildren();
    $('#label-dialog-title').textContent=`${label} · ${example.city.split('/').pop()} · f${example.feature}`;
    body.append(el('div','四方向原图','label-row-title'),panoramaRow(example,false));
    body.append(el('div','激活高亮','label-row-title'),panoramaRow(example,true));
    dialog.showModal();
  }
  function renderActivationGallery(family,label){
    const box=$('#label-gallery');box.replaceChildren();
    if(!label){box.append(el('p','选择上方细标签查看全景激活实例。'));return;}
    box.append(el('h3',`${family.label} › ${label.label} · ${label.examples.length} / 5 例`));
    if(!label.eligible){box.append(el('p','这一组按证据状态归档，不是景观语义类别。请在下方逐维查看原始激活。','note'));return;}
    if(!label.examples.length){
      box.append(el('p','现有激活样本中没有该标签对应维度的正激活，无法提供真实高亮图像。','note'));
      return;
    }
    for(const [index,example] of label.examples.entries()){
      const item=el('article',undefined,'label-example');
      const header=el('div',undefined,'label-example-heading');
      const title=el('strong',`${index+1}. ${example.city.split('/').pop()} · f${example.feature}`);
      const link=el('a','查看逐维证据');link.href=`../semantic/feature.html?f=${example.feature}`;
      const enlarge=el('button','放大全景');enlarge.type='button';
      enlarge.addEventListener('click',()=>openPanorama(example,label.label));
      header.append(title,enlarge,link);item.append(header);
      item.append(el('p',`pano ${example.pano} · 峰值 ${example.activation.toFixed(4)} · ${example.source==='vlm_dimensions_top10_expanded'?'扩大样本补充':'固定验证样本'}`,'label-example-meta'));
      item.append(el('div','四方向原图','label-row-title'),panoramaRow(example,false));
      item.append(el('div','激活高亮 · 黄色越亮响应越强','label-row-title'),panoramaRow(example,true));
      box.append(item);
    }
  }
  function renderCategories() {
    const families=$('#category-families'), labels=$('#category-labels');
    const family=categoryGraph.families.find(item=>item.id===activeFamily);
    families.replaceChildren();labels.replaceChildren();
    $('#category-family-total').textContent=`${categoryGraph.family_count} 类 · ${categoryGraph.total.toLocaleString()} 维`;
    $('#category-path-family').textContent=family.label;
    $('#category-path-label').textContent=activeLabel||'选择细标签';
    $('#category-label-heading').textContent=`${family.label} · ${family.label_count} 个细标签`;
    for(const item of categoryGraph.families){
      const button=categoryButton(item.label,item.count,categoryGraph.families[0].count,item.id===activeFamily);
      button.addEventListener('click',()=>{activeFamily=item.id;activeLabel=item.labels[0]?.label;renderCategories();});
      families.append(button);
    }
    const maximum=family.labels[0]?.count||1;
    for(const item of family.labels){
      const button=categoryButton(item.label,item.count,maximum,item.label===activeLabel);
      button.addEventListener('click',()=>{activeLabel=item.label;renderCategories();});
      labels.append(button);
    }
    const label=family.labels.find(item=>item.label===activeLabel);
    renderCategoryDetail(family,label);renderActivationGallery(family,label);
  }
  function filtered() {
    const query=$('#element-search').value.trim().toLowerCase(), family=$('#element-family').value, status=$('#element-status').value, conflict=$('#element-conflicts').checked;
    return data.dimensions.filter(r=>(!query||[String(r.feature),'f'+r.feature,r.label_zh,r.raw_qwen_label_zh,r.family_zh,r.reference_label_zh].some(v=>String(v||'').toLowerCase().includes(query)))&&(!family||r.family===family)&&(!status||r.status===status)&&(!conflict||r.family_disagreement)).sort((a,b)=>a.review_priority-b.review_priority||a.feature-b.feature);
  }
  function listing() {
    const rows=filtered(), list=$('#element-list');list.replaceChildren();
    for(const row of rows.slice(0,limit)) {
      const button=el('button',`f${row.feature} · ${row.display_label_zh||'未解释'}`);
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
    const detail=$('#element-detail');detail.replaceChildren(el('h3',`f${row.feature} · ${row.display_label_zh||'无解释证据'}`));
    detail.append(el('p',`细标签：${row.fine_label_zh||'—'} → 父类：${row.family_zh}`),el('p',`${states[row.status]} · ${row.crop_count}个 Qwen 输入实例 · ${row.source==='vlm_dimensions_top10_expanded'?'扩大样本补充':'固定验证样本'} · 人工独立验证未完成`));
    if(row.screening){detail.append(el('p',`正激活：${row.screening.positive_panoramas} / ${row.screening.support_scope==='fixed_validation'?'3,712 个固定验证':'86,038 个扩大样本'}全景；Top 图像涉及 ${row.screening.top_crop_cities} 个城市。${row.screening.position_review?'峰值位置集中，待受控检验；':''}位置编码尚未确认。`));}
    if(!row.screening.label_eligible&&row.raw_qwen_label_zh){detail.append(el('p',`旧 Qwen 猜测：“${row.raw_qwen_label_zh}”。样本不足，不能作为景观标签。`,'element-flags'));}
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
    const [response, graphResponse]=await Promise.all([fetch('hierarchy.json'),fetch('category-graph.json')]);
    if(!response.ok||!graphResponse.ok)throw Error('要素表或类别图谱载入失败');
    [data,categoryGraph]=await Promise.all([response.json(),graphResponse.json()]);
    if(data.identity!==categoryGraph.identity||data.dimension_screening_identity!==categoryGraph.dimension_screening_identity||data.dimensions.length!==categoryGraph.total)throw Error('类别图谱与维度表不匹配');
    const query=new URLSearchParams(location.search);
    const requestedFamily=categoryGraph.families.find(family=>family.id===query.get('family'));
    activeFamily=requestedFamily?.id||categoryGraph.families.find(family=>family.id==='building')?.id||categoryGraph.families[0].id;
    const initialFamily=categoryGraph.families.find(family=>family.id===activeFamily);
    activeLabel=initialFamily.labels.find(label=>label.label===query.get('label'))?.label||initialFamily.labels[0].label;
    renderCategories();
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
