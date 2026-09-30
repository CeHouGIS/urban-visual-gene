#!/usr/bin/env python3
"""Post-hoc Qwen interpretation of fixed Leiden community evidence sheets."""
from __future__ import annotations

import scripts._env  # noqa: F401

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.multicity.interpret_sae_dimensions import DimensionInterpreter, extract_json_object
from .utils import DEFAULT_CONFIG, assert_safe_affinity, ensure_layout, load_config, save_json

SEMANTIC_TYPES={"object","material","color_light","texture","geometry","spatial_layout","environment","artifact","mixed","unclear"}
def normalize(value:dict,community_id:str)->dict:
    kind=str(value.get('semantic_type','unclear')).strip().lower()
    if kind not in SEMANTIC_TYPES: kind='mixed' if '|' in kind else 'unclear'
    cues=value.get('visual_cues_zh',[]); cues=cues if isinstance(cues,list) else [str(cues)]
    def probability(key,default):
        try: return min(1.,max(0.,float(value.get(key,default))))
        except (TypeError,ValueError): return default
    return {"community_id":community_id,"name_zh":str(value.get('name_zh','未明确社区')).strip(),"name_en":str(value.get('name_en','unclear community')).strip(),"semantic_type":kind,"summary_zh":str(value.get('summary_zh','')).strip(),"visual_cues_zh":[str(x).strip() for x in cues if str(x).strip()][:2],"urban_meaning_zh":str(value.get('urban_meaning_zh','不明确')).strip(),"artifact_probability":probability('artifact_probability',.5),"confidence":probability('confidence',0.)}


def prompt(community_id: str, size: int, medoid: int) -> str:
    return f"""你是一名城市街景视觉表征研究助手。图中展示已经由无监督图社区发现固定的视觉社区 {community_id}（{size} 个 latent dimensions，graph medoid D{medoid:03d}）的代表证据。每个样本格依次为四方向拼接原始全景、激活叠加图、最大激活局部裁剪。语义标签绝未参与社区发现。

请保守概括跨样本反复出现的共同视觉模式。热力图色彩是人工叠加，不是原图语义；若证据混杂或主要是接缝、黑边、模糊、文字、设备等，应标记 artifact 或 unclear。只输出合法 JSON，不要 Markdown：
{{"community_id":"{community_id}","name_zh":"2-10字","name_en":"short label","semantic_type":"object|material|color_light|texture|geometry|spatial_layout|environment|artifact|mixed|unclear","summary_zh":"不超过50字","visual_cues_zh":["证据1","证据2"],"urban_meaning_zh":"不超过40字","artifact_probability":0.0,"confidence":0.0}}"""


def run(config: dict, force: bool = False) -> dict:
    assert_safe_affinity(); ensure_layout(config)
    data=config["paths"]["paper_data_root"]; figures=config["paths"]["paper_figure_root"]
    metrics=pd.read_csv(data/"validation/community_graph_metrics.csv")
    raw_dir=data/"semantic/raw_outputs"; raw_dir.mkdir(parents=True,exist_ok=True)
    pending=[row for row in metrics.itertuples(index=False) if force or not (raw_dir/f"{row.community_id}.json").exists()]
    interpreter=None
    if pending:
        interpreter=DimensionInterpreter(config["semantic"]["model_dir"],int(config["semantic"]["max_new_tokens"]),int(config["semantic"]["image_max_side"]))
    for index,row in enumerate(pending,1):
        image=figures/"community_evidence"/f"{row.community_id}.jpg"
        response=interpreter.generate(image,prompt(row.community_id,int(row.community_size),int(row.medoid_dimension_id)))
        try:
            parsed=normalize(extract_json_object(response),row.community_id); error=""
        except Exception as exc:
            parsed=normalize({"name_zh":"解析失败","name_en":"parse failure","semantic_type":"unclear","artifact_probability":.5},row.community_id); error=str(exc)
        save_json(raw_dir/f"{row.community_id}.json",{"result":parsed,"raw_response":response,"parse_error":error,"post_hoc_only":True})
        print(f"semantic [{index:02d}/{len(pending)}] {row.community_id}: {parsed.get('name_zh','')}",flush=True)
    records=[]
    for row in metrics.itertuples(index=False):
        path=raw_dir/f"{row.community_id}.json"; payload=json.loads(path.read_text()); value=normalize(payload["result"],row.community_id)
        if value != payload["result"]: payload["result"]=value; save_json(path,payload)
        records.append({"community_id":row.community_id,"community_size":int(row.community_size),"medoid_dimension":int(row.medoid_dimension_id),"name_zh":value.get("name_zh",""),"name_en":value.get("name_en",""),"semantic_type":value.get("semantic_type","unclear"),"summary_zh":value.get("summary_zh",""),"visual_cues_zh":json.dumps(value.get("visual_cues_zh",[]),ensure_ascii=False),"urban_meaning_zh":value.get("urban_meaning_zh",""),"artifact_probability":value.get("artifact_probability",.5),"confidence":value.get("confidence",0),"semantic_role":"post-hoc interpretation only"})
    pd.DataFrame(records).to_csv(data/"semantic/community_labels.csv",index=False)
    return {"communities":len(records),"newly_interpreted":len(pending),"labels_used_for_discovery":False}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--force",action="store_true"); args=parser.parse_args(); print(run(load_config(args.config),args.force))
if __name__=="__main__": main()
