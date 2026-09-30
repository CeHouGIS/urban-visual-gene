#!/usr/bin/env python3
"""Resumable subprocess runner for the graph vocabulary experiment."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from .utils import DEFAULT_CONFIG

STAGES=[
 '01_prepare_views','02_build_graphs','03_run_leiden_sweep','04_select_stable_scales',
 '05_export_communities','06_validate_visual_coherence','07_generate_semantic_evidence',
 '08_rerun_city_composition','09_rerun_cooccurrence','10_rerun_spatial_analysis',
 '11_compare_old_new','12_make_figures']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=DEFAULT_CONFIG); p.add_argument('--from-stage',type=int,default=1); p.add_argument('--to-stage',type=int,default=12); p.add_argument('--force',action='store_true'); a=p.parse_args()
 for number,name in enumerate(STAGES,1):
  if not a.from_stage<=number<=a.to_stage: continue
  executable=sys.executable
  if number==7:
   import yaml
   raw=yaml.safe_load(a.config.read_text()); executable=str(raw['semantic'].get('python_executable',sys.executable))
  cmd=[executable,'-m',f'scripts.multicity.graph_visual_vocabulary.{name}','--config',str(a.config)]
  if a.force and number not in (5,11,12): cmd.append('--force')
  print(f'\n=== stage {number:02d}: {name} ===',flush=True); subprocess.run(cmd,check=True)
if __name__=='__main__': main()
