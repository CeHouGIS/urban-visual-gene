#!/bin/bash
set -e
cd /workplace/urban_visual_gene
echo "===== VIENNA $(date +%T) =====" 
/opt/conda/bin/python -u run_experiment.py --city Vienna --max-panos 500 --K 32 --epochs 50 --batch-size 32 --skip-stage1
echo "===== HONGKONG $(date +%T) ====="
/opt/conda/bin/python -u run_experiment.py --city HongKong --max-panos 500 --K 32 --epochs 50 --batch-size 32 --skip-stage1
echo "===== ALL DONE $(date +%T) ====="
