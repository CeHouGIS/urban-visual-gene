#!/bin/bash
cd /workplace/urban_visual_gene
echo "########## VIENNA SWEEP $(date +%T) ##########"
/opt/conda/bin/python -u -m scripts.run_sampling_sweep --city Vienna --sizes 500 2000 5000 --K 32 --epochs 50
echo "########## HONGKONG SWEEP $(date +%T) ##########"
/opt/conda/bin/python -u -m scripts.run_sampling_sweep --city HongKong --sizes 500 2000 --K 32 --epochs 50
echo "########## ALL SWEEPS DONE $(date +%T) ##########"
