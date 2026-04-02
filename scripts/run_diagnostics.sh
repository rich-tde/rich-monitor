#!/bin/bash
source ~/.bashrc
conda activate rich
python diagnostics.py /home/yujiehe/rich-analysis/YujieSnellius/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR/snap_full_68.h5 /data1/yujiehe/snellius-logs "$@"