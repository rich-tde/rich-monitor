#!/bin/bash
source /home/yujiehe/.bashrc
conda activate rich
python "$(dirname "$0")/dispatch_diagnostics.py" "$@"
