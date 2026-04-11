#!/bin/bash
rclone copy snellius:/scratch-shared/yhe/rich-data /disks/emrdata/YujieSnellius --log-file=/data1/yujiehe/snellius-logs/rclone-scratch.log --log-file-max-size 1M  --log-file-max-backups 3