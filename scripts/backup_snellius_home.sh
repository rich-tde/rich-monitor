#!/bin/bash
rclone copy snellius:/home/yhe/snellius /data1/yujiehe/snellius-backup --exclude="/rich-data**" --log-file=/data1/yujiehe/snellius-logs/rclone-home.log  --log-file-max-size 1M  --log-file-max-backups 3
