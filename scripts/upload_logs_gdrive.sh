#!/bin/bash
rclone sync /data1/yujiehe/snellius-logs/ gdrive:snellius-logs \
    --exclude "*.log" \
    --exclude "*.json" \
    --exclude "archived/**" \
    --log-file=/data1/yujiehe/snellius-logs/rclone-gdrive.log \
    --log-file-max-size 1M \
    --log-file-max-backups 3
