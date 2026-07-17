#!/usr/bin/env bash
set -e

cd /home/pi/yolo
source /home/pi/yolo/venv313/bin/activate

exec python3 /home/pi/yolo/production/QR_detection_colors.py \
  --source picamera0 \
  --resolution 1920x1080 \
  --decode_width 640 \
  --detector yolo \
  --primary_detector hybrid \
  --model qryolo26n_ncnn_model2 \
  --conf 0.20 \
  --yolo_decode_crop \
  --strong_crop_decode \
  --yolo_pad 60 \
  --yolo_imgsz 640 \
  --yolo_fallback_mode presence \
  --yolo_decode_topk 3 \
  --max_det 8 \
  --scan_every 10 \
  --max_lost 150 \
  --trail_seconds 30 \
  --trail_len 60 \
  --track_match_dist 200 \
  --color_fix none \
  --camera_fps 15 \
  --headless \
  --quiet \
  --backend_url "https://tesis-backend-service.onrender.com/iot/v1/automation/inventory/merchandise/" \
  --auth_url "https://tesis-backend-service.onrender.com/iot/v1/auth/login/camera"