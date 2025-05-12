#!/bin/bash

LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"

echo "Running preprocess_subject.py..."
python src/preprocess_subject_data.py 2>&1 | tee "$LOG_DIR/preprocess.log"

echo "Running main.py..."
python src/main.py 2>&1 | tee "$LOG_DIR/main.log"

echo "Scripts finished. Container will stay alive for inspection."

# Keep container alive
while true; do sleep 1000; done
