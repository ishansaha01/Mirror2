#!/bin/bash
# MotionGPT3 Activation Script for Mirror2
# This script activates the MotionGPT3 environment

# Activate the venv on scratch
source /scratch/network/is1893/MotionGPT3_env/venv/bin/activate

# Set pip cache to scratch to avoid quota issues  
export PIP_CACHE_DIR=/scratch/network/is1893/pip_cache

# Navigate to MotionGPT3 directory
cd /home/is1893/Mirror2/MotionGPT3

echo "=========================================="
echo "MotionGPT3 Environment Activated"
echo "=========================================="
echo "Python: $(python --version 2>&1)"
echo "Working directory: $(pwd)"
echo ""
echo "Available commands:"
echo "  python demo.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt"
echo "  python app.py  # Launch WebUI on port 8888"
echo "=========================================="

