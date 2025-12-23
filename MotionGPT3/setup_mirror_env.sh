#!/bin/bash
# MotionGPT3 Setup Script for Mirror2 Repository
# Large files (deps, checkpoints) are stored on /scratch and symlinked

set -e

echo "=========================================="
echo "MotionGPT3 Setup for Mirror2"
echo "=========================================="

# Paths
SCRATCH_DATA="/scratch/network/is1893/motiongpt3_data"
SCRATCH_VENV="/scratch/network/is1893/MotionGPT3_env/venv"
PROJECT_DIR="/home/is1893/Mirror2/MotionGPT3"
PIP_CACHE="/scratch/network/is1893/pip_cache"

# Set pip cache to scratch
export PIP_CACHE_DIR=$PIP_CACHE
mkdir -p $PIP_CACHE_DIR

# Create venv on scratch if it doesn't exist
if [ ! -d "$SCRATCH_VENV" ]; then
    echo "Creating virtual environment on scratch..."
    mkdir -p /scratch/network/is1893/MotionGPT3_env
    python3.9 -m venv --without-pip $SCRATCH_VENV
    source $SCRATCH_VENV/bin/activate
    curl -sS https://bootstrap.pypa.io/get-pip.py | python
else
    source $SCRATCH_VENV/bin/activate
fi

echo "✓ Activated venv (Python $(python --version 2>&1 | cut -d' ' -f2))"

cd $PROJECT_DIR

echo ""
echo "Step 1: Installing Python dependencies..."
echo "-----------------------------------------"

# Install PyTorch 2.0 with CUDA 11.8
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.0.0+cu118 \
    torchtext==0.15.0

# Install core dependencies
pip install \
    bert_score==0.3.13 \
    einops==0.8.1 \
    hydra-core==1.3.2 \
    imageio==2.36.1 \
    gdown==5.2.0 \
    moviepy==1.0.3 \
    omegaconf==2.3.0 \
    orjson \
    pandas \
    peft \
    Pillow \
    psutil \
    pyrender==0.1.45 \
    pytorch_lightning==2.0.0 \
    rich \
    scikit-learn \
    scipy \
    Shapely \
    shortuuid \
    spacy==3.7.6 \
    torchmetrics==0.7.0 \
    transformers==4.47.1 \
    trimesh==3.9.24

echo ""
echo "Step 2: Downloading spaCy model..."
echo "-----------------------------------"
python -m spacy download en_core_web_sm

# Create data directory on scratch if needed
mkdir -p $SCRATCH_DATA

echo ""
echo "Step 3: Downloading dependencies..."
echo "------------------------------------"

# Temporarily cd to scratch for downloads, then symlink back
cd $SCRATCH_DATA

# Download SMPL model
if [ ! -d "$SCRATCH_DATA/deps/smpl_models" ]; then
    cd $PROJECT_DIR
    bash prepare/download_smpl_model.sh
    [ -d "deps" ] && mv deps/* $SCRATCH_DATA/deps/ 2>/dev/null || true
fi

# Prepare GPT2
if [ ! -d "$SCRATCH_DATA/deps/gpt2" ]; then
    cd $PROJECT_DIR  
    bash prepare/prepare_gpt2.sh
    [ -d "deps" ] && mv deps/* $SCRATCH_DATA/deps/ 2>/dev/null || true
fi

# Download T2M evaluators
if [ ! -d "$SCRATCH_DATA/deps/t2m" ]; then
    cd $PROJECT_DIR
    bash prepare/download_t2m_evaluators.sh
    [ -d "deps" ] && mv deps/* $SCRATCH_DATA/deps/ 2>/dev/null || true
fi

# Download MLD pretrained models  
if [ ! -f "$SCRATCH_DATA/checkpoints/1222_mld_humanml3d_FID041.ckpt" ]; then
    cd $PROJECT_DIR
    bash prepare/download_mld_pretrained_models.sh
    [ -d "checkpoints" ] && mv checkpoints/* $SCRATCH_DATA/checkpoints/ 2>/dev/null || true
fi

cd $PROJECT_DIR

echo ""
echo "Step 4: Processing checkpoints..."
echo "-----------------------------------"
python -m scripts.gen_mot_gpt

echo ""
echo "Step 5: Downloading pretrained MotionGPT3 model..."
echo "----------------------------------------------------"
if [ ! -f "$SCRATCH_DATA/checkpoints/motiongpt3.ckpt" ]; then
    gdown --fuzzy "https://drive.google.com/file/d/1Wvx5PGJjVKPRvjcl8firChw1UVjUj36l/view?usp=drive_link" -O $SCRATCH_DATA/checkpoints/motiongpt3.ckpt
fi

# Ensure symlinks exist
rm -f $PROJECT_DIR/deps $PROJECT_DIR/checkpoints 2>/dev/null || true
ln -sf $SCRATCH_DATA/deps $PROJECT_DIR/deps
ln -sf $SCRATCH_DATA/checkpoints $PROJECT_DIR/checkpoints

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "To use MotionGPT3:"
echo "  source /home/is1893/Mirror2/MotionGPT3/activate.sh"
echo ""
echo "Or manually:"
echo "  source /scratch/network/is1893/MotionGPT3_env/venv/bin/activate"
echo "  cd /home/is1893/Mirror2/MotionGPT3"
echo ""
