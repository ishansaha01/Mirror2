#!/bin/bash

# Setup script for Princeton Adroit cluster
# This script sets up the environment for the eventfulness project on Adroit

# Load necessary modules
module purge
module load anaconda3/2023.3
module load cudatoolkit/11.1
module load cudnn/cuda-11.1/8.2.0

# Create and activate conda environment
conda create -n eventfulness python=3.9 -y
source activate eventfulness

# Install PyTorch with CUDA 11.1 support
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install librosa matplotlib
pip install psutil
pip install torchmetrics
pip install tensorboard
pip install setuptools==59.5.0
pip install setuptools
pip install protobuf
pip install audioread
pip install six
pip install av

echo "Environment setup complete!"
