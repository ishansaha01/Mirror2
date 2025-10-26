#!/bin/bash

# Create a Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv eventfulness_env

# Activate the virtual environment
echo "Activating virtual environment..."
source eventfulness_env/bin/activate

# Install PyTorch for Mac (CPU version since most Macs don't have CUDA)
echo "Installing PyTorch for Mac..."
pip3 install torch torchvision torchaudio

# Install other required packages
echo "Installing other dependencies..."
pip3 install librosa matplotlib
pip3 install psutil
pip3 install torchmetrics
pip3 install tensorboard
pip3 install setuptools==59.5.0
pip3 install setuptools
pip3 install protobuf
pip3 install audioread
pip3 install six
pip3 install av

echo "Setup complete! To activate the environment, run:"
echo "source eventfulness_env/bin/activate"