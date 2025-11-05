#!/bin/bash

# Script to create conda environment in scratch space
# This version uses the scratch directory which has a much larger quota (95GB)

# Note: We don't use 'set -e' here because some pip commands may return non-zero
# exit codes for warnings, but we handle errors explicitly below

# Load anaconda module
echo "Loading anaconda module..."
module purge 2>/dev/null || true
module load anaconda3/2024.10

# Clear pip cache first to save space (if it exists and is accessible)
echo "Clearing pip cache..."
rm -rf ~/.cache/pip/* 2>/dev/null || echo "  Note: Could not clear pip cache (may not exist or already empty)"

# Define scratch directory path
SCRATCH_DIR="/scratch/network/is1893"
CONDA_ENV_PATH="$SCRATCH_DIR/conda_envs/eventfulness"

# Configure cache directories to use scratch space (avoid home quota issues)
export TORCH_HOME="$SCRATCH_DIR/.cache/torch"
export MPLCONFIGDIR="$SCRATCH_DIR/.config/matplotlib"
mkdir -p "$SCRATCH_DIR/conda_envs" "$TORCH_HOME" "$MPLCONFIGDIR"
echo "Configured cache directories: TORCH_HOME=$TORCH_HOME, MPLCONFIGDIR=$MPLCONFIGDIR"

# Create Environment Eventfulness in scratch space
echo "Creating conda environment at $CONDA_ENV_PATH..."
conda create -p $CONDA_ENV_PATH python=3.9 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_PATH
echo "Environment created and activated"

# Set pip to not cache downloads to save space
export PIP_NO_CACHE_DIR=1

# CRITICAL: Ensure pip installs into conda environment, not user site-packages
# Unset PYTHONUSERBASE to prevent installing to ~/.local
unset PYTHONUSERBASE
# Explicitly set PYTHONPATH to prioritize conda environment (but don't override it completely)
export PYTHONPATH="$CONDA_ENV_PATH/lib/python3.9/site-packages:${PYTHONPATH:-}"

# Check if packages exist in user site-packages and warn
USER_SITE_PACKAGES="$HOME/.local/lib/python3.9/site-packages"
if [ -d "$USER_SITE_PACKAGES" ]; then
    if [ -f "$USER_SITE_PACKAGES/torch/__init__.py" ] || [ -f "$USER_SITE_PACKAGES/torchvision/__init__.py" ]; then
        echo "WARNING: PyTorch/torchvision found in $USER_SITE_PACKAGES"
        echo "  These may interfere with conda environment. Consider removing them or using --ignore-installed"
    fi
fi

# CRITICAL: Install NumPy 1.x FIRST - PyTorch 1.9.1 requires NumPy <2.0
# Many packages (librosa, matplotlib) may pull in NumPy 2.x, so we pin it early
echo "Installing NumPy 1.x (required for PyTorch 1.9.1 compatibility)..."
pip install --ignore-installed "numpy<2.0" || {
    echo "ERROR: NumPy installation failed"
    exit 1
}

# Change this pytorch1.9.1 installation line based on your machine specification: https://pytorch.org/get-started/previous-versions/#v191
# Also, please use the pip option provided by pytorch since conda would take forever to resolve the conflicts
# The program assume that a CUDA GPU is available, so you should build this on an environment with CUDA GPU
# Use --ignore-installed to force reinstall into conda environment even if found elsewhere
echo "Installing PyTorch 1.9.1 with CUDA 11.1 support..."
pip install --ignore-installed torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html || {
    echo "ERROR: PyTorch installation failed. Please check your network connection and CUDA compatibility."
    exit 1
}

# Ensure NumPy stays at 1.x (constrain it to prevent upgrades)
pip install --upgrade --force-reinstall "numpy<2.0" || {
    echo "WARNING: Could not ensure NumPy <2.0"
}

echo "Installing other dependencies..."
# Create constraint file for NumPy to prevent upgrades to 2.x
CONSTRAINT_FILE=$(mktemp)
echo "numpy<2.0" > "$CONSTRAINT_FILE"
# Install with numpy constraint to prevent NumPy 2.x from being installed
pip install --constraint "$CONSTRAINT_FILE" librosa matplotlib || {
    # If constraint doesn't work, install and then downgrade numpy
    echo "  Retrying without constraint, will fix NumPy version after..."
    pip install librosa matplotlib || { echo "ERROR: Failed to install librosa/matplotlib"; exit 1; }
    pip install "numpy<2.0" --force-reinstall || { echo "ERROR: Could not downgrade NumPy"; exit 1; }
}
rm -f "$CONSTRAINT_FILE"

pip install psutil || { echo "ERROR: Failed to install psutil"; exit 1; }
pip install torchmetrics==0.6.0 || { echo "ERROR: Failed to install torchmetrics"; exit 1; }
pip install tensorboard || { echo "ERROR: Failed to install tensorboard"; exit 1; }
# CRITICAL: Ensure NumPy remains <2.0 after all installations (other packages may upgrade it)
pip install "numpy<2.0" --upgrade --force-reinstall || { echo "ERROR: Could not maintain NumPy <2.0"; exit 1; }
# Install setuptools version that includes distutils (required by PyTorch's tensorboard utils)
# setuptools < 60 includes distutils, which is needed for PyTorch's tensorboard utils
# DO NOT upgrade setuptools beyond 59.x or distutils will be removed
pip install "setuptools==59.5.0" || { echo "ERROR: Failed to install setuptools with distutils support"; exit 1; }
pip install protobuf || { echo "ERROR: Failed to install protobuf"; exit 1; }
pip install audioread || { echo "ERROR: Failed to install audioread"; exit 1; }
pip install six || { echo "ERROR: Failed to install six"; exit 1; }
pip install av || { echo "ERROR: Failed to install av"; exit 1; }
# Install scipy (required by sklearn/scikit-learn) - use version compatible with NumPy <2.0
pip install "scipy<1.12" || { echo "ERROR: Failed to install scipy"; exit 1; }

echo "Verifying installations..."
# First check NumPy version
python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); assert numpy.__version__.startswith('1.'), f'ERROR: NumPy {numpy.__version__} is incompatible with PyTorch 1.9.1. Expected NumPy <2.0'" || {
    echo "ERROR: NumPy version is incompatible. Attempting to fix..."
    pip install "numpy<2.0" --force-reinstall || exit 1
}

# Verify packages are imported from the correct location and work correctly
python -c "import torch; import numpy; import os; loc = os.path.dirname(torch.__file__); print(f'✓ PyTorch {torch.__version__} installed successfully'); assert 'conda_envs' in loc or 'scratch' in loc, f'PyTorch found in wrong location: {loc}'; print(f'  Location: {loc}'); print(f'  NumPy version: {numpy.__version__}')" || { 
    echo "ERROR: PyTorch import failed or installed in wrong location"; 
    python -c "import sys; print('Python path:'); [print(p) for p in sys.path]" 2>/dev/null || true;
    exit 1; 
}
python -c "import torchvision; import os; loc = os.path.dirname(torchvision.__file__); print(f'✓ Torchvision {torchvision.__version__} installed successfully'); assert 'conda_envs' in loc or 'scratch' in loc, f'Torchvision found in wrong location: {loc}'; print(f'  Location: {loc}')" || { 
    echo "ERROR: Torchvision import failed or installed in wrong location"; 
    exit 1; 
}
python -c "import librosa; print('✓ Librosa installed successfully')" || { echo "ERROR: Librosa import failed"; exit 1; }
python -c "import matplotlib; print('✓ Matplotlib installed successfully')" || { echo "ERROR: Matplotlib import failed"; exit 1; }

# Clean conda cache to save space
echo "Cleaning conda cache..."
conda clean --all -y || echo "WARNING: Could not clean conda cache"

echo ""
echo "✓ Environment setup complete in $CONDA_ENV_PATH"
echo "To activate this environment, use: conda activate $CONDA_ENV_PATH"
