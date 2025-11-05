#!/bin/bash
#SBATCH --job-name=eventfulness-predict
#SBATCH --output=/scratch/network/is1893/mirror2_data/logs/predict-%j.out
#SBATCH --error=/scratch/network/is1893/mirror2_data/logs/predict-%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100  # Use A100 GPUs (change to v100 if needed)

# Load modules
module purge
module load anaconda3/2024.10

# Load CUDA modules
module load cudatoolkit/11.8
module load cudnn/cuda-11.8/8.6.0 2>/dev/null || echo "cudnn/cuda-11.8/8.6.0 not available, trying alternative"
module load cudnn/8.6.0 2>/dev/null || echo "cudnn/8.6.0 not available, continuing without it"

# Set conda environment path to scratch
export CONDA_ENVS_PATH=/scratch/network/is1893/conda_envs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate eventfulness

# Install required packages directly in the conda environment only if not already installed
echo "Checking for missing dependencies..."
python -c "import psutil" 2>/dev/null || pip install --no-cache-dir psutil --target=/scratch/network/is1893/conda_envs/eventfulness/lib/python3.9/site-packages/

# Configure cache directories to use scratch space (avoid home quota issues)
export TORCH_HOME=/scratch/network/is1893/.cache/torch
export MPLCONFIGDIR=/scratch/network/is1893/.config/matplotlib
mkdir -p $TORCH_HOME $MPLCONFIGDIR

# Set PYTHONPATH to prioritize conda environment
export PYTHONPATH=/scratch/network/is1893/conda_envs/eventfulness/lib/python3.9/site-packages:${PYTHONPATH:-}

# Print Python path for debugging
echo "PYTHONPATH: $PYTHONPATH"
echo "Python executable: $(which python)"
python -c "import sys; print('Python sys.path:', sys.path)"

# Create log and results directories if they don't exist
mkdir -p /scratch/network/is1893/mirror2_data/logs
mkdir -p /scratch/network/is1893/mirror2_data/dataSets/test_data/results

# Navigate to scripts directory
cd ~/Mirror2/scripts

# Run prediction script
# Data structure expected: data_dir/val/category_name/video_files.mp4
# Current test data: /scratch/network/is1893/mirror2_data/dataSets/test_data/val/fireworks/FireworkVertical.mov
python predict.py --data_dir /scratch/network/is1893/mirror2_data/dataSets/test_data \
    --ngpu 1 --nepoch 1 --nworker 4 \
    --label_type none \
    --num_accS_dir 4 --num_velS_dir 4 --num_blurrs 4 \
    --prediction_window_step 24 \
    --load_model --load_model_dir /scratch/network/is1893/mirror2_data/checkpoints --load_epoch 61
