#!/bin/bash
#SBATCH --job-name=eventfulness-train
#SBATCH --output=/scratch/network/is1893/mirror2_data/logs/train-%j.out
#SBATCH --error=/scratch/network/is1893/mirror2_data/logs/train-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100  # Use A100 GPUs (change to v100 if needed)

# Load modules
module purge
module load anaconda3/2024.10
module load cudatoolkit/11.8
# Try loading cudnn with simpler path if available (comment out if module doesn't exist)
module load cudnn/8.6.0 2>/dev/null || echo "cudnn/8.6.0 not available, continuing without it"

# Set conda environment path to scratch
export CONDA_ENVS_PATH=/scratch/network/is1893/conda_envs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate eventfulness

# Configure cache directories to use scratch space (avoid home quota issues)
export TORCH_HOME=/scratch/network/is1893/.cache/torch
export MPLCONFIGDIR=/scratch/network/is1893/.config/matplotlib
mkdir -p $TORCH_HOME $MPLCONFIGDIR

# CRITICAL: Ensure Python uses conda environment packages, not user site-packages
# Prevent Python from using user site-packages (~/.local)
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1
# Set PYTHONPATH to prioritize conda environment
export PYTHONPATH=/scratch/network/is1893/conda_envs/eventfulness/lib/python3.9/site-packages:${PYTHONPATH:-}

# Create log directory if it doesn't exist
mkdir -p /scratch/network/is1893/mirror2_data/logs

# Set output directories to scratch
export RESULTS_DIR=/scratch/network/is1893/mirror2_data/results
export RUNS_DIR=/scratch/network/is1893/mirror2_data/runs
mkdir -p $RESULTS_DIR $RUNS_DIR

# Navigate to scripts directory
cd ~/Mirror2/scripts

# Run training script
python train.py --train_clips_per_video 6 --val_clips_per_video 1 \
    --data_dir /scratch/network/is1893/mirror2_data/dataSets/train_data \
    --ngpu 2 --nepoch 300 --nworker 8 \
    --label_type beat_random_time_c_shift_scaled \
    --lr 1e-6 --num_accS_dir 4 --num_velS_dir 4 --num_blurrs 4 \
    --clip_size 72 \
    --data_augmentation ~/Mirror2/dataAugParams/data_augmentation_list.json \
    --batch_size 16
