#!/bin/bash

# Request an interactive session with GPU
echo "Requesting interactive GPU session on Adroit..."
echo "This will give you a shell on a compute node with GPU access"
echo "Usage: ./adroit_interactive.sh [time in hours] [memory in GB]"

# Default values
TIME=${1:-2}  # Default 2 hours
MEM=${2:-16}  # Default 16GB

# Convert time to SLURM format
TIME_SLURM="${TIME}:00:00"

# Request interactive session
salloc --nodes=1 --ntasks=1 --gres=gpu:1 --time=$TIME_SLURM --mem=${MEM}G

# Note: After allocation is granted, you'll need to:
# module load anaconda3/2023.3 cudatoolkit/11.1 cudnn/cuda-11.1/8.2.0
# source activate eventfulness
