#!/bin/bash
# MotionGPT3 SLURM Submission Script
# 
# Usage:
#   SCRIPT_TYPE=demo sbatch run_motiongpt.slurm
#   SCRIPT_TYPE=train sbatch run_motiongpt.slurm
#   SCRIPT_TYPE=test sbatch run_motiongpt.slurm
#   SCRIPT_TYPE=app sbatch run_motiongpt.slurm
#
# To customize parameters, edit the commands in the case statement below
# or modify this script to accept additional environment variables.
#
#SBATCH --job-name=motiongpt
#SBATCH --output=/scratch/network/is1893/mirror2_data/logs/motiongpt-%j.out
#SBATCH --error=/scratch/network/is1893/mirror2_data/logs/motiongpt-%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100  # Use A100 GPUs (change to v100 if needed)

# Load modules
module purge
module load anaconda3/2024.10

# Load CUDA modules
module load cudatoolkit/11.8
module load cudnn/cuda-11.8/8.6.0 2>/dev/null || echo "cudnn/cuda-11.8/8.6.0 not available, trying alternative"
module load cudnn/8.6.0 2>/dev/null || echo "cudnn/8.6.0 not available, continuing without it"

# Activate MotionGPT3 virtual environment
source /scratch/network/is1893/MotionGPT3_env/venv/bin/activate

# Set pip cache to scratch to avoid quota issues
export PIP_CACHE_DIR=/scratch/network/is1893/pip_cache

# Configure cache directories to use scratch space (avoid home quota issues)
export TORCH_HOME=/scratch/network/is1893/.cache/torch
export MPLCONFIGDIR=/scratch/network/is1893/.config/matplotlib
export HF_HOME=/scratch/network/is1893/.cache/huggingface
mkdir -p $TORCH_HOME $MPLCONFIGDIR $HF_HOME

# Set PYTHONPATH to prioritize venv
export PYTHONPATH=/scratch/network/is1893/MotionGPT3_env/venv/lib/python3.9/site-packages:${PYTHONPATH:-}

# Print environment info for debugging
echo "=========================================="
echo "MotionGPT3 SLURM Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
echo "=========================================="

# Create log and results directories if they don't exist
mkdir -p /scratch/network/is1893/mirror2_data/logs
mkdir -p /scratch/network/is1893/mirror2_data/dataSets/test_data/results

# Navigate to MotionGPT3 directory
cd /home/is1893/Mirror2/MotionGPT3

# ============================================
# Configure which script to run and parameters
# ============================================
# Options: demo, train, test, app
SCRIPT_TYPE=${SCRIPT_TYPE:-demo}

case $SCRIPT_TYPE in
    demo)
        echo "Running demo.py..."
        # Example: python demo.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt
        python demo.py --cfg ./configs/test.yaml --example ./assets/texts/t2m.txt
        ;;
    train)
        echo "Running train.py..."
        # Example: python train.py --cfg ./configs/MoT_vae_stage3.yaml
        python train.py --cfg ./configs/MoT_vae_stage3.yaml
        ;;
    test)
        echo "Running test.py..."
        # Example: python test.py --cfg ./configs/test.yaml
        python test.py --cfg ./configs/test.yaml
        ;;
    app)
        echo "Running app.py (WebUI)..."
        # Note: WebUI may need port forwarding. Consider using --port and --share for Gradio
        python app.py
        ;;
    *)
        echo "Unknown script type: $SCRIPT_TYPE"
        echo "Available options: demo, train, test, app"
        exit 1
        ;;
esac

echo "=========================================="
echo "Job completed"
echo "=========================================="

