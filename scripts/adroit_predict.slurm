#!/bin/bash
#SBATCH --job-name=eventfulness-predict
#SBATCH --output=eventfulness-predict-%j.out
#SBATCH --error=eventfulness-predict-%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100  # Use A100 GPUs (change to v100 if needed)

# Load modules
module purge
module load anaconda3/2023.3
module load cudatoolkit/11.1
module load cudnn/cuda-11.1/8.2.0

# Activate conda environment
source activate eventfulness

# Navigate to scripts directory
cd $SLURM_SUBMIT_DIR

# Run prediction script
python predict.py --data_dir ../dataSets/myVideos \
    --ngpu 1 --nepoch 1 --nworker 4 \
    --label_type none \
    --num_accS_dir 4 --num_velS_dir 4 --num_blurrs 4 \
    --prediction_window_step 24 \
    --load_model --load_model_dir ../checkpoints/caccvel_full --load_epoch 61
