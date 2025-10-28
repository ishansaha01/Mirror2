#!/bin/bash
#SBATCH --job-name=eventfulness-train
#SBATCH --output=eventfulness-train-%j.out
#SBATCH --error=eventfulness-train-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
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

# Run training script
python train.py --train_clips_per_video 6 --val_clips_per_video 1 \
    --data_dir ../dataSets/train_syn_data \
    --ngpu 2 --nepoch 300 --nworker 8 \
    --label_type beat_random_time_c_shift_scaled \
    --lr 1e-6 --num_accS_dir 4 --num_velS_dir 4 --num_blurrs 4 \
    --clip_size 72 \
    --data_augmentation ../dataAugParams/data_augmentation_list.json \
    --batch_size 16
