#!/bin/bash
#SBATCH --job-name=eventfulness-predict-cpu
#SBATCH --output=/scratch/network/is1893/mirror2_data/logs/predict-%j.out
#SBATCH --error=/scratch/network/is1893/mirror2_data/logs/predict-%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# No GPU requested - running on CPU only

# Load modules
module purge
module load anaconda3/2024.10

# Set conda environment path to scratch
export CONDA_ENVS_PATH=/scratch/network/is1893/conda_envs
source $(conda info --base)/etc/profile.d/conda.sh
conda activate eventfulness

# Configure cache directories to use scratch space (avoid home quota issues)
export TORCH_HOME=/scratch/network/is1893/.cache/torch
export MPLCONFIGDIR=/scratch/network/is1893/.config/matplotlib
mkdir -p $TORCH_HOME $MPLCONFIGDIR

# Set PYTHONPATH to include user site-packages (for psutil)
export PYTHONPATH=/scratch/network/is1893/conda_envs/eventfulness/lib/python3.9/site-packages:~/.local/lib/python3.9/site-packages:${PYTHONPATH:-}

# Print Python path for debugging
echo "PYTHONPATH: $PYTHONPATH"
echo "Python executable: $(which python)"
python -c "import sys; print('Python sys.path:', sys.path)"

# Create log directory if it doesn't exist
mkdir -p /scratch/network/is1893/mirror2_data/logs

# Navigate to scripts directory
cd ~/Mirror2/scripts

# Create a temporary modified version of the model.py file
cat > /tmp/model_patch.py << 'EOF'
import torch
import os

# Original function path
original_file = os.path.expanduser('~/Mirror2/eventfulness/deepVisualBeatUtil/model.py')

# Read the original file
with open(original_file, 'r') as f:
    content = f.read()

# Replace the map_location parameter
modified_content = content.replace(
    'checkpoint = torch.load(model_path, map_location = "cuda:0")',
    'checkpoint = torch.load(model_path, map_location = "cpu")'
)

# Write the modified file
with open(original_file, 'w') as f:
    f.write(modified_content)

print("Modified model.py to use CPU for loading checkpoints")
EOF

# Apply the patch
python /tmp/model_patch.py

# Run prediction script
# Data structure expected: data_dir/val/category_name/video_files.mp4
python predict.py --data_dir /scratch/network/is1893/mirror2_data/dataSets/test_data \
    --ngpu 0 --nepoch 1 --nworker 4 \
    --label_type none \
    --num_accS_dir 4 --num_velS_dir 4 --num_blurrs 4 \
    --prediction_window_step 24 \
    --load_model --load_model_dir /scratch/network/is1893/mirror2_data/checkpoints --load_epoch 61

# Restore the original model.py file
cat > /tmp/model_restore.py << 'EOF'
import os

# Original function path
original_file = os.path.expanduser('~/Mirror2/eventfulness/deepVisualBeatUtil/model.py')

# Read the original file
with open(original_file, 'r') as f:
    content = f.read()

# Replace the map_location parameter back to original
modified_content = content.replace(
    'checkpoint = torch.load(model_path, map_location = "cpu")',
    'checkpoint = torch.load(model_path, map_location = "cuda:0")'
)

# Write the modified file
with open(original_file, 'w') as f:
    f.write(modified_content)

print("Restored original model.py file")
EOF

# Apply the restoration
python /tmp/model_restore.py
