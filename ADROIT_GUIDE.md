# Running Mirror2 on Princeton's Adroit Cluster

This guide explains how to set up and run the Mirror2 project on Princeton's Adroit GPU cluster.

## 1. Request Access to Adroit

First, request access to the Adroit cluster:
- Fill out the [Adroit Registration Form](https://forms.rc.princeton.edu/registration/?q=adroit) to request an account
- If you're using Adroit for a course, use the dedicated form for course access

## 2. Connect to Adroit

Once your account is approved, connect via SSH:
```
ssh <NetID>@adroit.princeton.edu
```
If you're off-campus, you'll need to connect through Princeton's VPN first.

## 3. Set Up Scratch Directory for Data and Environments

For better performance and more storage space, set up a scratch directory for data, outputs, and environments:
```
mkdir -p /scratch/network/<NetID>/mirror2_data
mkdir -p /scratch/network/<NetID>/conda_envs
```

## 4. Clone the Repository (Home Directory)

The repository is already cloned in your home directory. Navigate to it:
```
cd ~/Mirror2
```

## 5. Set Up the Environment in Scratch Area

Create a setup script that will configure your environment to use the scratch area:

```
cd ~/Mirror2
mkdir -p setup
```

Create a new file `setup/adroit_scratch_setup.sh`:

```bash
#!/bin/bash

# Set conda environment path to scratch area
export CONDA_ENVS_PATH=/scratch/network/<NetID>/conda_envs

# Load necessary modules
module load anaconda3/2023.3
module load cudatoolkit/11.7
module load cudnn/8.6.0

# Create conda environment in scratch area
conda create -y -n mirror2 python=3.9

# Activate environment
source activate mirror2

# Install required packages
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib tensorboard
# Add any other required packages for your project
```

Make the script executable and run it:
```
chmod +x setup/adroit_scratch_setup.sh
./setup/adroit_scratch_setup.sh
```

This script will:
- Load necessary modules (anaconda3, cudatoolkit, cudnn)
- Create a conda environment called "mirror2" in your scratch directory
- Install PyTorch with CUDA support and all dependencies

## 6. Prepare Your Data in Scratch Area

Download the necessary datasets directly to your scratch area:
- For training: Download the training dataset from the project website
- For prediction: Either use the provided datasets or upload your own data

Create the data directory structure in your scratch area:
```
mkdir -p /scratch/network/<NetID>/mirror2_data/dataSets/train_data
mkdir -p /scratch/network/<NetID>/mirror2_data/dataSets/test_data
```

Place the datasets in this structure:
```
/scratch/network/<NetID>/mirror2_data/
  dataSets/
    train_data/  # For training
    test_data/   # For prediction
      val/
        category1/
          data1.mp4
          data2.mp4
        category2/
          data3.mp4
          ...
```

Create symbolic links to access the data from your repository:
```
ln -s /scratch/network/<NetID>/mirror2_data/dataSets ~/Mirror2/dataSets
```

## 7. Download Pre-trained Models to Scratch Area

If you want to use pre-trained models:
1. Create a directory for checkpoints in your scratch area:
```
mkdir -p /scratch/network/<NetID>/mirror2_data/checkpoints
```

2. Download the checkpoints from the project website to this directory

3. Create a symbolic link to access the checkpoints from your repository:
```
ln -s /scratch/network/<NetID>/mirror2_data/checkpoints ~/Mirror2/checkpoints
```

## 8. Submit Training Job

To train a model on the cluster, create a Slurm script that uses your home directory repository but with scratch area for outputs:

Create a file named `adroit_train_scratch.slurm` in your repository:
```
#!/bin/bash
#SBATCH --job-name=mirror2_train
#SBATCH --output=/scratch/network/<NetID>/mirror2_data/logs/train_%j.log
#SBATCH --error=/scratch/network/<NetID>/mirror2_data/logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100  # or a100 for A100 GPUs

# Create log directory in scratch
mkdir -p /scratch/network/<NetID>/mirror2_data/logs

# Change to your repository directory
cd ~/Mirror2/scripts

# Load modules
module load anaconda3/2023.3
module load cudatoolkit/11.7
module load cudnn/8.6.0

# Set output directories to scratch
export RESULTS_DIR=/scratch/network/<NetID>/mirror2_data/results
export RUNS_DIR=/scratch/network/<NetID>/mirror2_data/runs
mkdir -p $RESULTS_DIR $RUNS_DIR

# Activate the conda environment from scratch
export CONDA_ENVS_PATH=/scratch/network/<NetID>/conda_envs
source activate mirror2

# Run the training script with outputs directed to scratch
python train.py --output-dir $RESULTS_DIR --runs-dir $RUNS_DIR --your-training-arguments
```

Submit the job:
```
cd ~/Mirror2
sbatch adroit_train_scratch.slurm
```

You can monitor your job status with:
```
squeue -u <NetID>
```

## 9. Submit Prediction Job

To run prediction on your data, create a similar Slurm script for prediction:

Create a file named `adroit_predict_scratch.slurm` in your repository:
```
#!/bin/bash
#SBATCH --job-name=mirror2_predict
#SBATCH --output=/scratch/network/<NetID>/mirror2_data/logs/predict_%j.log
#SBATCH --error=/scratch/network/<NetID>/mirror2_data/logs/predict_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100  # or a100 for A100 GPUs

# Create log directory in scratch
mkdir -p /scratch/network/<NetID>/mirror2_data/logs

# Change to your repository directory
cd ~/Mirror2/scripts

# Load modules
module load anaconda3/2023.3
module load cudatoolkit/11.7
module load cudnn/8.6.0

# Set output directory to scratch
export RESULTS_DIR=/scratch/network/<NetID>/mirror2_data/prediction_results
mkdir -p $RESULTS_DIR

# Activate the conda environment from scratch
export CONDA_ENVS_PATH=/scratch/network/<NetID>/conda_envs
source activate mirror2

# Run the prediction script with outputs directed to scratch
python predict.py --output-dir $RESULTS_DIR --your-prediction-arguments
```

Submit the job:
```
cd ~/Mirror2
sbatch adroit_predict_scratch.slurm
```

## 10. Check Results

All results are stored in your scratch directory:
- Training results will be in `/scratch/network/<NetID>/mirror2_data/results/<timestamp>/prediction`
- Training/validation data will be in `/scratch/network/<NetID>/mirror2_data/runs/<timestamp>`
- Prediction results will be saved as JSON files in `/scratch/network/<NetID>/mirror2_data/prediction_results/`

## 11. Visualize Results with TensorBoard

You can use TensorBoard to visualize training progress. First, start an interactive session:
```
salloc --nodes=1 --ntasks=1 --time=01:00:00 --mem=4G
```

Then run TensorBoard pointing to your scratch directory:
```
module load anaconda3/2023.3
export CONDA_ENVS_PATH=/scratch/network/<NetID>/conda_envs
source activate mirror2
tensorboard --logdir=/scratch/network/<NetID>/mirror2_data/runs
```

Access TensorBoard through the MyAdroit web portal or by setting up port forwarding.

## 12. For Long-Running Sessions

For repeated or long-running sessions, use tmux to keep your session active:
```
# Start a new tmux session
tmux new -s mirror2

# To detach from the session: press Ctrl+b, then d
# To reattach to the session:
tmux attach -t mirror2
```

## Additional Notes

- Use `snodes` or `shownodes -p gpu` to check available GPU nodes
- For visualization tasks, connect to the visualization node: `ssh <NetID>@adroit-vis.princeton.edu`
- Always use `/scratch/network/` for large datasets and outputs
- Remember to periodically back up important results from scratch to a more permanent storage
- For more help, see Princeton Research Computing support and knowledge base pages

## GPU Availability

Check GPU availability with:
```
shownodes -p gpu
```

Adroit has both V100 and A100 GPUs. To specifically request one type, use the constraint flag in your SLURM script:
```
#SBATCH --constraint=a100  # For A100 GPUs
```
or
```
#SBATCH --constraint=v100  # For V100 GPUs
```
