# Running Eventfulness on Princeton's Adroit Cluster

This guide explains how to set up and run the Eventfulness project on Princeton's Adroit GPU cluster.

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

## 3. Clone the Repository

Clone this repository to your home directory on Adroit:
```
git clone https://github.com/jiatiansun/Eventfulness.git
cd Eventfulness
```

## 4. Set Up the Environment

Use our Adroit-specific setup script:
```
cd setup
chmod +x adroit_setup.sh
./adroit_setup.sh
```

This script will:
- Load necessary modules (anaconda3, cudatoolkit, cudnn)
- Create a conda environment called "eventfulness"
- Install PyTorch with CUDA support and all dependencies

## 5. Prepare Your Data

Download the necessary datasets:
- For training: Download the synthetic dataset from the [eventfulness website](https://www.cs.cornell.edu/abe/projects/eventfulness/)
- For prediction: Either use the provided datasets or upload your own videos

Place the datasets in the appropriate directory structure:
```
Eventfulness/
  dataSets/
    train_syn_data/  # For training
    myVideos/        # For prediction
      val/
        category1/
          video1.mp4
          video2.mp4
        category2/
          video3.mp4
          ...
```

## 6. Download Pre-trained Models (for Prediction)

If you want to use pre-trained models:
1. Download the checkpoints from the [eventfulness website](https://www.cs.cornell.edu/abe/projects/eventfulness/)
2. Extract and place them in the `checkpoints` directory

## 7. Submit Training Job

To train a model on the cluster:
```
cd scripts
sbatch adroit_train.slurm
```

You can monitor your job status with:
```
squeue -u <NetID>
```

## 8. Submit Prediction Job

To run prediction on your videos:
```
cd scripts
sbatch adroit_predict.slurm
```

## 9. Check Results

- Training results will be in `scripts/lossAccuracyReport/<timestamp>/prediction`
- Training/validation loss data will be in `scripts/runs/<timestamp>`
- Prediction results will be saved as JSON files in `dataSets/myVideos/results/`

## 10. Visualize Results with TensorBoard

You can use TensorBoard to visualize training progress. First, start an interactive session:
```
salloc --nodes=1 --ntasks=1 --time=01:00:00 --mem=4G
```

Then run TensorBoard:
```
cd scripts
module load anaconda3/2023.3
source activate eventfulness
tensorboard --logdir=runs
```

Access TensorBoard through the MyAdroit web portal or by setting up port forwarding.

## Additional Notes

- Use `snodes` or `shownodes -p gpu` to check available GPU nodes
- For visualization tasks, connect to the visualization node: `ssh <NetID>@adroit-vis.princeton.edu`
- Use `/scratch/network/` for large datasets and outputs
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
