#!/bin/bash
# Example usage of run_motiongpt.slurm
# This file shows how to submit different MotionGPT3 jobs to SLURM

# Example 1: Run demo with default config
# SCRIPT_TYPE=demo sbatch run_motiongpt.slurm

# Example 2: Run training
# SCRIPT_TYPE=train sbatch run_motiongpt.slurm

# Example 3: Run testing
# SCRIPT_TYPE=test sbatch run_motiongpt.slurm

# Example 4: Run WebUI (app.py)
# Note: For WebUI, you may need to set up port forwarding
# SCRIPT_TYPE=app sbatch run_motiongpt.slurm

# Example 5: Custom demo with specific config and example file
# You can modify the script to accept additional parameters, or edit run_motiongpt.slurm directly

echo "To submit a job, use one of these commands:"
echo ""
echo "  # Run demo"
echo "  SCRIPT_TYPE=demo sbatch run_motiongpt.slurm"
echo ""
echo "  # Run training"
echo "  SCRIPT_TYPE=train sbatch run_motiongpt.slurm"
echo ""
echo "  # Run testing"
echo "  SCRIPT_TYPE=test sbatch run_motiongpt.slurm"
echo ""
echo "  # Run WebUI"
echo "  SCRIPT_TYPE=app sbatch run_motiongpt.slurm"
echo ""
echo "To customize parameters, edit the run_motiongpt.slurm file directly"
echo "or modify it to accept command-line arguments."

