#!/bin/bash
#SBATCH --job-name=install-megatron
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=develbooster
#SBATCH --account=envcomp
#SBATCH --output=logs/install-megatron-%j.out

# ============================================================
# Install Megatron dependencies on a compute node with GPU/CUDA
# ============================================================

set -e

# Module environment
module purge
module load GCC/13.3.0
module load OpenMPI/5.0.5
module load CUDA/12
module load cuDNN/9.5.0.50
module load git

# CUDA settings for compilation
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_HOME=$CUDA_ROOT

echo "============================================================"
echo "Installing Megatron dependencies"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CUDA: $CUDA_HOME"
echo "nvcc: $(which nvcc)"
echo "============================================================"

cd /p/project1/envcomp/yll/adaptive-compute-rewrite/nemo-rl

# Activate the venv
source .venv/bin/activate

# Install mcore extras (includes megatron-core, megatron-bridge, transformer-engine, flash-attn, mamba-ssm)
echo "Running: uv sync --extra mcore"
uv sync --extra mcore

# Fix for Ray worker compatibility - convert editable installs to regular installs
# Ray workers don't process .pth files, so namespace packages break with editable installs
echo "============================================================"
echo "Reinstalling megatron packages as non-editable for Ray compatibility..."
echo "============================================================"
pip uninstall megatron-core megatron-bridge -y
pip install --no-deps --no-cache-dir ./3rdparty/Megatron-LM-workspace/
pip install --no-deps --no-cache-dir ./3rdparty/Megatron-Bridge-workspace/

echo "============================================================"
echo "Installation completed!"
echo "============================================================"

# Verify installation
echo "Checking installed packages:"
pip list | grep -iE "megatron|mamba|flash|transformer-engine"
