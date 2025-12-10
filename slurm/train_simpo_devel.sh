#!/bin/bash
#SBATCH --job-name=simpo-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --account=envcomp
#SBATCH --output=logs/%x-%j.out

# --- Environment Setup ---
module --force purge
module load Stages/2025
module load GCC/13.3.0
module load OpenMPI/5.0.5
module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12

# Activate Virtual Environment
cd /p/project1/envcomp/yll/adaptive-compute-rewrite
source .venv/bin/activate

# --- Network & Distributed Setup ---
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)  # Single node, just use current hostname
echo "MASTER_ADDR: $MASTER_ADDR"

# Network Interfaces
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=INFO

# --- Hugging Face & Caching ---
export HF_HOME="/p/project1/envcomp/yll/.cache/huggingface"
export HF_DATASETS_CACHE="/p/project1/envcomp/yll/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/p/project1/envcomp/yll/.cache/huggingface/transformers"

# Ensure directories exist
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE slurm/logs

# --- Training Configuration ---
CONFIG_FILE="examples/train_full/qwen2_full_simpo.yaml"

# --- Execution ---
echo "Starting training on 1 node (develbooster test)..."
echo "Configuration: $CONFIG_FILE"

export FORCE_TORCHRUN=1
export NNODES=1
export NPROC_PER_NODE=4

# Launch - single node, simplified setup
srun \
    --label \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --gpus-per-node=4 \
    bash -c '
export NODE_RANK=0
export MASTER_ADDR='$(hostname)'
export MASTER_PORT='"$MASTER_PORT"'

echo "Node: $(hostname) Rank: $NODE_RANK Master: $MASTER_ADDR"

cd /p/project1/envcomp/yll/adaptive-compute-rewrite
source .venv/bin/activate

echo "Python: $(which python)"
llamafactory-cli train skythought/train/LLaMA-Factory/examples/train_full/qwen2_full_simpo.yaml
'
