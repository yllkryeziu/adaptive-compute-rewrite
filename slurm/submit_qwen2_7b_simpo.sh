#!/bin/bash
#SBATCH --job-name=qwen2-7b-simpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=logs/%x-%j.out

# --- Environment Setup (Copied from adaptive-compute-rewrite standards) ---
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
export MASTER_PORT=29501
# Use 'i' suffix for InfiniBand interface on JUWELS Booster if needed, or standard hostname
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR"

# Network Interfaces
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

# --- Hugging Face & Caching ---
export HF_HOME="/p/project1/envcomp/yll/.cache/huggingface"
export HF_DATASETS_CACHE="/p/project1/envcomp/yll/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/p/project1/envcomp/yll/.cache/huggingface/transformers"

# --- Training Configuration ---
CONFIG_FILE="examples/train_full/qwen2_7b_simpo.yaml"
export WANDB_MODE=offline
export DS_SKIP_CUDA_CHECK=1
export FORCE_TORCHRUN=1
export NNODES=$SLURM_JOB_NUM_NODES
export NPROC_PER_NODE=4

echo "Starting training on $NNODES nodes..."
echo "Configuration: $CONFIG_FILE"

# --- Execution ---
srun \
    --label \
    --nodes=$SLURM_JOB_NUM_NODES \
    --ntasks-per-node=1 \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --gpus-per-node=4 \
    bash -c '
export NODE_RANK=$SLURM_NODEID
export MASTER_ADDR='"$MASTER_ADDR"'
export MASTER_PORT='"$MASTER_PORT"'

echo "Node: $(hostname) Rank: $NODE_RANK Master: $MASTER_ADDR"

cd /p/project1/envcomp/yll/adaptive-compute-rewrite
source .venv/bin/activate
cd skythought/train/LLaMA-Factory

# Ensure we use the correct config file path relative to root
unset OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE
llamafactory-cli train examples/train_full/qwen2_7b_simpo.yaml
'
