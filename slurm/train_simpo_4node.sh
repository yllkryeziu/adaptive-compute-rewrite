#!/bin/bash
#SBATCH --job-name=simpo-4node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

# --- Environment Setup ---
module purge
module load Stages/2025
module load GCC/12.3.0
module load OpenMPI/4.1.6
module load CUDA/12.1
module load cuDNN/8.9.7.29-CUDA-12

# Activate Virtual Environment
source .venv/bin/activate

# --- Network & Distributed Setup ---
export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i  # Append 'i' for InfiniBand if needed on JUWELS
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
echo "Starting training on 4 nodes..."
echo "Configuration: $CONFIG_FILE"

# We use srun to launch one task per node.
# llamafactory-cli with FORCE_TORCHRUN=1 will handle the distributed setup 
# but we need to ensure it knows the node rank and master address.
# A more robust way for SLURM is often to let `srun` handle the launch and set specific env vars.

export FORCE_TORCHRUN=1
export NNODES=$SLURM_JOB_NUM_NODES
export NPROC_PER_NODE=4

# Launch
srun \
    --label \
    --nodes=$SLURM_JOB_NUM_NODES \
    --ntasks-per-node=1 \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --gpus-per-node=4 \
    bash -c '
    export NODE_RANK=$SLURM_PROCID
    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    
    echo "Node: $(hostname) Rank: $NODE_RANK Master: $MASTER_ADDR"
    
    llamafactory-cli train examples/train_full/qwen2_full_simpo.yaml
    '
