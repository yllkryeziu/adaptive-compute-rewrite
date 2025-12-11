#!/bin/bash
#SBATCH --job-name=simpo-lora-4node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=booster
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
export TORCH_EXTENSIONS_DIR="/p/project1/envcomp/yll/adaptive-compute-rewrite/.cache/torch_extensions"
export TRITON_CACHE_DIR="/p/project1/envcomp/yll/adaptive-compute-rewrite/.cache/triton"

# Ensure directories exist
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $TORCH_EXTENSIONS_DIR $TRITON_CACHE_DIR slurm/logs

# --- Training Configuration ---
CONFIG_FILE="skythought/train/LLaMA-Factory/examples/train_lora/qwen2_lora_simpo.yaml"

# --- Execution ---
echo "Starting LoRA training on 4 nodes..."
echo "Configuration: $CONFIG_FILE"
export WANDB_MODE=offline
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
export NODE_RANK=$SLURM_NODEID
export MASTER_ADDR='"$MASTER_ADDR"'
export MASTER_PORT='"$MASTER_PORT"'

echo "Node: $(hostname) Rank: $NODE_RANK Master: $MASTER_ADDR"

cd /p/project1/envcomp/yll/adaptive-compute-rewrite
source .venv/bin/activate

echo "Python: $(which python)"
unset OMPI_COMM_WORLD_LOCAL_RANK OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE
llamafactory-cli train skythought/train/LLaMA-Factory/examples/train_lora/qwen2_lora_simpo.yaml
'
