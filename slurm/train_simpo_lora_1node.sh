#!/bin/bash
#SBATCH --job-name=simpo-lora-1node
#SBATCH --nodes=1
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
echo "Starting LoRA training on 1 node..."
echo "Configuration: $CONFIG_FILE"
export WANDB_MODE=offline
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch with torchrun for single node
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    -m llamafactory.launcher \
    skythought/train/LLaMA-Factory/examples/train_lora/qwen2_lora_simpo.yaml
