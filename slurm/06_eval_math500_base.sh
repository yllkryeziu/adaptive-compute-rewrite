#!/bin/bash
#SBATCH --job-name=eval_math500_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
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

# Activate Environment
cd /p/project1/envcomp/yll/adaptive-compute-rewrite
source .venv/bin/activate

# --- Configuration ---
# Base Qwen2-7B-Instruct Path
MODEL_PATH="/p/project1/envcomp/yll/.cache/huggingface/hub/models--Qwen--Qwen2-7B-Instruct/snapshots/f2826a00ceef68f0f2b946d945ecc0477ce4450c"
TASK_NAME="math500"
OUTPUT_DIR="/p/project1/envcomp/yll/adaptive-compute-rewrite/outputs/eval_math500/Qwen2-7B-Instruct"

export HF_HOME="/p/project1/envcomp/yll/.cache/huggingface"
export HF_DATASETS_OFFLINE=1

# Network & Distributed Setup (Standard for single node execution)
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

echo "Starting evaluation for Base Model..."
echo "Model: $MODEL_PATH"
echo "Task: $TASK_NAME"
echo "Output Directory: $OUTPUT_DIR"

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

# --- Evaluation ---
echo "============================================================"
echo "Evaluating Base Qwen2-7B on $TASK_NAME with skythought prompt..."
echo "============================================================"

# Using 'skythought' system prompt to force reasoning structure
skythought evaluate \
    --model $MODEL_PATH \
    --task $TASK_NAME \
    --backend vllm \
    --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.85 \
    --sampling-params "temperature=0,top_p=1.0,max_tokens=16384" \
    --system-prompt-name skythought \
    --batch-size 16 \
    --n 1 \
    --result-dir $OUTPUT_DIR \
    --overwrite

echo "Evaluation complete."
