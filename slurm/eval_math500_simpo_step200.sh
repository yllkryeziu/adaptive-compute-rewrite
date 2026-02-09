#!/bin/bash
#SBATCH --job-name=eval_math500_step200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=slurm/logs/%j-%x.out

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
MODEL_PATH="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b/step_200"
TASK_NAME="math500"
OUTPUT_DIR="/p/project1/envcomp/yll/adaptive-compute-rewrite/outputs/eval_math500"
export HF_HOME="/p/project1/envcomp/yll/.cache/huggingface"
export HF_DATASETS_OFFLINE=1

export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

echo "Starting evaluation: SimPO Step 200 Checkpoint"
echo "Model: $MODEL_PATH"
echo "Task: $TASK_NAME"

mkdir -p $OUTPUT_DIR

skythought evaluate \
    --model $MODEL_PATH \
    --task $TASK_NAME \
    --backend vllm \
    --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.85 \
    --sampling-params "temperature=0,top_p=1.0,max_tokens=16384" \
    --system-prompt-name skythought \
    --batch-size 16 \
    --n 1 \
    --result-dir $OUTPUT_DIR/Sky-T1-32B-SimPO-Step200 \
    --overwrite

echo "Evaluation complete."
