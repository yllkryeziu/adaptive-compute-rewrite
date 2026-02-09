#!/bin/bash
#SBATCH --job-name=eval_simpo_ckpts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=logs/%j-%x.out

# --- Environment Setup ---
module --force purge
module load Stages/2025
module load GCC/13.3.0
module load OpenMPI/5.0.5
module load CUDA/12
module load cuDNN/9.5.0.50-CUDA-12

cd /p/project1/envcomp/yll/adaptive-compute-rewrite
source .venv/bin/activate

export HF_HOME="/p/project1/envcomp/yll/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

HF_OUTPUT_BASE="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b_hf"
EVAL_OUTPUT="/p/project1/envcomp/yll/adaptive-compute-rewrite/outputs/eval_math500"
BASE_MODEL="/p/project1/envcomp/yll/.cache/huggingface/hub/models--NovaSky-AI--Sky-T1-32B-Preview/snapshots/1e3f4c62a30c7ce70f4b3a3b952895d866943551"

mkdir -p $EVAL_OUTPUT

# ============================================================
# Step 1: Evaluate BASE model (Sky-T1-32B-Preview)
# ============================================================
echo "============================================================"
echo "Evaluating BASE model: Sky-T1-32B-Preview"
echo "============================================================"

skythought evaluate \
    --model $BASE_MODEL \
    --task math500 \
    --backend vllm \
    --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.90 \
    --sampling-params "temperature=0,top_p=1.0,max_tokens=16384" \
    --system-prompt-name skythought \
    --batch-size 8 \
    --n 1 \
    --result-dir $EVAL_OUTPUT/Sky-T1-32B-Base \
    --overwrite

# ============================================================
# Step 2: Evaluate step_200 checkpoint
# ============================================================
echo "============================================================"
echo "Evaluating SimPO step_200 checkpoint"
echo "============================================================"

skythought evaluate \
    --model $HF_OUTPUT_BASE/step_200 \
    --task math500 \
    --backend vllm \
    --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.90 \
    --sampling-params "temperature=0,top_p=1.0,max_tokens=16384" \
    --system-prompt-name skythought \
    --batch-size 8 \
    --n 1 \
    --result-dir $EVAL_OUTPUT/Sky-T1-32B-SimPO-Step200 \
    --overwrite

# ============================================================
# Step 3: Evaluate step_589 (final) checkpoint
# ============================================================
echo "============================================================"
echo "Evaluating SimPO step_589 (final) checkpoint"
echo "============================================================"

skythought evaluate \
    --model $HF_OUTPUT_BASE/step_589 \
    --task math500 \
    --backend vllm \
    --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.90 \
    --sampling-params "temperature=0,top_p=1.0,max_tokens=16384" \
    --system-prompt-name skythought \
    --batch-size 8 \
    --n 1 \
    --result-dir $EVAL_OUTPUT/Sky-T1-32B-SimPO-Final \
    --overwrite

echo "============================================================"
echo "All evaluations complete!"
echo "Results saved to: $EVAL_OUTPUT"
echo "============================================================"
