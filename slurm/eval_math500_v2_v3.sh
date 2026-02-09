#!/bin/bash
#SBATCH --job-name=eval_math500_v2_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
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

TASK_NAME="math500"
OUTPUT_BASE="/p/project1/envcomp/yll/adaptive-compute-rewrite/outputs/eval_math500"
export HF_HOME="/p/project1/envcomp/yll/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# Function to evaluate checkpoint
eval_checkpoint() {
    local VERSION=$1
    local STEP=$2
    
    MODEL_PATH="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b_${VERSION}_hf/step_${STEP}"
    # Check if model path exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Model path not found: $MODEL_PATH"
        return
    fi

    RESULT_DIR="$OUTPUT_BASE/nemo_simpo_32b_${VERSION}_step${STEP}"
    
    echo "------------------------------------------------------------"
    echo "Starting evaluation for ${VERSION} Step ${STEP}"
    echo "Model: $MODEL_PATH"
    echo "Output: $RESULT_DIR"
    echo "------------------------------------------------------------"

    mkdir -p $RESULT_DIR

    skythought evaluate \
        --model $MODEL_PATH \
        --task $TASK_NAME \
        --backend vllm \
        --backend-args tensor_parallel_size=4,gpu_memory_utilization=0.85 \
        --sampling-params "temperature=0,top_p=1.0,max_tokens=16384" \
        --system-prompt-name skythought \
        --batch-size 16 \
        --n 1 \
        --result-dir $RESULT_DIR \
        --as-test \
        --overwrite
}

# Evaluate v2 steps
for STEP in 80 90 98; do
    eval_checkpoint "v2" $STEP
done

# Evaluate v3 steps
for STEP in 80 90 98; do
    eval_checkpoint "v3" $STEP
done

echo "All evaluations complete."
