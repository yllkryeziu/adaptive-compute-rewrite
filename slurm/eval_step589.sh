#!/bin/bash
#SBATCH --job-name=eval_simpo_step589
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=logs/%j-%x.out

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

HF_OUTPUT_BASE="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b_hf"
EVAL_OUTPUT="/p/project1/envcomp/yll/adaptive-compute-rewrite/outputs/eval_math500"

echo "Evaluating SimPO step_589 (final) checkpoint"

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

echo "Done!"
