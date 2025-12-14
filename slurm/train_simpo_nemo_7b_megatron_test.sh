#!/bin/bash
#SBATCH --job-name=nemo-simpo-7b-megatron
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --account=envcomp
#SBATCH --output=logs/nemo-simpo-7b-megatron-%j.out

# ============================================================
# NeMo-RL SimPO Test Training for Qwen2.5-7B with Megatron
# 1 node x 4 A100 40GB GPUs @ 16k sequences
# ============================================================

set -e

# Module environment
module purge
module load GCC/13.3.0
module load OpenMPI/5.0.5
module load CUDA/12
# cuDNN loaded from pip package (nvidia-cudnn-cu12) - do NOT load system cuDNN module
module load git

# Environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# HuggingFace cache
export HF_HOME=/p/project1/envcomp/yll/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM=false

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 architecture, required for Megatron

# Ray settings
export RAY_ADDRESS=""
export RAY_DEDUP_LOGS=0
# Use system Python (activated venv) instead of uv run for Ray workers
# This ensures workers use the non-editable megatron packages we installed
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# W&B settings (offline mode for cluster)
export WANDB_MODE=offline
export WANDB_DIR=/p/project1/envcomp/yll/adaptive-compute-rewrite/wandb

# Suppress warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore::UserWarning"

# Working directory
cd /p/project1/envcomp/yll/adaptive-compute-rewrite

# Activate NeMo-RL venv
source nemo-rl/.venv/bin/activate

# Use pip-installed cuDNN (avoid module cuDNN conflicts)
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

echo "============================================================"
echo "NeMo-RL SimPO Test (7B Megatron @ 16k)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: 4"
echo "Config: configs/nemo_simpo_7b_megatron_16k.yaml"
echo "============================================================"

# Create logs directory if needed
mkdir -p slurm/logs

# Run SimPO training test with Megatron backend
python nemo-rl/examples/run_simpo.py \
    --config configs/nemo_simpo_7b_megatron_16k.yaml \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=4

echo "Test training completed!"
