#!/bin/bash
#SBATCH --job-name=nemo-simpo-32b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=slurm/logs/nemo-simpo-32b-%j.out
#SBATCH --error=slurm/logs/nemo-simpo-32b-%j.err

# ============================================================
# NeMo-RL SimPO Training for Sky-T1-32B
# 4 nodes x 4 A100 40GB GPUs with Context Parallelism for 16k sequences
# ============================================================

set -e

# Module environment
module purge
module load GCC/13.3.0
module load OpenMPI/5.0.5
module load CUDA/12
module load cuDNN/9.5.0.50

# Environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# HuggingFace cache
export HF_HOME=/p/project1/envcomp/yll/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Disable tokenizers parallelism (causes issues with multiprocessing)
export TOKENIZERS_PARALLELISM=false

# NCCL settings for InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=WARN

# Ray settings
export RAY_ADDRESS=""  # Let Ray auto-detect
export RAY_DEDUP_LOGS=0

# Working directory
cd /p/project1/envcomp/yll/adaptive-compute-rewrite

# Activate NeMo-RL venv
source nemo-rl/.venv/bin/activate

echo "============================================================"
echo "NeMo-RL SimPO Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Master: $MASTER_ADDR"
echo "Config: configs/nemo_simpo_32b.yaml"
echo "============================================================"

# Create logs directory if needed
mkdir -p slurm/logs

# Run SimPO training
python nemo-rl/examples/run_simpo.py \
    --config configs/nemo_simpo_32b.yaml \
    cluster.num_nodes=4 \
    cluster.gpus_per_node=4

echo "Training completed!"
