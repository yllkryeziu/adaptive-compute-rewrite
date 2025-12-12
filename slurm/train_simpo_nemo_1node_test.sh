#!/bin/bash
#SBATCH --job-name=nemo-simpo-7b-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --account=envcomp
#SBATCH --output=slurm/logs/nemo-simpo-7b-test-%j.out
#SBATCH --error=slurm/logs/nemo-simpo-7b-test-%j.err

# ============================================================
# NeMo-RL SimPO Test Training for Qwen2-7B
# Single node test to validate setup before scaling
# ============================================================

set -e

# Module environment
module purge
module load GCC/13.3.0
module load OpenMPI/5.0.5
module load CUDA/12
module load cuDNN/9.5.0.50

# Environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# HuggingFace cache
export HF_HOME=/p/project1/envcomp/yll/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_HOME

# Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM=false

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Ray settings
export RAY_ADDRESS=""
export RAY_DEDUP_LOGS=0

# Working directory
cd /p/project1/envcomp/yll/adaptive-compute-rewrite

# Activate NeMo-RL venv
source nemo-rl/.venv/bin/activate

echo "============================================================"
echo "NeMo-RL SimPO Test Training (7B)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: 4"
echo "Config: configs/nemo_simpo_7b_test.yaml"
echo "============================================================"

# Create logs directory if needed
mkdir -p slurm/logs

# Run SimPO training test
python nemo-rl/examples/run_simpo.py \
    --config configs/nemo_simpo_7b_test.yaml \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=4

echo "Test training completed!"
