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

# CUDA settings for Megatron
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 architecture

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

# ----------------------------
# Start a multi-node Ray cluster
# ----------------------------
GPUS_PER_NODE=4
RAY_PORT=6379
HEAD_NODE="$MASTER_ADDR"

# Use the first IP on the head node (Ray needs an IP, not just hostname)
HEAD_IP="$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" bash -lc 'hostname -I | cut -d" " -f1')"
export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "Ray head node: $HEAD_NODE"
echo "Ray head IP:   $HEAD_IP"
echo "Ray address:   $RAY_ADDRESS"

echo "Starting Ray on all allocated nodes..."
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-node="$GPUS_PER_NODE" \
  bash -lc '
set -e
cd /p/project1/envcomp/yll/adaptive-compute-rewrite
source nemo-rl/.venv/bin/activate

# Always stop any stale Ray first (safe if none running)
ray stop --force >/dev/null 2>&1 || true

NODE_HOSTNAME="$(hostname)"
if [ "$SLURM_NODEID" -eq 0 ]; then
  echo "[Ray] Starting head on ${NODE_HOSTNAME}..."
  ray start --head --node-ip-address="'"$HEAD_IP"'" --port="'"$RAY_PORT"'" \
    --num-cpus="'"$SLURM_CPUS_PER_TASK"'" --num-gpus="'"$GPUS_PER_NODE"'" --disable-usage-stats
else
  echo "[Ray] Starting worker on ${NODE_HOSTNAME}..."
  ray start --address="'"$HEAD_IP"':'"'"$RAY_PORT"'" \
    --num-cpus="'"$SLURM_CPUS_PER_TASK"'" --num-gpus="'"$GPUS_PER_NODE"'" --disable-usage-stats
fi
'

echo "Ray cluster started. Launching training on head..."

# Run SimPO training with Megatron backend (connects to Ray via RAY_ADDRESS)
python nemo-rl/examples/run_simpo.py \
  --config configs/nemo_simpo_32b.yaml \
  cluster.num_nodes=4 \
  cluster.gpus_per_node=4

echo "Stopping Ray cluster..."
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --cpus-per-task=1 bash -lc 'ray stop --force >/dev/null 2>&1 || true'

echo "Training completed!"
