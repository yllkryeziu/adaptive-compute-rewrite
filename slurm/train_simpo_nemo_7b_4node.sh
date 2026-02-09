#!/bin/bash
#SBATCH --job-name=nemo-simpo-7b-4node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=logs/%j-nemo-simpo-7b-4node.out

# ============================================================
# NeMo-RL SimPO Training for Qwen2.5-7B
# 4 nodes x 4 A100 40GB GPUs @ 16k sequences
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
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# HuggingFace cache
export HF_HOME=/p/project1/envcomp/yll/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Disable tokenizers parallelism
export TOKENIZERS_PARALLELISM=false

# NCCL settings for InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=WARN

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="8.0"

# Ray settings
export RAY_ADDRESS=""
export RAY_DEDUP_LOGS=0
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
echo "NeMo-RL SimPO Training (7B @ 4 nodes)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Master: $MASTER_ADDR"
echo "Config: configs/nemo_simpo_7b_megatron_16k.yaml"
echo "============================================================"

# Create logs directory if needed
mkdir -p slurm/logs

# ----------------------------
# Start Ray cluster and run training
# All in one srun so Ray stays alive
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

# Export variables for srun subshells
export HEAD_IP RAY_PORT GPUS_PER_NODE

# Create a shared signal file for coordination
SIGNAL_FILE="/tmp/ray_training_done_${SLURM_JOB_ID}"

echo "Starting Ray cluster and training (all nodes in single srun)..."

srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" --gpus-per-node="$GPUS_PER_NODE" \
  bash -lc "
set -e
cd /p/project1/envcomp/yll/adaptive-compute-rewrite
source nemo-rl/.venv/bin/activate

# Stop any stale Ray first
ray stop --force >/dev/null 2>&1 || true

NODE_HOSTNAME=\"\$(hostname)\"
SIGNAL_FILE=\"$SIGNAL_FILE\"

if [ \"\$SLURM_NODEID\" -eq 0 ]; then
  # === HEAD NODE ===
  echo \"[Ray] Starting head on \${NODE_HOSTNAME}...\"
  ray start --head --node-ip-address=\"$HEAD_IP\" --port=\"$RAY_PORT\" \
    --num-cpus=\"\$SLURM_CPUS_PER_TASK\" --num-gpus=\"$GPUS_PER_NODE\" --disable-usage-stats

  # Wait for workers to join (simple wait)
  echo \"[Head] Waiting for workers to join...\"
  sleep 15

  # Run the training
  echo \"[Head] Starting SimPO training...\"
  python nemo-rl/examples/run_simpo.py \
    --config configs/nemo_simpo_7b_megatron_16k.yaml \
    cluster.num_nodes=4 \
    cluster.gpus_per_node=4

  # Signal completion to workers
  echo \"[Head] Training complete, signaling workers...\"
  touch \"\$SIGNAL_FILE\"

  # Stop Ray on head
  ray stop --force >/dev/null 2>&1 || true
  echo \"[Head] Done.\"

else
  # === WORKER NODES ===
  echo \"[Ray] Starting worker on \${NODE_HOSTNAME}...\"
  
  # Wait a moment for head to start
  sleep 5
  
  ray start --address=\"$HEAD_IP:$RAY_PORT\" \
    --num-cpus=\"\$SLURM_CPUS_PER_TASK\" --num-gpus=\"$GPUS_PER_NODE\" --disable-usage-stats

  # Wait for training to complete (poll for signal file)
  echo \"[Worker \$SLURM_NODEID] Waiting for training to complete...\"
  while [ ! -f \"\$SIGNAL_FILE\" ]; do
    sleep 10
  done

  # Stop Ray on worker
  ray stop --force >/dev/null 2>&1 || true
  echo \"[Worker \$SLURM_NODEID] Done.\"
fi
"

# Cleanup signal file
rm -f "$SIGNAL_FILE"

echo "============================================================"
echo "Training completed!"
echo "============================================================"
