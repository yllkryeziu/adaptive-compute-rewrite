#!/bin/bash
#SBATCH --job-name=convert_eval_simpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=logs/%j-%x.out

# --- Environment Setup (minimal for conversion) ---
module --force purge
module load Stages/2025
module load GCC/13.3.0
module load OpenMPI/5.0.5
module load CUDA/12
# NOTE: Do NOT load cuDNN here - it conflicts with nemo-rl's transformer_engine

cd /p/project1/envcomp/yll/adaptive-compute-rewrite

export HF_HOME="/p/project1/envcomp/yll/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

CKPT_BASE="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b"
HF_OUTPUT_BASE="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b_hf"
EVAL_OUTPUT="/p/project1/envcomp/yll/adaptive-compute-rewrite/outputs/eval_math500"

mkdir -p $HF_OUTPUT_BASE
mkdir -p $EVAL_OUTPUT

# ============================================================
# Step 1: Convert step_200 checkpoint to HF format
# ============================================================
echo "============================================================"
echo "Converting step_200 checkpoint to HuggingFace format..."
echo "============================================================"

# Use nemo-rl venv for conversion (has megatron-core)
source nemo-rl/.venv/bin/activate

# Run conversion with safe globals set
# Use local cached model path since compute nodes have no internet
LOCAL_MODEL_PATH="/p/project1/envcomp/yll/.cache/huggingface/hub/models--NovaSky-AI--Sky-T1-32B-Preview/snapshots/1e3f4c62a30c7ce70f4b3a3b952895d866943551"

python -c "
import torch
import sys
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
torch.serialization.add_safe_globals([torch.optim.AdamW, FusedAdam])

# Now run the actual conversion
import yaml
from nemo_rl.models.megatron.community_import import export_model_from_megatron

# Use local path instead of HF ID
local_model_path = '$LOCAL_MODEL_PATH'

export_model_from_megatron(
    hf_model_name=local_model_path,
    input_path='$CKPT_BASE/step_200/policy/weights/iter_0000000',
    output_path='$HF_OUTPUT_BASE/step_200',
    hf_tokenizer_path=local_model_path,
    hf_overrides={},
    overwrite=True,
)
"

# ============================================================
# Step 2: Convert step_589 (final) checkpoint to HF format
# ============================================================
echo "============================================================"
echo "Converting step_589 checkpoint to HuggingFace format..."
echo "============================================================"

python -c "
import torch
import sys
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
torch.serialization.add_safe_globals([torch.optim.AdamW, FusedAdam])

# Now run the actual conversion
from nemo_rl.models.megatron.community_import import export_model_from_megatron

# Use local path instead of HF ID
local_model_path = '$LOCAL_MODEL_PATH'

export_model_from_megatron(
    hf_model_name=local_model_path,
    input_path='$CKPT_BASE/step_589/policy/weights/iter_0000000',
    output_path='$HF_OUTPUT_BASE/step_589',
    hf_tokenizer_path=local_model_path,
    hf_overrides={},
    overwrite=True,
)
"

deactivate
echo "Checkpoint conversion complete!"
echo "HF checkpoints saved to: $HF_OUTPUT_BASE"

# ============================================================
# Step 3: Evaluate BASE model (Sky-T1-32B-Preview)
# ============================================================
echo "============================================================"
echo "Evaluating BASE model: Sky-T1-32B-Preview"
echo "============================================================"

# Load cuDNN for evaluation (safe now that conversion is done)
module load cuDNN/9.5.0.50-CUDA-12

# Use main .venv for skythought evaluation
source .venv/bin/activate

skythought evaluate \
    --model /p/project1/envcomp/yll/.cache/huggingface/hub/models--NovaSky-AI--Sky-T1-32B-Preview/snapshots/1e3f4c62a30c7ce70f4b3a3b952895d866943551 \
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
# Step 4: Evaluate step_200 checkpoint
# ============================================================
echo "============================================================"
echo "Evaluating SimPO step_200 checkpoint"
echo "============================================================"

# Use same system prompt (skythought) for fair comparison
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
# Step 5: Evaluate step_589 (final) checkpoint
# ============================================================
echo "============================================================"
echo "Evaluating SimPO step_589 (final) checkpoint"
echo "============================================================"

# Use same system prompt (skythought) for fair comparison
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
