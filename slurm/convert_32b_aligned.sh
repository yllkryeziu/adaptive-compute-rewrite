#!/bin/bash
#SBATCH --job-name=convert_32b_aligned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=slurm/logs/%j-%x.out

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

CKPT_BASE="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b_aligned"
HF_OUTPUT_BASE="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b_aligned_hf"

mkdir -p $HF_OUTPUT_BASE

# Use nemo-rl venv for conversion (has megatron-core)
source nemo-rl/.venv/bin/activate

# Use local cached model path since compute nodes have no internet
LOCAL_MODEL_PATH="/p/project1/envcomp/yll/.cache/huggingface/hub/models--NovaSky-AI--Sky-T1-32B-Preview/snapshots/1e3f4c62a30c7ce70f4b3a3b952895d866943551"

# ============================================================
# Convert step_400 checkpoint to HF format
# ============================================================
echo "============================================================"
echo "Converting step_400 checkpoint to HuggingFace format..."
echo "============================================================"

python -c "
import torch
import sys
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
torch.serialization.add_safe_globals([torch.optim.AdamW, FusedAdam])

from nemo_rl.models.megatron.community_import import export_model_from_megatron

local_model_path = '$LOCAL_MODEL_PATH'

export_model_from_megatron(
    hf_model_name=local_model_path,
    input_path='$CKPT_BASE/step_400/policy/weights/iter_0000000',
    output_path='$HF_OUTPUT_BASE/step_400',
    hf_tokenizer_path=local_model_path,
    hf_overrides={},
    overwrite=True,
)
"

# ============================================================
# Convert step_600 checkpoint to HF format
# ============================================================
echo "============================================================"
echo "Converting step_600 checkpoint to HuggingFace format..."
echo "============================================================"

python -c "
import torch
import sys
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
torch.serialization.add_safe_globals([torch.optim.AdamW, FusedAdam])

from nemo_rl.models.megatron.community_import import export_model_from_megatron

local_model_path = '$LOCAL_MODEL_PATH'

export_model_from_megatron(
    hf_model_name=local_model_path,
    input_path='$CKPT_BASE/step_600/policy/weights/iter_0000000',
    output_path='$HF_OUTPUT_BASE/step_600',
    hf_tokenizer_path=local_model_path,
    hf_overrides={},
    overwrite=True,
)
"

# ============================================================
# Convert step_786 checkpoint (final) to HF format
# ============================================================
echo "============================================================"
echo "Converting step_786 checkpoint to HuggingFace format..."
echo "============================================================"

python -c "
import torch
import sys
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformer_engine.pytorch.optimizers.fused_adam import FusedAdam
torch.serialization.add_safe_globals([torch.optim.AdamW, FusedAdam])

from nemo_rl.models.megatron.community_import import export_model_from_megatron

local_model_path = '$LOCAL_MODEL_PATH'

export_model_from_megatron(
    hf_model_name=local_model_path,
    input_path='$CKPT_BASE/step_786/policy/weights/iter_0000000',
    output_path='$HF_OUTPUT_BASE/step_786',
    hf_tokenizer_path=local_model_path,
    hf_overrides={},
    overwrite=True,
)
"

# ============================================================
# Copy tokenizer files with chat template to all converted checkpoints
# (The conversion doesn't include the chat_template, causing vLLM errors)
# ============================================================
echo "============================================================"
echo "Copying tokenizer files with chat template..."
echo "============================================================"

for step_dir in $HF_OUTPUT_BASE/step_*; do
    echo "Copying tokenizer to $step_dir"
    cp $LOCAL_MODEL_PATH/tokenizer*.json "$step_dir/"
done

deactivate
echo "============================================================"
echo "Checkpoint conversion complete!"
echo "HF checkpoints saved to: $HF_OUTPUT_BASE"
echo "============================================================"
