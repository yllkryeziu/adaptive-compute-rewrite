#!/bin/bash
#SBATCH --job-name=convert_v4
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
# Do NOT load cuDNN here - conflicts with nemo-rl

cd /p/project1/envcomp/yll/adaptive-compute-rewrite

export HF_HOME="/p/project1/envcomp/yll/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN

# Activate environment
source nemo-rl/.venv/bin/activate

# Local model path for tokenizer
LOCAL_MODEL_PATH="/p/project1/envcomp/yll/.cache/huggingface/hub/models--NovaSky-AI--Sky-T1-32B-Preview/snapshots/1e3f4c62a30c7ce70f4b3a3b952895d866943551"

# Function to convert checkpoint
convert_checkpoint() {
    local VERSION=$1
    local STEP=$2
    
    CKPT_BASE="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b_${VERSION}"
    HF_OUTPUT_BASE="/p/project1/envcomp/yll/adaptive-compute-rewrite/results/nemo_simpo_32b_${VERSION}_hf"
    
    mkdir -p $HF_OUTPUT_BASE

    echo "============================================================"
    echo "Converting ${VERSION} step_${STEP} to HuggingFace format..."
    echo "Input: $CKPT_BASE/step_${STEP}/policy/weights/iter_0000000"
    echo "Output: $HF_OUTPUT_BASE/step_${STEP}"
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
    input_path='$CKPT_BASE/step_${STEP}/policy/weights/iter_0000000',
    output_path='$HF_OUTPUT_BASE/step_${STEP}',
    hf_tokenizer_path=local_model_path,
    hf_overrides={},
    overwrite=True,
)
"
    # Copy tokenizer files
    echo "Copying tokenizer files to $HF_OUTPUT_BASE/step_${STEP}..."
    cp $LOCAL_MODEL_PATH/tokenizer*.json "$HF_OUTPUT_BASE/step_${STEP}/"
}

# Convert v4 steps (latest training run with NovaSky-equivalent config)
for STEP in 80 90 98; do
    convert_checkpoint "v4" $STEP
done

deactivate
echo "All conversions complete."
