#!/bin/bash
#SBATCH --job-name=convert_ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
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

cd /p/project1/envcomp/yll/adaptive-compute-rewrite
source nemo-rl/.venv/bin/activate

export HF_HOME="/p/project1/envcomp/yll/.cache/huggingface"
export HF_DATASETS_OFFLINE=1

# Convert step_200 checkpoint
echo "Converting step_200 checkpoint..."
python -c "
from megatron.bridge.models.hf_pretrained.causal_lm import HFCausalLMFromPretrained
from megatron.bridge.training.checkpointing import load_checkpoint
import torch

# This is a placeholder - actual conversion depends on NeMo-RL checkpoint format
# You may need to use the megatron-bridge converter or custom script
print('Checkpoint conversion needed - checking format...')
"

echo "Done"
