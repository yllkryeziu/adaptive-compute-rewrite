#!/bin/bash
#SBATCH --job-name=move-to-scratch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --account=envcomp
#SBATCH --output=logs/%j-move-to-scratch.out

# ============================================================
# Move large checkpoint files from project to scratch
# ============================================================

set -e

echo "============================================================"
echo "Moving checkpoints from project to scratch"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo "============================================================"

# Source and destination directories
PROJECT_DIR="/p/project1/envcomp/yll"
SCRATCH_DIR="/p/scratch/envcomp/yll"

# Create destination directories
mkdir -p "$SCRATCH_DIR/adaptive-compute/results"
mkdir -p "$SCRATCH_DIR/adaptive-compute/logs"
mkdir -p "$SCRATCH_DIR/adaptive-compute/wandb"
mkdir -p "$SCRATCH_DIR/pThought/outputs"

# Function to move with progress
move_with_progress() {
    local src="$1"
    local dst="$2"
    local name="$3"
    
    if [ -d "$src" ] && [ "$(ls -A "$src" 2>/dev/null)" ]; then
        echo ""
        echo ">>> Moving $name..."
        echo "    From: $src"
        echo "    To:   $dst"
        echo "    Size: $(du -sh "$src" 2>/dev/null | cut -f1)"
        
        # Use rsync for resumable transfer, then remove source
        rsync -av --progress --remove-source-files "$src/" "$dst/"
        
        # Remove empty directories left behind
        find "$src" -type d -empty -delete 2>/dev/null || true
        
        echo "    Done: $(date)"
    else
        echo ">>> Skipping $name (empty or doesn't exist)"
    fi
}

# Move adaptive-compute-rewrite files (~8TB)
move_with_progress \
    "$PROJECT_DIR/adaptive-compute-rewrite/results" \
    "$SCRATCH_DIR/adaptive-compute/results" \
    "adaptive-compute-rewrite/results"

move_with_progress \
    "$PROJECT_DIR/adaptive-compute-rewrite/logs" \
    "$SCRATCH_DIR/adaptive-compute/logs" \
    "adaptive-compute-rewrite/logs"

move_with_progress \
    "$PROJECT_DIR/adaptive-compute-rewrite/wandb" \
    "$SCRATCH_DIR/adaptive-compute/wandb" \
    "adaptive-compute-rewrite/wandb"

# Move pThought outputs (~2.7TB)
move_with_progress \
    "$PROJECT_DIR/pThought/outputs" \
    "$SCRATCH_DIR/pThought/outputs" \
    "pThought/outputs"

echo ""
echo "============================================================"
echo "Move completed!"
echo "Finished: $(date)"
echo ""
echo "Space freed from project:"
du -sh "$PROJECT_DIR/adaptive-compute-rewrite" 2>/dev/null || true
du -sh "$PROJECT_DIR/pThought" 2>/dev/null || true
echo ""
echo "Space used on scratch:"
du -sh "$SCRATCH_DIR/adaptive-compute" 2>/dev/null || true
du -sh "$SCRATCH_DIR/pThought" 2>/dev/null || true
echo "============================================================"
