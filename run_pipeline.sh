#!/bin/bash
# ==============================================================================
# Full pipeline: preprocessing -> training -> latent extraction/SMOTE -> visualization
# Usage: bash run_pipeline.sh [--skip-preprocess]
# ==============================================================================
set -e

CONDA_ENV="cuda_tda"
EPOCHS=100
BATCH_SIZE=64
TOPO_WEIGHT=1.0
DIMS=(16 32 64)
METRICS=("euclidean" "pearson" "cosine")

# Parse arguments
SKIP_PREPROCESS=false
for arg in "$@"; do
    if [ "$arg" == "--skip-preprocess" ]; then
        SKIP_PREPROCESS=true
    fi
done

echo "=============================================="
echo "  TCGA-BRCA TAE Pipeline"
echo "=============================================="

# --------------------------------------------------
# Step 1: Preprocessing (BRCA-filtered, GPU ComBat)
# --------------------------------------------------
if [ "$SKIP_PREPROCESS" = true ]; then
    echo ""
    echo "[Step 1/4] Skipping preprocessing as requested."
else
    echo ""
    echo "[Step 1/4] Preprocessing..."
    conda run -n $CONDA_ENV python data_preprocessing/preprocess_pipeline_gpu.py
fi

# --------------------------------------------------
# Step 2: Train TAE for all (dim, metric) combos
# --------------------------------------------------
echo ""
echo "[Step 2/4] Training TAE models..."

for metric in "${METRICS[@]}"; do
    for dim in "${DIMS[@]}"; do
        if [ "$metric" = "euclidean" ]; then
            output="TAE/models/tae_dim${dim}.pth"
        else
            output="TAE/models/tae_dim${dim}_${metric}.pth"
        fi

        echo ""
        echo "--- Training: dim=${dim}, metric=${metric} ---"
        conda run -n $CONDA_ENV python TAE/training/train.py \
            --dimension $dim \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --topo-weight $TOPO_WEIGHT \
            --distance-metric $metric \
            --output $output

        # Add Sinkhorn variant (recommended for high-dim spaces)
        if [ "$metric" != "euclidean" ]; then
            echo ""
            echo "--- Training Sinkhorn Sweep: dim=${dim}, metric=${metric} ---"
            
            # Run 3 random multipliers for each config to find the best scale
            for i in {1..3}; do
                # Generate random exponent between 1.0 and 5.0
                # Using python for clean float math
                EXP=$(python3 -c "import random; print(round(random.uniform(1.0, 5.0), 2))")
                MULT=$(python3 -c "print(round(10**$EXP, 1))")
                
                echo "  Run $i: topo_multiplier = $MULT (10^$EXP)"
                
                sinkhorn_output="TAE/models/tae_dim${dim}_${metric}_sinkhorn_m${MULT}.pth"
                
                conda run -n $CONDA_ENV python TAE/training/train.py \
                    --dimension $dim \
                    --epochs $EPOCHS \
                    --batch-size $BATCH_SIZE \
                    --distance-metric $metric \
                    --sinkhorn \
                    --topo-multiplier $MULT \
                    --output $sinkhorn_output
            done
        fi
    done
done

# --------------------------------------------------
# Step 3: Latent extraction, SMOTE, UMAP
# --------------------------------------------------
# Step 3: Latent extraction, SMOTE, UMAP
echo ""
echo "[Step 3/4] Latent extraction & SMOTE & UMAP..."
# Standard models
conda run -n $CONDA_ENV python TAE/training/latent_vis.py --metric euclidean pearson cosine
# Sinkhorn sweep models
conda run -n $CONDA_ENV python TAE/training/latent_vis.py --metric pearson cosine --sinkhorn

# --------------------------------------------------
# Step 4: Training visualization
# --------------------------------------------------
echo ""
echo "[Step 4/4] Generating training visualizations..."
conda run -n $CONDA_ENV python TAE/training/visualization.py

echo ""
echo "=============================================="
echo "  Pipeline complete!"
echo "=============================================="
