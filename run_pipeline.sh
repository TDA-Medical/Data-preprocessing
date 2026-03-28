#!/bin/bash
# ==============================================================================
# Full pipeline: preprocessing -> training -> latent extraction/SMOTE -> visualization
# Usage: bash run_pipeline.sh
# ==============================================================================
set -e

CONDA_ENV="cuda_tda"
EPOCHS=100
BATCH_SIZE=64
TOPO_WEIGHT=0.5
DIMS=(16 32 64)
METRICS=("euclidean" "pearson" "cosine")

echo "=============================================="
echo "  TCGA-BRCA TAE Pipeline"
echo "=============================================="

# --------------------------------------------------
# Step 1: Preprocessing (BRCA-filtered, GPU ComBat)
# --------------------------------------------------
echo ""
echo "[Step 1/4] Preprocessing..."
conda run -n $CONDA_ENV python data_preprocessing/preprocess_pipeline_gpu.py

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
    done
done

# --------------------------------------------------
# Step 3: Latent extraction, SMOTE, UMAP
# --------------------------------------------------
echo ""
echo "[Step 3/4] Latent extraction & SMOTE & UMAP..."
conda run -n $CONDA_ENV python TAE/training/latent_vis.py

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
