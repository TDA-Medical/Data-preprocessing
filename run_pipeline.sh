#!/bin/bash
# run full pipeline

set -e

CONDA_ENV="cuda_tda"
EPOCHS=100
BATCH_SIZE=64
TOPO_WEIGHT=1.0
DIMS=(16 32 64)
METRICS=("euclidean" "pearson" "cosine")

SKIP_PREPROCESS=false
for arg in "$@"; do
    if [ "$arg" == "--skip-preprocess" ]; then
        SKIP_PREPROCESS=true
    fi
done

echo "Starting pipeline..."

# 1. Preprocessing
if [ "$SKIP_PREPROCESS" = true ]; then
    echo "Skipping preprocessing."
else
    echo "Preprocessing data..."
    conda run -n $CONDA_ENV python data_preprocessing/preprocess_pipeline_gpu.py
fi

# 2. Train TAE models
echo "Training TAE models..."

for metric in "${METRICS[@]}"; do
    for dim in "${DIMS[@]}"; do
        if [ "$metric" = "euclidean" ]; then
            output="TAE/models/tae_dim${dim}.pth"
        else
            output="TAE/models/tae_dim${dim}_${metric}.pth"
        fi

        echo "Training: dim=${dim}, metric=${metric}"
        conda run -n $CONDA_ENV python TAE/training/train.py \
            --dimension $dim \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --topo-weight $TOPO_WEIGHT \
            --distance-metric $metric \
            --output $output

        if [ "$metric" != "euclidean" ]; then
            echo "Running Sinkhorn Sweep for dim=${dim}, metric=${metric}"
            
            for i in {1..3}; do
                EXP=$(python3 -c "import random; print(round(random.uniform(1.0, 5.0), 2))")
                MULT=$(python3 -c "print(round(10**$EXP, 1))")
                
                echo "Run $i: multiplier = $MULT"
                
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

# 3. Latent extraction and UMAP
echo "Extracting latents and running SMOTE/UMAP..."
conda run -n $CONDA_ENV python TAE/training/latent_vis.py --metric euclidean pearson cosine
conda run -n $CONDA_ENV python TAE/training/latent_vis.py --metric pearson cosine --sinkhorn

# 4. Visualizations
echo "Plotting results..."
conda run -n $CONDA_ENV python TAE/training/visualization.py

echo "Done!"
