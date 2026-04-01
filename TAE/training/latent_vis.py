"""
Extract latent vectors, run Borderline-SMOTE, and generate UMAP plots.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import BorderlineSMOTE
from umap import UMAP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import TopologicalAutoencoder

# Config
DATA_PATH = 'data_preprocessing/cleaned_tcga_tpm_for_TAE.csv'
RESULTS_DIR = 'TAE/results'


LATENT_DIMS = [16, 32, 64]
DISTANCE_METRICS = ['euclidean', 'pearson', 'cosine']

WEIGHT_PATHS = {
    'euclidean': {16: 'TAE/models/tae_dim16.pth', 32: 'TAE/models/tae_dim32.pth', 64: 'TAE/models/tae_dim64.pth'},
    'pearson': {16: 'TAE/models/tae_dim16_pearson.pth', 32: 'TAE/models/tae_dim32_pearson.pth', 64: 'TAE/models/tae_dim64_pearson.pth'},
    'cosine': {16: 'TAE/models/tae_dim16_cosine.pth', 32: 'TAE/models/tae_dim32_cosine.pth', 64: 'TAE/models/tae_dim64_cosine.pth'},
}

UMAP_PARAMS = dict(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
SMOTE_PARAMS = dict(random_state=42, kind='borderline-1')

COLORS = {0: '#2196F3', 1: '#F44336'}
LABELS = {0: 'Normal', 1: 'Tumor'}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path, index_col=0)
    X = df.drop(columns=['Target']).values.astype(np.float32)
    y = df['Target'].values.astype(int)
    return X, y

def extract_latent(X, model, device, batch_size=512):
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32, device=device)
    latent_parts = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i + batch_size]
            _, z = model(batch)
            latent_parts.append(z.cpu().numpy())
    return np.vstack(latent_parts)

def apply_borderline_smote(Z, y):
    smote = BorderlineSMOTE(**SMOTE_PARAMS)
    Z_resampled, y_resampled = smote.fit_resample(Z, y)
    return Z_resampled, y_resampled

def save_latent(Z, y, path, dim):
    cols = [f'z{i}' for i in range(Z.shape[1])]
    df = pd.DataFrame(Z, columns=cols)
    df['Target'] = y
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")

def plot_before_after(Z_before, y_before, Z_after, y_after, dim, metric, save_path, sinkhorn=False, multiplier=1.0):
    umap_before = UMAP(**UMAP_PARAMS).fit_transform(Z_before)
    umap_after = UMAP(**UMAP_PARAMS).fit_transform(Z_after)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    title_suffix = f" (Sinkhorn, m={multiplier})" if sinkhorn else f" ({metric})"
    fig.suptitle(f'Latent Space {dim}D{title_suffix} — Borderline-SMOTE', fontsize=15, fontweight='bold', y=0.98)

    for ax, embedding, labels, title in [(axes[0], umap_before, y_before, 'Before SMOTE'), (axes[1], umap_after, y_after, 'After SMOTE')]:
        for cls in [1, 0]:
            mask = labels == cls
            ax.scatter(embedding[mask, 0], embedding[mask, 1], c=COLORS[cls], label=LABELS[cls], s=8, alpha=0.5)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def _process_model(X, y, input_dim, dim, metric, weight_path, device, sinkhorn=False, multiplier=1.0):
    model = TopologicalAutoencoder(input_dim=input_dim, latent_dim=dim).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    print(f"  Loaded weights: {weight_path}")

    Z = extract_latent(X, model, device)
    suffix = f"_sinkhorn_m{multiplier}" if sinkhorn else f"_{metric}"
    if not sinkhorn and metric == 'euclidean': suffix = ""
    
    save_latent(Z, y, os.path.join(RESULTS_DIR, 'woutSMOTE', f'latent_{dim}d{suffix}.csv'), dim)
    Z_smote, y_smote = apply_borderline_smote(Z, y)
    save_latent(Z_smote, y_smote, os.path.join(RESULTS_DIR, 'wSMOTE', f'latent_{dim}d{suffix}_smote.csv'), dim)
    
    fig_path = os.path.join(RESULTS_DIR, f'umap_borderline_smote_{dim}d{suffix}.png')
    plot_before_after(Z, y, Z_smote, y_smote, dim, metric, fig_path, sinkhorn=sinkhorn, multiplier=multiplier)
    del model
    torch.cuda.empty_cache()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, nargs='+', default=DISTANCE_METRICS)
    parser.add_argument('--sinkhorn', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = load_data(DATA_PATH)
    input_dim = X.shape[1]

    for metric in args.metric:
        for dim in LATENT_DIMS:
            if args.sinkhorn:
                if metric == 'euclidean': continue
                pattern = f'TAE/models/tae_dim{dim}_{metric}_sinkhorn_m*.pth'
                found_weights = glob.glob(pattern)
                for weight_path in found_weights:
                    try:
                        mult = float(weight_path.split('_m')[-1].replace('.pth', ''))
                    except:
                        mult = 1.0
                    print(f"\n--- Processing Sinkhorn Sweep: dim={dim}, metric={metric}, m={mult} ---")
                    _process_model(X, y, input_dim, dim, metric, weight_path, device, sinkhorn=True, multiplier=mult)
            else:
                weight_path = WEIGHT_PATHS.get(metric, {}).get(dim)
                if weight_path and os.path.exists(weight_path):
                    print(f"\n--- Processing Standard Model: dim={dim}, metric={metric} ---")
                    _process_model(X, y, input_dim, dim, metric, weight_path, device, sinkhorn=False)

    print("\nAll done.")

if __name__ == "__main__":
    main()
