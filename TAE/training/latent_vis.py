"""
[Step 3] Latent vector extraction, Borderline-SMOTE, and UMAP visualization.

For each latent dimension, this script:
  1. Extracts latent vectors Z from a pre-trained TAE encoder
  2. Saves the original latent vectors as CSV
  3. Applies Borderline-SMOTE to balance Normal/Tumor
  4. Saves the SMOTE-augmented latent vectors as CSV
  5. Generates before/after UMAP scatter plots

Output files (per dimension):
  TAE/results/woutSMOTE/latent_{dim}d.csv         — original latent vectors
  TAE/results/wSMOTE/latent_{dim}d_smote.csv      — SMOTE-augmented latent vectors
  TAE/results/umap_borderline_smote_{dim}d.png    — UMAP figure

Requirements:
    pip install torch imbalanced-learn umap-learn matplotlib seaborn

Usage (from project root):
    python TAE/training/latent_vis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import BorderlineSMOTE
from umap import UMAP

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import TopologicalAutoencoder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = 'data_preprocessing/cleaned_tcga_tpm_for_TAE.csv'
RESULTS_DIR = 'TAE/results'

LATENT_DIMS = [16, 32, 64]

# Mapping from latent_dim -> weight file path
# Adjust these paths if your weight files are named differently
WEIGHT_PATHS = {
    16: 'TAE/models/tae_dim16.pth',
    32: 'TAE/models/tae_dim32.pth',
    64: 'TAE/models/tae_dim64.pth',
}

UMAP_PARAMS = dict(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
SMOTE_PARAMS = dict(random_state=42, kind='borderline-1')

COLORS = {0: '#2196F3', 1: '#F44336'}  # Normal=blue, Tumor=red
LABELS = {0: 'Normal', 1: 'Tumor'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(path):
    """Load preprocessed CSV and split into features / labels."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, index_col=0)
    X = df.drop(columns=['Target']).values.astype(np.float32)
    y = df['Target'].values.astype(int)
    print(f"  Samples: {len(y)}  |  Genes: {X.shape[1]}")
    print(f"  Normal: {(y == 0).sum()}  |  Tumor: {(y == 1).sum()}")
    return X, y


def extract_latent(X, model, device, batch_size=512):
    """Pass data through the encoder and return latent vectors Z."""
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
    """Apply Borderline-SMOTE to balance Normal/Tumor in latent space."""
    smote = BorderlineSMOTE(**SMOTE_PARAMS)
    Z_resampled, y_resampled = smote.fit_resample(Z, y)

    n_original = len(y)
    n_synthetic = len(y_resampled) - n_original
    print(f"  SMOTE: generated {n_synthetic} synthetic Normal samples")
    print(f"  After: Normal={( y_resampled == 0).sum()}  |  Tumor={(y_resampled == 1).sum()}")

    return Z_resampled, y_resampled


def save_latent(Z, y, path, dim):
    """Save latent vectors + labels to CSV."""
    cols = [f'z{i}' for i in range(Z.shape[1])]
    df = pd.DataFrame(Z, columns=cols)
    df['Target'] = y
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({df.shape[0]} samples, {dim}D)")


def plot_before_after(Z_before, y_before, Z_after, y_after, dim, save_path):
    """Create 1x2 UMAP scatter plot: before vs after Borderline-SMOTE."""
    print(f"  Computing UMAP projections...")
    umap_before = UMAP(**UMAP_PARAMS).fit_transform(Z_before)
    umap_after = UMAP(**UMAP_PARAMS).fit_transform(Z_after)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Latent Space ({dim}D) — Borderline-SMOTE Augmentation',
                 fontsize=15, fontweight='bold', y=0.98)

    for ax, embedding, labels, title in [
        (axes[0], umap_before, y_before, 'Before Borderline-SMOTE'),
        (axes[1], umap_after, y_after, 'After Borderline-SMOTE\n(Danger Zone Augmented)'),
    ]:
        for cls in [1, 0]:  # Draw Tumor first so Normal points are on top
            mask = labels == cls
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=COLORS[cls], label=LABELS[cls],
                       s=8, alpha=0.5, edgecolors='none')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(markerscale=3, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    X, y = load_data(DATA_PATH)
    input_dim = X.shape[1]

    for dim in LATENT_DIMS:
        print(f"\n{'='*50}")
        print(f"  Processing latent_dim = {dim}")
        print(f"{'='*50}")

        weight_path = WEIGHT_PATHS.get(dim, f'TAE/models/tae_dim{dim}.pth')
        if not os.path.exists(weight_path):
            print(f"  [SKIP] Weight file not found: {weight_path}")
            print(f"         Train first: python TAE/training/train.py --dimension {dim} --output {weight_path}")
            continue

        # Load model
        model = TopologicalAutoencoder(input_dim=input_dim, latent_dim=dim).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print(f"  Loaded weights: {weight_path}")

        # Extract latent vectors
        Z = extract_latent(X, model, device)
        print(f"  Latent shape: {Z.shape}")

        # Save original latent vectors
        save_latent(Z, y, os.path.join(RESULTS_DIR, 'woutSMOTE', f'latent_{dim}d.csv'), dim)

        # Borderline-SMOTE
        Z_smote, y_smote = apply_borderline_smote(Z, y)

        # Save SMOTE-augmented latent vectors
        save_latent(Z_smote, y_smote, os.path.join(RESULTS_DIR, 'wSMOTE', f'latent_{dim}d_smote.csv'), dim)

        # UMAP visualization
        fig_path = os.path.join(RESULTS_DIR, f'umap_borderline_smote_{dim}d.png')
        plot_before_after(Z, y, Z_smote, y_smote, dim, fig_path)

        del model
        torch.cuda.empty_cache()

    print("\nAll done.")


if __name__ == "__main__":
    main()
