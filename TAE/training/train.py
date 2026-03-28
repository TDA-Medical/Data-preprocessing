import os
import sys
import copy
import json
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import TopologicalAutoencoder
from models.loss import TopologicalLoss
from models.loss_alternative import PearsonTopologicalLoss, CosineTopologicalLoss

LOSS_CLASSES = {
    'euclidean': TopologicalLoss,
    'pearson': PearsonTopologicalLoss,
    'cosine': CosineTopologicalLoss,
}


def _extract_latent(model, X, device, batch_size=512):
    """Extract latent vectors in batches to avoid GPU OOM."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size].to(device)
            _, z = model(batch)
            parts.append(z.cpu().numpy())
    return np.vstack(parts)


def _evaluate_latent_classifier(model, X_train, y_train, X_val, y_val, device):
    """
    Fit logistic regression on train latent vectors, evaluate on val.
    Returns dict with accuracy, auc, f1, precision, recall.
    """
    z_train = _extract_latent(model, X_train, device)
    z_val = _extract_latent(model, X_val, device)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(z_train, y_train)

    y_pred = clf.predict(z_val)
    y_prob = clf.predict_proba(z_val)[:, 1]

    return {
        'accuracy': accuracy_score(y_val, y_pred),
        'auc': roc_auc_score(y_val, y_prob),
        'f1': f1_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
    }


def train_tae(data_tensor, input_dim, latent_dim=16, epochs=100, batch_size=64, topo_weight=1.0,
              labels=None, val_split=0.2, clf_every=10, log_dir='TAE/results',
              distance_metric='euclidean'):
    """
    Train a Topological Autoencoder on gene expression data.

    Args:
        data_tensor: Input tensor of shape (n_samples, n_genes).
        input_dim:   Number of genes (features).
        latent_dim:  Latent space dimension (default: 16).
        epochs:      Number of training epochs (default: 100).
        batch_size:  Mini-batch size. Should be >= 64 for stable topological loss.
        topo_weight: Weight for the topological loss term (default: 1.0).
        labels:      Class labels array (0=Normal, 1=Tumor). When provided,
                     enables stratified train/val split and classification metrics.
        val_split:   Fraction of data for validation (default: 0.2).
        clf_every:   Compute classification metrics every N epochs (default: 10).
        distance_metric: Distance metric for topological loss. One of
                     'euclidean', 'pearson', or 'cosine' (default: 'euclidean').

    Returns:
        (model, history) — trained model and dict of per-epoch losses / metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TopologicalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    LossClass = LOSS_CLASSES[distance_metric]
    criterion = LossClass(topo_weight=topo_weight).to(device)
    print(f"Distance metric: {distance_metric} ({LossClass.__name__})")
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    history = {
        'train_total': [], 'train_recon': [], 'train_topo': [],
    }

    best_val_loss = float('inf')
    best_val_recon = None
    best_val_topo = None
    best_model_state = None
    best_epoch = -1

    # --- Data splitting ---
    if labels is not None:
        y_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
        indices = np.arange(len(data_tensor))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_split, stratify=y_np, random_state=42
        )

        X_train_tensor = data_tensor[train_idx]
        X_val_tensor = data_tensor[val_idx]
        y_train = y_np[train_idx]
        y_val = y_np[val_idx]

        train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor), batch_size=batch_size, shuffle=False)

        history.update({
            'val_total': [], 'val_recon': [], 'val_topo': [],
            'clf_epochs': [], 'accuracy': [], 'auc': [], 'f1': [], 'precision': [], 'recall': [],
        })

        print(f"Train/Val split: {len(train_idx)} train, {len(val_idx)} val "
              f"(Normal: {(y_train == 0).sum()}/{(y_val == 0).sum()}, "
              f"Tumor: {(y_train == 1).sum()}/{(y_val == 1).sum()})")
    else:
        train_loader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)
        val_loader = None

    # --- Training loop ---
    for epoch in range(epochs):
        # Training pass
        model.train()
        total, recon, topo = 0.0, 0.0, 0.0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            x_recon, z = model(x_batch)
            loss, loss_recon, loss_topo = criterion(x_batch, x_recon, z)
            loss.backward()
            optimizer.step()

            total += loss.item()
            recon += loss_recon.item()
            topo += loss_topo.item()

        n_train = len(train_loader)
        history['train_total'].append(total / n_train)
        history['train_recon'].append(recon / n_train)
        history['train_topo'].append(topo / n_train)

        # Validation pass
        if val_loader is not None:
            model.eval()
            val_total, val_recon, val_topo = 0.0, 0.0, 0.0
            with torch.no_grad():
                for (x_batch,) in val_loader:
                    x_batch = x_batch.to(device)
                    x_recon, z = model(x_batch)
                    loss, loss_recon, loss_topo = criterion(x_batch, x_recon, z)
                    val_total += loss.item()
                    val_recon += loss_recon.item()
                    val_topo += loss_topo.item()
            n_val = len(val_loader)
            history['val_total'].append(val_total / n_val)
            history['val_recon'].append(val_recon / n_val)
            history['val_topo'].append(val_topo / n_val)

            # Best model checkpoint
            current_val_loss = val_total / n_val
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_val_recon = val_recon / n_val
                best_val_topo = val_topo / n_val
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  [Checkpoint] New best val_loss: {best_val_loss:.4f} (epoch {best_epoch})")

        # Logging
        if (epoch + 1) % 10 == 0:
            log = (f"Epoch [{epoch+1}/{epochs}] | "
                   f"Train - Total: {history['train_total'][-1]:.4f}  "
                   f"Recon: {history['train_recon'][-1]:.4f}  "
                   f"Topo: {history['train_topo'][-1]:.4f}")
            if val_loader is not None:
                log += (f" | Val - Total: {history['val_total'][-1]:.4f}  "
                        f"Recon: {history['val_recon'][-1]:.4f}  "
                        f"Topo: {history['val_topo'][-1]:.4f}")
            print(log)

        # Classification metrics (periodic)
        if labels is not None and (epoch + 1) % clf_every == 0:
            metrics = _evaluate_latent_classifier(
                model, X_train_tensor, y_train, X_val_tensor, y_val, device
            )
            history['clf_epochs'].append(epoch + 1)
            for k, v in metrics.items():
                history[k].append(v)
            print(f"  Classifier | Acc: {metrics['accuracy']:.4f}  "
                  f"AUC: {metrics['auc']:.4f}  F1: {metrics['f1']:.4f}  "
                  f"Prec: {metrics['precision']:.4f}  Rec: {metrics['recall']:.4f}")

    # --- Restore best model if validation was used ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")

    # --- Save training log ---
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # CSV log (per-epoch)
    import pandas as pd
    log_df = pd.DataFrame({'epoch': list(range(1, epochs + 1))})
    log_df['train_total'] = history['train_total']
    log_df['train_recon'] = history['train_recon']
    log_df['train_topo'] = history['train_topo']
    if 'val_total' in history:
        log_df['val_total'] = history['val_total']
        log_df['val_recon'] = history['val_recon']
        log_df['val_topo'] = history['val_topo']
    csv_path = os.path.join(log_dir, f'training_log_dim{latent_dim}_{timestamp}.csv')
    log_df.to_csv(csv_path, index=False)
    print(f"Training log saved: {csv_path}")

    # JSON summary
    summary = {
        'timestamp': timestamp,
        'latent_dim': latent_dim,
        'epochs': epochs,
        'batch_size': batch_size,
        'topo_weight': topo_weight,
        'distance_metric': distance_metric,
        'val_split': val_split,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
        'best_val_recon': best_val_recon,
        'best_val_topo': best_val_topo,
        'final_train_loss': history['train_total'][-1],
        'final_train_recon': history['train_recon'][-1],
        'final_train_topo': history['train_topo'][-1],
        'final_val_loss': history['val_total'][-1] if 'val_total' in history else None,
        'final_val_recon': history['val_recon'][-1] if 'val_recon' in history else None,
        'final_val_topo': history['val_topo'][-1] if 'val_topo' in history else None,
    }
    if 'clf_epochs' in history and history['clf_epochs']:
        summary['final_clf_metrics'] = {
            'accuracy': history['accuracy'][-1],
            'auc': history['auc'][-1],
            'f1': history['f1'][-1],
            'precision': history['precision'][-1],
            'recall': history['recall'][-1],
        }
    json_path = os.path.join(log_dir, f'training_summary_dim{latent_dim}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved: {json_path}")

    return model, history


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Train Topological Autoencoder")
    parser.add_argument('--dimension', type=int, default=16, help='Latent space dimension (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--topo-weight', type=float, default=0.5, help='Topological loss weight (default: 0.5)')
    parser.add_argument('--distance-metric', type=str, default='euclidean',
                        choices=['euclidean', 'pearson', 'cosine'],
                        help='Distance metric for topological loss (default: euclidean)')
    parser.add_argument('--output', type=str, default='TAE/models/tae_dim16.pth', help='Output model path (default: TAE/models/tae_dim16.pth)')
    args = parser.parse_args()

    data_path = 'data_preprocessing/cleaned_tcga_tpm_for_TAE.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop(columns=['Target']).values
    y = df['Target'].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    print(f"Data loaded: {X_tensor.shape[0]} samples, {X_tensor.shape[1]} genes")
    print(f"Class distribution: Normal={(y == 0).sum()}, Tumor={(y == 1).sum()}")
    print(f"Config: latent_dim={args.dimension}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, topo_weight={args.topo_weight}, "
          f"distance_metric={args.distance_metric}")

    model, history = train_tae(X_tensor, input_dim=X_tensor.shape[1], latent_dim=args.dimension,
                               epochs=args.epochs, batch_size=args.batch_size,
                               topo_weight=args.topo_weight, labels=y,
                               log_dir='TAE/results',
                               distance_metric=args.distance_metric)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")
