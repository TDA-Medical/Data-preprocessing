import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import TopologicalAutoencoder
from models.loss import TopologicalLoss


def train_tae(data_tensor, input_dim, latent_dim=16, epochs=100, batch_size=64, topo_weight=1.0):
    """
    Train a Topological Autoencoder on gene expression data.

    Args:
        data_tensor: Input tensor of shape (n_samples, n_genes).
        input_dim:   Number of genes (features).
        latent_dim:  Latent space dimension (default: 16).
        epochs:      Number of training epochs (default: 100).
        batch_size:  Mini-batch size. Should be >= 64 for stable topological loss.
        topo_weight: Weight for the topological loss term (default: 1.0).

    Returns:
        Trained TopologicalAutoencoder model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = TopologicalAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = TopologicalLoss(topo_weight=topo_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    dataloader = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total, recon, topo = 0.0, 0.0, 0.0

        for (x_batch,) in dataloader:
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            x_recon, z = model(x_batch)
            loss, loss_recon, loss_topo = criterion(x_batch, x_recon, z)
            loss.backward()
            optimizer.step()

            total += loss.item()
            recon += loss_recon.item()
            topo += loss_topo.item()

        if (epoch + 1) % 10 == 0:
            n = len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Total: {total/n:.4f} | Recon: {recon/n:.4f} | Topo: {topo/n:.4f}")

    return model


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Train Topological Autoencoder")
    parser.add_argument('--dimension', type=int, default=16, help='Latent space dimension (default: 16)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--topo-weight', type=float, default=0.5, help='Topological loss weight (default: 0.5)')
    parser.add_argument('--output', type=str, default='TAE/tae_trained.pth', help='Output model path (default: TAE/tae_trained.pth)')
    args = parser.parse_args()

    data_path = 'data_preprocessing/cleaned_tcga_tpm_for_TAE.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0)
    X = df.drop(columns=['Target']).values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    print(f"Data loaded: {X_tensor.shape[0]} samples, {X_tensor.shape[1]} genes")
    print(f"Config: latent_dim={args.dimension}, epochs={args.epochs}, "
          f"batch_size={args.batch_size}, topo_weight={args.topo_weight}")

    model = train_tae(X_tensor, input_dim=X_tensor.shape[1], latent_dim=args.dimension,
                      epochs=args.epochs, batch_size=args.batch_size, topo_weight=args.topo_weight)

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")
