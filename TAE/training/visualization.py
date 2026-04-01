import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_latest_logs(results_dir='TAE/results'):
    """Load the latest JSON summary and corresponding CSV log for each (dimension, metric, sinkhorn, multiplier) config."""
    json_files = glob.glob(os.path.join(results_dir, 'training_summary_dim*.json'))

    # Group by (dimension, metric, sinkhorn, multiplier) and find the latest
    dim_logs = {}
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)

        dim = data['latent_dim']
        metric = data.get('distance_metric', 'euclidean')
        sinkhorn = data.get('sinkhorn', False)
        multiplier = data.get('topo_multiplier', 1.0) if sinkhorn else 1.0
        timestamp = data['timestamp']
        key = (dim, metric, sinkhorn, multiplier)

        # Check if we already have a newer one for this config
        if key not in dim_logs or dim_logs[key]['timestamp'] < timestamp:
            csv_path = os.path.join(results_dir, f'training_log_dim{dim}_{timestamp}.csv')
            if os.path.exists(csv_path):
                data['csv_path'] = csv_path
                data['distance_metric'] = metric
                data['sinkhorn'] = sinkhorn
                data['topo_multiplier'] = multiplier
                dim_logs[key] = data

    return dim_logs

def plot_learning_curves(dim_logs, save_dir='TAE/results'):
    """Plot Train vs Val losses across epochs for each config."""
    keys = sorted(dim_logs.keys())
    if not keys:
        print("No log files found.")
        return

    for (dim, metric, sinkhorn, mult) in keys:
        data = dim_logs[(dim, metric, sinkhorn, mult)]
        df = pd.read_csv(data['csv_path'])

        suffix = f" (Sinkhorn, m={mult})" if sinkhorn else ""
        save_suffix = f"_sinkhorn_m{mult}" if sinkhorn else ""

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Learning Curves - Dim {dim}, {metric}{suffix}', fontsize=16, fontweight='bold')

        # Total Loss
        axes[0].plot(df['epoch'], df['train_total'], label='Train Total', c='#2196F3', linewidth=2)
        if 'val_total' in df.columns:
            axes[0].plot(df['epoch'], df['val_total'], label='Val Total', c='#FF9800', linewidth=2)
        axes[0].set_title('Total Loss', fontsize=14)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # Recon Loss
        axes[1].plot(df['epoch'], df['train_recon'], label='Train Recon', c='#4CAF50', linewidth=2)
        if 'val_recon' in df.columns:
            axes[1].plot(df['epoch'], df['val_recon'], label='Val Recon', c='#FF5722', linewidth=2)
        axes[1].set_title('Reconstruction Loss', fontsize=14)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        # Topo Loss
        axes[2].plot(df['epoch'], df['train_topo'], label='Train Topo', c='#9C27B0', linewidth=2)
        if 'val_topo' in df.columns:
            axes[2].plot(df['epoch'], df['val_topo'], label='Val Topo', c='#E91E63', linewidth=2)
        axes[2].set_title('Topological Loss', fontsize=14)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Loss', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'learning_curves_dim{dim}_{metric}{save_suffix}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

def plot_classifier_metrics(dim_logs, save_dir='TAE/results'):
    """Plot separate bar charts for each classifier metric across configs."""
    records = []
    for (dim, metric, sinkhorn, mult), data in dim_logs.items():
        if 'final_clf_metrics' in data:
            row = dict(data['final_clf_metrics'])
            sink_str = f"-sink-m{mult}" if sinkhorn else ""
            row['Config'] = f'{dim}D-{metric}{sink_str}'
            records.append(row)

    if not records:
        print("No classifier metrics found in logs.")
        return

    df = pd.DataFrame(records).sort_values('Config')
    metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    for metric_name, color in zip(metrics, colors):
        if metric_name not in df.columns: continue
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Config', y=metric_name, hue='Config', palette=[color], legend=False)
        plt.title(f'Classifier {metric_name.upper()} Comparison', fontsize=15, fontweight='bold')
        plt.ylim(0.0, 1.05)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'clf_{metric_name}_cmp.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

def plot_best_losses(dim_logs, save_dir='TAE/results'):
    """Plot separate bar charts for best validation losses across configs."""
    records = []
    for (dim, metric, sinkhorn, mult), data in dim_logs.items():
        best_recon = data.get('best_val_recon')
        best_topo = data.get('best_val_topo')
        sink_str = f"-sink-m{mult}" if sinkhorn else ""

        records.append({
            'Config': f'{dim}D-{metric}{sink_str}',
            'Total': data.get('best_val_loss', 0) or 0,
            'Recon': best_recon if best_recon is not None else data.get('final_val_recon', 0),
            'Topo': best_topo if best_topo is not None else data.get('final_val_topo', 0)
        })

    if not records:
        return

    df = pd.DataFrame(records).sort_values('Config')
    
    tasks = [
        ('Total', 'best_val_total_cmp.png', 'Blues_d'),
        ('Recon', 'best_val_recon_cmp.png', 'Greens_d'),
        ('Topo', 'best_val_topo_cmp.png', 'Reds_d')
    ]

    for column, filename, palette in tasks:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Config', y=column, hue='Config', palette=palette, legend=False)
        plt.title(f'Best Validation {column} Loss Comparison', fontsize=15, fontweight='bold')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

def main():
    print(f"[{'='*40}]\nGenerating Visualizations from Logs\n[{'='*40}]")
    dim_logs = load_latest_logs()
    
    if not dim_logs:
        print("No valid logs found in TAE/results/.")
        return

    configs = []
    for d, m, s, mult in dim_logs.keys():
        s_str = f"-sink-m{mult}" if s else ""
        configs.append(f"{d}D-{m}{s_str}")
    
    print(f"Found logs for configs: {sorted(configs)}\n")
    
    print("Generating learning curve plots...")
    plot_learning_curves(dim_logs)
    
    print("\nGenerating classifier comparisons...")
    plot_classifier_metrics(dim_logs)
    
    print("\nGenerating validation loss comparisons...")
    plot_best_losses(dim_logs)
    
    print("\nVisualization complete. Check TAE/results/ for PNG files.")

if __name__ == '__main__':
    main()
