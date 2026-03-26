import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_latest_logs(results_dir='TAE/results'):
    """Load the latest JSON summary and corresponding CSV log for each latent dimension."""
    json_files = glob.glob(os.path.join(results_dir, 'training_summary_dim*.json'))
    
    # Group by dimension and find the latest
    dim_logs = {}
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
        
        dim = data['latent_dim']
        timestamp = data['timestamp']
        
        # Check if we already have a newer one for this dim
        if dim not in dim_logs or dim_logs[dim]['timestamp'] < timestamp:
            csv_path = os.path.join(results_dir, f'training_log_dim{dim}_{timestamp}.csv')
            if os.path.exists(csv_path):
                data['csv_path'] = csv_path
                dim_logs[dim] = data
                
    return dim_logs

def plot_learning_curves(dim_logs, save_dir='TAE/results'):
    """Plot Train vs Val losses across epochs for each dimension."""
    dims = sorted(dim_logs.keys())
    if not dims:
        print("No log files found.")
        return
        
    for dim in dims:
        data = dim_logs[dim]
        df = pd.read_csv(data['csv_path'])
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Learning Curves - Latent Dimension {dim}', fontsize=16, fontweight='bold')
        
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
        save_path = os.path.join(save_dir, f'learning_curves_dim{dim}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

def plot_classifier_metrics(dim_logs, save_dir='TAE/results'):
    """Plot bar chart comparing classifier metrics across dimensions."""
    records = []
    for dim, data in dim_logs.items():
        if 'final_clf_metrics' in data:
            metrics = data['final_clf_metrics']
            metrics['Dimension'] = str(dim)
            records.append(metrics)
            
    if not records:
        print("No classifier metrics found in logs.")
        return
        
    df = pd.DataFrame(records).sort_values('Dimension')
    df_melted = df.melt(id_vars='Dimension', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Dimension', palette='viridis')
    plt.title('Latent Space Linear Classifier Performance by Dimension', fontsize=15, fontweight='bold')
    plt.ylim(0.5, 1.05)
    plt.legend(title='Latent Dim', loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'classifier_performance_cmp.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_best_losses(dim_logs, save_dir='TAE/results'):
    """Plot bar chart comparing best validation losses across dimensions."""
    records = []
    for dim, data in dim_logs.items():
        # Handle the case where best_val_recon might still be None in older logs
        best_recon = data.get('best_val_recon')
        best_topo = data.get('best_val_topo')
        
        records.append({
            'Dimension': str(dim),
            'Best Val Total': data.get('best_val_loss', 0) or 0,
            # Fallback to final validation loss if best_val_recon is None (older logs)
            'Best Val Recon': best_recon if best_recon is not None else data.get('final_val_recon', 0),
            'Best Val Topo': best_topo if best_topo is not None else data.get('final_val_topo', 0)
        })
        
    if not records:
        return
        
    df = pd.DataFrame(records).sort_values('Dimension')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Best Validation Losses by Latent Dimension', fontsize=16, fontweight='bold', y=1.05)
    
    sns.barplot(ax=axes[0], data=df, x='Dimension', y='Best Val Total', palette='Blues_d')
    axes[0].set_title('Total Loss', fontsize=13)
    
    sns.barplot(ax=axes[1], data=df, x='Dimension', y='Best Val Recon', palette='Greens_d')
    axes[1].set_title('Recon Loss', fontsize=13)
    
    sns.barplot(ax=axes[2], data=df, x='Dimension', y='Best Val Topo', palette='Reds_d')
    axes[2].set_title('Topo Loss', fontsize=13)
    
    for ax in axes:
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'best_validation_losses_cmp.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def main():
    print(f"[{'='*40}]\nGenerating Visualizations from Logs\n[{'='*40}]")
    dim_logs = load_latest_logs()
    
    if not dim_logs:
        print("No valid logs found in TAE/results/.")
        return
        
    print(f"Found logs for dimensions: {sorted(dim_logs.keys())}\n")
    
    print("Generating learning curve plots...")
    plot_learning_curves(dim_logs)
    
    print("\nGenerating classifier comparisons...")
    plot_classifier_metrics(dim_logs)
    
    print("\nGenerating validation loss comparisons...")
    plot_best_losses(dim_logs)
    
    print("\nVisualization complete. Check TAE/results/ for PNG files.")

if __name__ == '__main__':
    main()
