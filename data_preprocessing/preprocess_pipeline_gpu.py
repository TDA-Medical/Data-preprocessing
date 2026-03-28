"""
GPU-accelerated TCGA RNA-seq preprocessing pipeline.
Usage: python data_preprocessing/preprocess_pipeline_gpu.py
"""

import os
import numpy as np
import pandas as pd
import time


def compute_skew_kurtosis(values):
    """Compute skewness and excess kurtosis using pure numpy (no scipy overhead)."""
    mean = values.mean(axis=0)
    diff = values - mean
    n = values.shape[0]
    var = (diff ** 2).sum(axis=0) / n
    std = np.sqrt(var)
    std[std == 0] = 1.0
    skewness = ((diff ** 3).sum(axis=0) / n) / (std ** 3)
    kurtosis = ((diff ** 4).sum(axis=0) / n) / (std ** 4) - 3.0
    return skewness, kurtosis


# TCGA-BRCA project TSS (Tissue Source Site) codes
BRCA_TSS = {'A1','A2','A7','A8','AC','AN','AO','AQ','AR','B6','BH',
            'C8','D8','E2','E9','EW','GM','GI','HN','LD','LL','MS',
            'OL','PE','PL','S3','UL','UU','WT','XX','Z7'}


def _is_brca(barcode):
    """Check if a TCGA barcode belongs to the BRCA project via its TSS code."""
    parts = str(barcode).split('-')
    return len(parts) > 1 and parts[1] in BRCA_TSS


def load_and_merge(tumor_file, normal_file):
    """Load tumor/normal TPM data, transpose to (samples x genes), merge, and filter to BRCA."""
    print("Loading Tumor TPM data...")
    tumor_df = pd.read_csv(tumor_file, sep='\t', index_col=0).T
    print("Loading Normal TPM data...")
    normal_df = pd.read_csv(normal_file, sep='\t', index_col=0).T

    tumor_df['Target'] = 1
    normal_df['Target'] = 0

    print("Merging data...")
    merged = pd.concat([tumor_df, normal_df])
    del tumor_df, normal_df

    # Filter to BRCA patients only
    brca_mask = merged.index.map(_is_brca)
    n_before = len(merged)
    merged = merged[brca_mask]
    print(f"BRCA filter: {n_before} -> {len(merged)} samples "
          f"(Tumor={(merged['Target']==1).sum()}, Normal={(merged['Target']==0).sum()})")
    return merged


def filter_blacklist(df, bad_genes_file):
    """Remove known noisy genes from the dataset."""
    print("Applying blacklist filtering...")
    with open(bad_genes_file, 'r') as f:
        bad_genes = [line.strip() for line in f if line.strip()]

    to_drop = [g for g in bad_genes if g in df.columns]
    df.drop(columns=to_drop, inplace=True)
    print(f"Removed {len(to_drop)} bad genes.")
    return len(to_drop)


def selective_log_transform(df, gene_cols, chunk_size=1000):
    """
    Apply log1p to genes with high skewness (|skew| > 2) or kurtosis (> 10).
    Processes in chunks to keep memory usage low.
    """
    print("Performing selective log transformation...")
    transform_count = 0
    total = len(gene_cols)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        cols = gene_cols[start:end]
        values = df[cols].values

        skew, kurt = compute_skew_kurtosis(values)
        needs_transform = (np.abs(skew) > 2.0) | (kurt > 10.0)
        cols_to_log = [c for c, flag in zip(cols, needs_transform) if flag]

        if cols_to_log:
            df[cols_to_log] = np.log1p(df[cols_to_log])
            transform_count += len(cols_to_log)

        print(f"  Processed {end}/{total} genes...")

    print(f"Applied log1p to {transform_count} genes.")
    return transform_count


def extract_batch_labels(df, clin_file):
    """Extract TSS (Tissue Source Site) codes as batch labels from clinical data."""
    print("Extracting batch information from clinical data...")
    clin_df = pd.read_csv(clin_file, sep='\t', index_col=0, low_memory=False).T
    clin_df = clin_df.iloc[2:]  # Skip metadata rows

    # Map patient barcodes to TSS codes
    patient_to_tss = {
        idx[:12]: str(idx).split('-')[1]
        for idx in clin_df.index
        if isinstance(idx, str) and '-' in idx
    }
    del clin_df

    df['TSS_Code'] = df.index.str[:12].map(patient_to_tss)

    # Fallback: extract TSS directly from barcode for unmapped samples
    missing = df['TSS_Code'].isna()
    if missing.any():
        df.loc[missing, 'TSS_Code'] = df.index[missing].str.split('-').str[1]

    return df


def drop_singleton_batches(df):
    """Remove samples whose batch has only 1 sample (ComBat requires >= 2)."""
    counts = df['TSS_Code'].value_counts()
    valid = counts[counts > 1].index
    n_dropped = counts[counts == 1].sum() if (counts == 1).any() else 0

    if n_dropped > 0:
        print(f"Dropping {n_dropped} samples due to batch size of 1.")
        df = df[df['TSS_Code'].isin(valid)]

    return df, len(valid), n_dropped


def drop_zero_variance_genes(df, gene_cols):
    """Remove genes with zero variance (uninformative and break ComBat)."""
    variances = df[gene_cols].var()
    zero_var = variances[variances == 0].index

    if len(zero_var) > 0:
        print(f"Dropping {len(zero_var)} zero-variance genes.")
        df.drop(columns=zero_var, inplace=True)
        gene_cols = [c for c in df.columns if c not in ['Target', 'TSS_Code']]

    return df, gene_cols, len(zero_var)


def main():
    print("=== Statistical Preprocessing Pipeline (GPU) ===")
    pipeline_start = time.time()

    # Paths (relative to project root)
    clin_file = 'GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt'
    tumor_file = 'GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz'
    normal_file = 'GSE62944_RAW/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt.gz'
    bad_genes_file = 'data_analysis/overall_bad_genes_4fold.txt'
    out_file = 'data_preprocessing/cleaned_tcga_tpm_for_TAE.csv'

    # Step 1: Load and merge
    df = load_and_merge(tumor_file, normal_file)
    initial_samples = df.shape[0]
    initial_genes = df.shape[1] - 1
    print(f"Merged: {initial_samples} samples, {initial_genes} genes")

    # Step 2: Remove noisy genes
    n_bad = filter_blacklist(df, bad_genes_file)
    gene_cols = [c for c in df.columns if c != 'Target']

    # Step 3: Log-transform skewed genes
    n_transformed = selective_log_transform(df, gene_cols)
    df = df.copy()  # Defragment after column-wise modifications

    # Step 4: Batch effect correction with GPU ComBat
    df = extract_batch_labels(df, clin_file)
    df, n_valid_batches, n_dropped = drop_singleton_batches(df)
    df, gene_cols, n_zero_var = drop_zero_variance_genes(df, gene_cols)

    print(f"Running ComBat with {n_valid_batches} batches...")

    covars = df[['TSS_Code', 'Target']].copy()
    expression_matrix = df[gene_cols].values.T  # (genes, samples)

    from gpu_combat import gpu_combat

    combat_start = time.time()
    result = gpu_combat(
        dat=expression_matrix,
        covars=covars,
        batch_col='TSS_Code',
        categorical_cols=['Target']
    )
    combat_time = time.time() - combat_start
    print(f"GPU ComBat completed in {combat_time:.1f}s")

    # Step 5: Export
    output_df = pd.DataFrame(result['data'].T, index=df.index, columns=gene_cols)
    output_df['Target'] = df['Target']

    print(f"Exporting to {out_file}...")
    output_df.to_csv(out_file)

    # Report
    total_time = time.time() - pipeline_start
    print(f"\n{'='*40}")
    print("      PREPROCESSING PIPELINE REPORT")
    print(f"{'='*40}")
    print(f"Initial Samples:        {initial_samples}")
    print(f"Initial Genes:          {initial_genes}")
    print(f"Bad Genes Removed:      {n_bad}")
    print(f"Genes Log-Transformed:  {n_transformed}")
    print(f"Zero-Var Genes Removed: {n_zero_var}")
    print(f"Valid Batches (TSS):    {n_valid_batches}")
    print(f"Samples Dropped(Batch): {n_dropped}")
    print(f"Final Samples:          {output_df.shape[0]}")
    print(f"Final Genes:            {output_df.shape[1] - 1}")
    print(f"ComBat Time:            {combat_time:.1f}s")
    print(f"Total Pipeline Time:    {total_time:.1f}s")
    print(f"{'='*40}")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
