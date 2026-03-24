"""
CPU preprocessing pipeline with parallel neuroCombat.
For GPU version, use preprocess_pipeline_gpu.py instead.

Usage: python data_preprocessing/preprocess_pipeline.py
"""

import os
import sys
import io
import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from multiprocessing import Pool, cpu_count
import time


def _run_combat_chunk(args):
    """Worker: run neuroCombat on a subset of genes (suppresses internal output)."""
    chunk_genes, data_values, sample_index, covars_df = args
    import pandas as pd
    import numpy as np
    from neuroCombat import neuroCombat
    import sys, io

    dat = pd.DataFrame(data_values, index=chunk_genes, columns=sample_index)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = neuroCombat(dat=dat, covars=covars_df,
                             batch_col='TSS_Code', categorical_cols=['Target'])
    finally:
        sys.stdout = old_stdout

    return result['data']


def main():
    print("=== Statistical Preprocessing Pipeline (CPU) ===")

    # Paths (relative to project root)
    clin_file = 'GSE62944_06_01_15_TCGA_24_548_Clinical_Variables_9264_Samples.txt'
    tumor_file = 'GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz'
    normal_file = 'GSE62944_RAW/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt.gz'
    bad_genes_file = 'data_analysis/overall_bad_genes_4fold.txt'
    out_file = 'data_preprocessing/cleaned_tcga_tpm_for_TAE.csv'

    # Step 1: Load and merge
    print("Loading Tumor TPM data...")
    tumor_df = pd.read_csv(tumor_file, sep='\t', index_col=0).T
    print("Loading Normal TPM data...")
    normal_df = pd.read_csv(normal_file, sep='\t', index_col=0).T

    tumor_df['Target'] = 1
    normal_df['Target'] = 0

    print("Merging data...")
    df = pd.concat([tumor_df, normal_df])
    del tumor_df, normal_df
    initial_samples = df.shape[0]
    initial_genes = df.shape[1] - 1
    print(f"Merged: {initial_samples} samples, {initial_genes} genes")

    # Step 2: Remove noisy genes
    print("Applying blacklist filtering...")
    with open(bad_genes_file, 'r') as f:
        bad_genes = [line.strip() for line in f if line.strip()]
    genes_to_drop = [g for g in bad_genes if g in df.columns]
    df.drop(columns=genes_to_drop, inplace=True)
    print(f"Removed {len(genes_to_drop)} bad genes.")

    # Step 3: Selective log transformation (chunked numpy, no scipy)
    print("Performing selective log transformation...")
    gene_cols = [c for c in df.columns if c != 'Target']
    transform_count = 0
    total_genes = len(gene_cols)

    for start in range(0, total_genes, 1000):
        end = min(start + 1000, total_genes)
        cols = gene_cols[start:end]
        vals = df[cols].values
        mean = vals.mean(axis=0)
        diff = vals - mean
        n = vals.shape[0]
        std = np.sqrt((diff ** 2).sum(axis=0) / n)
        std[std == 0] = 1.0
        skew = ((diff ** 3).sum(axis=0) / n) / (std ** 3)
        kurt = ((diff ** 4).sum(axis=0) / n) / (std ** 4) - 3.0

        to_log = [c for c, s, k in zip(cols, skew, kurt) if abs(s) > 2.0 or k > 10.0]
        if to_log:
            df[to_log] = np.log1p(df[to_log])
            transform_count += len(to_log)
        print(f"  Processed {end}/{total_genes} genes...")

    print(f"Applied log1p to {transform_count} genes.")
    df = df.copy()  # Defragment

    # Step 4: Batch effect correction (parallel neuroCombat)
    print("Extracting batch information from clinical data...")
    clin_df = pd.read_csv(clin_file, sep='\t', index_col=0, low_memory=False).T.iloc[2:]
    patient_to_tss = {
        idx[:12]: str(idx).split('-')[1]
        for idx in clin_df.index if isinstance(idx, str) and '-' in idx
    }
    del clin_df

    df['TSS_Code'] = df.index.str[:12].map(patient_to_tss)
    missing = df['TSS_Code'].isna()
    if missing.any():
        df.loc[missing, 'TSS_Code'] = df.index[missing].str.split('-').str[1]

    # Drop singleton batches
    counts = df['TSS_Code'].value_counts()
    valid_batches = counts[counts > 1].index
    dropped_samples = counts[counts == 1].sum() if (counts == 1).any() else 0
    if dropped_samples > 0:
        print(f"Dropping {dropped_samples} samples due to batch size of 1.")
        df = df[df['TSS_Code'].isin(valid_batches)]

    # Drop zero-variance genes
    variances = df[gene_cols].var()
    zero_var_genes = variances[variances == 0].index
    if len(zero_var_genes) > 0:
        print(f"Dropping {len(zero_var_genes)} zero-variance genes.")
        df.drop(columns=zero_var_genes, inplace=True)
        gene_cols = [c for c in df.columns if c not in ['Target', 'TSS_Code']]

    print(f"Running neuroCombat with {len(valid_batches)} batches...")

    covars = df[['TSS_Code', 'Target']].copy()
    sample_index = df.index

    n_workers = min(4, max(1, cpu_count() - 1))
    n_chunks = n_workers * 2
    gene_chunks = np.array_split(gene_cols, n_chunks)
    print(f"Parallel: {n_workers} workers, {n_chunks} chunks (~{len(gene_chunks[0])} genes/chunk)")

    def gen_args():
        for chunk in gene_chunks:
            cols = list(chunk)
            yield (cols, df[cols].values.T, sample_index, covars)

    with Pool(processes=n_workers) as pool:
        results = []
        combat_start = time.time()
        for result in pool.imap(_run_combat_chunk, gen_args()):
            results.append(result)
            done = len(results)
            elapsed = time.time() - combat_start
            eta_sec = int((elapsed / done) * (n_chunks - done))
            eta_m, eta_s = divmod(eta_sec, 60)
            print(f"\r  neuroCombat progress: {done}/{n_chunks} "
                  f"({done/n_chunks*100:.1f}%) | ETA: {eta_m}m {eta_s}s   ",
                  end="", flush=True)
        print()

    # Reassemble
    corrected = np.vstack(results).T
    output_df = pd.DataFrame(corrected, index=df.index, columns=gene_cols)
    output_df['Target'] = df['Target']

    # Step 5: Export
    print(f"Exporting to {out_file}...")
    output_df.to_csv(out_file)

    print(f"\n{'='*40}")
    print("      PREPROCESSING PIPELINE REPORT")
    print(f"{'='*40}")
    print(f"Initial Samples:        {initial_samples}")
    print(f"Initial Genes:          {initial_genes}")
    print(f"Bad Genes Removed:      {len(genes_to_drop)}")
    print(f"Genes Log-Transformed:  {transform_count}")
    print(f"Zero-Var Genes Removed: {len(zero_var_genes)}")
    print(f"Valid Batches (TSS):    {len(valid_batches)}")
    print(f"Samples Dropped(Batch): {dropped_samples}")
    print(f"Final Samples:          {output_df.shape[0]}")
    print(f"Final Genes:            {output_df.shape[1] - 1}")
    print(f"{'='*40}")
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
