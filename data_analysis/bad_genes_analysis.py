import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gzip
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 0. Setup
# ---------------------------------------------------------
OUTPUT_DIR = './output_bad_genes/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

TUMOR_FILE = 'GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz'
NORMAL_FILE = 'GSE62944_RAW/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt.gz'

# ---------------------------------------------------------
# 1. Load "bad genes" lists
# ---------------------------------------------------------
with open('tcga_bad_genes_4fold.txt', 'r') as f:
    tcga_bad_genes = set(line.strip() for line in f if line.strip())

with open('overall_bad_genes_4fold.txt', 'r') as f:
    overall_bad_genes = set(line.strip() for line in f if line.strip())

# Set analysis
shared_bad = tcga_bad_genes & overall_bad_genes
tcga_only = tcga_bad_genes - overall_bad_genes
overall_only = overall_bad_genes - tcga_bad_genes
all_bad_genes = tcga_bad_genes | overall_bad_genes

print(f"TCGA bad genes: {len(tcga_bad_genes)}")
print(f"Overall bad genes: {len(overall_bad_genes)}")
print(f"Shared (both lists): {len(shared_bad)}")
print(f"TCGA-only: {len(tcga_only)}")
print(f"Overall-only: {len(overall_only)}")
print(f"Union (all bad genes): {len(all_bad_genes)}")

# Check if BRCA genes are included
for label, gene_set in [('TCGA bad', tcga_bad_genes), ('Overall bad', overall_bad_genes)]:
    brca_in = [g for g in gene_set if 'BRCA' in g.upper()]
    if brca_in:
        print(f"  BRCA-related genes in {label}: {brca_in}")

# ---------------------------------------------------------
# 2. Full gene scan: collect statistics by bad gene status
# ---------------------------------------------------------
print("\nScanning all genes and collecting bad gene statistics...")

gene_stats = []
sample_barcodes = []
target_labels = []

with gzip.open(TUMOR_FILE, 'rt') as f_t, gzip.open(NORMAL_FILE, 'rt') as f_n:
    header_t = f_t.readline().strip().split('\t')
    header_n = f_n.readline().strip().split('\t')

    samples_t = header_t if header_t[0].startswith('TCGA') else header_t[1:]
    samples_n = header_n if header_n[0].startswith('TCGA') else header_n[1:]

    sample_barcodes = samples_t + samples_n
    target_labels = np.array([1]*len(samples_t) + [0]*len(samples_n))

    for line_t, line_n in zip(f_t, f_n):
        parts_t = line_t.strip().split('\t')
        parts_n = line_n.strip().split('\t')

        gene_name_t = parts_t[0]
        gene_name_n = parts_n[0]
        if gene_name_t != gene_name_n:
            continue

        gene_name = gene_name_t
        expr_t = np.array(parts_t[1:], dtype=np.float32)
        expr_n = np.array(parts_n[1:], dtype=np.float32)

        expr_t_log = np.log2(expr_t + 1)
        expr_n_log = np.log2(expr_n + 1)
        expr_all_log = np.concatenate([expr_t_log, expr_n_log])

        t_stat, p_val = stats.ttest_ind(expr_t_log, expr_n_log, equal_var=False)
        pb_corr, _ = stats.pointbiserialr(target_labels, expr_all_log)

        in_tcga_bad = gene_name in tcga_bad_genes
        in_overall_bad = gene_name in overall_bad_genes

        gene_stats.append({
            'Gene': gene_name,
            'Mean_Normal': np.mean(expr_n),
            'Mean_Tumor': np.mean(expr_t),
            'Log2FC': np.mean(expr_t_log) - np.mean(expr_n_log),
            'T_Stat': t_stat,
            'P_Value': p_val,
            'PB_Corr': pb_corr,
            'Abs_PB_Corr': abs(pb_corr),
            'In_TCGA_Bad': in_tcga_bad,
            'In_Overall_Bad': in_overall_bad,
            'Bad_Category': 'Both' if (in_tcga_bad and in_overall_bad)
                            else 'TCGA_Only' if in_tcga_bad
                            else 'Overall_Only' if in_overall_bad
                            else 'Not_Bad'
        })

df = pd.DataFrame(gene_stats)
print(f"Full gene analysis complete: {len(df)} genes")

# ---------------------------------------------------------
# 3. Venn diagram-style summary table
# ---------------------------------------------------------
venn_summary = pd.DataFrame({
    'Category': ['TCGA Bad Genes', 'Overall Bad Genes', 'Shared', 'TCGA Only', 'Overall Only', 'Union', 'Not Bad'],
    'Count': [len(tcga_bad_genes), len(overall_bad_genes), len(shared_bad),
              len(tcga_only), len(overall_only), len(all_bad_genes),
              len(df[df['Bad_Category'] == 'Not_Bad'])],
})
# Number of matches found in actual data
matched_tcga = df['In_TCGA_Bad'].sum()
matched_overall = df['In_Overall_Bad'].sum()
venn_summary['Matched_in_Data'] = [matched_tcga, matched_overall,
                                    df[(df['In_TCGA_Bad']) & (df['In_Overall_Bad'])].shape[0],
                                    df[(df['In_TCGA_Bad']) & (~df['In_Overall_Bad'])].shape[0],
                                    df[(~df['In_TCGA_Bad']) & (df['In_Overall_Bad'])].shape[0],
                                    df[df['Bad_Category'] != 'Not_Bad'].shape[0],
                                    df[df['Bad_Category'] == 'Not_Bad'].shape[0]]
venn_summary.to_csv(f'{OUTPUT_DIR}01_Venn_Summary.csv', index=False)

# ---------------------------------------------------------
# 4. Bad vs Not-Bad gene comparison statistics
# ---------------------------------------------------------
print("Comparing bad vs not-bad genes...")

df_bad = df[df['Bad_Category'] != 'Not_Bad']
df_notbad = df[df['Bad_Category'] == 'Not_Bad']

comparison = pd.DataFrame({
    'Metric': ['Count', 'Mean |PB_Corr|', 'Median |PB_Corr|',
               'Mean |Log2FC|', 'Median |Log2FC|',
               'Mean |T_Stat|', '% with p<0.05', '% with p<0.001'],
    'Bad_Genes': [
        len(df_bad),
        df_bad['Abs_PB_Corr'].mean(),
        df_bad['Abs_PB_Corr'].median(),
        df_bad['Log2FC'].abs().mean(),
        df_bad['Log2FC'].abs().median(),
        df_bad['T_Stat'].abs().mean(),
        (df_bad['P_Value'] < 0.05).mean() * 100,
        (df_bad['P_Value'] < 0.001).mean() * 100,
    ],
    'Not_Bad_Genes': [
        len(df_notbad),
        df_notbad['Abs_PB_Corr'].mean(),
        df_notbad['Abs_PB_Corr'].median(),
        df_notbad['Log2FC'].abs().mean(),
        df_notbad['Log2FC'].abs().median(),
        df_notbad['T_Stat'].abs().mean(),
        (df_notbad['P_Value'] < 0.05).mean() * 100,
        (df_notbad['P_Value'] < 0.001).mean() * 100,
    ]
})
comparison.to_csv(f'{OUTPUT_DIR}02_Bad_vs_NotBad_Comparison.csv', index=False)

# Mann-Whitney U test: are bad genes' correlations systematically different?
u_stat, u_pval = stats.mannwhitneyu(
    df_bad['Abs_PB_Corr'].dropna(), df_notbad['Abs_PB_Corr'].dropna(), alternative='two-sided'
)
print(f"  Mann-Whitney U (|PB_Corr| bad vs not-bad): U={u_stat:.0f}, p={u_pval:.2e}")

# ---------------------------------------------------------
# 5. Distribution comparison visualization
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 5a) |Point-Biserial Corr| distribution
for label, subset, color in [('Bad', df_bad, 'salmon'), ('Not Bad', df_notbad, 'steelblue')]:
    sns.histplot(subset['Abs_PB_Corr'], kde=True, ax=axes[0], color=color, label=label, alpha=0.5, stat='density')
axes[0].set_title('|Point-Biserial Corr| Distribution')
axes[0].legend()

# 5b) |Log2 Fold Change| distribution
for label, subset, color in [('Bad', df_bad, 'salmon'), ('Not Bad', df_notbad, 'steelblue')]:
    sns.histplot(subset['Log2FC'].abs(), kde=True, ax=axes[1], color=color, label=label, alpha=0.5, stat='density')
axes[1].set_title('|Log2 Fold Change| Distribution')
axes[1].legend()

# 5c) -log10(p-value) distribution
for label, subset, color in [('Bad', df_bad, 'salmon'), ('Not Bad', df_notbad, 'steelblue')]:
    log_p = -np.log10(subset['P_Value'].clip(lower=1e-300))
    sns.histplot(log_p, kde=True, ax=axes[2], color=color, label=label, alpha=0.5, stat='density')
axes[2].set_title('-log10(P-Value) Distribution')
axes[2].legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}03_Bad_vs_NotBad_Distributions.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# 6. 3-category comparison (TCGA-only, Overall-only, Both)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

categories = ['Both', 'TCGA_Only', 'Overall_Only']
df_cats = df[df['Bad_Category'].isin(categories)]

sns.boxplot(data=df_cats, x='Bad_Category', y='Abs_PB_Corr', ax=axes[0], palette='Set2', order=categories)
axes[0].set_title('|PB Corr| by Bad Gene Category')

sns.boxplot(data=df_cats, x='Bad_Category', y='Log2FC', ax=axes[1], palette='Set2', order=categories)
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Log2 Fold Change by Bad Gene Category')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_Bad_Category_Comparison.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# 7. Volcano Plot: all genes, bad genes highlighted
# ---------------------------------------------------------
print("Generating volcano plot...")
plt.figure(figsize=(12, 8))

df['neg_log10_p'] = -np.log10(df['P_Value'].clip(lower=1e-300))

# Not bad genes (background)
mask_notbad = df['Bad_Category'] == 'Not_Bad'
plt.scatter(df.loc[mask_notbad, 'Log2FC'], df.loc[mask_notbad, 'neg_log10_p'],
            c='lightgray', s=3, alpha=0.3, label='Not Bad')

# Bad genes
for cat, color, marker in [('TCGA_Only', 'blue', 'o'), ('Overall_Only', 'green', 's'), ('Both', 'red', '^')]:
    mask = df['Bad_Category'] == cat
    plt.scatter(df.loc[mask, 'Log2FC'], df.loc[mask, 'neg_log10_p'],
                c=color, s=15, alpha=0.6, label=cat, marker=marker)

# Highlight BRCA-related genes
brca_genes = df[df['Gene'].str.contains('BRCA', case=False)]
for _, row in brca_genes.iterrows():
    plt.annotate(row['Gene'], (row['Log2FC'], row['neg_log10_p']),
                 fontsize=9, fontweight='bold', color='darkred',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8),
                 xytext=(row['Log2FC']+0.3, row['neg_log10_p']+5))

plt.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='p=0.05')
plt.xlabel('Log2 Fold Change (Tumor vs Normal)')
plt.ylabel('-log10(P-Value)')
plt.title('Volcano Plot: Bad Genes Highlighted (BRCA annotated)')
plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}05_Volcano_Bad_Genes.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# 8. Top bad genes ranked by significance
# ---------------------------------------------------------
df_bad_sorted = df[df['Bad_Category'] != 'Not_Bad'].sort_values('P_Value')
df_bad_sorted.to_csv(f'{OUTPUT_DIR}06_All_Bad_Genes_Ranked.csv', index=False)

# Top 20 most significant bad genes
top20 = df_bad_sorted.head(20)
top20_neg_log_p = -np.log10(top20['P_Value'].clip(lower=1e-300))

plt.figure(figsize=(12, 6))
colors = top20['Bad_Category'].map({'Both': 'red', 'TCGA_Only': 'blue', 'Overall_Only': 'green'})
plt.barh(range(len(top20)), top20_neg_log_p, color=colors)
plt.yticks(range(len(top20)), top20['Gene'])
plt.xlabel('-log10(P-Value)')
plt.title('Top 20 Most Significant Bad Genes (Tumor vs Normal)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}07_Top20_Bad_Genes.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# 9. BRCA-specific analysis
# ---------------------------------------------------------
print("\nAnalyzing BRCA-related genes...")
brca_related = df[df['Gene'].str.contains('BRCA', case=False)]
if not brca_related.empty:
    print(brca_related[['Gene', 'Mean_Normal', 'Mean_Tumor', 'Log2FC', 'P_Value', 'PB_Corr', 'Bad_Category']].to_string())
    brca_related.to_csv(f'{OUTPUT_DIR}08_BRCA_Genes_Detail.csv', index=False)
else:
    print("  No BRCA-related genes found.")

# ---------------------------------------------------------
# 10. Bad genes enrichment by chromosome/pathway proxy (gene name patterns)
# ---------------------------------------------------------
# Count HLA genes in bad list (immune-related)
hla_bad = df_bad[df_bad['Gene'].str.startswith('HLA-')]
hist_bad = df_bad[df_bad['Gene'].str.startswith('HIST')]
krt_bad = df_bad[df_bad['Gene'].str.startswith('KRT')]
rpl_bad = df_bad[df_bad['Gene'].str.startswith('RPL')]
rps_bad = df_bad[df_bad['Gene'].str.startswith('RPS')]

gene_family_summary = pd.DataFrame({
    'Gene_Family': ['HLA (Immune)', 'HIST (Histone)', 'KRT (Keratin)', 'RPL (Ribosomal L)', 'RPS (Ribosomal S)'],
    'Count_in_Bad': [len(hla_bad), len(hist_bad), len(krt_bad), len(rpl_bad), len(rps_bad)],
    'Total_in_Dataset': [
        df['Gene'].str.startswith('HLA-').sum(),
        df['Gene'].str.startswith('HIST').sum(),
        df['Gene'].str.startswith('KRT').sum(),
        df['Gene'].str.startswith('RPL').sum(),
        df['Gene'].str.startswith('RPS').sum(),
    ]
})
gene_family_summary['Pct_Bad'] = (gene_family_summary['Count_in_Bad'] / gene_family_summary['Total_in_Dataset'] * 100).round(1)
gene_family_summary.to_csv(f'{OUTPUT_DIR}09_Gene_Family_Enrichment.csv', index=False)

print(f"\nAll analyses complete. Results have been saved to the {OUTPUT_DIR} folder.")
