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
OUTPUT_DIR = './output_full_genes/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

TUMOR_FILE = 'GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz'
NORMAL_FILE = 'GSE62944_RAW/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt.gz'

# =========================================================
# [Pass 1] Sequential scan and statistical testing of all 23,368 genes
# =========================================================
print("1. Starting sequential analysis of all genes. (This may take some time)")

all_genes_stats = []
sample_barcodes = []
target_labels = []  # 1: Tumor, 0: Normal
mismatched_genes = 0

with gzip.open(TUMOR_FILE, 'rt') as f_t, gzip.open(NORMAL_FILE, 'rt') as f_n:
    # 1. Process headers (sample barcodes)
    header_t = f_t.readline().strip().split('\t')
    header_n = f_n.readline().strip().split('\t')

    samples_t = header_t if header_t[0].startswith('TCGA') else header_t[1:]
    samples_n = header_n if header_n[0].startswith('TCGA') else header_n[1:]

    sample_barcodes = samples_t + samples_n
    target_labels = np.array([1]*len(samples_t) + [0]*len(samples_n))

    # 2. Iterate line by line (one gene per line) and compute statistics
    for line_t, line_n in zip(f_t, f_n):
        parts_t = line_t.strip().split('\t')
        parts_n = line_n.strip().split('\t')

        gene_name_t = parts_t[0]
        gene_name_n = parts_n[0]

        # Verify gene order matches between files
        if gene_name_t != gene_name_n:
            mismatched_genes += 1
            continue

        gene_name = gene_name_t

        # Convert data to numpy arrays (optimized for speed and memory)
        expr_t = np.array(parts_t[1:], dtype=np.float32)
        expr_n = np.array(parts_n[1:], dtype=np.float32)
        expr_all = np.concatenate([expr_t, expr_n])

        # Log2 transformation (skewness correction)
        expr_all_log = np.log2(expr_all + 1)
        expr_t_log = np.log2(expr_t + 1)
        expr_n_log = np.log2(expr_n + 1)

        # T-test and Point-Biserial correlation coefficient
        t_stat, p_val_t = stats.ttest_ind(expr_t_log, expr_n_log, equal_var=False)
        pb_corr, p_val_pb = stats.pointbiserialr(target_labels, expr_all_log)

        all_genes_stats.append({
            'Gene': gene_name,
            'Mean_Normal': np.mean(expr_n),
            'Mean_Tumor': np.mean(expr_t),
            'Log2_T_Stat': t_stat,
            'Log2_T_PValue': p_val_t,
            'Point_Biserial_Corr': pb_corr
        })

# Convert full scan results to DataFrame and save
df_all_stats = pd.DataFrame(all_genes_stats)
df_all_stats.to_csv(f'{OUTPUT_DIR}00_All_23368_Genes_Statistics.csv', index=False)
print(f"Full gene statistical testing complete! ({len(all_genes_stats)} genes analyzed)")
if mismatched_genes > 0:
    print(f"  Warning: {mismatched_genes} genes were skipped due to order mismatch between files.")

# =========================================================
# [Feature Selection] Identify Top 5 genes most associated with the target
# =========================================================
df_all_stats['Abs_Corr'] = df_all_stats['Point_Biserial_Corr'].abs()
top_5_genes = df_all_stats.nlargest(5, 'Abs_Corr')['Gene'].tolist()

# Since this is a BRCA study, force-include BRCA1 if not already in Top 5
if 'BRCA1' not in top_5_genes:
    top_5_genes.append('BRCA1')
print(f"Automatically discovered top 5 key genes (+BRCA1): {top_5_genes}")

# =========================================================
# [Pass 2] Re-extract raw data for discovered Top 5 genes for in-depth visualization
# =========================================================
print("2. Extracting and merging data for key genes for in-depth visualization...")
data_dict = {gene: [] for gene in top_5_genes}

with gzip.open(TUMOR_FILE, 'rt') as f_t, gzip.open(NORMAL_FILE, 'rt') as f_n:
    f_t.readline()  # Skip header
    f_n.readline()

    for line_t, line_n in zip(f_t, f_n):
        gene_name = line_t.strip().split('\t')[0]
        if gene_name in top_5_genes:
            parts_t = line_t.strip().split('\t')[1:]
            parts_n = line_n.strip().split('\t')[1:]
            data_dict[gene_name] = [float(x) for x in parts_t] + [float(x) for x in parts_n]

# Create DataFrame
df_top = pd.DataFrame(data_dict)
df_top['Sample_Barcode'] = sample_barcodes
df_top['Is_Tumor'] = target_labels

# Derived variable: tissue type (TSS Code)
df_top['TSS_Code'] = df_top['Sample_Barcode'].apply(lambda x: x.split('-')[1] if len(x.split('-')) > 1 else 'Unknown')

# ---------------------------------------------------------
# Interaction analysis
# ---------------------------------------------------------
top_tss_list = df_top['TSS_Code'].value_counts().nlargest(5).index
df_sub = df_top[df_top['TSS_Code'].isin(top_tss_list)].copy()

# 1) Categorical (TSS) x Categorical (Target) -> Mean expression of a specific gene (Barplot)
target_gene_1 = top_5_genes[0]
plt.figure(figsize=(10, 6))
sns.barplot(data=df_sub, x='TSS_Code', y=target_gene_1, hue='Is_Tumor')
plt.title(f'Interaction: {target_gene_1} Expression by TSS Code & Tumor Status')
plt.savefig(f'{OUTPUT_DIR}04_Interaction_TSS_Target_{target_gene_1}.png')
plt.close()

# 2) Categorical (TSS) x Numeric (binned) -> Target ratio (table and visualization)
# rank(method='first') breaks ties, guaranteeing 3 distinct quantile bins
df_sub[f'{target_gene_1}_Group'] = pd.qcut(
    df_sub[target_gene_1].rank(method='first'), q=3, labels=['Low', 'Medium', 'High']
)
inter_cat_num = df_sub.groupby(['TSS_Code', f'{target_gene_1}_Group'])['Is_Tumor'].mean().unstack()
inter_cat_num.to_csv(f'{OUTPUT_DIR}04_Interaction_TSS_{target_gene_1}Group.csv')

plt.figure(figsize=(8, 5))
sns.heatmap(inter_cat_num, annot=True, cmap='coolwarm')
plt.title(f'Tumor Ratio by TSS & {target_gene_1} Group')
plt.savefig(f'{OUTPUT_DIR}04_Interaction_Heatmap.png')
plt.close()

# ---------------------------------------------------------
# Cross-tabulation & chi-square test
# ---------------------------------------------------------
contingency = pd.crosstab(df_top['TSS_Code'], df_top['Is_Tumor'])
chi2, p, dof, ex = stats.chi2_contingency(contingency)
n = contingency.sum().sum()
phi2 = chi2 / n
cramers_v = np.sqrt(phi2 / min(contingency.shape[0]-1, contingency.shape[1]-1))

pd.DataFrame([{'Feature': 'TSS_Code', 'Chi2_Stat': chi2, 'p-value': p, 'Cramers_V': cramers_v}]).to_csv(
    f'{OUTPUT_DIR}05_ChiSquare_Results.csv', index=False
)

# ---------------------------------------------------------
# Distribution visualization
# ---------------------------------------------------------
# 1) & 2) Histograms and boxplots by target for top 3 numeric variables
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, col in enumerate(top_5_genes[:3]):
    log_data = np.log2(df_top[col] + 1)

    # Histogram (with mean/median)
    sns.histplot(log_data, kde=True, ax=axes[0, i], color='teal')
    axes[0, i].axvline(log_data.mean(), color='red', linestyle='--', label='Mean')
    axes[0, i].axvline(np.median(log_data), color='blue', linestyle='-', label='Median')
    axes[0, i].set_title(f'Dist: {col} (Log2)')
    axes[0, i].legend()

    # Boxplot
    df_top['log_'+col] = log_data
    sns.boxplot(x='Is_Tumor', y='log_'+col, data=df_top, ax=axes[1, i], palette='Set2')
    axes[1, i].set_title(f'Boxplot: {col} by Status')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}06_Distributions_and_Boxplots.png')
plt.close()

# 3) Scatter plot of two key features colored by target variable (with transparency)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_top, x='log_'+top_5_genes[0], y='log_'+top_5_genes[1],
                hue='Is_Tumor', alpha=0.3, palette='coolwarm', s=30)
plt.title(f'Scatter Plot: {top_5_genes[0]} vs {top_5_genes[1]} (Log2 TPM)')
plt.savefig(f'{OUTPUT_DIR}06_Scatter_Top2_Genes.png')
plt.close()

print(f"All analyses complete. Results have been saved to the {OUTPUT_DIR} folder.")
