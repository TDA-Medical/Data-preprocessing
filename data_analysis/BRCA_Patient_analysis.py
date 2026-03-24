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
OUTPUT_DIR = './output_brca_patients/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

TUMOR_FILE = 'GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz'
NORMAL_FILE = 'GSE62944_RAW/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt.gz'

# TCGA-BRCA project TSS code list
BRCA_TSS = {'A1','A2','A7','A8','AC','AN','AO','AQ','AR','B6','BH',
            'C8','D8','E2','E9','EW','GM','GI','HN','LD','LL','MS',
            'OL','PE','PL','S3','UL','UU','WT','XX','Z7'}

def is_brca_sample(barcode):
    parts = barcode.split('-')
    return len(parts) > 1 and parts[1] in BRCA_TSS

# ---------------------------------------------------------
# 1. Extract BRCA sample indices from header
# ---------------------------------------------------------
print("1. Identifying BRCA patient samples...")

with gzip.open(TUMOR_FILE, 'rt') as f:
    header_t = f.readline().strip().split('\t')
with gzip.open(NORMAL_FILE, 'rt') as f:
    header_n = f.readline().strip().split('\t')

samples_t = header_t if header_t[0].startswith('TCGA') else header_t[1:]
samples_n = header_n if header_n[0].startswith('TCGA') else header_n[1:]

brca_tumor_idx = [i for i, s in enumerate(samples_t) if is_brca_sample(s)]
brca_normal_idx = [i for i, s in enumerate(samples_n) if is_brca_sample(s)]

brca_tumor_barcodes = [samples_t[i] for i in brca_tumor_idx]
brca_normal_barcodes = [samples_n[i] for i in brca_normal_idx]

n_tumor = len(brca_tumor_idx)
n_normal = len(brca_normal_idx)
n_total = n_tumor + n_normal

print(f"  BRCA tumor samples: {n_tumor}")
print(f"  BRCA normal samples: {n_normal}")
print(f"  BRCA total samples: {n_total}")

all_brca_barcodes = brca_tumor_barcodes + brca_normal_barcodes
brca_labels = np.array([1]*n_tumor + [0]*n_normal)

# =========================================================
# [Pass 1] Full scan of all 23,368 genes — extract BRCA patients only for statistics
# =========================================================
print("\n2. Starting full gene scan for BRCA patients...")

all_genes_stats = []

with gzip.open(TUMOR_FILE, 'rt') as f_t, gzip.open(NORMAL_FILE, 'rt') as f_n:
    f_t.readline()
    f_n.readline()

    for line_t, line_n in zip(f_t, f_n):
        parts_t = line_t.strip().split('\t')
        parts_n = line_n.strip().split('\t')

        gene_t = parts_t[0]
        gene_n = parts_n[0]
        if gene_t != gene_n:
            continue

        gene_name = gene_t
        vals_t = parts_t[1:]
        vals_n = parts_n[1:]

        # Extract BRCA patients only
        expr_tumor = np.array([float(vals_t[i]) for i in brca_tumor_idx], dtype=np.float32)
        expr_normal = np.array([float(vals_n[i]) for i in brca_normal_idx], dtype=np.float32)

        expr_t_log = np.log2(expr_tumor + 1)
        expr_n_log = np.log2(expr_normal + 1)
        expr_all_log = np.concatenate([expr_t_log, expr_n_log])

        t_stat, p_val = stats.ttest_ind(expr_t_log, expr_n_log, equal_var=False)
        pb_corr, p_pb = stats.pointbiserialr(brca_labels, expr_all_log)

        log2fc = np.mean(expr_t_log) - np.mean(expr_n_log)

        all_genes_stats.append({
            'Gene': gene_name,
            'Mean_Normal': np.mean(expr_normal),
            'Mean_Tumor': np.mean(expr_tumor),
            'Log2FC': log2fc,
            'T_Stat': t_stat,
            'P_Value': p_val,
            'PB_Corr': pb_corr,
            'Abs_PB_Corr': abs(pb_corr),
        })

df_all = pd.DataFrame(all_genes_stats)
df_all.to_csv(f'{OUTPUT_DIR}00_BRCA_All_23368_Genes_Statistics.csv', index=False)
print(f"  Full scan complete: {len(df_all)} genes")

# =========================================================
# Data overview
# =========================================================
overview = pd.DataFrame({
    'Metric': ['BRCA Tumor Samples', 'BRCA Normal Samples', 'Total BRCA Samples',
               'Total Genes Analyzed', 'Target Ratio (Tumor:Normal)'],
    'Value': [n_tumor, n_normal, n_total, len(df_all),
              f'{n_tumor/n_total*100:.1f}% : {n_normal/n_total*100:.1f}%']
})
overview.to_csv(f'{OUTPUT_DIR}01_BRCA_Data_Overview.csv', index=False)

# =========================================================
# Feature Selection: Top genes + force-include BRCA1/BRCA2
# =========================================================
top_genes = df_all.nlargest(10, 'Abs_PB_Corr')['Gene'].tolist()
for g in ['BRCA1', 'BRCA2']:
    if g not in top_genes:
        top_genes.append(g)

print(f"\n  Key genes (Top 10 + BRCA1/2): {top_genes}")

# =========================================================
# [Pass 2] Extract data for key genes
# =========================================================
print("\n3. Extracting key gene data for in-depth analysis...")
target_set = set(top_genes)
# Also include core cancer-related genes
known_cancer_genes = ['TP53', 'PTEN', 'EGFR', 'MYC', 'PIK3CA', 'CDH1', 'GATA3', 'MAP3K1', 'ESR1', 'PGR']
for g in known_cancer_genes:
    target_set.add(g)

data_dict = {gene: [] for gene in target_set}

with gzip.open(TUMOR_FILE, 'rt') as f_t, gzip.open(NORMAL_FILE, 'rt') as f_n:
    f_t.readline()
    f_n.readline()

    for line_t, line_n in zip(f_t, f_n):
        gene_name = line_t.strip().split('\t')[0]
        if gene_name in target_set:
            vals_t = line_t.strip().split('\t')[1:]
            vals_n = line_n.strip().split('\t')[1:]
            brca_vals = [float(vals_t[i]) for i in brca_tumor_idx] + \
                        [float(vals_n[i]) for i in brca_normal_idx]
            data_dict[gene_name] = brca_vals

# Remove genes with no data
data_dict = {k: v for k, v in data_dict.items() if len(v) == n_total}

df = pd.DataFrame(data_dict)
df['Sample_Barcode'] = all_brca_barcodes
df['Is_Tumor'] = brca_labels
df['TSS_Code'] = df['Sample_Barcode'].apply(lambda x: x.split('-')[1])

# ---------------------------------------------------------
# 2. Descriptive statistics
# ---------------------------------------------------------
print("  Computing descriptive statistics...")

# Descriptive statistics for BRCA1/BRCA2 + known cancer genes
stat_genes = [g for g in ['BRCA1','BRCA2'] + known_cancer_genes if g in df.columns]
num_stats = df[stat_genes].describe(percentiles=[.01, .25, .5, .75, .95, .99]).T
num_stats['Skewness'] = df[stat_genes].skew()
num_stats['Kurtosis'] = df[stat_genes].kurt()
num_stats.to_csv(f'{OUTPUT_DIR}02_BRCA_Descriptive_Stats.csv')

# T-test for known genes
ttest_results = []
for col in stat_genes:
    g0 = np.log2(df[df['Is_Tumor'] == 0][col] + 1).dropna()
    g1 = np.log2(df[df['Is_Tumor'] == 1][col] + 1).dropna()
    t, p = stats.ttest_ind(g1, g0, equal_var=False)
    ttest_results.append({
        'Gene': col,
        'Mean_TPM_Normal': df[df['Is_Tumor']==0][col].mean(),
        'Mean_TPM_Tumor': df[df['Is_Tumor']==1][col].mean(),
        'Fold_Change': df[df['Is_Tumor']==1][col].mean() / max(df[df['Is_Tumor']==0][col].mean(), 0.001),
        'T_Stat_Log2': t,
        'P_Value': p,
    })
pd.DataFrame(ttest_results).to_csv(f'{OUTPUT_DIR}02_BRCA_TTest_Results.csv', index=False)

# ---------------------------------------------------------
# 3. Correlation analysis
# ---------------------------------------------------------
print("  Running correlation analysis...")

corr_genes = [g for g in ['BRCA1','BRCA2','TP53','PTEN','EGFR','MYC','ESR1','PGR','PIK3CA','CDH1','GATA3'] if g in df.columns]

# Pearson & Spearman on log2
df_log = np.log2(df[corr_genes] + 1)
corr_p = df_log.corr(method='pearson')
corr_s = df_log.corr(method='spearman')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(corr_p, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('Pearson Correlation (BRCA Patients, Log2 TPM)')
sns.heatmap(corr_s, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title('Spearman Correlation (BRCA Patients, Log2 TPM)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}03_BRCA_Correlation_Heatmaps.png', dpi=150)
plt.close()

# Point-Biserial
pb_results = []
for col in corr_genes:
    r, p = stats.pointbiserialr(df['Is_Tumor'], np.log2(df[col] + 1))
    pb_results.append({'Gene': col, 'PB_Corr': r, 'p-value': p})
pb_df = pd.DataFrame(pb_results).sort_values('PB_Corr', key=abs, ascending=False)
pb_df.to_csv(f'{OUTPUT_DIR}03_BRCA_Point_Biserial.csv', index=False)

# ---------------------------------------------------------
# 4. Interaction analysis: TSS x Tumor -> BRCA1 expression
# ---------------------------------------------------------
print("  Running interaction analysis...")

top_tss = df['TSS_Code'].value_counts().nlargest(8).index
df_sub = df[df['TSS_Code'].isin(top_tss)].copy()

# BRCA1 by TSS x Tumor
plt.figure(figsize=(12, 6))
sns.barplot(data=df_sub, x='TSS_Code', y='BRCA1', hue='Is_Tumor')
plt.title('BRCA1 Expression by TSS Code & Tumor Status (BRCA Patients Only)')
plt.ylabel('BRCA1 TPM')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_Interaction_TSS_BRCA1.png', dpi=150)
plt.close()

# BRCA2 by TSS x Tumor
plt.figure(figsize=(12, 6))
sns.barplot(data=df_sub, x='TSS_Code', y='BRCA2', hue='Is_Tumor')
plt.title('BRCA2 Expression by TSS Code & Tumor Status (BRCA Patients Only)')
plt.ylabel('BRCA2 TPM')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_Interaction_TSS_BRCA2.png', dpi=150)
plt.close()

# ESR1 (estrogen receptor) by TSS x Tumor — related to breast cancer subtype
if 'ESR1' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_sub, x='TSS_Code', y='ESR1', hue='Is_Tumor')
    plt.title('ESR1 (Estrogen Receptor) by TSS Code & Tumor Status')
    plt.ylabel('ESR1 TPM')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}04_Interaction_TSS_ESR1.png', dpi=150)
    plt.close()

# BRCA1 Group x Tumor ratio heatmap
df_sub['BRCA1_Group'] = pd.qcut(df_sub['BRCA1'].rank(method='first'), q=3, labels=['Low','Medium','High'])
inter = df_sub.groupby(['TSS_Code','BRCA1_Group'])['Is_Tumor'].mean().unstack()
inter.to_csv(f'{OUTPUT_DIR}04_Interaction_TSS_BRCA1Group.csv')

plt.figure(figsize=(10, 6))
sns.heatmap(inter, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Tumor Ratio by TSS Code & BRCA1 Expression Group')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_Interaction_Heatmap_BRCA1.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# 5. Cross-tabulation & chi-square test
# ---------------------------------------------------------
print("  Running cross-tabulation...")

contingency = pd.crosstab(df['TSS_Code'], df['Is_Tumor'])
chi2, p, dof, _ = stats.chi2_contingency(contingency)
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / n / min(contingency.shape[0]-1, contingency.shape[1]-1))

chi_result = pd.DataFrame([{
    'Feature': 'TSS_Code', 'Chi2': chi2, 'p-value': p, 'dof': dof, 'Cramers_V': cramers_v
}])
chi_result.to_csv(f'{OUTPUT_DIR}05_BRCA_ChiSquare.csv', index=False)

# Tumor/normal distribution by TSS
tss_dist = df.groupby('TSS_Code')['Is_Tumor'].agg(['sum','count'])
tss_dist.columns = ['Tumor_Count', 'Total']
tss_dist['Normal_Count'] = tss_dist['Total'] - tss_dist['Tumor_Count']
tss_dist['Tumor_Ratio'] = (tss_dist['Tumor_Count'] / tss_dist['Total'] * 100).round(1)
tss_dist = tss_dist.sort_values('Total', ascending=False)
tss_dist.to_csv(f'{OUTPUT_DIR}05_BRCA_TSS_Distribution.csv')

# ---------------------------------------------------------
# 6. Distribution visualization
# ---------------------------------------------------------
print("  Generating distribution plots...")

# 6-1. BRCA1, BRCA2, ESR1, PGR histograms
plot_genes = [g for g in ['BRCA1','BRCA2','ESR1','PGR'] if g in df.columns]
n_plots = len(plot_genes)

fig, axes = plt.subplots(2, n_plots, figsize=(5*n_plots, 10))
for i, col in enumerate(plot_genes):
    log_data = np.log2(df[col] + 1)

    sns.histplot(log_data, kde=True, ax=axes[0, i], color='teal', bins=40)
    axes[0, i].axvline(log_data.mean(), color='red', linestyle='--', label='Mean')
    axes[0, i].axvline(np.median(log_data), color='blue', linestyle='-', label='Median')
    axes[0, i].set_title(f'Log2(TPM+1): {col}')
    axes[0, i].legend(fontsize=8)

    df['log_'+col] = log_data
    sns.boxplot(x='Is_Tumor', y='log_'+col, data=df, ax=axes[1, i], palette='Set2')
    axes[1, i].set_title(f'{col} by Tumor Status')
    axes[1, i].set_xlabel('0=Normal, 1=Tumor')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}06_BRCA_Distributions_Boxplots.png', dpi=150)
plt.close()

# 6-2. BRCA1 vs BRCA2 scatter plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='log_BRCA1', y='log_BRCA2',
                hue='Is_Tumor', alpha=0.4, palette='coolwarm', s=40)
plt.title('BRCA1 vs BRCA2 Expression (BRCA Patients, Log2 TPM)')
plt.xlabel('Log2(BRCA1 TPM + 1)')
plt.ylabel('Log2(BRCA2 TPM + 1)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}06_Scatter_BRCA1_vs_BRCA2.png', dpi=150)
plt.close()

# 6-3. ESR1 vs PGR (hormone receptors)
if 'ESR1' in df.columns and 'PGR' in df.columns:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='log_ESR1', y='log_PGR',
                    hue='Is_Tumor', alpha=0.4, palette='coolwarm', s=40)
    plt.title('ESR1 vs PGR (Hormone Receptors, BRCA Patients)')
    plt.xlabel('Log2(ESR1 TPM + 1)')
    plt.ylabel('Log2(PGR TPM + 1)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06_Scatter_ESR1_vs_PGR.png', dpi=150)
    plt.close()

# ---------------------------------------------------------
# 7. Volcano Plot: BRCA patients only
# ---------------------------------------------------------
print("  Generating volcano plot...")

df_all['neg_log10_p'] = -np.log10(df_all['P_Value'].clip(lower=1e-300))

plt.figure(figsize=(14, 8))
# Background
plt.scatter(df_all['Log2FC'], df_all['neg_log10_p'],
            c='lightgray', s=3, alpha=0.3, label='All genes')

# Significant genes
sig = df_all[(df_all['P_Value'] < 0.05) & (df_all['Log2FC'].abs() > 1)]
plt.scatter(sig['Log2FC'], sig['neg_log10_p'],
            c='salmon', s=8, alpha=0.5, label=f'|Log2FC|>1 & p<0.05 (n={len(sig)})')

# Annotate key genes
for gene in ['BRCA1','BRCA2','TP53','PTEN','EGFR','ESR1','PGR','PIK3CA','CDH1','GATA3','MYC']:
    row = df_all[df_all['Gene'] == gene]
    if not row.empty:
        r = row.iloc[0]
        plt.annotate(gene, (r['Log2FC'], r['neg_log10_p']),
                     fontsize=8, fontweight='bold', color='darkred',
                     arrowprops=dict(arrowstyle='->', color='darkred', lw=0.7),
                     xytext=(r['Log2FC'] + 0.2, r['neg_log10_p'] + 8))

plt.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
plt.axvline(-1, color='blue', linestyle=':', alpha=0.3)
plt.axvline(1, color='blue', linestyle=':', alpha=0.3)
plt.xlabel('Log2 Fold Change (Tumor vs Normal)')
plt.ylabel('-log10(P-Value)')
plt.title('Volcano Plot: BRCA Patients Only (Key Cancer Genes Annotated)')
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}07_BRCA_Volcano_Plot.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# 8. Full scan Top 20 genes
# ---------------------------------------------------------
top20 = df_all.nlargest(20, 'Abs_PB_Corr')
top20.to_csv(f'{OUTPUT_DIR}08_BRCA_Top20_Genes.csv', index=False)

top20_plot = top20.copy()
top20_plot['neg_log10_p'] = -np.log10(top20_plot['P_Value'].clip(lower=1e-300))
colors = ['salmon' if fc > 0 else 'steelblue' for fc in top20_plot['Log2FC']]

plt.figure(figsize=(12, 7))
plt.barh(range(len(top20_plot)), top20_plot['Abs_PB_Corr'], color=colors)
plt.yticks(range(len(top20_plot)), top20_plot['Gene'])
plt.xlabel('|Point-Biserial Correlation|')
plt.title('Top 20 Most Discriminating Genes (BRCA Patients)\nRed=Up in Tumor, Blue=Down in Tumor')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}08_BRCA_Top20_BarChart.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# 9. BRCA1/BRCA2 in-depth analysis
# ---------------------------------------------------------
print("  Running BRCA1/BRCA2 in-depth analysis...")

brca_detail = df_all[df_all['Gene'].isin(['BRCA1','BRCA2'])].copy()
brca_detail.to_csv(f'{OUTPUT_DIR}09_BRCA1_BRCA2_Detail.csv', index=False)

# Classify samples by BRCA1 expression level
df['BRCA1_Level'] = pd.qcut(df['BRCA1'].rank(method='first'), q=4,
                             labels=['Q1(Low)','Q2','Q3','Q4(High)'])

# Statistics per quartile
q_stats = df.groupby('BRCA1_Level').agg(
    n=('BRCA1', 'count'),
    tumor_ratio=('Is_Tumor', 'mean'),
    brca1_mean=('BRCA1', 'mean'),
    brca2_mean=('BRCA2', 'mean'),
).round(4)
q_stats['tumor_ratio'] = (q_stats['tumor_ratio'] * 100).round(1)
q_stats.to_csv(f'{OUTPUT_DIR}09_BRCA1_Quartile_Analysis.csv')

# BRCA1 quartile barplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
q_stats['tumor_ratio'].plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title('Tumor Ratio (%) by BRCA1 Quartile')
axes[0].set_ylabel('Tumor Ratio (%)')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

q_stats[['brca1_mean','brca2_mean']].plot(kind='bar', ax=axes[1], color=['#e74c3c','#3498db'])
axes[1].set_title('Mean Expression by BRCA1 Quartile')
axes[1].set_ylabel('Mean TPM')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}09_BRCA1_Quartile_Plot.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# 10. BRCA vs other cancer types comparison (from full dataset)
# ---------------------------------------------------------
print("  Comparing BRCA vs other cancer types...")

# Compare BRCA1/BRCA2 expression: BRCA tumor vs non-BRCA tumor
brca_vs_other = []

with gzip.open(TUMOR_FILE, 'rt') as f:
    header = f.readline().strip().split('\t')
    all_samples = header if header[0].startswith('TCGA') else header[1:]

    non_brca_idx = [i for i, s in enumerate(all_samples) if not is_brca_sample(s)]

    for line in f:
        parts = line.strip().split('\t')
        gene = parts[0]
        if gene in ('BRCA1', 'BRCA2', 'ESR1', 'PGR', 'TP53'):
            vals = parts[1:]
            brca_vals = np.array([float(vals[i]) for i in brca_tumor_idx], dtype=np.float32)
            other_vals = np.array([float(vals[i]) for i in non_brca_idx], dtype=np.float32)

            t, p = stats.ttest_ind(np.log2(brca_vals+1), np.log2(other_vals+1), equal_var=False)
            brca_vs_other.append({
                'Gene': gene,
                'Mean_BRCA_Tumor': np.mean(brca_vals),
                'Mean_Other_Tumor': np.mean(other_vals),
                'Ratio': np.mean(brca_vals) / max(np.mean(other_vals), 0.001),
                'T_Stat': t,
                'P_Value': p,
            })

brca_vs_other_df = pd.DataFrame(brca_vs_other)
brca_vs_other_df.to_csv(f'{OUTPUT_DIR}10_BRCA_vs_OtherCancers.csv', index=False)

# Barplot comparison
plt.figure(figsize=(10, 6))
x = np.arange(len(brca_vs_other_df))
w = 0.35
plt.bar(x - w/2, brca_vs_other_df['Mean_BRCA_Tumor'], w, label='BRCA Tumor', color='coral')
plt.bar(x + w/2, brca_vs_other_df['Mean_Other_Tumor'], w, label='Other Tumors', color='steelblue')
plt.xticks(x, brca_vs_other_df['Gene'])
plt.ylabel('Mean TPM')
plt.title('Gene Expression: BRCA Tumor vs Other Cancer Types')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}10_BRCA_vs_OtherCancers.png', dpi=150)
plt.close()

print(f"\nAll analyses complete. Results have been saved to the {OUTPUT_DIR} folder.")
