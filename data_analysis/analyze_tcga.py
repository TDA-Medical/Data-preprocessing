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
# 0. Setup and data loading (memory-optimized)
# ---------------------------------------------------------
OUTPUT_DIR = './output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("muted")

# Core cancer-related genes for analysis
TARGET_GENES = ['TP53', 'BRCA1', 'PTEN', 'EGFR', 'MYC']
TUMOR_FILE = 'GSE62944_RAW/GSM1536837_06_01_15_TCGA_24.tumor_Rsubread_TPM.txt.gz'
NORMAL_FILE = 'GSE62944_RAW/GSM1697009_06_01_15_TCGA_24.normal_Rsubread_TPM.txt.gz'

def extract_genes_from_gz(file_path, target_genes, is_tumor):
    data_dict = {}
    header = []

    with gzip.open(file_path, 'rt') as f:
        header_line = f.readline().strip().split('\t')
        sample_barcodes = header_line if header_line[0].startswith('TCGA') else header_line[1:]

        for line in f:
            parts = line.strip().split('\t')
            gene_name = parts[0]
            if gene_name in target_genes:
                data_dict[gene_name] = [float(x) for x in parts[1:]]
            if len(data_dict) == len(target_genes):
                break

    df = pd.DataFrame(data_dict, index=sample_barcodes)
    df['Is_Tumor'] = 1 if is_tumor else 0
    return df

print("1. Parsing and merging large RNA-Seq files...")
df_tumor = extract_genes_from_gz(TUMOR_FILE, TARGET_GENES, is_tumor=True)
df_normal = extract_genes_from_gz(NORMAL_FILE, TARGET_GENES, is_tumor=False)

df = pd.concat([df_tumor, df_normal])
df.index.name = 'Sample_Barcode'
df.reset_index(inplace=True)

# Derived features
df['TSS_Code'] = df['Sample_Barcode'].apply(lambda x: x.split('-')[1] if len(x.split('-')) > 1 else 'Unknown')
# Second categorical variable (tissue type: normal vs tumor, temporary variable for other analyses)
df['Sample_Type'] = df['Sample_Barcode'].apply(lambda x: x.split('-')[3][:2] if len(x.split('-')) > 3 else 'Unknown')


num_vars = TARGET_GENES
cat_vars = ['TSS_Code', 'Sample_Type']
target = 'Is_Tumor'

print("2. Starting statistical analysis and visualization...")

# ---------------------------------------------------------
# 1. Data overview
# ---------------------------------------------------------
print(" - Generating data overview...")
data_overview = pd.DataFrame({
    'Metric': ['Total Samples', 'Total Genes Analyzed', 'Missing Values Ratio (%)', 'Target Ratio (Tumor : Normal)'],
    'Value': [
        len(df),
        len(TARGET_GENES),
        round(df.isnull().sum().sum() / (len(df)*len(df.columns)) * 100, 4),
        f"{df[target].mean()*100:.1f}% : {(1-df[target].mean())*100:.1f}%"
    ]
})
data_overview.to_csv(f'{OUTPUT_DIR}01_Data_Overview.csv', index=False)

# ---------------------------------------------------------
# 2. Descriptive statistics
# ---------------------------------------------------------
print(" - Computing descriptive statistics...")
num_stats = df[num_vars].describe(percentiles=[.01, .25, .5, .75, .95, .99]).T
num_stats['Skewness'] = df[num_vars].skew()
num_stats['Kurtosis'] = df[num_vars].kurt()
num_stats.to_csv(f'{OUTPUT_DIR}02_Numeric_Descriptive_Stats.csv')

top_tss = df['TSS_Code'].value_counts().head(20).reset_index()
top_tss.columns = ['TSS_Code', 'Count']
top_tss['Ratio(%)'] = (top_tss['Count'] / len(df)) * 100
top_tss.to_csv(f'{OUTPUT_DIR}02_Categorical_Frequencies.csv', index=False)

t_test_results = []
for col in num_vars:
    group0 = np.log2(df[df[target] == 0][col] + 1).dropna()
    group1 = np.log2(df[df[target] == 1][col] + 1).dropna()
    t_stat, p_val = stats.ttest_ind(group0, group1, equal_var=False)
    t_test_results.append({
        'Gene': col,
        'Mean_TPM_Normal': df[df[target] == 0][col].mean(),
        'Mean_TPM_Tumor': df[df[target] == 1][col].mean(),
        't-statistic (Log2)': t_stat,
        'p-value': p_val
    })
pd.DataFrame(t_test_results).to_csv(f'{OUTPUT_DIR}02_T_Test_Results.csv', index=False)

# ---------------------------------------------------------
# 3. Correlation analysis
# ---------------------------------------------------------
print(" - Running correlation analysis...")
corr_pearson = df[num_vars].corr(method='pearson')
corr_spearman = df[num_vars].corr(method='spearman')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(corr_pearson, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1)
plt.title('Pearson Correlation (Genes)')
plt.subplot(1, 2, 2)
sns.heatmap(corr_spearman, annot=True, cmap='RdBu_r', fmt=".2f", vmin=-1, vmax=1)
plt.title('Spearman Correlation (Genes)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}03_Correlation_Heatmaps.png')
plt.close()

pb_results = []
for col in num_vars:
    pb_corr, pb_p = stats.pointbiserialr(df[target], df[col])
    pb_results.append({'Gene': col, 'Point_Biserial_Corr': pb_corr, 'p-value': pb_p})
pd.DataFrame(pb_results).to_csv(f'{OUTPUT_DIR}03_Point_Biserial_Corr.csv', index=False)

# ---------------------------------------------------------
# 4. Interaction analysis
# ---------------------------------------------------------
print(" - Running interaction analysis...")
top_10_tss = df['TSS_Code'].value_counts().nlargest(10).index
df_top_tss = df[df['TSS_Code'].isin(top_10_tss)].copy()

# 1. [Categorical 1] x [Categorical 2] -> Target ratio (TSS_Code x Sample_Type)
inter_cat_cat = df_top_tss.groupby(['TSS_Code', 'Sample_Type'])[target].mean().unstack()
inter_cat_cat.to_csv(f'{OUTPUT_DIR}04_Interaction_TSS_SampleType.csv')

plt.figure(figsize=(10, 6))
sns.heatmap(inter_cat_cat, annot=True, cmap='YlGnBu')
plt.title('Tumor Ratio by TSS_Code and Sample_Type')
plt.savefig(f'{OUTPUT_DIR}04_Interaction_Cat_Cat.png')
plt.close()


# 2. [Categorical 1] x [Numeric 1 binned] -> Target change (TSS_Code x TP53 Group)
df_top_tss['TP53_Group'] = pd.qcut(df_top_tss['TP53'], q=3, labels=['Low', 'Medium', 'High'])
inter_cat_num = df_top_tss.groupby(['TSS_Code', 'TP53_Group'])[target].mean().unstack()
inter_cat_num.to_csv(f'{OUTPUT_DIR}04_Interaction_TSS_TP53Group.csv')

plt.figure(figsize=(10, 6))
inter_cat_num.plot(kind='bar', stacked=False)
plt.title('Tumor Ratio by TSS_Code and TP53 Expression Group')
plt.ylabel('Tumor Ratio')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_Interaction_Cat_NumGroup.png')
plt.close()

# 3. Additional interaction (TSS_Code x EGFR)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_top_tss, x='TSS_Code', y='EGFR', hue=target, marker='o', ci=None)
plt.title('Interaction: Mean EGFR Expression by TSS Code & Tumor Status')
plt.xticks(rotation=45)
plt.ylabel('Mean EGFR TPM')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_Interaction_TSS_EGFR.png')
plt.close()

# ---------------------------------------------------------
# 5. Cross-tabulation & chi-square test
# ---------------------------------------------------------
print(" - Running cross-tabulation and chi-square test...")
def calc_cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    if min(k - 1, r - 1) == 0: return 0
    return np.sqrt(phi2 / min(k - 1, r - 1))

chi_results_list = []
for col in cat_vars:
    contingency = pd.crosstab(df[col], df[target])
    chi2, p, dof, ex = stats.chi2_contingency(contingency)
    v = calc_cramers_v(contingency)
    chi_results_list.append({
        'Feature': col,
        'Chi2_Stat': chi2,
        'p-value': p,
        'Cramers_V': v
    })

chi_results = pd.DataFrame(chi_results_list)
chi_results.to_csv(f'{OUTPUT_DIR}05_ChiSquare_Results.csv', index=False)

# ---------------------------------------------------------
# 6. Distribution visualization
# ---------------------------------------------------------
print(" - Generating distribution plots...")
# 1) Histograms for numeric variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_genes = ['TP53', 'EGFR', 'MYC']
for i, col in enumerate(plot_genes):
    log_data = np.log2(df[col] + 1)
    sns.histplot(log_data, kde=True, ax=axes[i], color='teal', bins=50)
    axes[i].axvline(log_data.mean(), color='red', linestyle='--', label=f'Mean')
    axes[i].axvline(np.median(log_data), color='blue', linestyle='-', label=f'Median')
    axes[i].set_title(f'Log2(TPM+1) Dist: {col}')
    axes[i].legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}06_Distributions_Histograms_Log.png')
plt.close()

# 2) Boxplots by target group
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(plot_genes):
    df['log_'+col] = np.log2(df[col] + 1)
    sns.boxplot(x=target, y='log_'+col, data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} (Log2 TPM) by Tumor Status')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}06_Boxplots_by_Target.png')
plt.close()

# 3) Scatter plot of two key features colored by target variable (with transparency)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='log_EGFR', y='log_MYC', hue=target, alpha=0.3, palette='coolwarm', s=30)
plt.title('Scatter Plot: EGFR vs MYC (Log2 TPM, colored by Tumor)')
plt.savefig(f'{OUTPUT_DIR}06_Scatter_EGFR_MYC.png')
plt.close()

print(f"Done! All results (CSV, PNG) have been saved to the {OUTPUT_DIR} folder.")
