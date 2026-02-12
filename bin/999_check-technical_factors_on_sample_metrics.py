#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2025-12-11'
__version__ = '0.0.1'

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import argparse
from anndata.experimental import read_elem
from h5py import File
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


####################
# Define custom functions
####################
def build_ca(df, label, group_col):
    ca_local = df[[group_col, label]].groupby([group_col, label]).size().reset_index()
    ca_local.columns = [group_col, label, 'count']
    total_cells = df[[group_col]].groupby(group_col).size().reset_index()
    total_cells.columns = [group_col, 'total_cells']
    ca_local = ca_local.merge(total_cells, on=group_col, how='left')
    ca_local['proportion'] = ca_local['count'] / ca_local['total_cells']
    # small pseudocount to avoid log(0)
    eps = 1e-6
    prop_pivot = ca_local.pivot_table(index=group_col, columns=label, values='proportion', aggfunc='sum', fill_value=0)
    prop_pivot = prop_pivot.clip(lower=eps)
    log_prop = np.log(prop_pivot)
    clr_pivot = log_prop.subtract(log_prop.mean(axis=1), axis=0)
    clr_long = clr_pivot.reset_index().melt(id_vars=group_col, var_name=label, value_name='proportion_clr')
    ca_local = ca_local.merge(clr_long, on=[group_col, label], how='left')
    return ca_local

def _is_categorical(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    )

def _r2_for_predictor(y, x, categorical=False):
    mask = np.isfinite(y) & pd.notna(x)
    if mask.sum() < 3:
        return np.nan
    y = y[mask]
    x = x[mask]
    if categorical:
        # one-way ANOVA R^2 (eta^2)
        groups = [y[x == lvl] for lvl in pd.unique(x) if np.sum(x == lvl) > 0]
        if len(groups) < 2:
            return np.nan
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_between = np.sum([len(g) * (np.mean(g) - np.mean(y)) ** 2 for g in groups])
        return ss_between / ss_total if ss_total > 0 else np.nan
    # numeric: R^2 via correlation
    if np.std(x) == 0:
        return np.nan
    r = np.corrcoef(y, x)[0, 1]
    return r ** 2

def _weighted_r2(pc_meta_df, predictor, evr):
    x = pc_meta_df[predictor]
    categorical = _is_categorical(x)
    r2s = []
    for pc in pcs:
        y = pc_meta_df[pc].to_numpy(dtype=float)
        r2s.append(_r2_for_predictor(y, x, categorical=categorical))
    r2s = np.array(r2s, dtype=float)
    valid = np.isfinite(r2s)
    if valid.sum() == 0:
        return np.nan
    w = evr[:len(r2s)][valid]
    w = w / w.sum()
    return np.sum(r2s[valid] * w)

def permutation_pvalue(pc_meta_df, predictor, evr, n_perm=1000, seed=1):
    rng = np.random.default_rng(seed)
    obs_stat = _weighted_r2(pc_meta_df, predictor, evr)
    if not np.isfinite(obs_stat):
        return np.nan, np.nan
    perm_stats = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = pc_meta_df.copy()
        shuffled[predictor] = rng.permutation(shuffled[predictor].to_numpy())
        perm_stats[i] = _weighted_r2(shuffled, predictor, evr)
    p = (np.sum(perm_stats >= obs_stat) + 1) / (n_perm + 1)
    return obs_stat, p


####################
# Options
####################
h5ad="/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/bradley_analysis/atlassing_IBDR/results/IBDRbatch1-8-conf_gt0pt5-immune-baseline-nobadsamps/objects/adata_PCAd_batched_umap.h5ad"
outdir="results/IBDRbatch1-8-conf_gt0pt5-immune-baseline-nobadsamps/figures/check_QC"
nonmerged_wetlab_meta="/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/bradley_analysis/IBDR_prep/processed_data/wetlab_metadata/2026-01-26-WETLAB-IBD-R_Sample_Record.csv"

####################
# Import the .obs from the post-QC h5ad
h5ad="/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/bradley_analysis/atlassing_IBDR/results/IBDRbatch1-8-conf_gt0pt5-immune-baseline-nobadsamps/objects/adata_PCAd_batched_umap.h5ad"
f2 = File(h5ad, 'r')
obs = read_elem(f2['obs'])

##################
# Add some derived covariates to .obs
##################
# Annotate .obs with the number of samples in each pool (make sure this comes from the wetlab metadata, not the .obs itself, as this is filtered for perfect matches)
wetlab_meta = pd.read_csv(nonmerged_wetlab_meta)
wetlab_meta = wetlab_meta[~(wetlab_meta['Sequencing_ID'] == "Not_sequenced")]
wetlab_meta = wetlab_meta[~(wetlab_meta['Sequencing_ID'].isna())]
wetlab_meta['pool_participant'] = wetlab_meta['Sequencing_ID'].astype(str) + "_" + wetlab_meta['Participant_Study_ID'].astype(str)
wetlab_meta = wetlab_meta.rename(columns={'Sequencing_ID': 'convoluted_samplename'})
poolcount = pd.DataFrame(wetlab_meta[['convoluted_samplename', 'pool_participant']].groupby('convoluted_samplename').nunique()).reset_index()
poolcount.columns = ['convoluted_samplename', 'num_samples_in_pool']
obs = obs.merge(poolcount, on='convoluted_samplename', how='left')

plt.figure(figsize=(8,6))
sns.histplot(poolcount['num_samples_in_pool'].dropna(), bins=40, kde=False)
plt.title(f'Distribution of Num Samples In Pool')
plt.xlabel('Number of Samples')
plt.ylabel('Count')
plt.savefig(f'{outdir}/num_samples_in_pool_hist.png')
plt.close()

# Also add the number of cells per sample (and median per pool)
cellcount = pd.DataFrame(obs[['convoluted_samplename', 'pool_participant']].groupby('pool_participant').size()).reset_index()
cellcount.columns = ['pool_participant', 'num_cells_in_sample']
obs = obs.merge(cellcount, on='pool_participant', how='left')

# Also add the proportion of each category per sample
celltype_counts = pd.DataFrame(obs[['pool_participant', 'IBDverse_eqtl:Category']].groupby(['pool_participant', 'IBDverse_eqtl:Category']).size()).reset_index()
celltype_counts.columns = ['pool_participant', 'IBDverse_eqtl:Category', 'count']
total_counts = pd.DataFrame(obs[['pool_participant']].groupby('pool_participant').size()).reset_index()
total_counts.columns = ['pool_participant', 'total_count']
celltype_counts = celltype_counts.merge(total_counts, on='pool_participant', how='left')
celltype_counts['proportion'] = celltype_counts['count'] / celltype_counts['total_count']
celltype_counts_add = celltype_counts[['pool_participant', 'IBDverse_eqtl:Category', 'proportion']].pivot(index='pool_participant', columns='IBDverse_eqtl:Category', values='proportion').reset_index()
obs = obs.merge(celltype_counts_add, on='pool_participant', how='left')


# Manually correct some errors
obs['meta-Date_of_receipt_at_Laboratory_.yyyy.mm.dd.'] = obs['meta-Date_of_receipt_at_Laboratory_.yyyy.mm.dd.'].astype('string')
mask = (obs['convoluted_samplename'] == "IBD-RESPONSE13764153") & (obs['participant_id'] == "CAM0046") # Typo of the month, updated in newest version of the metadata
obs.loc[mask, 'meta-Date_of_receipt_at_Laboratory_.yyyy.mm.dd.'] = "2023-04-13"
mask = (obs['convoluted_samplename'] == "IBD-RESPONSE14552622") & (obs['participant_id'] == "OXF0006") # Typo of the month, updated in newest version of the metadata
obs.loc[mask, 'meta-Date_of_receipt_at_Laboratory_.yyyy.mm.dd.'] = "2023-11-08"
mask = (obs['convoluted_samplename'] == "IBD-RESPONSE15443671")  # Typo of the days, so was received before collected
obs.loc[mask, 'meta-Date_of_receipt_at_Laboratory_.yyyy.mm.dd.'] = "2025-01-16"

# And the time between dispatch and delivery (in days)
date_cols = ['meta-Date_of_receipt_at_Laboratory_.yyyy.mm.dd.', 'meta-Date_of_shipment', 'meta-Sample_Collection']
for col in date_cols:
    obs[col] = pd.to_datetime(obs[col].astype('string'), errors='coerce')

# Extract the info per sample to plot
obs['collection_to_wetlab'] = (obs['meta-Date_of_receipt_at_Laboratory_.yyyy.mm.dd.'] - obs['meta-Sample_Collection']).dt.days
obs['shipment_to_wetlab'] = (obs['meta-Date_of_receipt_at_Laboratory_.yyyy.mm.dd.'] - obs['meta-Date_of_shipment']).dt.days
test_collection_to_wetlab = obs[['convoluted_samplename', 'participant_id', 'meta-Date_of_shipment', 'meta-date_blood_sent_sanger', 'meta-Sample_Collection', 'meta-blood_date_collected', 'meta-Sender', 'meta-Date_of_receipt_at_Laboratory_.yyyy.mm.dd.', 'collection_to_wetlab', 'shipment_to_wetlab']].drop_duplicates().reset_index(drop=True)

# Distribution of collection_to_wetlab (days)
plotcols = ['shipment_to_wetlab', 'collection_to_wetlab']
os.makedirs(outdir, exist_ok=True)
for c in plotcols:
    plt.figure(figsize=(8,6))
    sns.histplot(test_collection_to_wetlab[c].dropna(), bins=40, kde=False)
    plt.title(f'Distribution of {c.replace("_", " ").title()} (days)')
    plt.xlabel('Days')
    plt.ylabel('Count')
    plt.savefig(f'{outdir}/{c}_days_hist.png')
    plt.close()

# Recalculate QC metrics PER POOL, not sample
qc_metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_gene_group__mito_transcript', 'num_cells_in_sample', 'Myeloid', 'T']
for metric in qc_metrics:
    if metric + '-pool_median' not in obs.columns:
        print(f"... calculating {metric} pool median")
        temp = obs[[metric, 'convoluted_samplename']].groupby('convoluted_samplename').median().reset_index()
        temp.columns = ['convoluted_samplename', metric + '-pool_median']
        obs = obs.merge(temp, on='convoluted_samplename', how='left')

##################
# Plot
##################
# Plot QC metrices per sample, dividing samples by the number of samples in each pool
obs['num_samples_in_pool'] = obs['num_samples_in_pool'].astype(str)
pool_numbers = np.sort(np.unique(obs['num_samples_in_pool']))
for metric in qc_metrics:
    # Plot distributions
    plt.figure(figsize=(8,6))
    for pool_num in pool_numbers:
        subset = obs[obs['num_samples_in_pool'] == pool_num]
        sns.kdeplot(subset[metric], label=f'Pool Size: {pool_num}', fill=True, alpha=0.5)
    plt.legend()
    plt.title(f'{metric} by Number of Samples in Pool')
    plt.xlabel(metric)
    plt.ylabel('Density')
    plt.savefig(f'{outdir}/qc_{metric}_dist.png')
    plt.close() 
    # Boxplots
    plt.figure(figsize=(8,6))
    sns.boxplot(x='num_samples_in_pool', y=metric, data=obs)
    plt.title(f'{metric} by Number of Samples in Pool')
    plt.xlabel('Number of Samples in Pool')
    plt.ylabel(metric)
    plt.savefig(f'{outdir}/qc_{metric}_boxplot.png')
    plt.close()
    
##################
# Look more strictly at cell-type abundances
# Is the abundance of any cell-type associated with pool size, or the time taken to arrive at the wetlab?
##################
labels = ['Celltypist:IBDverse_eqtl:predicted_labels', 'IBDverse_eqtl:Category']

# Containers to collect per-label regression DataFrames for programmatic use
ca_dict_pool = {}
ca_dict_sample = {}
results_list = []
results_dict = {}

for label in labels:
    celltypes = obs[label].unique()
    obs[['num_samples_in_pool']] = obs[['num_samples_in_pool']].astype(int)
    # Pool-level (convoluted_samplename)
    ca_pool = build_ca(obs, label = label, group_col='convoluted_samplename')
    num_samples = obs[['convoluted_samplename', 'num_samples_in_pool']].drop_duplicates()
    ca_pool = ca_pool.merge(num_samples, on='convoluted_samplename', how='left')
    coll_to_wetlab_pool = obs[['convoluted_samplename', 'collection_to_wetlab', 'shipment_to_wetlab']].groupby('convoluted_samplename').median().reset_index()
    ca_pool = ca_pool.merge(coll_to_wetlab_pool, on='convoluted_samplename', how='left')
    ca_pool['num_samples_in_pool'] = ca_pool['num_samples_in_pool'].astype(float)
    ca_pool['collection_to_wetlab'] = ca_pool['collection_to_wetlab'].astype(float)
    ca_pool['shipment_to_wetlab'] = ca_pool['shipment_to_wetlab'].astype(float)
    ca_dict_pool[label] = ca_pool
    # Sample-level (pool_participant)
    ca_sample = build_ca(obs, label = label, group_col='pool_participant')
    sample_meta = obs[['pool_participant', 'collection_to_wetlab', 'shipment_to_wetlab']].groupby('pool_participant').median().reset_index()
    ca_sample = ca_sample.merge(sample_meta, on='pool_participant', how='left')
    ca_sample['collection_to_wetlab'] = ca_sample['collection_to_wetlab'].astype(float)
    ca_sample['shipment_to_wetlab'] = ca_sample['shipment_to_wetlab'].astype(float)
    ca_dict_sample[label] = ca_sample
    # Run linear regression of CLR ~ num_samples_in_pool / collection_to_wetlab / shipment_to_wetlab for each cell-type label
    os.makedirs(outdir, exist_ok=True)
    results = []
    celltypelabels = obs[label].unique()
    for celltypelabel in celltypelabels:
        for predictor in ['num_samples_in_pool', 'collection_to_wetlab', 'shipment_to_wetlab']:
            ca_use = ca_pool if predictor == 'num_samples_in_pool' else ca_sample
            subset = ca_use[ca_use[label] == celltypelabel]
            y = subset['proportion_clr'].values.astype(float)
            x = subset[predictor].values.astype(float)
            # require at least 3 observations with finite values
            finite_mask = np.isfinite(x) & np.isfinite(y)
            if finite_mask.sum() < 3:
                results.append({'celltypelabel': celltypelabel, 'predictor': predictor, 'n': int(finite_mask.sum()), 'slope': np.nan, 'intercept': np.nan, 'r_value': np.nan, 'p_value': np.nan, 'std_err': np.nan})
                continue
            try:
                lr = stats.linregress(x[finite_mask], y[finite_mask])
                results.append({'celltypelabel': celltypelabel, 'predictor': predictor, 'n': int(finite_mask.sum()), 'slope': lr.slope, 'intercept': lr.intercept, 'r_value': lr.rvalue, 'p_value': lr.pvalue, 'std_err': lr.stderr})
            except Exception as e:
                results.append({'celltypelabel': celltypelabel, 'predictor': predictor, 'n': int(finite_mask.sum()), 'slope': np.nan, 'intercept': np.nan, 'r_value': np.nan, 'p_value': np.nan, 'std_err': np.nan})
    results_df = pd.DataFrame(results).sort_values('p_value')
    # Benjamini-Hochberg FDR correction for the p-values (adds `p_value_adj`)
    pvals = results_df['p_value'].to_numpy()
    mask = np.isfinite(pvals)
    # default filler for non-finite p-values
    adj = np.full(pvals.shape, np.nan)
    if mask.sum() > 0:
        p_for_adj = pvals[mask]
        # BH procedure
        order = np.argsort(p_for_adj)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(p_for_adj) + 1)
        q = p_for_adj * float(len(p_for_adj)) / ranks
        q = np.minimum.accumulate(q[::-1])[::-1]
        q[q > 1] = 1.0
        adj[mask] = q
    results_df['p_value_adj'] = adj
    results_df.to_csv(f'{outdir}/{label}_clr_vs_technical_factors_regression_results.csv', index=False)
    # collect into list/dict while keeping CSVs
    results_list.append(results_df)
    results_dict[label] = results_df


# Plot the significant cell types
for label in labels:
    results_df = results_dict[label]
    sig_results = results_df[results_df['p_value'] < 0.05]
    for _, row in sig_results.iterrows():
        celltypelabel = row['celltypelabel']
        predictor = row['predictor']
        ca_df = ca_dict_pool[label] if predictor == 'num_samples_in_pool' else ca_dict_sample[label]
        subset = ca_df[ca_df[label] == celltypelabel]
        # skip empty subsets
        if subset.shape[0] == 0:
            continue
        plt.figure(figsize=(8,6))
        # prepare category order (string) so violin/strip align
        uniq = sorted(subset[predictor].dropna().unique())
        order = [str(x) for x in uniq]
        # violin plot (no inner points)
        sns.violinplot(x=subset[predictor].astype(str), y='proportion_clr', data=subset, order=order, inner=None, color='lightgray')
        # jittered points on top
        sns.stripplot(x=subset[predictor].astype(str), y='proportion_clr', data=subset, order=order, jitter=0.25, size=4, color='black', alpha=0.6)
        # Plot regression line mapped to category positions
        x_pos = np.arange(len(uniq))
        y_pred = [row['intercept'] + row['slope'] * v for v in uniq]
        plt.plot(x_pos, y_pred, '--', color='red')
        # Add median and IQR as horizontal lines per pool size
        ax = plt.gca()
        for i, pool_size in enumerate(uniq):
            pool_data = subset[subset[predictor] == pool_size]['proportion_clr']
            med = pool_data.median()
            q25 = pool_data.quantile(0.25)
            q75 = pool_data.quantile(0.75)
            # Draw median as solid line
            ax.hlines(med, i - 0.4, i + 0.4, colors='blue', linewidth=2, linestyle='-', label='Median' if i == 0 else '')
            # Draw IQR bounds as dashed lines
            ax.hlines(q25, i - 0.4, i + 0.4, colors='green', linewidth=1.5, linestyle='--', label='IQR' if i == 0 else '')
            ax.hlines(q75, i - 0.4, i + 0.4, colors='green', linewidth=1.5, linestyle='--')
        plt.xlabel(predictor.replace("_", " "))
        plt.title(f'CLR of {celltypelabel} vs {predictor.replace("_", " ")} ({label})\n p={row["p_value"]:.3e}, slope={row["slope"]:.3f}')
        plt.ylabel('CLR Proportion')
        celltypelabel = str(celltypelabel).replace('/', '.')
        plt.savefig(f'{outdir}/{label}_{celltypelabel}_clr_vs_{predictor}.png')
        plt.close() 

# Manually inspect results
results_all = pd.concat(results_list, ignore_index=True)
results_all[results_all['p_value_adj'] < 0.05].sort_values('p_value').shape[0] # 15


##################
# Compute a PCA based on the cell-type proportions per sample, and compute the variance explained by each of the technical factors.
##################
label_for_pca = 'Celltypist:IBDverse_eqtl:predicted_labels'
group_col = 'pool_participant'
ca = build_ca(obs, label=label_for_pca, group_col=group_col)

# Build wide CLR matrix for PCA
clr_wide = ca.pivot_table(index=group_col, columns=label_for_pca, values='proportion_clr', aggfunc='mean', fill_value=0)
X = clr_wide.to_numpy()
X_centered = X - X.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
scores = U * S
explained_variance = (S ** 2) / (X.shape[0] - 1)
explained_variance_ratio = explained_variance / explained_variance.sum()

pcs = [f'PC{i+1}' for i in range(scores.shape[1])]
pc_scores = pd.DataFrame(scores, index=clr_wide.index, columns=pcs).reset_index()

# Merge with metadata (per sample)
meta_cols = [
    'collection_to_wetlab',
    'shipment_to_wetlab',
    'num_samples_in_pool',
    'meta-Sex',
    'meta-BMI',
    'meta-Months_since_Dx',
    'meta-Months_since_Symptoms',
    'meta-Restricted_diet',
    'meta-Smoking',
    'meta-Baseline_Previous_immunomodulator',
    'meta-Baseline_Number_previous_immunomodulator',
    'meta-Previous_biologic',
    'meta-Biologic_starting',
    'meta-Steroids',
    'meta-Diagnosis'
]
meta_df = obs[[group_col] + meta_cols].drop_duplicates()
pc_meta = pc_scores.merge(meta_df, on=group_col, how='left')

# Compute regression of PCs vs meta columns
results = []
n_perm = 1000
for predictor in meta_cols:
    print(f"... testing {predictor}")
    obs_stat, p_val = permutation_pvalue(pc_meta, predictor, explained_variance_ratio, n_perm=n_perm, seed=1)
    results.append({
        'predictor': predictor,
        'weighted_r2': obs_stat,
        'p_value_perm': p_val,
        'n_perm': n_perm
    })

results_df = pd.DataFrame(results).sort_values('p_value_perm')
if results_df.shape[0] > 0:
    m = results_df.shape[0]
    results_df['p_value_perm_bonf'] = np.minimum(results_df['p_value_perm'] * m, 1.0)

results_df.to_csv(f'{outdir}/cell_abundance_pca_variance_explained_by_technical_factors.csv', index=False)

# Plot the skee plot for the PC factors
plt.figure(figsize=(8,6))
pc_idx = np.arange(1, len(explained_variance_ratio) + 1)
plt.plot(pc_idx, explained_variance_ratio, marker='o')
# Knee detection: max distance to line between first and last points
if len(explained_variance_ratio) >= 3:
    x1, y1 = pc_idx[0], explained_variance_ratio[0]
    x2, y2 = pc_idx[-1], explained_variance_ratio[-1]
    denom = np.hypot(y2 - y1, x2 - x1)
    if denom > 0:
        distances = np.abs((y2 - y1) * pc_idx - (x2 - x1) * explained_variance_ratio + x2 * y1 - y2 * x1) / denom
        knee_idx = int(np.argmax(distances))
        knee_pc = pc_idx[knee_idx]
        plt.axvline(knee_pc, color='red', linestyle='--', label=f'Knee = PC{knee_pc}')
        plt.legend()

plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.tight_layout()
plt.savefig(f'{outdir}/cell_abundance_pca_scree_plot.png')
plt.close()


# Plot the PCA, colouring the points by any value from results_df which has p_value_perm < 0.05
sig_predictors = results_df.loc[results_df['p_value_perm'] < 0.05, 'predictor'].tolist()
if len(sig_predictors) == 0:
    sig_predictors = []

for predictor in sig_predictors:
    plt.figure(figsize=(8,6))
    x = pc_meta['PC1']
    y = pc_meta['PC2']
    values = pc_meta[predictor]
    if _is_categorical(values):
        # categorical coloring
        sns.scatterplot(x=x, y=y, hue=values, palette='tab10', s=30, alpha=0.8)
        plt.legend(title=predictor, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
    else:
        # continuous coloring
        plt.scatter(x, y, c=values, cmap='viridis', s=30, alpha=0.8)
        plt.colorbar(label=predictor)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'PCA (PC1 vs PC2) colored by {predictor}')
    plt.tight_layout()
    safe_pred = str(predictor).replace('/', '.').replace(' ', '_')
    plt.savefig(f'{outdir}/pca_pc1_pc2_colored_by_{safe_pred}.png')
    plt.close()

# Heatmap of PC vs predictor associations (R2 with p<0.05 stars)
pcs_use = pcs[:knee_idx+1]
heat_r2 = pd.DataFrame(index=meta_cols, columns=pcs_use, dtype=float)
heat_p = pd.DataFrame(index=meta_cols, columns=pcs_use, dtype=float)

for predictor in meta_cols:
    x = pc_meta[predictor]
    categorical = _is_categorical(x)
    for pc in pcs_use:
        y = pc_meta[pc].to_numpy(dtype=float)
        mask = np.isfinite(y) & pd.notna(x)
        if mask.sum() < 3:
            heat_r2.loc[predictor, pc] = np.nan
            heat_p.loc[predictor, pc] = np.nan
            continue
        yv = y[mask]
        xv = x[mask]
        if categorical:
            groups = [yv[xv == lvl] for lvl in pd.unique(xv) if np.sum(xv == lvl) > 0]
            if len(groups) < 2:
                heat_r2.loc[predictor, pc] = np.nan
                heat_p.loc[predictor, pc] = np.nan
                continue
            # one-way ANOVA
            try:
                f_stat, p_val = stats.f_oneway(*groups)
            except Exception:
                p_val = np.nan
            heat_r2.loc[predictor, pc] = _r2_for_predictor(yv, xv, categorical=True)
            heat_p.loc[predictor, pc] = p_val
        else:
            if np.std(xv) == 0:
                heat_r2.loc[predictor, pc] = np.nan
                heat_p.loc[predictor, pc] = np.nan
                continue
            r, p_val = stats.pearsonr(yv, xv.astype(float))
            heat_r2.loc[predictor, pc] = r ** 2
            heat_p.loc[predictor, pc] = p_val

heat_r2_t = heat_r2.T
heat_p_t = heat_p.T
annot = heat_r2_t.copy().astype(object)
for pc in heat_r2_t.index:
    for predictor in heat_r2_t.columns:
        p_val = heat_p_t.loc[pc, predictor]
        annot.loc[pc, predictor] = '*' if (pd.notna(p_val) and p_val < 0.05) else ''

# Keep both as DataFrames for proper alignment in seaborn
annot = annot.astype(str)
heat_r2_plot = heat_r2_t.fillna(0)

plt.figure(figsize=(0.8 * len(meta_cols), 0.25 * len(pcs_use) + 4))
sns.heatmap(
    heat_r2_plot,
    annot=annot,
    fmt='',
    cmap='viridis',
    cbar_kws={'label': 'R2'},
    annot_kws={'color': 'white', 'fontsize': 9, 'fontweight': 'bold'}
)
plt.xlabel('Predictor')
plt.ylabel('Principal Component')
plt.title('PC vs Predictor Associations (R2, * p<0.05)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{outdir}/pc_predictor_r2_heatmap.png')
plt.close()

###############
# Do some more formal association testing - Linear model per cell-type (sample-level CLR)
# y ~ response + num_samples_in_pool + collection_to_wetlab
###############
other_variables = ['num_samples_in_pool', 'collection_to_wetlab']
label_lm = 'Celltypist:IBDverse_eqtl:predicted_labels'
group_col_lm = 'pool_participant'
target_vars = ["meta-Diagnosis", "meta-W14_RESPONSE"]

# Build CLR per sample
ca_sample_lm = build_ca(obs, label=label_lm, group_col=group_col_lm)
sample_meta_lm = obs[[group_col_lm] + target_vars + other_variables].drop_duplicates().reset_index(drop=True)
ca_sample_lm = ca_sample_lm.merge(sample_meta_lm, on=group_col_lm, how='left')

lm_results = []
target = target_vars[0]
target_var_format = target.replace("meta-", "")

celltypes = ca_sample_lm[label_lm].dropna().unique()
for target in target_vars:
    print(f"*** testing {target} ***")
    target_var_format = target.replace("meta-", "")
    for celltypelabel in celltypes:
        print(f"... cell type: ", celltypelabel)
        subset = ca_sample_lm[ca_sample_lm[label_lm] == celltypelabel].copy()
        subset = subset[['proportion_clr'] + target_vars + other_variables].dropna()
        subset = subset.rename(columns={target: target_var_format})
        # require at least 3 observations and >=2 response levels
        if subset.shape[0] < 3 or subset[target_var_format].nunique() < 2:
            lm_results.append({
                'celltypelabel': celltypelabel,
                'n': int(subset.shape[0]),
                'p_response': np.nan,
                'coef_num_samples_in_pool': np.nan,
                'p_num_samples_in_pool': np.nan,
                'coef_collection_to_wetlab': np.nan,
                'p_collection_to_wetlab': np.nan,
                'r2': np.nan
            })
            continue
        try:
            if subset[target_var_format].nunique() == 2:
                subset['_target_bin'] = pd.Categorical(subset[target_var_format]).codes
                formula = "_target_bin ~ proportion_clr + num_samples_in_pool + collection_to_wetlab"
                model = smf.logit(formula, data=subset).fit(disp=False)
                lm_results.append({
                    'celltypelabel': celltypelabel,
                    'target': target,
                    'model': formula,
                    'n': int(subset.shape[0]),
                    'coef_target': model.params.get('proportion_clr', np.nan),
                    'p_target': model.pvalues.get('proportion_clr', np.nan),
                    'r2': model.prsquared
                })
            else:
                formula = f"proportion_clr ~ C({response_var_format}) + num_samples_in_pool + collection_to_wetlab"
                model = smf.ols(formula, data=subset).fit()
                anova_res = anova_lm(model, typ=2)
                p_response = anova_res.loc[f"C({response_var_format})", 'PR(>F)'] if f"C({response_var_format})" in anova_res.index else np.nan
                lm_results.append({
                    'celltypelabel': celltypelabel,
                    'target': target,
                    'model': formula,
                    'n': int(subset.shape[0]),
                    'coef_target': model.params.get('proportion_clr', np.nan),
                    'p_target': model.pvalues.get('proportion_clr', np.nan),
                    'r2': model.prsquared
                })
        except Exception:
            lm_results.append({
                'celltypelabel': celltypelabel,
                'n': int(subset.shape[0]),
                'p_response': np.nan,
                'coef_num_samples_in_pool': np.nan,
                'p_num_samples_in_pool': np.nan,
                'coef_collection_to_wetlab': np.nan,
                'p_collection_to_wetlab': np.nan,
                'r2': np.nan
            })

lm_results_df = pd.DataFrame(lm_results)
lm_results_df["p_target_adj"] = lm_results_df.groupby("target")["p_target"].transform(lambda p: sm.stats.multipletests(p, method="fdr_bh")[1])

lm_results_df.to_csv(
    f'{outdir}/celltype_clr_linear_model_results.csv',
    index=False
)