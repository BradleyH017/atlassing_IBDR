#!/usr/bin/env python

__author__ = 'Bradley Harris'
__date__ = '2025-12-11'
__version__ = '0.0.1'

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import os
import pickle
import argparse
from anndata.experimental import read_elem
from h5py import File
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from plotnine import (
    ggplot, aes, geom_tile, geom_text, scale_fill_gradient,
    theme, theme_bw, element_text, element_blank, labs
)
import sys
import os
sys.path.append("bin")
from qc_check_helpers import (
    load_and_format_obs,
    build_ca, _is_categorical, _r2_for_predictor, _weighted_r2,
    permutation_pvalue, _sig_stars, pseudobulk_by_label, pca_on_expression,
    pca_variance_explained, variance_explained_by_covariates,
)

####################
# Options
####################
h5ad="results_round1/IBDRbatch1-8-conf_gt0pt5-immune-baseline-nobadsamps/objects/adata_PCAd_batched_umap.h5ad"
h5ad_full="results_round1/IBDRbatch1-8-conf_gt0pt5-immune-baseline-nobadsamps/objects/adata_PCAd_batched_umap_all_genes_exprgt5cells.h5ad"
outdir="results_round1/IBDRbatch1-8-conf_gt0pt5-immune-baseline-nobadsamps/figures/check_QC"
nonmerged_wetlab_meta="/lustre/scratch125/humgen/projects_v2/ibdresponse/analysis/bradley_analysis/IBDR_prep/processed_data/wetlab_metadata/2026-01-26-WETLAB-IBD-R_Sample_Record.csv"
repo_dir = "/lustre/scratch127/humgen/projects_v2/sc-eqtl-ibd/analysis/bradley_analysis/IBDverse/IBDVerse-sc-eQTL-code/"
qc_metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_gene_group__mito_transcript', 'num_cells_in_sample', 'Myeloid', 'T']
plot_distributions = False   # QC metric plots, pool-size and transit-time histograms
plot_abundance = True        # Pairwise CLR vs predictor scatter plots and volcano
use_omega2 = True            # True → ω²/adj-R² (penalises for #levels); False → η²/R² (uncorrected)
effect_size_label = 'Weighted ω²' if use_omega2 else 'Weighted η² / R²'
effect_size_col = 'weighted_omega2' if use_omega2 else 'weighted_r2'
palette = pd.read_csv(f"{repo_dir}/data/palette.csv")
color_map = dict(zip(palette['category'], palette['category_color']))
annot_mapping = pd.read_csv(f"{repo_dir}/data/all_IBDverse_annotation_mastersheet.csv")
annot_mapping = annot_mapping.rename(
    columns={
        "leiden": "label_machine",
        "JAMBOREE_ANNOTATION": "label_new",
        "Category": "category"
    }
)
annot_mapping['label_new'] = annot_mapping['label_new'].replace({"_": " "}, regex=True)
level_mapping = {
    '0': 'All Cells',
    '1': 'Major population',
    '2': 'Cell type'
}
tissue_mapping = {
    'ct': 'Cross-site',
    'ti': 'Terminal ileum',
    'r': 'Rectum',
    'blood': 'Blood'
}
annot_mapping = annot_mapping.copy()
annot_mapping['Level'] = 2
cat_col = 'category_new' if 'category_new' in annot_mapping.columns else 'category'
major = annot_mapping[[cat_col]].dropna().drop_duplicates()
major['label_new'] = major[cat_col]
major['label_machine'] = major[cat_col]
major['Level'] = 1
annot_mapping = annot_mapping[major.columns]
annot_mapping = pd.concat([annot_mapping, major], ignore_index=True, sort=False)
annot_mapping = annot_mapping.dropna()
unannotated = {
    'label_machine': 'unannotated',
    'label_new': level_mapping['0'],
    cat_col: level_mapping['0'],
    'Level': 0
}
annot_mapping = pd.concat([annot_mapping, pd.DataFrame([unannotated])], ignore_index=True, sort=False)
tissue_df = pd.DataFrame({'tissue': list(tissue_mapping.keys())})
annot_mapping = annot_mapping.assign(_key=1).merge(tissue_df.assign(_key=1), on='_key').drop(columns=['_key'])
annot_mapping['label_machine'] = annot_mapping['label_machine'].astype(str) + '_' + annot_mapping['tissue'].astype(str)
annot_mapping['tissue'] = annot_mapping['tissue'].map(tissue_mapping)
annot_mapping['annotation_type'] = annot_mapping['Level'].map({0: 'All Cells', 1: 'Major population', 2: 'Cell type'})

meta_cols = [
    'convoluted_samplename',
    'collection_to_wetlab',
    'shipment_to_wetlab',
    'num_samples_in_pool',
    'site'
] + ["meta-" + x for x in [
    "Age", "BMI", "Months_since_Dx", "Months_since_Symptoms",
    "Baseline_PRO2_CD", "Baseline_PRO2_UC", "Calprotectin",
    "Bristol", "CRP", "Number_previous_biologic", "W14_RESPONSE", "W14_REMISSION", "Sex", "Diagnosis",
    "Restricted_diet", "Smoking", "Montreal_location",
    "Montreal_behaviour", "Montreal_extent", "Max_extent",
    "Baseline_Previous_immunomodulator", "Baseline_Mesalazine",
    "Previous_biologic", "Biologic_stopping", "Steroids",
    "CD_Surgery", "Appendectomy"
]]

####################
# Output directories
####################
abundance_dir  = os.path.join(outdir, 'abundance')
pairwise_dir   = os.path.join(abundance_dir, 'pairwise')
expression_dir = os.path.join(outdir, 'expression')
for _d in [outdir, abundance_dir, pairwise_dir, expression_dir]:
    os.makedirs(_d, exist_ok=True)

####################
# Load and format metadata
####################
obs, poolcount, test_collection_to_wetlab = load_and_format_obs(h5ad, nonmerged_wetlab_meta, qc_metrics)
meta_df = obs[['pool_participant'] + meta_cols].drop_duplicates()

##################
# Overall distributions
##################
obs['num_samples_in_pool'] = obs['num_samples_in_pool'].astype(str)
pool_numbers = np.sort(np.unique(obs['num_samples_in_pool']))

if plot_distributions:
    os.makedirs(outdir, exist_ok=True)
    # Pool size histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(poolcount['num_samples_in_pool'].dropna(), bins=40, kde=False)
    plt.title('Distribution of Num Samples In Pool')
    plt.xlabel('Number of Samples')
    plt.ylabel('Count')
    plt.savefig(f'{outdir}/num_samples_in_pool_hist.png')
    plt.close()
    # Transit-time histograms
    for c in ['shipment_to_wetlab', 'collection_to_wetlab', 'site']:
        plt.figure(figsize=(8, 6))
        sns.histplot(meta_df[c].dropna(), bins=40, kde=False)
        plt.title(f'Distribution of {c.replace("_", " ").title()} (days)')
        plt.xlabel('Days')
        plt.ylabel('Count')
        if c == 'site':
            plt.setp(plt.gca().get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        plt.savefig(f'{outdir}/{c}_days_hist.png')
        plt.close()
    # QC metrics by pool size
    for metric in qc_metrics:
        plt.figure(figsize=(8, 6))
        for pool_num in pool_numbers:
            subset = obs[obs['num_samples_in_pool'] == pool_num]
            sns.kdeplot(subset[metric], label=f'Pool Size: {pool_num}', fill=True, alpha=0.5)
        plt.legend()
        plt.title(f'{metric} by Number of Samples in Pool')
        plt.xlabel(metric)
        plt.ylabel('Density')
        plt.savefig(f'{outdir}/qc_{metric}_dist.png')
        plt.close()
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='num_samples_in_pool', y=metric, data=obs)
        plt.title(f'{metric} by Number of Samples in Pool')
        plt.xlabel('Number of Samples in Pool')
        plt.ylabel(metric)
        plt.savefig(f'{outdir}/qc_{metric}_boxplot.png')
        plt.close()
    # Scatter: collection to wetlab vs pool size
    spec_comp = obs[['pool_participant', 'num_samples_in_pool', 'collection_to_wetlab']].drop_duplicates()
    plt.figure(figsize=(8, 6))
    valid = spec_comp[['collection_to_wetlab', 'num_samples_in_pool']].dropna()
    sns.histplot(data=valid, x='collection_to_wetlab', y='num_samples_in_pool', bins=30, cmap='viridis', cbar=True)
    mask = np.isfinite(spec_comp['collection_to_wetlab']) & np.isfinite(spec_comp['num_samples_in_pool'])
    if mask.sum() >= 3:
        x = spec_comp.loc[mask, 'collection_to_wetlab'].astype(float).values
        y = spec_comp.loc[mask, 'num_samples_in_pool'].astype(float).values
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red', linestyle='--', linewidth=1.5)
        rho, pval = stats.pearsonr(x, y)
        plt.text(0.02, 0.98, f'r={rho:.3f}, p={pval:.3e}', transform=plt.gca().transAxes,
                 ha='left', va='top', fontsize=15)
    plt.xlabel('collection_to_wetlab')
    plt.ylabel('num_samples_in_pool')
    plt.tight_layout()
    plt.savefig(f'{outdir}/scatter_collection_to_wetlab_vs_num_samples_in_pool.png')
    plt.close()
    
##################
# Cell-type abundance vs technical factors
##################
labels = ['Celltypist:IBDverse_eqtl:predicted_labels', 'IBDverse_eqtl:Category']

ca_dict_pool = {}
ca_dict_sample = {}
results_list = []
results_dict = {}

for label in labels:
    obs[['num_samples_in_pool']] = obs[['num_samples_in_pool']].astype(int)
    ca_pool = build_ca(obs, label=label, group_col='convoluted_samplename')
    num_samples = obs[['convoluted_samplename', 'num_samples_in_pool']].drop_duplicates()
    ca_pool = ca_pool.merge(num_samples, on='convoluted_samplename', how='left')
    coll_to_wetlab_pool = obs[['convoluted_samplename', 'collection_to_wetlab', 'shipment_to_wetlab']].groupby('convoluted_samplename').median().reset_index()
    ca_pool = ca_pool.merge(coll_to_wetlab_pool, on='convoluted_samplename', how='left')
    ca_pool[['num_samples_in_pool', 'collection_to_wetlab', 'shipment_to_wetlab']] = ca_pool[['num_samples_in_pool', 'collection_to_wetlab', 'shipment_to_wetlab']].astype(float)
    ca_dict_pool[label] = ca_pool
    ca_sample = build_ca(obs, label=label, group_col='pool_participant')
    sample_meta = obs[['pool_participant', 'collection_to_wetlab', 'shipment_to_wetlab']].groupby('pool_participant').median().reset_index()
    ca_sample = ca_sample.merge(sample_meta, on='pool_participant', how='left')
    ca_sample[['collection_to_wetlab', 'shipment_to_wetlab']] = ca_sample[['collection_to_wetlab', 'shipment_to_wetlab']].astype(float)
    ca_dict_sample[label] = ca_sample
    results = []
    for celltypelabel in obs[label].unique():
        for predictor in ['num_samples_in_pool', 'collection_to_wetlab', 'shipment_to_wetlab']:
            ca_use = ca_pool if predictor == 'num_samples_in_pool' else ca_sample
            subset = ca_use[ca_use[label] == celltypelabel]
            subset = subset[subset['proportion_clr'].notna() & subset[predictor].notna()]
            valid_groups = subset[predictor].value_counts()
            subset = subset[subset[predictor].isin(valid_groups[valid_groups >= 3].index)]
            y = subset['proportion_clr'].values.astype(float)
            x = subset[predictor].values.astype(float)
            finite_mask = np.isfinite(x) & np.isfinite(y)
            if finite_mask.sum() < 3:
                results.append({'celltypelabel': celltypelabel, 'predictor': predictor, 'n': int(finite_mask.sum()), 'slope': np.nan, 'intercept': np.nan, 'r_value': np.nan, 'p_value': np.nan, 'std_err': np.nan})
                continue
            try:
                lr = stats.linregress(x[finite_mask], y[finite_mask])
                results.append({'celltypelabel': celltypelabel, 'predictor': predictor, 'n': int(finite_mask.sum()), 'slope': lr.slope, 'intercept': lr.intercept, 'r_value': lr.rvalue, 'p_value': lr.pvalue, 'std_err': lr.stderr})
            except Exception:
                results.append({'celltypelabel': celltypelabel, 'predictor': predictor, 'n': int(finite_mask.sum()), 'slope': np.nan, 'intercept': np.nan, 'r_value': np.nan, 'p_value': np.nan, 'std_err': np.nan})
    results_df = pd.DataFrame(results).sort_values('p_value')
    pvals = results_df['p_value'].to_numpy()
    mask = np.isfinite(pvals)
    adj = np.full(pvals.shape, np.nan)
    if mask.sum() > 0:
        p_for_adj = pvals[mask]
        order = np.argsort(p_for_adj)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(p_for_adj) + 1)
        q = p_for_adj * float(len(p_for_adj)) / ranks
        q = np.minimum.accumulate(q[::-1])[::-1]
        q[q > 1] = 1.0
        adj[mask] = q
    results_df['p_value_adj'] = adj
    results_df.to_csv(f'{abundance_dir}/{label}_clr_vs_technical_factors_regression_results.csv', index=False)
    results_list.append(results_df)
    results_dict[label] = results_df

results_all = pd.concat(results_list, ignore_index=True)
results_all[results_all['p_value_adj'] < 0.05].sort_values('p_value').to_csv(
    f'{abundance_dir}/significant_celltype_abundance_vs_technical_factors_univariate.csv', index=False)
results_all = results_all.merge(
    annot_mapping[['label_new', 'category', 'annotation_type']].drop_duplicates(),
    left_on="celltypelabel", right_on="label_new", how='left')

if plot_abundance:
    # Pairwise scatter plots for significant cell types
    for label in labels:
        results_df = results_dict[label]
        sig_results = results_df[results_df['p_value'] < 0.05]
        for _, row in sig_results.iterrows():
            celltypelabel = row['celltypelabel']
            predictor = row['predictor']
            ca_df = ca_dict_pool[label] if predictor == 'num_samples_in_pool' else ca_dict_sample[label]
            subset = ca_df[ca_df[label] == celltypelabel]
            subset = subset[subset['proportion_clr'].notna() & subset[predictor].notna()]
            valid_groups = subset[predictor].value_counts()
            subset = subset[subset[predictor].isin(valid_groups[valid_groups >= 3].index)]
            if subset.shape[0] == 0:
                continue
            plt.figure(figsize=(8, 6))
            uniq = sorted(subset[predictor].dropna().unique())
            order = [str(x) for x in uniq]
            sns.violinplot(x=subset[predictor].astype(str), y='proportion_clr', data=subset, order=order, inner=None, color='lightgray')
            sns.stripplot(x=subset[predictor].astype(str), y='proportion_clr', data=subset, order=order, jitter=0.25, size=4, color='black', alpha=0.6)
            x_pos = np.arange(len(uniq))
            y_pred = [row['intercept'] + row['slope'] * v for v in uniq]
            plt.plot(x_pos, y_pred, '--', color='red')
            ax = plt.gca()
            for i, pool_size in enumerate(uniq):
                pool_data = subset[subset[predictor] == pool_size]['proportion_clr']
                med = pool_data.median()
                q25 = pool_data.quantile(0.25)
                q75 = pool_data.quantile(0.75)
                ax.hlines(med, i - 0.4, i + 0.4, colors='blue', linewidth=2, linestyle='-', label='Median' if i == 0 else '')
                ax.hlines(q25, i - 0.4, i + 0.4, colors='green', linewidth=1.5, linestyle='--', label='IQR' if i == 0 else '')
                ax.hlines(q75, i - 0.4, i + 0.4, colors='green', linewidth=1.5, linestyle='--')
            plt.xlabel(predictor.replace("_", " "))
            plt.title(f'CLR of {celltypelabel} vs {predictor.replace("_", " ")} ({label})\n p={row["p_value"]:.3e}, slope={row["slope"]:.3f}')
            plt.ylabel('CLR Proportion')
            safe_ct = str(celltypelabel).replace('/', '.')
            plt.savefig(f'{pairwise_dir}/{label}_{safe_ct}_clr_vs_{predictor}.png')
            plt.close()
    
    # Volcano plots
    size_map = {'Cell type': 100, 'Major population': 200, 'All Cells': 400}
    for predictor in results_all['predictor'].dropna().unique():
        sub = results_all[results_all['predictor'] == predictor].copy()
        sub = sub[np.isfinite(sub['slope']) & np.isfinite(sub['p_value'])]
        if sub.shape[0] == 0:
            continue
        sub['point_color'] = np.where(
            sub['p_value_adj'] < 0.05,
            sub['category'].map(color_map).fillna('#9e9e9e'),
            '#9e9e9e'
        )
        sub['point_size'] = sub['annotation_type'].map(size_map).fillna(25)
        sub['neglog10_p'] = -np.log10(sub['p_value'])
        plt.figure(figsize=(8, 6))
        plt.scatter(sub['slope'], sub['neglog10_p'], c=sub['point_color'], s=sub['point_size'], alpha=0.8)
        sig = sub[sub['p_value_adj'] < 0.05]
        for _, row in sig.iterrows():
            plt.text(row['slope'], row['neglog10_p'], str(row['celltypelabel']), fontsize=7, alpha=0.9)
        plt.xlabel('Slope')
        plt.ylabel('-log10(p-value)')
        plt.title(str(predictor))
        sig_cats = sub.loc[sub['p_value_adj'] < 0.05, 'category'].dropna().unique().tolist()
        color_handles = [Line2D([0], [0], marker='o', color='w', label=cat,
                    markerfacecolor=color_map.get(cat, '#9e9e9e'), markersize=8)
                for cat in sig_cats]
        color_handles.append(Line2D([0], [0], marker='o', color='w', label='Not significant',
                        markerfacecolor='#9e9e9e', markersize=8))
        size_handles = [Line2D([0], [0], marker='o', color='w', label=lbl,
                        markerfacecolor='#666666', markersize=np.sqrt(sz))
                    for lbl, sz in size_map.items()]
        first_legend = plt.legend(handles=color_handles, title='Major pop. (p_adj<0.05)', loc='upper right', frameon=True)
        plt.gca().add_artist(first_legend)
        plt.legend(handles=size_handles, title='Annotation type', loc='lower right', frameon=True)
        plt.tight_layout()
        plt.savefig(f'{abundance_dir}/volcano_{predictor}.png')
        plt.close()

##################
# Compute a PCA based on the cell-type proportions per sample, and compute the variance explained by each of the technical factors.
##################
label_for_pca = 'Celltypist:IBDverse_eqtl:predicted_labels'
group_col = 'pool_participant'
ca = build_ca(obs, label=label_for_pca, group_col=group_col)

# Build wide CLR matrix for PCA
clr_wide = ca.pivot_table(index=group_col, columns=label_for_pca, values='proportion_clr', aggfunc='mean', fill_value=0)

results_df, pc_scores, explained_variance_ratio, knee_idx = pca_variance_explained(
    clr_wide, meta_df, meta_cols, group_col,
    use_knee=True, scale=False, n_perm=1000, seed=1,
)

results_df.to_csv(f'{abundance_dir}/cell_abundance_pca_variance_explained_by_technical_factors.csv', index=False)

pc_meta = pc_scores.merge(meta_df, on=group_col, how='left')

# Plot the scree plot for the PC factors
plt.figure(figsize=(8,6))
pc_idx = np.arange(1, len(explained_variance_ratio) + 1)
plt.plot(pc_idx, explained_variance_ratio, marker='o')
knee_pc = pc_idx[knee_idx]
plt.axvline(knee_pc, color='red', linestyle='--', label=f'Knee = PC{knee_pc}')
plt.legend()

plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.tight_layout()
plt.savefig(f'{abundance_dir}/cell_abundance_pca_scree_plot.png')
plt.close()


# Plot the PCA, colouring the points by any value from results_df which has p_value_perm < 0.05 + a dummy
sig_predictors = results_df.loc[results_df['p_value_perm'] < 0.05, 'predictor'].tolist()
if len(sig_predictors) == 0:
    sig_predictors = []

pc_meta['dummy'] = ""
sig_predictors.append('dummy')  

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
    plt.savefig(f'{abundance_dir}/pca_pc1_pc2_colored_by_{safe_pred}.png')
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
            try:
                _, p_val = stats.kruskal(*groups)
            except Exception:
                p_val = np.nan
            heat_r2.loc[predictor, pc] = _r2_for_predictor(yv, xv, categorical=True, use_omega2=use_omega2)
            heat_p.loc[predictor, pc] = p_val
        else:
            xv_f = xv.astype(float)
            if np.std(xv_f) == 0:
                heat_r2.loc[predictor, pc] = np.nan
                heat_p.loc[predictor, pc] = np.nan
                continue
            # linear regression: PC ~ variable
            xv_design = sm.add_constant(xv_f)
            try:
                ols_res = sm.OLS(yv, xv_design).fit()
                p_val = ols_res.pvalues[1]
                r2 = ols_res.rsquared
                if use_omega2:
                    r2 = 1.0 - (1.0 - r2) * (len(yv) - 1) / (len(yv) - 2)
            except Exception:
                p_val, r2 = np.nan, np.nan
            heat_r2.loc[predictor, pc] = r2
            heat_p.loc[predictor, pc] = p_val

# Melt to long format for plotnine
heat_r2.index.name = 'predictor'
heat_p.index.name = 'predictor'
_hm_pc = heat_r2.reset_index().melt(id_vars='predictor', var_name='PC', value_name='r2')
_hmp_pc = heat_p.reset_index().melt(id_vars='predictor', var_name='PC', value_name='p_val')
_hm_pc = _hm_pc.merge(_hmp_pc, on=['predictor', 'PC'])
_hm_pc['sig'] = _hm_pc['p_val'].apply(_sig_stars)
_hm_pc['predictor_clean'] = _hm_pc['predictor'].str.replace('meta-', '', regex=False)
_hm_pc['PC'] = pd.Categorical(_hm_pc['PC'], categories=pcs_use, ordered=True)

_p_pc = (
    ggplot(_hm_pc, aes(x='predictor_clean', y='PC', fill=effect_size_col))
    + geom_tile(color='white', size=0.3)
    + geom_text(aes(label='sig'), size=7, color='black', va='center')
    + scale_fill_gradient(low='white', high='#2166ac', name=effect_size_label, na_value='lightgrey')
    + theme_bw()
    + theme(
        axis_text_x=element_text(rotation=45, hjust=1),
        panel_grid=element_blank(),
        figure_size=(max(6, 0.6 * len(meta_cols)), max(4, 0.4 * len(pcs_use))),
    )
    + labs(x='Predictor', y='PC', title=f'PC vs Predictor Associations ({effect_size_label}, * p<0.05)')
)
_p_pc.save(f'{abundance_dir}/pc_predictor_r2_heatmap.png', dpi=150, bbox_inches='tight')

# Plot a regression for the strongest association
pc_plot = "PC2"
predictor_plot = "collection_to_wetlab"
subset = pc_meta[[pc_plot, predictor_plot]].dropna()

# Convert predictor to categorical (sorted by value)
subset['predictor_cat'] = subset[predictor_plot].astype(str)
unique_vals = sorted(subset[predictor_plot].unique())
cat_order = [str(v) for v in unique_vals]

plt.figure(figsize=(max(10, 0.5 * len(unique_vals)), 6))
ax = plt.gca()
# Create violin plot
sns.violinplot(x='predictor_cat', y=pc_plot, data=subset, order=cat_order, inner=None, color='lightgray')
# Add strip plot for individual points
sns.stripplot(x='predictor_cat', y=pc_plot, data=subset, order=cat_order, color='black', size=4, alpha=0.5, jitter=0.2)

# Add median and quartile lines for each category
for i, val in enumerate(unique_vals):
    val_data = subset[subset[predictor_plot] == val][pc_plot]
    if len(val_data) > 0:
        med = val_data.median()
        q25 = val_data.quantile(0.25)
        q75 = val_data.quantile(0.75)
        # Draw median as solid line
        ax.hlines(med, i - 0.4, i + 0.4, colors='blue', linewidth=2.5, linestyle='-', label='Median' if i == 0 else '')
        # Draw quartile bounds as dashed lines
        ax.hlines(q25, i - 0.4, i + 0.4, colors='green', linewidth=2, linestyle='--', label='Q25/Q75' if i == 0 else '')
        ax.hlines(q75, i - 0.4, i + 0.4, colors='green', linewidth=2, linestyle='--')

plt.xlabel(predictor_plot.replace("_", " ") + " (days)")
plt.ylabel(pc_plot)
plt.title(f'{pc_plot} vs {predictor_plot.replace("_", " ")}')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{abundance_dir}/{pc_plot}_vs_{predictor_plot}_violin.png')
plt.close()

###############
# Do some more formal association testing - Linear model per cell-type (sample-level CLR)
# response ~ clr + num_samples_in_pool + collection_to_wetlab
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
                    'intercept': model.params.get('Intercept', np.nan),
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
                    'intercept': model.params.get('Intercept', np.nan),
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
    f'{abundance_dir}/celltype_clr_linear_model_results.csv',
    index=False
)

# Plot any significant results
sig_results = lm_results_df[lm_results_df['p_target'] < 0.05]
for _, row in sig_results.iterrows():
    celltypelabel = row['celltypelabel']
    target_var = row['target']
    subset = ca_sample_lm[ca_sample_lm[label_lm] == celltypelabel]
    # skip empty subsets
    if subset.shape[0] == 0:
        continue
    plt.figure(figsize=(8,6))
    # prepare category order (string) so violin/strip align
    uniq = sorted(subset[target_var].dropna().unique())
    order = [str(x) for x in uniq]
    # violin plot (no inner points)
    sns.violinplot(x=subset[target_var].astype(str), y='proportion_clr', data=subset, order=order, inner=None, color='lightgray')
    # jittered points on top
    sns.stripplot(x=subset[target_var].astype(str), y='proportion_clr', data=subset, order=order, jitter=0.25, size=4, color='black', alpha=0.6)
    # Plot regression line mapped to category positions
    x_pos = np.arange(len(uniq))
    y_pred = [row['intercept'] + row['coef_target'] * v for v in x_pos]
    # Add median and IQR as horizontal lines per pool size
    ax = plt.gca()
    for i, value in enumerate(uniq):
        pool_data = subset[subset[target_var] == value]['proportion_clr']
        med = pool_data.median()
        q25 = pool_data.quantile(0.25)
        q75 = pool_data.quantile(0.75)
        # Draw median as solid line
        ax.hlines(med, i - 0.4, i + 0.4, colors='blue', linewidth=2, linestyle='-', label='Median' if i == 0 else '')
        # Draw IQR bounds as dashed lines
        ax.hlines(q25, i - 0.4, i + 0.4, colors='green', linewidth=1.5, linestyle='--', label='IQR' if i == 0 else '')
        ax.hlines(q75, i - 0.4, i + 0.4, colors='green', linewidth=1.5, linestyle='--')
    plt.xlabel(target_var.replace("_", " "))
    plt.title(f'CLR of {celltypelabel} vs {target_var.replace("_", " ")}\n p={row["p_target"]:.3e}, slope={row["coef_target"]:.3f}')
    plt.ylabel('CLR Proportion')
    celltypelabel = str(celltypelabel).replace('/', '.')
    plt.savefig(f'{pairwise_dir}/{label_lm}_{celltypelabel}_clr_vs_{target_var}.png')
    plt.close() 
    
    
    
##################
# Calculate the amount of variance in gene expression that is explained by the technical factors
##################
# Specify pseudobulking options
pseudobulk_options = {
    'groupby': 'pool_participant',
    'label': 'Celltypist:IBDverse_eqtl:predicted_labels',
    'min_cells': 5,
    'min_samples': 30,
    'layer': 'counts',
    'method': 'sum',
    'min_samples_per_gene': 0.2,
    'nhvgs': 0.2
}

pbfile = f'{expression_dir}/pseudobulk_expr_matrices-{pseudobulk_options["label"]}-method_{pseudobulk_options["method"]}.pkl'
if os.path.exists(pbfile):
    print(f"Loading existing pseudobulk matrices from {pbfile}")
    with open(pbfile, 'rb') as f:
        pseudobulk_matrices = pickle.load(f)
else:
    # Load actual expression (FULL)
    print("Loading full expression data for pseudobulking")
    adata = sc.read_h5ad(h5ad_full)
    # Pseudobulk using the custom function
    pseudobulk_matrices = pseudobulk_by_label(adata, **pseudobulk_options)
    # Print shapes of all pseudobulk matrices
    print(f".. Pseudobulk matrix shapes")
    for cell_type, matrix in pseudobulk_matrices.items():
        print(f"{cell_type}: {matrix.shape} (samples x genes)")
    # Save this as a pickle for later use (since it can be time-consuming to generate)
    with open(pbfile, 'wb') as f:
        pickle.dump(pseudobulk_matrices, f)


# Calculate PCA for each pseudobulk matrix and calculate variance explained by each technical factor
n_perm = 1000
expr_pca_results_list = []
expr_pc_scores_dict = {}
group_col= 'pool_participant'
meta_df = obs[[group_col] + meta_cols].drop_duplicates()

expr_res_file = f'{expression_dir}/expression_pca_variance_explained_by_technical_factors.csv'
if os.path.exists(expr_res_file):
    print(f"Loading existing expression PCA variance explained results from {expr_res_file}")
    expr_pca_results = pd.read_csv(expr_res_file)
else:
    expr_pca_results_list = []
    for cell_type, expr_matrix in pseudobulk_matrices.items():
        print(f"..PCA for {cell_type}")
        ct_results, pc_scores, evr, _ = pca_variance_explained(
            expr_matrix, meta_df, meta_cols, group_col,
            use_knee=True, scale=True, n_perm=n_perm, seed=1,
        )
        ct_results.insert(0, 'celltype', cell_type)
        expr_pc_scores_dict[cell_type] = (pc_scores, evr)
        expr_pca_results_list.append(ct_results)
    expr_pca_results = pd.concat(expr_pca_results_list, ignore_index=True)
    expr_pca_results.to_csv(expr_res_file, index=False)

# Attach annotations
expr_pca_results = expr_pca_results.merge(annot_mapping[['label_new', 'category']].drop_duplicates(), left_on="celltype", right_on="label_new", how='left')

# Plot a big heatmap of this
_hm = expr_pca_results.copy()
_hm['sig'] = _hm['p_value_perm'].apply(_sig_stars)
_hm['predictor_clean'] = _hm['predictor'].str.replace('meta-', '', regex=False)

# Order cell types grouped by category
_ct_order = (
    _hm[['celltype', 'category']].drop_duplicates()
    .sort_values(['category', 'celltype'])['celltype'].tolist()
)
_hm['celltype'] = pd.Categorical(_hm['celltype'], categories=_ct_order, ordered=True)

_n_ct = _hm['celltype'].nunique()
_n_pred = _hm['predictor_clean'].nunique()
_fig_w = max(6, _n_pred * 0.6)
_fig_h = max(4, _n_ct * 0.35)

# Map each cell type to its category color (same color_map as volcano)
_ct_cat_map = _hm[['celltype', 'category']].drop_duplicates().set_index('celltype')['category'].to_dict()
_ytick_colors = [color_map.get(_ct_cat_map.get(ct), '#9e9e9e') for ct in _ct_order]

# Replace negative values in the weighted_omega2 column with 0 for better visualization (if using omega2)
if use_omega2:
    _hm['weighted_omega2'] = _hm['weighted_omega2'].apply(lambda x: max(x, 0) if pd.notna(x) else x)

_p = (
    ggplot(_hm, aes(x='predictor_clean', y='celltype', fill='weighted_omega2' if use_omega2 else 'weighted_r2'))
    + geom_tile(color='white', size=0.3)
    + geom_text(aes(label='sig'), size=7, color='black', va='center')
    + scale_fill_gradient(low='white', high='#d62728', name=effect_size_label)
    + theme_bw()
    + theme(
        axis_text_x=element_text(rotation=45, hjust=1, size=8),
        axis_text_y=element_text(size=7),
        panel_grid=element_blank(),
        figure_size=(_fig_w, _fig_h),
    )
    + labs(x='Predictor', y='Cell type', title=f'Expression PCA: {effect_size_label} by technical factors',
           subtitle='* p<0.05, ** p<0.01, *** p<0.001 (permutation p-value, uncorrected)')
)
_mpl_fig = _p.draw()
_ax = _mpl_fig.axes[0]
for tick, color in zip(_ax.get_yticklabels(), _ytick_colors):
    tick.set_color(color)

_mpl_fig.savefig(f'{expression_dir}/expr_pca_weighted_r2_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(_mpl_fig)


####################
# Variance in key technical covariates explained by other covariates
# meta_df is at pool_participant level (one row per sample), defined in the abundance section above
####################
query_cols = ['collection_to_wetlab', 'convoluted_samplename']
cov_pred_cols = [c for c in meta_cols if c not in query_cols]

cov_results = {}
for query in query_cols:
    print(f"Testing variance explained in: {query}")
    cov_pred_cols = [c for c in meta_cols if c != query]
    res = variance_explained_by_covariates(
        meta_df, query, cov_pred_cols,
        use_omega2=use_omega2, n_perm=1000, seed=1,
    )
    res['query'] = query
    cov_results[query] = res
    res.to_csv(f'{outdir}/covariate_variance_explained_by_{query}.csv', index=False)

# Combined heatmap: predictors on y, query on x, fill = effect size
_cov_hm = pd.concat(cov_results.values(), ignore_index=True)
_cov_hm['sig'] = _cov_hm['p_value_perm'].apply(_sig_stars)
_cov_hm['predictor_clean'] = _cov_hm['predictor'].str.replace('meta-', '', regex=False)

# Order predictors by mean effect size across queries (descending)
_pred_order = (
    _cov_hm.groupby('predictor_clean')['effect_size']
    .mean().sort_values(ascending=False).index.tolist()
)
_cov_hm['predictor_clean'] = pd.Categorical(_cov_hm['predictor_clean'], categories=_pred_order, ordered=True)

# Clip negatives to 0 for display
if use_omega2:
    _cov_hm['effect_size'] = _cov_hm['effect_size'].clip(lower=0)

_n_pred_cov = len(_pred_order)
_p_cov = (
    ggplot(_cov_hm, aes(x='predictor_clean', y='query', fill='effect_size'))
    + geom_tile(color='white', size=0.4)
    + geom_text(aes(label='sig'), size=9, color='black', va='center')
    + scale_fill_gradient(low='white', high='#2ca25f', name=effect_size_label, limits=[0, None])
    + theme_bw()
    + theme(
        axis_text_x=element_text(rotation=45, hjust=1, size=9),
        axis_text_y=element_text(size=8),
        panel_grid=element_blank(),
        figure_size=(max(6, _n_pred_cov * 0.6), 4),
    )
    + labs(
        x=None, y=None,
        title=f'Variance in technical covariates explained by other covariates\n({effect_size_label})',
        subtitle='* p<0.05, ** p<0.01, *** p<0.001 (permutation p-value, uncorrected)',
    )
)
_p_cov.save(f'{outdir}/covariate_variance_explained_heatmap.png', dpi=150, bbox_inches='tight')

