#!/usr/bin/env python
# Bradley September 2025
# Further analysis of the IBDR scRNAseq after round1 of atlassing/QC

# Packages
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('bin')
import nmad_qc # Import custom functions
import anndata as ad
from anndata.experimental import read_elem
from h5py import File
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind
import itertools
import numpy as np
from statsmodels.stats.multitest import multipletests

# Define tissue
tissue="IBDRbatch1-8"

# Load
fpath = f"results/{tissue}/objects/adata_PCAd_batched_umap.h5ad"
adata = sc.read_h5ad(fpath)

# Merge this with the full matrix
rawpath = f"input/adata_raw_input_{tissue}.h5ad"
raw = sc.read_h5ad(rawpath)
raw = raw[raw.obs.index.isin(adata.obs.index)]
raw.obsm['X_umap'] = adata.obsm['X_umap'].copy()
raw.obs = adata.obs
del adata
adata = raw

# Define out
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

###########################
# Other UMAP plots
###########################
sc.settings.figdir=f"results/{tissue}/figures/UMAP"
cols = ["meta-Diagnosis", "meta-Assessment_stage", "meta-Number_previous_biologic", "meta-Biologic_starting", "meta-W14_RESPONSE", "meta-Baseline_PRO2_CD", "meta-Baseline_PRO2_UC", "Celltypist:IBDverse_eqtl:conf_score", "samp_median_n_genes"]
numeric = ["Celltypist:IBDverse_eqtl:conf_score", "samp_median_n_genes"]
for c in cols:
    print(c)
    if c in numeric:
        adata.obs[c] = adata.obs[c].astype(float)
    
    sc.pl.umap(adata, color = c, save="_" + c + ".png")

# genes too
sc.settings.figdir=f"results/{tissue}/figures/UMAP/expr"
genes = "EPCAM,KRT8,KRT18,CDH5,COL1A1,COL1A2,COL6A2,VWF,PTPRC,CD3D,CD3G,CD3E,CD79A,CD79B,CD14,FCGR3A,CD68,CD83,CSF1R,FCER1G".split(",")
for g in genes:
    out = f"results/{tissue}/figures/UMAP/expr/umap_X_scVI_{g}.png"
    if os.path.exists(out) == False:
        if g in adata.var['gene_symbols'].values:
            print(f"..Plotting {g}")
            ens=adata.var[adata.var['gene_symbols'] == g].index[0]
            sc.pl.umap(adata, color = ens, save="_X_scVI_" + g + ".png")
    
###########################
# Dotplot for the canonical markers
###########################
# Also plot a dotplot
major = {"Epithelial": ["EPCAM" ,'CDH1', 'KRT19', 'EPCAM'], "Mesenchymal": ["COL1A1","COL1A2","COL6A2","VWF"], 'Immune':['PTPRC'], 'B':['CD79A', 'MS4A1', 'MS4A1', 'CD79B'], 'Plasma':['MZB1', 'JCHAIN'], 'T':['CD3D', 'CD3E', 'CD3G','CCR7','IL7R', 'TRAC'], 'Myeloid':['ITGAM', 'CD14', 'CSF1R', 'TYROBP'], 'DC':['ITGAX', 'CLEC4C','CD1C', 'FCER1A', 'CLEC10A'], 'Mac':['APOE', 'C1QA', 'CD68','AIF1'], 'Mono':['FCN1','S100A8', 'S100A9', "CD14", "FCGR3A", 'LYZ'], 'Mast':["TPSAB1", 'TPSB2', "CPA3" ], 'Platelet+RBC':['GATA1', 'TAL1', 'ITGA2B', 'ITGB3']}
sc.pl.dotplot(adata, major, layer="log1p_cp10k", gene_symbols = "gene_symbols", groupby='IBDverse_eqtl:Category', dendrogram=False, save="_major_markers.png")


###########################
# Checking the non-immune populations
###########################
qc_check_path = sc.settings.figdir=f"results/{tissue}/figures/check_QC"
if os.path.exists(qc_check_path) == False:
    os.mkdir(qc_check_path)

# it looks like some cells are labelled confidently as non-immune.
# Count these per individual/batch to see where these are coming from
conf = adata.obs[adata.obs['Celltypist:IBDverse_eqtl:conf_score'] > 0.5]

def summarize(df, group_col):
    grouped = (
        df.groupby(group_col)
          .apply(lambda g: pd.Series({
              "total": len(g),
              "nonimmune": (g["IBDverse_eqtl:Lineage"].isin(["Epithelial", "Mesenchymal"])).sum()
          }))
          .reset_index()
    )
    grouped["nonimmune_pct"] = grouped["nonimmune"] / grouped["total"] * 100
    grouped["group_type"] = group_col  # to distinguish later
    grouped = grouped.rename(columns={group_col: "group"})
    return grouped

# Summarize by both grouping columns
by_sample = summarize(conf, "convoluted_samplename")
by_pool = summarize(conf, "pool_participant")

# Combine into one final dataframe
final_df = pd.concat([by_sample, by_pool], ignore_index=True)

# Plot a distibution
for c in ["convoluted_samplename", "pool_participant"]:
    print(c)
    dat = final_df[final_df['group_type'] == c]
    dat.sort_values(by="nonimmune_pct", ascending=False)
    sns.distplot(dat['nonimmune_pct'], hist=False, rug=True, label=c)
    plt.xlabel(f"% high confidence non immune cells")
    plt.legend()
    plt.savefig(f"{qc_check_path}/high_confident_non-immune_cells_per_{c}.png", bbox_inches='tight')
    plt.clf()

# From this, it looks like there are a handful of samples for which we are detecting high confidence epithelial cells. 
# Find those with > 10% high confident epithelial cells
final_df[(final_df['group_type'] == "pool_participant") & (final_df['nonimmune_pct'] > 10)]
final_df[(final_df['group_type'] == "convoluted_samplename") & (final_df['nonimmune_pct'] > 10)]
bad_samps = final_df.loc[(final_df['group_type'] == "pool_participant") & (final_df['nonimmune_pct'] > 10), 'group']

# How many of each major population per high non-immune sample?
cat_per_sample = (
    conf.groupby(["pool_participant", "IBDverse_eqtl:Category"])
        .size()
        .reset_index(name="size")
)

cat_per_sample["pct"] = (
    cat_per_sample.groupby("pool_participant")["size"]
        .transform(lambda x: x / x.sum() * 100)
)

# Have a look at the percentage of colonocytes and enterocytes specifically
cat_per_sample[
    (cat_per_sample['pool_participant'].isin(bad_samps) ) & ( cat_per_sample['IBDverse_eqtl:Category'].isin(['Colonocyte', 'Enterocyte', 'Mesenchymal']) )
]

# CONCLUSION
# Comparing these with marker gene expression, it really does like these are indeed epithelial cells.
# Need to check these with Cristina. 
# Biggest concern is whether these samples are swapped with an epithelial sample from another study

###########################
# Exploring QC distributions per major population
###########################
major_pops = adata.obs['IBDverse_eqtl:Category'].unique().astype(str)
major_pops = ['Plasma', 'Myeloid', 'B', 'T'] # Forget the nonimmune for now

cols = ["pct_counts_gene_group__mito_transcript", "log_n_genes_by_counts", "log_total_counts"]
adata.obs['log_n_genes_by_counts'] = np.log10(adata.obs['n_genes_by_counts'])
adata.obs['log_total_counts'] = np.log10(adata.obs['total_counts'])

for c in cols:
    print(c)
    for cat in major_pops:
        dat = adata.obs[adata.obs['IBDverse_eqtl:Category'] == cat]
        sns.distplot(dat[c], hist=False, rug=True, label=cat)
    plt.xlabel(c)
    plt.legend()
    plt.savefig(f"{qc_check_path}/{c}_per_major_pop.png", bbox_inches='tight')
    plt.clf()

###########################
# How does this change with confidence score? And what if we compare to the thresholds used for IBDverse
###########################
# Load obs from old high qc and find the min/max per major population
h5ad="/lustre/scratch127/humgen/projects_v2/sc-eqtl-ibd/core_analysis_output/IBDverse_multi-tissue_eQTL_project/IBDverse_scRNA/adata_PCAd_batched_umap_add_expression.h5ad"
f2 = File(h5ad, 'r')
ibdvobs = read_elem(f2['obs'])
ibdvobs['log_n_genes_by_counts'] = np.log10(ibdvobs['n_genes_by_counts'])
ibdvobs['log_total_counts'] = np.log10(ibdvobs['total_counts'])
ibdvobsblood = ibdvobs[ibdvobs['tissue'] == "blood"]
ibdvobsblood['Category'] = ibdvobsblood['Category'].astype(str)
ranges = (
    ibdvobsblood.groupby("Category")[cols]
      .agg(["min", "max"])
      .reset_index()
)
ranges["Category"] = ranges["Category"].replace("T/ILC", "T")

for c in cols:
    print(c)
    for cat in major_pops:
        dat = adata.obs[adata.obs['IBDverse_eqtl:Category'] == cat]
        sns.distplot(dat[c], hist=False, rug=True, label=cat)
        dat2 = dat[dat['Celltypist:IBDverse_eqtl:conf_score'].astype(float) > 0.5]
        sns.distplot(dat2[c], hist=False, rug=True, label=f"{cat} - conf > 0.5")
        plt.axvline(x = ranges.loc[ranges['Category'] == cat, (c, "min")].iloc[0], linestyle = '--', color = "black", alpha = 0.5)
        plt.axvline(x = ranges.loc[ranges['Category'] == cat, (c, "max")].iloc[0], linestyle = '--', color = "black", alpha = 0.5)
        plt.title("Lines = IBDverse high QC cut off")
        plt.xlabel(c)
        plt.legend()
        plt.savefig(f"{qc_check_path}/{c}_per_{cat}.png", bbox_inches='tight')
        plt.clf()

#####################
# Checking the epithelial / poorly integrated samples
#####################
# It looks like a population of UC samples don't integrate very well. Could this be due to low median nGenes expressed in those samples? 
# Plot a subset of cells, applying splightly high min median cells/sample
# This might also adjust the number of predicted epithelial cells 
sc.settings.figdir=f"results/{tissue}/figures/UMAP"
thresholds = [250, 500, 750, 1000, 1250, 1500]
for thresh in thresholds:
    print(f"..Plotting {thresh}")
    sc.pl.umap(adata[adata.obs['samp_median_n_genes'] > thresh], color = c, save=f"_samp_median_n_genes_thresh{thresh}.png")



#####################
# Plot the category contributions per sample (ordered) - confident annotations
#####################
pivot_df = cat_per_sample.pivot(index='pool_participant', columns='IBDverse_eqtl:Category', values='pct').fillna(0)
pivot_df = pivot_df.sort_values('T')
colors = sns.color_palette("husl", len(pivot_df.columns))
bottom = np.zeros(len(pivot_df))
bottom = np.zeros(len(pivot_df))
fig, ax = plt.subplots(figsize=(10, 6))
for idx, category in enumerate(pivot_df.columns):
    ax.bar(
        pivot_df.index,
        pivot_df[category],
        bottom=bottom,
        color=colors[idx],
        label=category
    )
    bottom += pivot_df[category].values     

# Legend and labels
ax.legend(title='IBDverse_eqtl:Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title('Relative Proportions of IBDverse_eqtl:Category by sample')
ax.set_xlabel('pool_participant')
ax.set_ylabel('Proportion')
ax.set_xticks([])
plt.savefig(f"{qc_check_path}/Category_proportions_across_samples.png", bbox_inches='tight')
plt.clf()


#####################
# Performing PCA of cell contributions. Use only high confident immune cells, exclude the bad samples define above
#####################
label_col = "Celltypist:IBDverse_eqtl:predicted_labels"
conf[label_col] = conf[label_col].astype(str)
ct_per_sample = (
    conf[(~conf['pool_participant'].isin(bad_samps)) & (conf['IBDverse_eqtl:Category'].isin(['Plasma', 'Myeloid', 'B', 'T']))]
        .groupby(["pool_participant", label_col])
        .size()
        .reset_index(name="size")
)

ct_per_sample["pct"] = (
    ct_per_sample.groupby("pool_participant")["size"]
        .transform(lambda x: x / x.sum() * 100)
)

pivot_df = ct_per_sample.pivot(index='pool_participant', columns=label_col, values='pct').fillna(0)

# Run PCA
pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit_transform(pivot_df), columns = ['PC1', 'PC2'], index=pivot_df.index)

# Merge with metadata to colour by 
cols_to_keep = adata.obs.columns.str.startswith("meta") | (adata.obs.columns == "pool_participant")
pca_df = pca_df.merge(
    adata.obs.loc[:, cols_to_keep].reset_index(drop=True).drop_duplicates().set_index("pool_participant"),
    left_index=True,
    right_index=True
)

# Merge also with T cell proportion
tcat = cat_per_sample.pivot(index='pool_participant', columns='IBDverse_eqtl:Category', values='pct').fillna(0)
pca_df = pca_df.merge(tcat['T'], left_index=True, right_index=True)

# Plot
plot_cols = ["meta-Assessment_stage", "meta-Sender", "meta-W14_RESPONSE", "meta-W14_REMISSION", 
             "meta-Sex", "meta-Ethnicity", "meta-Baseline_PRO2_CD", "meta-Baseline_PRO2_UC", 
             "meta-Previous_biologic", "meta-Biologic_starting", "meta-Diagnosis", "T"]

quant_cols = ["meta-Baseline_PRO2_CD", "meta-Baseline_PRO2_UC", "T"]
for col in quant_cols:
    pca_df[col] = pca_df[col].astype(float)
else:
    pca_df[col] = pca_df[col].astype(str)


pca_contributions = sc.settings.figdir=f"results/{tissue}/figures/PCA_contributions"
if os.path.exists(pca_contributions) == False:
    os.mkdir(pca_contributions)

for col in plot_cols:
    print(f"..Plotting {col}")
    outf=f"{pca_contributions}/PCA-{label_col}-{col}.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    # If categorical, map
    if col not in quant_cols:
        # Get unique categories sorted (optional)
        categories = sorted(pca_df[col].astype(str).unique())
        colors = sns.color_palette("husl", len(categories))
        color_dict = dict(zip(categories, colors))
        for category in categories:
            subset = pca_df[pca_df[col] == category]
            ax.scatter(
                subset['PC1'], 
                subset['PC2'], 
                alpha=0.8, 
                c=[color_dict[category]],  # consistent color
                label=category
            )
        #
    else:
        sc = ax.scatter(
            pca_df['PC1'],
            pca_df['PC2'],
            alpha=0.8,
            c=pca_df[col].astype(float),
            cmap='viridis'
        )
        plt.colorbar(sc, ax=ax, label=col)
    #
    ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel('Cell contribution PC 1')
    ax.set_ylabel('Cell contribution PC 2')
    ax.set_title(f'PCA of {label_col} proportions - {col}')
    plt.savefig(outf, bbox_inches='tight')
    plt.clf() 

#####################
# Looking at cell-type abundance across disease
#####################
resultsout = f"results/{tissue}/figures/results"
if os.path.exists(resultsout) == False:
    os.mkdir(resultsout)
    

label_col = "Celltypist:IBDverse_eqtl:predicted_labels"
conf[label_col] = conf[label_col].astype(str)
ct_per_sample = (
    conf[(~conf['pool_participant'].isin(bad_samps)) & (conf['IBDverse_eqtl:Category'].isin(['Plasma', 'Myeloid', 'B', 'T']))]
        .groupby(["pool_participant", label_col])
        .size()
        .reset_index(name="size")
)

ct_per_sample["pct"] = (
    ct_per_sample.groupby("pool_participant")["size"]
        .transform(lambda x: x / x.sum() * 100)
)

ct_per_sample = ct_per_sample.set_index("pool_participant").merge(
    adata.obs.loc[:, cols_to_keep].reset_index(drop=True).drop_duplicates().set_index("pool_participant"),
        left_index=True,
        right_index=True
)

order = (
    ct_per_sample.groupby('Celltypist:IBDverse_eqtl:predicted_labels')['pct']
      .median()
      .sort_values(ascending=True)
      .index
)

# Perform t-test and adjustment
groups = ct_per_sample["meta-Diagnosis"].unique()
pairs = []
pvals = []

for celltype in order:
    subset = ct_per_sample[ct_per_sample["Celltypist:IBDverse_eqtl:predicted_labels"] == celltype]
    if len(groups) == 2:  # only compare 2 groups
        g1 = subset[subset["meta-Diagnosis"] == groups[0]]["pct"]
        g2 = subset[subset["meta-Diagnosis"] == groups[1]]["pct"]
        if len(g1) > 1 and len(g2) > 1:
            stat, p = ttest_ind(g1, g2, nan_policy="omit")
            pairs.append(celltype)
            pvals.append(p)

# Apply Bonferroni correction
reject, pvals_corr, _, _ = multipletests(pvals, method="bonferroni")

# Plot
plt.figure(figsize=(14, 7))
ax = sns.boxplot(
    data=ct_per_sample,
    x='Celltypist:IBDverse_eqtl:predicted_labels',
    y='pct',
    hue='meta-Diagnosis',
    order=order
)

for i, (celltype, p_corr) in enumerate(zip(pairs, pvals_corr)):
    # Get the numeric x-position of this celltype
    x_pos = i
    # Place annotation a bit above the max value
    y_max = ct_per_sample.loc[ct_per_sample['Celltypist:IBDverse_eqtl:predicted_labels'] == celltype, "pct"].max()
    height = y_max * 1.1  
    if p_corr < 0.001:
        label = "***"
    elif p_corr < 0.01:
        label = "**"
    elif p_corr < 0.05:
        label = "*"
    else:
        label = ""
    #
    if label:  # only annotate if significant
        ax.text(
            x_pos, height, label,
            ha="center", va="bottom",
            fontsize=14, color="black", fontweight="bold"
        )

plt.xticks(rotation=45, ha='right', fontsize=8)
plt.ylabel("Proportion (%)")
plt.xlabel("Cell type")
plt.legend(title='', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{resultsout}/celltype_abundance_across_diagnosis.png", bbox_inches='tight')
plt.clf()