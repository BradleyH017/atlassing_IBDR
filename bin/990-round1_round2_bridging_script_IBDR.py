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

# How many of each major population per high non-immune sample?


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
# Checking major population contributions
#####################

