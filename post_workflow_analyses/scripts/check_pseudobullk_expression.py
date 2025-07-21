# Bradley July 2025
# Calculate pseudo-bulk median expression for a given set of genes

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import math

# options
min_cells = 5
min_donors = 30
min_prop_expr = 0.2
method = "Sum"
want_meta = "sex,age"

###############
# Load data
###############
adata = sc.read_h5ad("/lustre/scratch127/humgen/projects_v2/sc-eqtl-ibd/core_analysis_output/IBDverse_multi-tissue_eQTL_project/IBDverse_scRNA/celltypist_0.5_ngene_ncount_mt_filt_nomiss.h5ad")
adata.X = adata.layers['counts']

# Extract the samples x celltypes use to use
cell_counts = adata.obs.groupby(['predicted_labels', 'Genotyping_ID']).size().reset_index(name='n_cells')
valid_pairs = cell_counts[cell_counts['n_cells'] >= min_cells]
valid_group_counts = valid_pairs['predicted_labels'].value_counts()
valid_groups = valid_group_counts[valid_group_counts >= min_donors].index
valid_pairs = valid_pairs[valid_pairs['predicted_labels'].isin(valid_groups)]
valid_combo_set = set(zip(valid_pairs['Genotyping_ID'], valid_pairs['predicted_labels']))

###############
# Pseudobulk
###############
if method == "Mean":
    adata.X = adata.layers['log1p_cp10k']

for label in valid_groups:
    print(f"Processing cell type: {label}")
    # Individuals with enough cells in this label
    individuals = valid_pairs[valid_pairs['predicted_labels'] == label]['Genotyping_ID']
    profiles = []
    indiv_index = []
    min_samps = math.ceil(len(individuals)*min_prop_expr)
    for indiv in individuals:
        # Get expression matrix for this celltype and indivudla
        cell_indices=adata.obs[(adata.obs['Genotyping_ID'].astype(str) == indiv) & (adata.obs['predicted_labels'].astype(str) == label)].index
        X = adata[cell_indices].X
        X_dense = X.toarray() if hasattr(X, 'toarray') else X
        if method == "Mean":
            want_expr = X_dense.mean(axis=0)
        else:
            want_expr = X_dense.sum(axis=0)
        profiles.append(want_expr)
        indiv_index.append(indiv)    
    #Aggregate
    pseudobulk_df = pd.DataFrame(
        np.vstack(profiles),
        index=indiv_index,
        columns=adata.var_names
    )
    # Subset for gene expressed in >20% samples
    pseudobulk_df = pseudobulk_df.loc[:, (pseudobulk_df > 0).sum(axis=0) >= min_samps]
    # Save
    pseudobulk_df.to_csv(f"/lustre/scratch127/humgen/projects_v2/sc-eqtl-ibd/analysis/jingling_analysis/IBDverse_data/pseudobulk/d{method}__{label}.tsv.gz", sep="\t", index=True, compression="gzip")

# Save metadata
meta_list = want_meta.split(",") 
meta_list.append("Genotyping_ID")  
to_save = adata.obs[meta_list].drop_duplicates().reset_index(drop=True).set_index("Genotyping_ID")
to_save.to_csv(f"/lustre/scratch127/humgen/projects_v2/sc-eqtl-ibd/analysis/jingling_analysis/IBDverse_data/pseudobulk/metadata_for_DE.tsv.gz", sep="\t", index=True, compression="gzip")


##################
# Plot dotplot
##################

# Define genes
markers=["IFNG-AS1", "ERAP2", "ETS1", "HOTTIP", "CCL20"]

# Add the labels
conv=pd.read_csv("/lustre/scratch127/humgen/projects_v2/sc-eqtl-ibd/analysis/bradley_analysis/IBDverse/IBDVerse-sc-eQTL-code/data/all_IBDverse_annotation_mastersheet.csv")
conv=conv[['leiden', 'JAMBOREE_ANNOTATION']]
conv.columns=['predicted_labels','Final_labels']
conv['predicted_labels'] = conv['predicted_labels'].astype(str) + '_ct'
conv = conv.dropna(subset=['Final_labels'])
adata.obs = adata.obs.merge(conv, on="predicted_labels", how="left")
adata.obs['category'] = adata.obs['predicted_category'].str.removesuffix('_ct')


# Make sure gene names are unique
adata.var['ENS'] = adata.var_names.copy()
adata.var_names = list(adata.var['gene_symbols'])
adata.var_names_make_unique()
adata.var['gene_symbols_unique'] = adata.var_names.copy()
adata.var.set_index("ENS", inplace=True)

# Set out
sc.settings.figdir="/lustre/scratch127/humgen/projects_v2/sc-eqtl-ibd/analysis/jingling_analysis/IBDverse_data/dotplots"
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=500, facecolor='white', format="png")

# Plot
sc.pl.dotplot(adata, markers, groupby='category',gene_symbols='gene_symbols_unique', dendrogram=False, standard_scale='var', show=True, save=f"_jingling_hits.png") 
sc.pl.dotplot(adata, markers, groupby='Final_labels',gene_symbols='gene_symbols_unique', dendrogram=False, standard_scale='var', show=True, save=f"_jingling_hits_celltype.png") 

# Subset for specific categories:
cats = np.unique(adata.obs['category'])
for c in cats:
    print(f"Plotting: {c}")
    mask = adata.obs['category'] == c
    sc.pl.dotplot(adata[mask,:].copy(), markers, groupby='Final_labels', gene_symbols='gene_symbols_unique', dendrogram=False, standard_scale='var', show=True, save=f"_jingling_markers_{c}.png") 
