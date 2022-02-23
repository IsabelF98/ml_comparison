# Manifold Learning Comparisons w/ fMRI Data
### By Isabel Fernandez NIH/NIMH/SFIM

This repository contains files that are used for the Frontier paper on manifold learning techniques and fMRI data. We will be comparing three different manifold learning techniques, Laplacian Eigenmaps, T-SNE, and UMAP. The techniques are applied to dynamic functional connectivity human brain fMRI data, in hopes of capturing shifts in externally imposed cognitive states through dimensionality reduction. In the paper we explore heuristics for choosing technique hyperparameter values, the intrinsic dimension of the data, and techniques for combining data across subjects.

## Repository Set Up
***
There are X sections to this repository\
1) Sliding Window Correlation Matrix (SWC_matrix)
2) Laplacian Eigenmap (LE_embedding)
3) T-SNE (TSNE_embedding)
4) UMAP (UMAP_embedding)

## Dynamic Functional Connectivity
***
We will be computing the sliding window correlation matrix to explore dynamic functional connectivity.\
File Names: SWC_matrix\
Data Output Name: {SUBJECT}_SWC_matrix_wl{WINDOW_LENGTH}.csv\