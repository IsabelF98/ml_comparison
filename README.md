# Manifold Learning Comparisons w/ fMRI Data
### By Isabel Fernandez NIH/NIMH/SFIM

This repository contains files that are used for the Frontier paper on manifold learning techniques and fMRI data. We will be comparing three different manifold learning techniques, Laplacian Eigenmaps, T-SNE, and UMAP. The techniques are applied to dynamic functional connectivity human brain fMRI data, in hopes of capturing shifts in externally imposed cognitive states through dimensionality reduction. In the paper we explore heuristics for choosing technique hyperparameter values, the intrinsic dimension of the data, and techniques for combining data across subjects.

## Repository Set Up
There are X sections to this repository\
1) Sliding Window Correlation Matrix (SWC_matrix)
2) Laplacian Eigenmap (LE_embedding)
3) T-SNE (TSNE_embedding)
4) UMAP (UMAP_embedding)

## Dynamic Functional Connectivity
We will be computing the sliding window correlation matrix to explore dynamic functional connectivity.\
File Names: SWC_matrix\
Data Output Name: {SUBJECT}_SWC_matrix_wl{WINDOW_LENGTH}.csv

## Laplacian Eigenmap
We will be computing the 3D embedding using the Laplacian Eigenmap algorithum.\
File Names: LE_embedding\
Data Output Name: {SUBJECT}_LE_embedding_wl{WINDOW_LENGTH}_k{k-NN}_n{DIMENSIONS}_{DISTANCE_METRIC}.csv

## T-distribution Stochastic Neighbor Embedding 
We will be computing the 3D embedding using the T-distribution Stochastic Neighbor Embedding algorithum.\
File Names: TSNE_embedding\
Data Output Name: {SUBJECT}_TSNE_embedding_wl{WINDOW_LENGTH}_p{PERPLEXITY}_n{DIMENSIONS}_{DISTANCE_METRIC}.csv

## Uniform Manifold Approximation and Projection Embedding 
We will be computing the 3D embedding using the Uniform Manifold Approximation and Projection algorithum.\
File Names: UMAP_embedding\
Data Output Name: {SUBJECT}_UMAP_embedding_wl{WINDOW_LENGTH}_k{NEIGHBORS}_n{DIMENSIONS}_{DISTANCE_METRIC}.csv

## Silhouette Index
We will be computing the silhouette index for all the embeddings we computed using LE, TSNE, and UMAP. For each subject and for each technique a data frame will be created that contains the silhouette index for a given distance metric and parameter values (either k or p).
File Names: Silhouette_index\
Data Output Name: {SUBJECT}_Silh_Idx_{EMBEDDING_METHOD}_wl{WINDOW_LENGTH}.csv