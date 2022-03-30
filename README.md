## Manifold Learning Comparisons w/ fMRI Data
### By Isabel Fernandez NIH/NIMH/SFIM

This repository contains files that are used for the Frontier paper on manifold learning techniques and fMRI data. We will be comparing three different manifold learning techniques, Laplacian Eigenmaps, T-SNE, and UMAP. The techniques are applied to dynamic functional connectivity human brain fMRI data, in hopes of capturing shifts in externally imposed cognitive states through dimensionality reduction. In the paper we explore heuristics for choosing technique hyperparameter values, the intrinsic dimension of the data, and techniques for combining data across subjects.

## Repository Set Up
There are X sections to this repository
1) Dynamic Functional Connectivity
2) Laplacian Eigenmap
3) T-SNE
4) UMAP


## Dynamic Functional Connectivity
We will be computing the sliding window correlation matrix to explore dynamic functional connectivity.\
File Names: SWC_matrix\
Data Output Name: {SUBJECT}_SWC_matrix_wl{WINDOW_LENGTH}.csv

### Method
1) Load a given subjects ROI time series of the data. The data should be saved as a pandas data frame *(time points x ROIs)*. For our data, each subject time series has 1017 time points and 157 ROIs.
2) We then compute the sliding window correlation SWC matrix of our ROI time series. The function that computes the SWC matrix *compute_SWC()* take in as input the ROI time series data frame (loaded in step 1), the window length in TRs, the window space in TRs (how many TRs between windows), and the window type. For our data we use a window length of 45 sec (30 TRs since our TR=1.5 sec), a window space of 1 TR, and a uniform window using *np.ones()*. The SWC matrix is stored as a pandas data frame *(windows x connections)*. For our data, each matrix had 988 windows and 12246 connections.
3) The SWC matrix is then saved as a csv file in the *derivatives/SWC/* directory.\

Note: These files should be computed before futher analysis. The following anlysis relies on this data.


## Laplacian Eigenmap
We will be computing the 3D embedding using the Laplacian Eigenmap algorithum.\
File Names: LE_embedding\
Data Output Name: {SUBJECT}_LE_embedding_wl{WINDOW_LENGTH}_k{k-NN}_n{DIMENSIONS}_{DISTANCE_METRIC}.csv

### Method
1) Load a given subjects SWC matrix as a pandas data frame *(windows x connections)* from the *derivatives/SWC/*.
2) Drop any windows that are between tasks. How the SWC matrix is computed, there are windows that will overlap more than one task. We remove these windows, so we focus only on pure task windows. This will bring the number of windows down. For our data, we removed 259 between task windows, and are left with 729 windows.
3) We then compute or low dimensional embedding using the function *Laplacain_Eigenmap()*. The function takes as input the SWC matrix (after the windows have been dropped in step 2), k (the value for the k-Nearest Neighbor algorithm), n (the number of dimensions the data will be reduced to), and the distance metric. For our data, we compute the embedding over a range of k values, reduce our data down to 3 dimensions, and we use three distance metrics (correlation, cosine, and Euclidean). The embedding is stored as a pandas data frame *(windows x dimensions/normalized dimensions)*. The data frame also has the accompanying normalized dimensions for each dimension computed (labeled *X_norm* where *X* is the dimension number). For our data, each data frame had 729 windows, 3 dimensions and 3 normalized dimensions.
4) The embedding is then saved as a csv file in the *derivatives/LE/* directory.\

Note: The Laplacian Eigenmap function is derived from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html?highlight=spectral%20embedding#sklearn.manifold.SpectralEmbedding).


## T-distribution Stochastic Neighbor Embedding 
We will be computing the 3D embedding using the T-distribution Stochastic Neighbor Embedding algorithum.\
File Names: TSNE_embedding\
Data Output Name: {SUBJECT}_TSNE_embedding_wl{WINDOW_LENGTH}_p{PERPLEXITY}_n{DIMENSIONS}_{DISTANCE_METRIC}.csv

### Method
1) Load a given subjects SWC matrix as a pandas data frame *(windows x connections)* from the *derivatives/SWC/*.
2) Drop any windows that are between tasks. How the SWC matrix is computed, there are windows that will overlap more than one task. We remove these windows, so we focus only on pure task windows. This will bring the number of windows down. For our data, we removed 259 between task windows, and are left with 729 windows.
3) We then compute or low dimensional embedding using the function *T_Stochastic_Neighbor_Embedding()*. The function takes as input the SWC matrix (after the windows have been dropped in step 2), p (the perplexity value), n (the number of dimensions the data will be reduced to), and the distance metric. For our data, we compute the embedding over a range of perplexity values, reduce our data down to 3 dimensions, and we use three distance metrics (correlation, cosine, and Euclidean). The embedding is stored as a pandas data frame *(windows x dimensions/normalized dimensions)*. The data frame also has the accompanying normalized dimensions for each dimension computed (labeled *X_norm* where *X* is the dimension number). For our data, each data frame had 729 windows, 3 dimensions and 3 normalized dimensions.
4) The embedding is then saved as a csv file in the *derivatives/TSNE/* directory.\

Note: The TSNE function is derived from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).


## Uniform Manifold Approximation and Projection Embedding 
We will be computing the 3D embedding using the Uniform Manifold Approximation and Projection algorithum.\
File Names: UMAP_embedding\
Data Output Name: {SUBJECT}_UMAP_embedding_wl{WINDOW_LENGTH}_k{NEIGHBORS}_n{DIMENSIONS}_{DISTANCE_METRIC}.csv

### Method
1) Load a given subjects SWC matrix as a pandas data frame *(windows x connections)* from the *derivatives/SWC/*.
2) Drop any windows that are between tasks. How the SWC matrix is computed, there are windows that will overlap more than one task. We remove these windows, so we focus only on pure task windows. This will bring the number of windows down. For our data, we removed 259 between task windows, and are left with 729 windows.
3) We then compute or low dimensional embedding using the function *Uniform_Manifold_Approximation_Projection()*. The function takes as input the SWC matrix (after the windows have been dropped in step 2), k (the value for the k-Nearest Neighbor algorithm), n (the number of dimensions the data will be reduced to), and the distance metric. For our data, we compute the embedding over a range of k values, reduce our data down to 3 dimensions, and we use three distance metrics (correlation, cosine, and Euclidean). The embedding is stored as a pandas data frame *(windows x dimensions/normalized dimensions)*. The data frame also has the accompanying normalized dimensions for each dimension computed (labeled *X_norm* where *X* is the dimension number). For our data, each data frame had 729 windows, 3 dimensions and 3 normalized dimensions.
4) The embedding is then saved as a csv file in the *derivatives/UMAP/* directory.\

Note: The UMAP function is derived from the original UMAP function on [github](https://github.com/lmcinnes/umap).


## Silhouette Index
We will be computing the silhouette index for all the embeddings we computed using LE, TSNE, and UMAP. For each subject and for each technique a data frame will be created that contains the silhouette index for a given distance metric and parameter values (either k or p).
File Names: Silhouette_index\
Data Output Name: {SUBJECT}_Silh_Idx_{EMBEDDING_METHOD}_wl{WINDOW_LENGTH}.csv

### Method
1) The user must first decide which embedding technique they want to evaluate (LE, TSNE, or UMAP).
2) Load a given subjects 3D embedding as a pandas data frame *(windows x dimensions/normalized dimensions)* for the selected embedding technique, parameter value, and distance metric. For our data, each data frame had 729 windows, 3 dimensions and 3 normalized dimensions. An embedding data frame is loaded based on parameter value (k-Nearest Neighbor or perplexity) and distance metric (correlation, cosine, and Euclidean).
3) The task labels are loaded as a pandas data frame *(windows x task labels)*. There should be the same number of labels as windows in the embedding data and one column for labels.
4) The silhouette index is computed for a single 3D embedding using the *silhouette_score()* function. The input data is the 3 normalized dimensions and the task data frame labels.
5) The silhouette indices are saved in a pandas data frame with index as the parameter value and columns as the distance matrix. For example, for a given subject and embedding technique, the silhouette index at index *55* and column *correlation* is the silhouette index for the embedding using the parameter value 55 and correlation distance metric. The dimensions of these data frames will be different for each embedding technique.
6) The embedding is then saved as a csv file in the *derivatives/Silh_Idx/* directory.\

Note: The silhouette score function is from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).


## Null Data
To test how well the techniques work, we will be "breaking" the data to see how the techniques perform. By breaking the data, we hope that the techniques will no longer cluster the data based on task and lead to low silhouette indices. We will compute the broken data (or the null data as we will refer to it) using two different methods. Method 1 is to shuffle the connections on for each window. Method 2 is to change the phase of the data for each window.\
File Names: Null_embedding\
Data Output Name: {SUBJECT}_{EMBEDDING}{NULL}_embedding_wl{WINDOW_LENGTH}_k/p{NEIGHBORS}_n{DIMENSIONS}_{DISTANCE_METRIC}.csv

### Method
1) The user must first decide which parameter values they wish to use for each embedding. We used the parameter values found to perform best from the previous section (Laplacian Eigenmap: k=50, metric=correlation, TSNE: p=55, metric=correlation, UMAP: k=130, metric=correlation).
2) Load a given subjects SWC matrix as a pandas data frame *(windows x connections)* from the *derivatives/SWC/*.
3) Drop any windows that are between tasks. How the SWC matrix is computed, there are windows that will overlap more than one task. We remove these windows, so we focus only on pure task windows. This will bring the number of windows down. For our data, we removed 259 between task windows, and are left with 729 windows.
4) Compute the null data using method 1 or 2. The data should be the same shape as the original SWC matrix *(windows x connections)*.
5) We then compute all three embeddings with the hyperparameters defined in step 1. You are left with three embeddings, all with dimensions *(windows x dimensions)*. For our analysis we reduced the data down to 3 dimensions (n=3).
6) Each embedding is saved as a csv file in the *derivatives/Null_Data/* directory.
7) In the notebook Null_embedding.Plot.ipynb we plot a bar plot of the average silhouette indices over all subjects for each technique and null method to compare data types.

Note: The shuffle function is from [numpy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html) and method 2 is defined in data_functions.py.