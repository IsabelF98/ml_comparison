## Manifold Learning Comparisons w/ fMRI Data
### By Isabel Fernandez NIH/NIMH/SFIM

This repository contains files that are used for the SFIM paper on manifold learning techniques and dFC fMRI data. We will be comparing three different manifold learning techniques, Laplacian Eigenmaps, T-SNE, and UMAP. The techniques are applied to dynamic functional connectivity (dFC) human brain fMRI data, in hopes of capturing shifts in externally imposed cognitive states through dimensionality reduction. In the paper we explore heuristics for choosing technique hyperparameter values, the intrinsic dimension of the data, and techniques for combining data across subjects.

## Repository Set Up
There are 8 sections to this repository
1) Dynamic Functional Connectivity
2) Laplacian Eigenmap
3) T-SNE
4) UMAP
5) Silhouette Index
6) Logistic Regression
7) Null Data
8) Dropping Windows

For those sections that have multiple files, the python file is where the main computation takes place. The sh file is to call the correct python environment. And the CreateSwarm.ipynb file is to create the SWARM file for the spersist node, as well as creates the directory for the outputs and logs. Stand alone notebooks, such as break_the_data.ipynb and Plot_Embeddings.ipynb are notebooks secific for certain plots. All files in utils are universal parameters and functions, such as the embedding functions.

## 1. Dynamic Functional Connectivity
Here we compute the sliding window correlation matrix to explore dynamic functional connectivity. The data is outputed as a csv file (windows X connections).\
File Names: SWC_matrix\
Output directory: derivatives/SWC/\
Output file names: {SUBJECT}_SWC_matrix_wl{WINDOW_LENGTH}_FullData.csv

### Parameters
* SBJ: the subject number
* wl_sec: the window lenth in seconds
* tr: the TR
* ws_trs: window spacing in TR's

Note: These files should be computed before futher analysis. The following anlysis relies on this data.


## 2. Laplacian Eigenmap
Here we compute the 3D embedding using the Laplacian Eigenmap algorithum. The data is outputed as a csv file (windows X dimensions).\
File Names: LE_embedding\
Output directory: derivatives/LE/\
Data Output Name: {SUBJECT}_LE_embedding_wl{WINDOW_LENGTH}_k{k-NN}_n{DIMENSIONS}_{DISTANCE_METRIC}_{DROP}.csv

### Parameters
* SBJ: the subject number
* wl_sec: the window lenth in seconds
* tr: the TR
* k: k-NN value
* n: number of dimensions to reduce to
* metric: distance metric
* drop: Type of data (FullData: all windows, DropData: only pure windows, DropX: keep every X window)

Note: The Laplacian Eigenmap function is derived from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html?highlight=spectral%20embedding#sklearn.manifold.SpectralEmbedding).


## 3. T-distribution Stochastic Neighbor Embedding 
Here we compute the 3D embedding using the T-distribution Stochastic Neighbor Embedding algorithum. The data is outputed as a csv file (windows X dimensions).\
File Names: TSNE_embedding\
Output directory: derivatives/TSNE/\
Data Output Name: {SUBJECT}_TSNE_embedding_wl{WINDOW_LENGTH}_p{PERPLEXITY}_n{DIMENSIONS}_{DISTANCE_METRIC}_{DROP}.csv

### Parameters
* SBJ: the subject number
* wl_sec: the window lenth in seconds
* tr: the TR
* p: perplexity value
* n: number of dimensions to reduce to
* metric: distance metric
* drop: Type of data (FullData: all windows, DropData: only pure windows, DropX: keep every X window)

Note: The TSNE function is derived from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).


## 4. Uniform Manifold Approximation and Projection Embedding 
We will be computing the 3D embedding using the Uniform Manifold Approximation and Projection algorithum. The data is outputed as a csv file (windows X dimensions).\
File Names: UMAP_embedding\
Output directory: derivatives/UMAP/\
Data Output Name: {SUBJECT}_UMAP_embedding_wl{WINDOW_LENGTH}_k{NEIGHBORS}_n{DIMENSIONS}_{DISTANCE_METRIC}_{DROP}.csv

### Parameters
* SBJ: the subject number
* wl_sec: the window lenth in seconds
* tr: the TR
* k: k-NN value
* n: number of dimensions to reduce to
* metric: distance metric
* drop: Type of data (FullData: all windows, DropData: only pure windows, DropX: keep every X window)

Note: The UMAP function is derived from the original UMAP function on [github](https://github.com/lmcinnes/umap).


## 5. Silhouette Index
We will be computing the silhouette index for all the embeddings we computed using LE, TSNE, and UMAP. For each subject and for each technique a data frame will be created that contains the silhouette index for a given distance metric and parameter values (either k-NN or perplexity). The data is outputed as a csv file (k or p values X distance metrics).\
File Names: Silhouette_index\
Output directory: derivatives/Silh_Idx/\
Data Output Name: {SUBJECT}_Silh_Idx_{EMBEDDING_METHOD}_wl{WINDOW_LENGTH}_{DROP}.csv

### Parameters
* SBJ: the subject number
* wl_sec: the window lenth in seconds
* tr: the TR
* embedding: Embedding technique (LE, TSNE, or UMAP)
* drop: Type of data (FullData: all windows, DropData: only pure windows, DropX: keep every X window)

Note: The silhouette score function is from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

## 6. Logistic Regression
To test how well the techniques work at predicting task labels, we will be using the Logistic Regression classifier. We will be splitting the data into a training and testing set. The classifier generates a set of coefficients for each label (i.e., one label for each task, in our case the four tasks) which we will use to predict the labels of the test data set. To split the data into a training and testing set we simply use the first half of the run as the training set and the second half as the testing set. We only look at the F1 accuracy score to determin how well the the predictive framework worked for now. The outputs are tw csv files. One being logistic regression coeffieients (dimesnions X labels). And the other being the classification report.\
File Names: Logistic_Regression\
Output directory: derivatives/Log_Reg/\
Data Output Name: {SUBJECT}_{EMBEDDING_METHOD}_LRcoef_wl{WINDOW_LENGTH}_k/p{NEIGHBORS}_n{DIMENSIONS}_{DISTANCE_METRIC}_{DROP}.csv (Logistic Regression Coefficients) and {SUBJECT}_{EMBEDDING_METHOD}_LRclassrep_wl{WINDOW_LENGTH}_k/p{NEIGHBORS}_n{DIMENSIONS}_{DISTANCE_METRIC}_{DROP}.csv (Classification Report)

### Parameters
* SBJ: the subject number
* wl_sec: the window lenth in seconds
* tr: the TR
* kp: k-NN or perplexity value
* n: number of embedding dimesnions
* metric: distance metric
* embedding: Embedding technique (LE, TSNE, or UMAP)
* drop: Type of data (FullData: all windows, DropData: only pure windows) NOTE: This file is not coded for DropX data

Note: The Logistic Regression function is from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

## 7. Null Data
To test how well the techniques work, we will be "breaking" the data to see how the techniques perform. By breaking the data, we hope that the techniques will no longer cluster the data based on task and lead to low silhouette indices and F1 accuracy scores. We will compute the broken data (or the null data as we will refer to it) using two different methods. Method 1 is to randomize the phase of the data for each window. Method 2 is to shuffle the connections for each window. The data outputed are the actual null embeddings (not the null ROI time series or SWC matrix) saved as csv files (windows X dimensions).\
File Names: Null_embedding\
Output directory: derivatives/Null_Data/\
Data Output Name: {SUBJECT}_{DATA}_{EMBEDDING}{NULL}_embedding_wl{WINDOW_LENGTH}_k/p{NEIGHBORS}_n{DIMENSIONS}_{DISTANCE_METRIC}_{DROP}.csv

### Parameters
* SBJ: the subject number
* wl_sec: the window lenth in seconds
* tr: the TR
* ws_trs: window spacing in TR's
* LE_k: Laplacian Eigenmap k-NN value
* LE_metric: Laplacian Eigenmap distance metric
* p: TSNE perplexity value
* TSNE_metric: TSNE distance metric
* UMAP_k: UMAP k-NN value
* UMAP_metric: UMAP distance metric
* n: number of dimensions to reduce to
* data: the data wich you want to break (ROI or SWC)
* drop: Type of data (FullData: all windows, DropData: only pure windows) NOTE: This file is not coded for DropX data

Note: The shuffle function is from [numpy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html) and method 2 is defined in data_functions.py.


## 8. Drop Windows
To test if the temporal correlation in our data effects the data, we drop winows in our SWC matrix in hopes of removing some of the temporal correlation. The file takes already existing SWC matricies and keeps ever X windows (look at slide 2 of Dropping_Windows.pptx on teams for a visual represntiation). The data is saved as a csv file same as the original SWC matrix (windows X connections) except there are less windows. For our analysis we looked mostly at keeping ever 5, 10, and 15 windows.\
File Names: Drop_Data\
Output directory: derivatives/SWC/\
Output file names: {SUBJECT}_SWC_matrix_wl{WINDOW_LENGTH}_{DROP}.csv

### Parameters
* SBJ: the subject number
* wl_sec: the window lenth in seconds
* tr: the TR
* n_drop: the windows index to keep (ex, if n_drop=5 keep every 5 window, if n_drop=1 its the full data set)

Note: If you want to compute the embeddings for these SWC matrix you must run these fiels first and the run the embedding with *drop=DropX*. 