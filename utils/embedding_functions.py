# Isabel Fernandez 02/22/2022

# This file contains the functions used to compute the lower dimension embeddings
# 1. Laplacian Eigenmap
# 2. T-SNE
# 3. UMAP

from sklearn.manifold  import SpectralEmbedding
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import correlation, cosine, euclidean
import pandas as pd
import numpy as np

# Laplacian Eigenmap Function
# ---------------------------
def Laplacain_Eigenmap(data_df,k,n,metric):
    """
    This function computes the lower dimensional embedding of data using the Laplacian Eigemap algorithum.
    1) Sets up the specrtal embedding transformation
    2) Constructs k-Nearest Neighboor affinity matrix
    3) Normalizes affinity matrix (restricted to only 1's and 0's)
    4) Transform data
    5) Normalize transofred data
    
    INPUTS
    ------
    data_df: (pd.DataFrame) data which you are dimensionally reducing
    k: (int) k value for k-Nearest Neighboor affinity matrix
    n: (int) number of dimensions you are reducing your data to
    metric: (str) distance metric you are using to construct affinity matrix (correlation, cosine, euclidean)
    
    OUTPUTS
    -------
    LE_data: (pd.DataFrame) dimensionaly reduced data (both original and normalized)
             Columns are labeld as dimesnsion number X. Normalized dimensions are label as 'X_norm'.
    """
    
    # Compute Embedding
    seed             = np.random.RandomState(seed=3) # Initialization seed
    embedding        = SpectralEmbedding(n_components=n, affinity='precomputed', n_jobs=32, random_state=seed) # Transformation
    X_affinity       = kneighbors_graph(data_df, n_neighbors=k, include_self=True, metric=metric, n_jobs=52) # Affinity matrix
    X_affinity       = 0.5 * (X_affinity + X_affinity.T) # Normalized affinity
    data_transformed = embedding.fit_transform(X_affinity.toarray()) # Transform data

    # Embedding data frame
    LE_data = pd.DataFrame(columns=[[str(i) for i in range(1,n+1)]]) # Empty data frame
    # Add LE data to data frame
    for i in range(1,n+1):
        LE_data[str(i)] = data_transformed[:,i-1]

    # Normalize data
    LE_data[[str(i)+'_norm' for i in range(1,n+1)]] = LE_data[[str(i) for i in range(1,n+1)]]/LE_data[[str(i) for i in range(1,n+1)]].max()

    return LE_data
