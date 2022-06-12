# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Manifold 2
#     language: python
#     name: manifold2
# ---

# # Dropping Windows Analysis
#
# * This notebook was used for testing different methods of dropping windows to remove autocorrelation.
# * Single run drop data: Computes UMAP embedding with dropped SWC matrix for a single run.
# * Split run drop data: Only drops windows in the second half of a run and then computes UMAP embedding for the entier data.
# * Same number of windows two runs: Dropping the windows for a single run to remove temporal correlation (same as "single run drop data") and dropping the windows for the same run to keep temporal correlation with the same number of windows as the first run. So there are two runs, but they come from the same SWC matrix, they both have the same number of windows, but one is to remove tempral correlation and the other is to keep it.
# * UMAP affinity matrix: Plotting the affinity matrix used in UMAP.
# * LE affinity matrix: Plotting the affinity matrix used in UMAP.
# * The results for this notbook can be found in Dropping_Windows.pptx and KC_FP_meeting_analysis.pptx (mostly this powerpoint).

import pandas as pd
import numpy as np
import os.path as osp
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import correlation, cosine, euclidean
from utils.embedding_functions import Laplacain_Eigenmap, Uniform_Manifold_Approximation_Projection
from utils.data_info import DATADIR, PRJDIR, load_task_ROI_TS, task_labels, SBJ_list, LE_k_list, wl_sec, tr, ws_trs
from utils.data_functions import compute_SWC, randomize_ROI
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import seaborn as sns
import panel as pn
import plotly.express as px
pn.extension('plotly')

# ## Single run drop data
# ***

SBJ = 'SBJ25' # Subject number
n_drop = 5 # Number of windows to drop

# Load full SWC data
file_name = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'_FullData.csv'
file_path = osp.join(PRJDIR,'derivatives','SWC',file_name)
SWC_df    = pd.read_csv(file_path)  
print(SWC_df.shape)

# Task labels
task_df = task_labels(int(wl_sec/tr), PURE=False)
print(task_df.shape)

# Drop windows (keep ever n windows)
SWC_df  = SWC_df.loc[range(0, SWC_df.shape[0], n_drop)].copy()
task_df = task_df.loc[range(0, task_df.shape[0], n_drop)].copy()
print(SWC_df.shape)
print(task_df.shape)

# Compute UMAP embedding with dropped SWC matrix
embed_df = Uniform_Manifold_Approximation_Projection(SWC_df,k=155,n=3,metric='correlation')
print(embed_df.shape)

embed_df.index=task_df.index # Make index the same for embedding and task labels

# +
# Plot embeddings
task_cmap = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'}

plot_input = pd.DataFrame(columns=['x','y','z'])
plot_input[['x','y','z']] = embed_df[['1_norm','2_norm','3_norm']].copy()
plot_input['Task'] = task_df['Task']

plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap, width=700, height=600, opacity=0.7)
plot = plot.update_traces(marker=dict(size=3,line=dict(width=0)))

plot
# -

# Compute SI for embedding using only pure windows
drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)

# ## Split run drop data
# ***

SBJ = 'SBJ25' # Subject number
n_drop1 = 1 # number of windows to drop for first half of run (ndrop=1 means no windows dropped)
n_drop2 = 15 # number of windows to drop for second half of run

# Load full SWC matrix
file_name = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'_FullData.csv'
file_path = osp.join(PRJDIR,'derivatives','SWC',file_name)
SWC_df    = pd.read_csv(file_path)  
print(SWC_df.shape)

# Task labels
task_df = task_labels(int(wl_sec/tr), PURE=False)
print(task_df.shape)

# Drop windows for first and second half of run
SWC_df  = pd.concat([SWC_df.loc[range(0, int(SWC_df.shape[0]/2), n_drop1)].copy(), SWC_df.loc[range(int(SWC_df.shape[0]/2), SWC_df.shape[0], n_drop2)].copy()])
task_df = pd.concat([task_df.loc[range(0, int(task_df.shape[0]/2), n_drop1)].copy(), task_df.loc[range(int(task_df.shape[0]/2), task_df.shape[0], n_drop2)].copy()])
print(SWC_df.shape)
print(task_df.shape)

# Compute embedding for dropped SWC
embed_df = Uniform_Manifold_Approximation_Projection(SWC_df,k=155,n=3,metric='correlation')
print(embed_df.shape)

embed_df.index=task_df.index # Make index the same for embedding and task labels

# +
# Plot embedding
task_cmap = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'}

plot_input = pd.DataFrame(columns=['x','y','z'])
plot_input[['x','y','z']] = embed_df[['1_norm','2_norm','3_norm']].copy()
plot_input['Task'] = task_df['Task']

plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap, width=700, height=600, opacity=0.7)
plot = plot.update_traces(marker=dict(size=3,line=dict(width=0)))

plot
# -

# Compute SI using only pure windows
drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)

# ## Same number of windows two runs
# ***

SBJ = 'SBJ07' # Subject number
n_drop = 15 # number of windows to drop

# Load full SWC matrix
file_name = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'_FullData.csv'
file_path = osp.join(PRJDIR,'derivatives','SWC',file_name)
SWC_df    = pd.read_csv(file_path)  
print(SWC_df.shape)

# Task labels
task_df = task_labels(int(wl_sec/tr), PURE=False)
print(task_df.shape)

# Run 1 which drops windows like we did before (remove temporal correlation)
SWC_drop_df  = SWC_df.loc[range(0, SWC_df.shape[0], n_drop)].copy()
task_drop_df = task_df.loc[range(0, task_df.shape[0], n_drop)].copy()
print(SWC_drop_df.shape)
print(task_drop_df.shape)

# Run 2 which drops same number of windows but keeps temporal correlation (Drop 5)
SWC_df  = pd.concat([SWC_df.loc[range(0,20)],SWC_df.loc[range(123,148)],SWC_df.loc[range(251,277)],SWC_df.loc[range(379,404)],SWC_df.loc[range(507,532)],
                     SWC_df.loc[range(635,661)],SWC_df.loc[range(763,789)],SWC_df.loc[range(891,916)]])
task_df = pd.concat([task_df.loc[range(0,20)],task_df.loc[range(123,148)],task_df.loc[range(251,277)],task_df.loc[range(379,404)],task_df.loc[range(507,532)],
                     task_df.loc[range(635,661)],task_df.loc[range(763,789)],task_df.loc[range(891,916)]])
print(SWC_df.shape)
print(task_df.shape)

# Run 2 which drops same number of windows but keeps temporal correlation (Drop 15)
SWC_df  = pd.concat([SWC_df.loc[range(0,7)],SWC_df.loc[range(126,135)],SWC_df.loc[range(254,263)],SWC_df.loc[range(382,391)],SWC_df.loc[range(510,518)],
                     SWC_df.loc[range(638,646)],SWC_df.loc[range(766,774)],SWC_df.loc[range(894,902)]])
task_df = pd.concat([task_df.loc[range(0,7)],task_df.loc[range(126,135)],task_df.loc[range(254,263)],task_df.loc[range(382,391)],task_df.loc[range(510,518)],
                     task_df.loc[range(638,646)],task_df.loc[range(766,774)],task_df.loc[range(894,902)]])
print(SWC_df.shape)
print(task_df.shape)

# Compute UMAP embedding for run 1
embed_drop_df = Uniform_Manifold_Approximation_Projection(SWC_drop_df,k=5,n=3,metric='correlation')
print(embed_drop_df.shape)

# Compute UMAP embedding for run 2
embed_df = Uniform_Manifold_Approximation_Projection(SWC_df,k=5,n=3,metric='correlation')
print(embed_df.shape)

# Make index the same for embedding and task labels
embed_df.index=task_df.index
embed_drop_df.index=task_drop_df.index

# +
# Plot both run embeddings
task_cmap = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'}

plot_input = pd.DataFrame(columns=['x','y','z'])
plot_input[['x','y','z']] = embed_df[['1_norm','2_norm','3_norm']].copy()
plot_input['Task'] = task_df['Task']

plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap, width=700, height=600, opacity=0.7)
plot = plot.update_traces(marker=dict(size=3,line=dict(width=0)))

drop_plot_input = pd.DataFrame(columns=['x','y','z'])
drop_plot_input[['x','y','z']] = embed_drop_df[['1_norm','2_norm','3_norm']].copy()
drop_plot_input['Task'] = task_drop_df['Task']

drop_plot = px.scatter_3d(drop_plot_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap, width=700, height=600, opacity=0.7)
drop_plot = drop_plot.update_traces(marker=dict(size=3,line=dict(width=0)))

pn.Row(plot, drop_plot)
# -

# Compute SI using pure windows for run 1
drop_index1    = task_df.index[task_df['Task'] == 'Inbetween']
drop_embed_df1 = embed_df.drop(drop_index1).reset_index(drop=True)
drop_task_df1  = task_df.drop(drop_index1).reset_index(drop=True)
silhouette_score(drop_embed_df1[['1_norm', '2_norm', '3_norm']], drop_task_df1['Task'].values)

# Compute SI using pure windows for run 2
drop_index2    = task_drop_df.index[task_drop_df['Task'] == 'Inbetween']
drop_embed_df2 = embed_drop_df.drop(drop_index2).reset_index(drop=True)
drop_task_df2  = task_drop_df.drop(drop_index2).reset_index(drop=True)
silhouette_score(drop_embed_df2[['1_norm', '2_norm', '3_norm']], drop_task_df2['Task'].values)

# ## UMAP Affinity Matrix
# ***

import umap
from umap.umap_ import nearest_neighbors

SBJ = 'SBJ06' # Subject number
n_drop = 5 # Number of windows to drop

# Load full SWC matrix
file_name = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'_FullData.csv'
file_path = osp.join(PRJDIR,'derivatives','SWC',file_name)
SWC_df    = pd.read_csv(file_path)  
print(SWC_df.shape)

# Task labesl
task_df = task_labels(int(wl_sec/tr), PURE=False)
print(task_df.shape)

# Drop windows (keep ever n windows)
SWC_df  = SWC_df.loc[range(0, SWC_df.shape[0], n_drop)].copy()
task_df = task_df.loc[range(0, task_df.shape[0], n_drop)].copy()
print(SWC_df.shape)
print(task_df.shape)

k = 160 # k-NN value
n = 3 # number of dimensions
metric = 'euclidean' # distance metric
seed = np.random.RandomState(seed=3) # random seed
random_state = check_random_state(seed) # chec random state (spacific for UMAP, not really sure what this does)

# Pre-computted K-NN
knn_indices, knn_dists, knn_search_index = nearest_neighbors(np.array(SWC_df),
                                                             n_neighbors=k,
                                                             metric=metric,
                                                             metric_kwds=None,
                                                             angular=False,
                                                             random_state=random_state
                                                            )

# Create k-NN matrix from knn_indices
knn_matrix = np.zeros((SWC_df.shape[0],SWC_df.shape[0]))+1
for i in range(knn_matrix.shape[0]):
    ind  = knn_indices[i]
    dist = knn_dists[i]
    for j in range(knn_matrix.shape[1]):
        if j in ind:
            knn_matrix[i,j] = 0 # make neighbor value 0

# Plot non-symmetric affinity matrix
fig, ax = plt.subplots(1, 1, figsize=(10,8))
sns.heatmap(knn_matrix, cmap='gray')

# Symmetrize affinity matrix (B = A + AT - A * AT)
sym_knn_matrix = knn_matrix + knn_matrix.T - np.multiply(knn_matrix,knn_matrix.T)

# Plot symmetric affinity matrix
fig, ax = plt.subplots(1, 1, figsize=(10,8))
sns.heatmap(sym_knn_matrix, cmap='gray')

# Normal UMAP (can't use precomputed k-nn, too small data set)
normal_umap = umap.UMAP(n_components=n,
                        n_neighbors=k,
                        min_dist=0.1,
                        metric=metric,
                        random_state=seed
                       ).fit_transform(SWC_df)

# +
# Plot embedding
task_cmap = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'}

plot_input = pd.DataFrame(normal_umap, columns=['x','y','z'])
task_df.index = plot_input.index
plot_input['Task'] = task_df['Task']

plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap, width=700, height=600, opacity=0.7)
plot = plot.update_traces(marker=dict(size=3,line=dict(width=0)))

plot
# -

# ## LE Matrix
# ***

from sklearn.neighbors import kneighbors_graph
from scipy.spatial import distance_matrix

SBJ = 'SBJ06' # Subject number
n_drop = 15 # Number of winows to drop

k = 80 # k-NN value
n = 3 # Number of dimensions
metric = 'correlation' # distance metric
dist_metric_dict = {'correlation':correlation, 'cosine':cosine, 'euclidean':euclidean} # dictionary for distance metrics

# Load full SWC matrix
file_name = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'_FullData.csv'
file_path = osp.join(PRJDIR,'derivatives','SWC',file_name)
SWC_df    = pd.read_csv(file_path)  
print(SWC_df.shape)

# Task labels
task_df = task_labels(int(wl_sec/tr), PURE=False)
print(task_df.shape)

# Drop windows (keep ever n windows)
SWC_df  = SWC_df.loc[range(0, SWC_df.shape[0], n_drop)].copy()
task_df = task_df.loc[range(0, task_df.shape[0], n_drop)].copy()
print(SWC_df.shape)
print(task_df.shape)

# Compute LE embedding and affinity matrix
# NOTE: The Laplacain_Eigenmap function in utils/embedding_functions.py does not output X-affinity.
#       You must change the function manually to output both X_affinity, LE_df
X_affinity, LE_df = Laplacain_Eigenmap(SWC_df,k=k,n=n,metric=dist_metric_dict[metric])
print(LE_df.shape)

# Plot symmetric affinity matrix
fig, ax = plt.subplots(1, 1, figsize=(10,8))
sns.heatmap(X_affinity.toarray(), cmap='binary')

# +
# Plot embedding
task_cmap = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'}

LE_df.index = task_df.index
plot_input = pd.DataFrame(columns=['x','y','z'])
plot_input[['x','y','z']] = LE_df[['1_norm','2_norm','3_norm']].copy()
plot_input['Task'] = task_df['Task']

plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap, width=700, height=600, opacity=0.7)
plot = plot.update_traces(marker=dict(size=3,line=dict(width=0)))

plot
# -


