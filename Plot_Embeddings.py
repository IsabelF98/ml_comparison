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

# # Plot Embeddings
#
# Only plots one embedding at a time.

import pandas as pd
import numpy as np
import os.path as osp
import umap
from sklearn.metrics import silhouette_score
from utils.data_info import PRJDIR, task_labels, SBJ_list, UMAP_k_list, wl_sec, tr
import panel as pn
import plotly.express as px
pn.extension('plotly')

SBJ = 'SBJ06' # Subject number
n = 3 # number of dimensions
drop = 'DropData' # Type of data (FullData: all windows, DropData: only pure windows, DropX: keep every X window)
ndrop = 15 # If using DropX embedding, X value (5, 10, or 15) MUST MATCH X
wl_trs = int(wl_sec/1.5) # Window length in TRs (TR = 1.5 sec)

if drop == 'FullData':
    task_df = task_labels(wl_trs, PURE=False)
elif drop == 'Drop5' or drop == 'Drop10' or drop == 'Drop15':
    print('in')
    task_df = task_labels(wl_trs, PURE=False)
    task_df = task_df.loc[range(0, task_df.shape[0], ndrop)].copy()
elif drop == 'DropData':
    task_df = task_labels(wl_trs, PURE=True)
print(task_df.shape)

# +
# Load LE Embedding
# -----------------
k = 60 # k-NN value
metric = 'correlation' # distance metric

file_name = SBJ+'_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
file_path = osp.join(PRJDIR,'derivatives','LE',file_name)
embed_df  = pd.read_csv(file_path)  
print('++ INFO: LE embedding loaded')
print('         Data shape:',embed_df.shape)

# + jupyter={"outputs_hidden": true, "source_hidden": true}
# Load TSNE Embedding
# -------------------
p = 13
metric = 'correlation'

file_name = SBJ+'_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
file_path = osp.join(PRJDIR,'derivatives','TSNE',file_name)
embed_df  = pd.read_csv(file_path)  
print('++ INFO: TSNE embedding loaded')
print('         Data shape:',embed_df.shape)

# + jupyter={"outputs_hidden": true, "source_hidden": true}
# Load UMAP Embedding
# -------------------
k = 11
metric = 'correlation'

file_name = SBJ+'_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
file_path = osp.join(PRJDIR,'derivatives','UMAP',file_name)
embed_df  = pd.read_csv(file_path)  
print('++ INFO: UMAP embedding loaded')
print('         Data shape:',embed_df.shape)
# -

embed_df.index = task_df.index # Make embeding data frame index and task label data frame index the same

# +
# Plot Embedding
task_cmap = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'}

plot_input = pd.DataFrame(columns=['x','y','z'])
plot_input[['x','y','z']] = embed_df[['1_norm','2_norm','3_norm']].copy()
plot_input['Task'] = task_df['Task']

plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap, width=700, height=600, opacity=0.7)
plot = plot.update_traces(marker=dict(size=3,line=dict(width=0)))

plot
# -

# Compute SI Value
if drop == 'FullData' or drop == 'Drop5' or drop == 'Drop10' or drop == 'Drop15':
    drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
    drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
    drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
    SI = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
elif drop == 'DropData':
    SI = silhouette_score(embed_df[['1_norm', '2_norm', '3_norm']], task_df['Task'].values)
print(SI)


