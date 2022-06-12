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

# # Null Data Plots
#
# * This notebook plots the null data embeddings along with the original data embeddings.
# * The null data is computed in the Null_embeddings.py script.
# * There are two null data types. Null data 1 is the phase randomized ROI times series embeddings and null data 2 is the randomized SWC connections embeddings.
# * Null embeddings are available for FullData (all wondows) or DropData (only pure windows)
# * These plots can be found in Null_Data.pptx (DropData) and Null_Data_V2.pptx (FullData) on Teams.

import pandas as pd
import numpy as np
import os.path as osp
from sklearn.metrics import silhouette_score
from utils.data_info import PRJDIR, LE_k_list, p_list, UMAP_k_list, task_labels, tr, wl_sec, ws_trs
import panel as pn
import plotly.express as px
pn.extension('plotly')

task_df = task_labels(int(wl_sec/tr), PURE=False) # Task data frame

# +
SBJ = 'SBJ26' # Subject number
emb = 'UMAP' # Embedding type (LE, TSNE, or UMAP)
k = 160 # k-NN or perplexity value
n = 3 # Number of dimensions
metric = 'euclidean' # Distance metric
drop = 'FullData' # Full data or drop data (only pure windows)

# Load Original Data
orig_file_name = SBJ+'_'+emb+'_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
orig_file_path = osp.join(PRJDIR,'derivatives',emb,orig_file_name)
Orig_df        = pd.read_csv(orig_file_path) 

# Load Null Data 1 (ROI)
null1_file_name = SBJ+'_ROI_Null_'+emb+'_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
null1_file_path = osp.join(PRJDIR,'derivatives','Null_Data',null1_file_name)
Null1_df        = pd.read_csv(null1_file_path)

# Load Null Data 2 (SWC)
null2_file_name = SBJ+'_SWC_Null_'+emb+'_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
null2_file_path = osp.join(PRJDIR,'derivatives','Null_Data',null2_file_name)
Null2_df        = pd.read_csv(null2_file_path)

print('++ INFO: Null embedding loaded')
print('         Orig data shape:', Orig_df.shape)
print('         Null data shape:', Null1_df.shape)
print('         Null data shape:', Null2_df.shape)

# +
# Brop inbetween windows to compute SI
drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
drop_Orig_df  = Orig_df.drop(drop_index).reset_index(drop=True)
drop_Null1_df = Null1_df.drop(drop_index).reset_index(drop=True)
drop_Null2_df = Null2_df.drop(drop_index).reset_index(drop=True)
drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)

orig_SI  = silhouette_score(drop_Orig_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values) # Original SI
null1_SI = silhouette_score(drop_Null1_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values) # Null 1 SI
null2_SI = silhouette_score(drop_Null2_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values) # Null 2 SI

# +
task_cmap = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'}

# Original Data Plot
plot1_input = pd.DataFrame(columns=['x','y','z'])
plot1_input[['x','y','z']] = Orig_df[['1_norm','2_norm','3_norm']].copy()
plot1_input['Task'] = task_df['Task']
plot1 = px.scatter_3d(plot1_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap, 
                      width=700, height=600, opacity=0.7, title='Original Data SI: '+str(orig_SI))
plot1 = plot1.update_traces(marker=dict(size=3,line=dict(width=0)))

# Null 1 Plot
plot2_input = pd.DataFrame(columns=['x','y','z'])
plot2_input[['x','y','z']] = Null1_df[['1_norm','2_norm','3_norm']].copy()
plot2_input['Task'] = task_df['Task']
plot2 = px.scatter_3d(plot2_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap,
                      width=700, height=600, opacity=0.7, title='Null Data SI: '+str(null1_SI))
plot2 = plot2.update_traces(marker=dict(size=3,line=dict(width=0)))

# Null 2 Plot
plot3_input = pd.DataFrame(columns=['x','y','z'])
plot3_input[['x','y','z']] = Null2_df[['1_norm','2_norm','3_norm']].copy()
plot3_input['Task'] = task_df['Task']
plot3 = px.scatter_3d(plot3_input, x='x', y='y', z='z', color='Task', color_discrete_map=task_cmap,
                      width=700, height=600, opacity=0.7, title='Null Data SI: '+str(null2_SI))
plot3 = plot3.update_traces(marker=dict(size=3,line=dict(width=0)))

pn.Row(plot1, plot2, plot3)
