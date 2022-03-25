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

# # Null Embedding Plots
#
# This notebook walks through the step for plotting the silhouette index (SI) bar graphs of the original embeddings v.s. the null embeddings for each of the three techniques. The embeddings that are used will be those that have the highest silhouette index for each techniques based on the hyperparameters.
#
# * Laplacain Eigenmap: k=50, metric=correlation
# * TSNE: p=55, metric=correlation
# * UMAP: k=130, metric=correlation

import pandas as pd
import numpy as np
import os.path as osp
from sklearn.metrics import silhouette_score
from utils.data_info import PRJDIR, wl_sec, tr, SBJ_list, task_labels
import matplotlib.pyplot as plt
import holoviews as hv
import panel as pn
hv.extension('bokeh')

# ## SI Function
# ***

wl_trs = int(wl_sec/tr)
task_df = task_labels(wl_trs, PURE=True)


def group_SI(data_dict, label_df, label):
    """
    This function computes the silhouette index for each embedding in a group of embeddings.
    
    INPUT
    -----
    data_dict: (dict) This is a dictionary of embeddings where the group labels are the key (e.x. subject number) 
               and the embeddings are the values (as pd.DataFrames)
    label_df: (pd.DataFrame) A data frame of the labels you wish score by (e.x. task labels)
    label: (str) The name of the label you are scoring by (e.x. task)
    
    OUTPUT
    ------
    SI_df: (pd.DataFrame) A data frame with group labels as the index (e.x. subjects) and embedding silhouette index 
           as the column values
    """
    
    group_list = list(data_dict.keys())
    SI_list    = []

    for key in group_list:
        embed_df = data_dict[key]
        silh_idx = silhouette_score(embed_df[['1_norm', '2_norm', '3_norm']], label_df[label].values)
        SI_list.append(silh_idx)

    SI_df = pd.DataFrame(SI_list, index=group_list, columns=['Silhouette Index'])
    return SI_df


# ## Laplacian Eigenmap SI
# ***

LE_k   = 50
n      = 3
metric = 'correlation'

# ### Original Data

# Load original LE embeddings
# ---------------------------
all_orig_LE = {}
for SBJ in SBJ_list:
    file_name  = SBJ+'_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path  = osp.join(PRJDIR,'derivatives','LE',file_name)
    orig_LE_df = pd.read_csv(file_path)
    all_orig_LE[SBJ] = orig_LE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
orig_LE_SI_df = group_SI(all_orig_LE, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', orig_LE_SI_df.shape)

# ### Null Data 1

# Load null LE embeddings 1
# -------------------------
all_null_LE = {}
for SBJ in SBJ_list:
    file_name  = SBJ+'_Null_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path  = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_LE_df = pd.read_csv(file_path)
    all_null_LE[SBJ] = null_LE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null_LE_SI_df = group_SI(all_null_LE, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', null_LE_SI_df.shape)

# ## TSNE SI
# ***

p = 55
n = 3
metric = 'correlation'

# ### Original Data

# Load original TSNE embeddings
# -----------------------------
all_orig_TSNE = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','TSNE',file_name)
    orig_TSNE_df = pd.read_csv(file_path)
    all_orig_TSNE[SBJ] = orig_TSNE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
orig_TSNE_SI_df = group_SI(all_orig_TSNE, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', orig_TSNE_SI_df.shape)

# ### Null Data 1

# Load null TSNE embeddings 1
# ---------------------------
all_null_TSNE = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_Null_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_TSNE_df = pd.read_csv(file_path)
    all_null_TSNE[SBJ] = null_TSNE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null_TSNE_SI_df = group_SI(all_null_TSNE, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', null_TSNE_SI_df.shape)

# ## UMAP SI
# ***

UMAP_k = 130
n      = 3
metric = 'correlation'

# ### Origianl Data

# Load original UMAP embeddings
# ---------------------------
all_orig_UMAP = {}
for SBJ in SBJ_list:
    file_name  = SBJ+'_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path  = osp.join(PRJDIR,'derivatives','UMAP',file_name)
    orig_UMAP_df = pd.read_csv(file_path)
    all_orig_UMAP[SBJ] = orig_UMAP_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
orig_UMAP_SI_df = group_SI(all_orig_UMAP, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', orig_UMAP_SI_df.shape)

# ### Null Data 1

# Load null UMAP embeddings 1
# ---------------------------
all_null_UMAP = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_Null_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_UMAP_df = pd.read_csv(file_path)
    all_null_UMAP[SBJ] = null_UMAP_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null_UMAP_SI_df = group_SI(all_null_UMAP, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', null_UMAP_SI_df.shape)

# ## Silhouette Index Bar Plot
# ***

# Create full data data frame
# ---------------------------
all_SI_df = pd.DataFrame(index=SBJ_list)
all_SI_df['LE','Original']   = orig_LE_SI_df['Silhouette Index'].copy()
all_SI_df['LE','Null 1']     = null_LE_SI_df['Silhouette Index'].copy()
all_SI_df['TSNE','Original'] = orig_TSNE_SI_df['Silhouette Index'].copy()
all_SI_df['TSNE','Null 1']   = null_TSNE_SI_df['Silhouette Index'].copy()
all_SI_df['UMAP','Original'] = orig_UMAP_SI_df['Silhouette Index'].copy()
all_SI_df['UMAP','Null 1']   = null_UMAP_SI_df['Silhouette Index'].copy()

all_SI_df

# +
# Mean and Standard Deviation
# ---------------------------
SI_stats = pd.DataFrame()

# Data Labels
SI_stats['Technique']  = np.array(['LE','LE','TSNE','TSNE','UMAP','UMAP'])
SI_stats['Data']       = np.array(['Original','Null 1','Original','Null 1','Original','Null 1'])

# Mean SI for each technqiues and data
SI_stats['SI_mean'] = all_SI_df[[('LE','Original'),('LE','Null 1'),('TSNE','Original'),('TSNE','Null 1'),('UMAP','Original'),('UMAP','Null 1')]].mean().values

# Computing std for each technqiues and data
SI_error = all_SI_df[[('LE','Original'),('LE','Null 1'),('TSNE','Original'),('TSNE','Null 1'),('UMAP','Original'),('UMAP','Null 1')]].std().values
SI_stats['SI+SD'] = SI_stats['SI_mean'].values + SI_error
SI_stats['SI-SD'] = SI_stats['SI_mean'].values - SI_error

SI_stats
# -

# Bar Plot
# --------
SI_bars = hv.Bars(SI_stats, kdims=['Technique', 'Data']).opts(width=600, ylabel='Avg Silhouette Index', xlabel='')
#          hv.ErrorBars(SI_stats, vdims=['SI_mean', 'SI+SD', 'SI-SD'], kdims=['Technique', 'Data']) # Error bars not working
SI_bars


