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
null = 'shuffle'
all_null1_LE = {}
for SBJ in SBJ_list:
    file_name  = SBJ+'_Null'+null+'_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path  = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_LE_df = pd.read_csv(file_path)
    all_null1_LE[SBJ] = null_LE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null1_LE_SI_df = group_SI(all_null1_LE, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', null1_LE_SI_df.shape)

# ### Null Data 2

# Load null LE embeddings 2
# -------------------------
null = 'phase'
all_null2_LE = {}
for SBJ in SBJ_list:
    file_name  = SBJ+'_Null'+null+'_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path  = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_LE_df = pd.read_csv(file_path)
    all_null2_LE[SBJ] = null_LE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null2_LE_SI_df = group_SI(all_null2_LE, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', null2_LE_SI_df.shape)

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
null = 'shuffle'
all_null1_TSNE = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_Null'+null+'_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_TSNE_df = pd.read_csv(file_path)
    all_null1_TSNE[SBJ] = null_TSNE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null1_TSNE_SI_df = group_SI(all_null1_TSNE, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', null1_TSNE_SI_df.shape)

# ### Null Data 2

# Load null TSNE embeddings 2
# ---------------------------
null = 'phase'
all_null2_TSNE = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_Null'+null+'_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_TSNE_df = pd.read_csv(file_path)
    all_null2_TSNE[SBJ] = null_TSNE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null2_TSNE_SI_df = group_SI(all_null2_TSNE, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', null2_TSNE_SI_df.shape)

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
null = 'shuffle'
all_null1_UMAP = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_Null'+null+'_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_UMAP_df = pd.read_csv(file_path)
    all_null1_UMAP[SBJ] = null_UMAP_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null1_UMAP_SI_df = group_SI(all_null1_UMAP, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', null1_UMAP_SI_df.shape)

# ### Null Data 2

# Load null UMAP embeddings 2
# ---------------------------
null = 'phase'
all_null2_UMAP = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_Null'+null+'_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_UMAP_df = pd.read_csv(file_path)
    all_null2_UMAP[SBJ] = null_UMAP_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null2_UMAP_SI_df = group_SI(all_null2_UMAP, task_df, 'Task')
print('++ INFO: SI data frame computed')
print('         Data shape', null2_UMAP_SI_df.shape)

# ## Full Data Frame
# ***

# Create full data data frame
# ---------------------------
all_SI_df = pd.DataFrame(index=SBJ_list)
all_SI_df['LE','Original']   = orig_LE_SI_df['Silhouette Index'].copy()
all_SI_df['LE','Null 1']     = null1_LE_SI_df['Silhouette Index'].copy()
all_SI_df['LE','Null 2']     = null2_LE_SI_df['Silhouette Index'].copy()
all_SI_df['TSNE','Original'] = orig_TSNE_SI_df['Silhouette Index'].copy()
all_SI_df['TSNE','Null 1']   = null1_TSNE_SI_df['Silhouette Index'].copy()
all_SI_df['TSNE','Null 2']   = null2_TSNE_SI_df['Silhouette Index'].copy()
all_SI_df['UMAP','Original'] = orig_UMAP_SI_df['Silhouette Index'].copy()
all_SI_df['UMAP','Null 1']   = null1_UMAP_SI_df['Silhouette Index'].copy()
all_SI_df['UMAP','Null 2']   = null2_UMAP_SI_df['Silhouette Index'].copy()

# ## Silhouette Index Bar Plot
# ***

# +
# Compute mean and std for each data type
# ---------------------------------------
orig_SI_mean  = list(all_SI_df[[('LE','Original'),('TSNE','Original'),('UMAP','Original')]].mean().values) # Mean SI for original data
null1_SI_mean = list(all_SI_df[[('LE','Null 1'),('TSNE','Null 1'),('UMAP','Null 1')]].mean().values) # Mean SI for null 1 data
null2_SI_mean = list(all_SI_df[[('LE','Null 2'),('TSNE','Null 2'),('UMAP','Null 2')]].mean().values) # Mean SI for null 2 data

orig_SI_error  = list(all_SI_df[[('LE','Original'),('TSNE','Original'),('UMAP','Original')]].std().values) # STD SI for original data
null1_SI_error = list(all_SI_df[[('LE','Null 1'),('TSNE','Null 1'),('UMAP','Null 1')]].std().values) # SID SI for null 1 data
null2_SI_error = list(all_SI_df[[('LE','Null 2'),('TSNE','Null 2'),('UMAP','Null 2')]].std().values) # SID SI for null 2 data

# +
# Bar plot with error bars
# ------------------------
X = np.arange(3)
fig, ax = plt.subplots(figsize=(14, 9))
ax.bar(X + 0.00, null1_SI_mean, yerr=null1_SI_error, ecolor='black', capsize=10, color='g', width=0.25,)
ax.bar(X + 0.25, null2_SI_mean, yerr=null2_SI_error, ecolor='black', capsize=10, color='r', width=0.25)
ax.bar(X + 0.50, orig_SI_mean, yerr=orig_SI_error, ecolor='black', capsize=10, color='b', width=0.25)
ax.set_ylabel('Avg Silhouette Index')
ax.set_xticks(X)
ax.set_xticklabels(['LE','TSNE','UMAP'])
ax.legend(labels=['Null 1', 'Null 2', 'Original'])

# Save the figure and show
plt.show()
# -

# ## T-Test
# ***

from scipy import stats

data_list = ['Null 1', 'Null 2', 'Original']

technique = 'UMAP'
T_stat  = np.zeros((len(data_list),len(data_list)))
P_value = np.zeros((len(data_list),len(data_list)))
for i in range(len(data_list)):
    for j in range(len(data_list)):
        t_stats = stats.ttest_rel(all_SI_df[technique,data_list[i]], all_SI_df[technique,data_list[j]])
        if i == j:
            T_stat[i,j]  = np.nan
            P_value[i,j] = np.nan
        else:
            T_stat[i,j]  = t_stats[0]
            T_stat[j,i]  = t_stats[0]
            P_value[i,j] = t_stats[1]
            P_value[j,i] = t_stats[1]

# +
plot_data = P_value

fig, ax = plt.subplots(figsize=(7, 7))
im = ax.imshow(plot_data, cmap=plt.cm.Blues)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(data_list)))
ax.set_xticklabels(data_list)
ax.set_yticks(np.arange(len(data_list)))
ax.set_yticklabels(data_list)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(data_list)):
    for j in range(len(data_list)):
        text = ax.text(j, i, round(plot_data[i, j],4), ha="center", va="center", color="black")

plt.rcParams.update({'font.size': 18})
fig.tight_layout()
plt.show()
# -


