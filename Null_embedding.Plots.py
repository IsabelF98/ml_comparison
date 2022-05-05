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
# * TSNE: p=50, metric=correlation
# * UMAP: k=130, metric=correlation

import pandas as pd
import numpy as np
import os.path as osp
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from utils.data_info import PRJDIR, wl_sec, tr, SBJ_list, task_labels
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import seaborn as sns

wl_trs = int(wl_sec/tr)
task_df = task_labels(wl_trs, PURE=False)


# ## Functions
# ***

def group_SI(data_dict, label_df, label, full_data):
    """
    This function computes the silhouette index for each embedding in a group of embeddings.
    
    INPUT
    -----
    data_dict: (dict) This is a dictionary of embeddings where the group labels are the key (e.x. subject number) 
               and the embeddings are the values (as pd.DataFrames)
    label_df: (pd.DataFrame) A data frame of the labels you wish score by (e.x. task labels)
    label: (str) The name of the label you are scoring by (e.x. task)
    full_data: (bool) Full data set (True) or dropped windows (False)
    
    OUTPUT
    ------
    SI_df: (pd.DataFrame) A data frame with group labels as the index (e.x. subjects) and embedding silhouette index 
           as the column values
    """
    
    group_list = list(data_dict.keys())
    SI_list    = []

    for key in group_list:
        embed_df = data_dict[key]
        if full_data:
            drop_index    = label_df.index[label_df[label] == 'Inbetween']
            drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
            drop_label_df = label_df.drop(drop_index).reset_index(drop=True)
            silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_label_df[label].values)
        else:
            silh_idx = silhouette_score(embed_df[['1_norm', '2_norm', '3_norm']], label_df[label].values)
        SI_list.append(silh_idx)
    
    SI_df = pd.DataFrame(SI_list, index=group_list, columns=['Silhouette Index'])
    
    return SI_df


def group_F1(data_dict, label_df, label, n):
    """
    This function computes the F1 accuracy for each embedding in a group of embeddings.
    
    INPUT
    -----
    data_dict: (dict) This is a dictionary of embeddings where the group labels are the key (e.x. subject number) 
               and the embeddings are the values (as pd.DataFrames)
    label_df: (pd.DataFrame) A data frame of the labels you wish score by (e.x. task labels)
    label: (str) The name of the label you are scoring by (e.x. task)
    n: (int) Number of dimensions for embedding
    
    OUTPUT
    ------
    F1_df: (pd.DataFrame) A data frame with group labels as the index (e.x. subjects) and embedding F1 accuracy 
           as the column values
    """
    
    train_idx = (0, 363) # Training set index range
    test_idx  = (364, 728) # Testing set index range
    
    train_label_df = label_df.loc[train_idx[0]:train_idx[1]].copy() # Training labels
    test_label_df  = label_df.loc[test_idx[0]:test_idx[1]].copy() # Testing labels
    
    group_list = list(data_dict.keys())
    F1_list    = []

    for key in group_list:
        embed_df = data_dict[key]
        # Split data
        train_embed_df = embed_df.loc[train_idx[0]:train_idx[1]].copy()
        test_embed_df  = embed_df.loc[test_idx[0]:test_idx[1]].copy()
        # Compute LR
        Log_Reg      = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
        data_Log_Reg = Log_Reg.fit(train_embed_df[[str(i)+'_norm' for i in range(1,n+1)]], train_label_df[label])
        Log_Reg_Coef = data_Log_Reg.coef_
        Log_Reg_Clas = data_Log_Reg.classes_
        # Predict test data
        predicted = data_Log_Reg.predict(test_embed_df[[str(i)+'_norm' for i in range(1,n+1)]])
        # Classification Report
        class_report_df = pd.DataFrame(metrics.classification_report(test_label_df, predicted, output_dict=True)).T
        F1_acur = class_report_df.loc['accuracy','f1-score']
        F1_list.append(F1_acur)
        
    F1_df = pd.DataFrame(F1_list, index=group_list, columns=['F1 Accuracy'])
    return F1_df


n         = 3
metric    = 'correlation'
drop      = 'FullData'
full_data = True

# ## Laplacian Eigenmap
# ***

LE_k = 80

# ### Original Data

# Load original LE embeddings
# ---------------------------
all_orig_LE = {}
for SBJ in SBJ_list:
    file_name  = SBJ+'_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    file_path  = osp.join(PRJDIR,'derivatives','LE',file_name)
    orig_LE_df = pd.read_csv(file_path)
    all_orig_LE[SBJ] = orig_LE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
orig_LE_SI_df = group_SI(all_orig_LE, task_df, 'Task', full_data=full_data)
print('++ INFO: SI data frame computed')
print('         Data shape', orig_LE_SI_df.shape)

# Compute group F1 Accuracy
# -------------------------
orig_LE_F1_df = group_F1(all_orig_LE, task_df, 'Task', n)
print('++ INFO: F1 data frame computed')
print('         Data shape', orig_LE_F1_df.shape)

# ### Null Data ROI

# Load null LE embeddings 1
# -------------------------
data = 'ROI'
all_null1_LE = {}
for SBJ in SBJ_list:
    file_name  = SBJ+'_'+data+'_Null_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    file_path  = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_LE_df = pd.read_csv(file_path)
    all_null1_LE[SBJ] = null_LE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null1_LE_SI_df = group_SI(all_null1_LE, task_df, 'Task', full_data=full_data)
print('++ INFO: SI data frame computed')
print('         Data shape', null1_LE_SI_df.shape)

# Compute group F1 Accuracy
# -------------------------
null1_LE_F1_df = group_F1(all_null1_LE, task_df, 'Task', n)
print('++ INFO: F1 data frame computed')
print('         Data shape', null1_LE_F1_df.shape)

# ### Null Data SWC

# Load null LE embeddings 2
# -------------------------
data = 'SWC'
all_null2_LE = {}
for SBJ in SBJ_list:
    file_name  = SBJ+'_'+data+'_Null_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    file_path  = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_LE_df = pd.read_csv(file_path)
    all_null2_LE[SBJ] = null_LE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null2_LE_SI_df = group_SI(all_null2_LE, task_df, 'Task', full_data=full_data)
print('++ INFO: SI data frame computed')
print('         Data shape', null2_LE_SI_df.shape)

# Compute group F1 Accuracy
# -------------------------
null2_LE_F1_df = group_F1(all_null2_LE, task_df, 'Task', n)
print('++ INFO: F1 data frame computed')
print('         Data shape', null2_LE_F1_df.shape)

# ## TSNE
# ***

p = 70

# ### Original Data

# Load original TSNE embeddings
# -----------------------------
all_orig_TSNE = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','TSNE',file_name)
    orig_TSNE_df = pd.read_csv(file_path)
    all_orig_TSNE[SBJ] = orig_TSNE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
orig_TSNE_SI_df = group_SI(all_orig_TSNE, task_df, 'Task', full_data=full_data)
print('++ INFO: SI data frame computed')
print('         Data shape', orig_TSNE_SI_df.shape)

# Compute group F1 Accuracy
# -------------------------
orig_TSNE_F1_df = group_F1(all_orig_TSNE, task_df, 'Task', n)
print('++ INFO: F1 data frame computed')
print('         Data shape', orig_TSNE_F1_df.shape)

# ### Null Data ROI

# Load null TSNE embeddings 1
# ---------------------------
data = 'ROI'
all_null1_TSNE = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_'+data+'_Null_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_TSNE_df = pd.read_csv(file_path)
    all_null1_TSNE[SBJ] = null_TSNE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null1_TSNE_SI_df = group_SI(all_null1_TSNE, task_df, 'Task', full_data=full_data)
print('++ INFO: SI data frame computed')
print('         Data shape', null1_TSNE_SI_df.shape)

# Compute group F1 Accuracy
# -------------------------
null1_TSNE_F1_df = group_F1(all_null1_TSNE, task_df, 'Task', n)
print('++ INFO: F1 data frame computed')
print('         Data shape', null1_TSNE_F1_df.shape)

# ### Null Data SWC

# Load null TSNE embeddings 2
# ---------------------------
data = 'SWC'
all_null2_TSNE = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_'+data+'_Null_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_TSNE_df = pd.read_csv(file_path)
    all_null2_TSNE[SBJ] = null_TSNE_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null2_TSNE_SI_df = group_SI(all_null2_TSNE, task_df, 'Task', full_data=full_data)
print('++ INFO: SI data frame computed')
print('         Data shape', null2_TSNE_SI_df.shape)

# Compute group F1 Accuracy
# -------------------------
null2_TSNE_F1_df = group_F1(all_null2_TSNE, task_df, 'Task', n)
print('++ INFO: F1 data frame computed')
print('         Data shape', null2_TSNE_F1_df.shape)

# ## UMAP
# ***

UMAP_k = 10

# ### Origianl Data

# Load original UMAP embeddings
# ---------------------------
all_orig_UMAP = {}
for SBJ in SBJ_list:
    file_name  = SBJ+'_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    file_path  = osp.join(PRJDIR,'derivatives','UMAP',file_name)
    orig_UMAP_df = pd.read_csv(file_path)
    all_orig_UMAP[SBJ] = orig_UMAP_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
orig_UMAP_SI_df = group_SI(all_orig_UMAP, task_df, 'Task', full_data=full_data)
print('++ INFO: SI data frame computed')
print('         Data shape', orig_UMAP_SI_df.shape)

# Compute group F1 Accuracy
# -------------------------
orig_UMAP_F1_df = group_F1(all_orig_UMAP, task_df, 'Task', n)
print('++ INFO: F1 data frame computed')
print('         Data shape', orig_UMAP_F1_df.shape)

# ### Null Data ROI

# Load null UMAP embeddings 1
# ---------------------------
data = 'ROI'
all_null1_UMAP = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_'+data+'_Null_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_UMAP_df = pd.read_csv(file_path)
    all_null1_UMAP[SBJ] = null_UMAP_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null1_UMAP_SI_df = group_SI(all_null1_UMAP, task_df, 'Task', full_data=full_data)
print('++ INFO: SI data frame computed')
print('         Data shape', null1_UMAP_SI_df.shape)

# Compute group F1 Accuracy
# -------------------------
null1_UMAP_F1_df = group_F1(all_null1_UMAP, task_df, 'Task', n)
print('++ INFO: F1 data frame computed')
print('         Data shape', null1_UMAP_F1_df.shape)

# ### Null Data SWC

# Load null UMAP embeddings 2
# ---------------------------
data = 'SWC'
all_null2_UMAP = {}
for SBJ in SBJ_list:
    file_name    = SBJ+'_'+data+'_Null_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    file_path    = osp.join(PRJDIR,'derivatives','Null_Data',file_name)
    null_UMAP_df = pd.read_csv(file_path)
    all_null2_UMAP[SBJ] = null_UMAP_df
    print('++ INFO: Data loaded for', SBJ)

# Compute group SI
# ----------------
null2_UMAP_SI_df = group_SI(all_null2_UMAP, task_df, 'Task', full_data=full_data)
print('++ INFO: SI data frame computed')
print('         Data shape', null2_UMAP_SI_df.shape)

# Compute group F1 Accuracy
# -------------------------
null2_UMAP_F1_df = group_F1(all_null2_UMAP, task_df, 'Task', n)
print('++ INFO: F1 data frame computed')
print('         Data shape', null2_UMAP_F1_df.shape)

# ## Full Silhouette Index Data Frame
# ***

# Full data frame of SI values
# ----------------------------
all_SI_df = pd.DataFrame(columns=['Technique','Data','Silhouette Index'])
all_SI_df = pd.concat([all_SI_df, pd.DataFrame({'Technique': ['LE' for i in range(orig_LE_SI_df.shape[0])],
                                                'Data': ['Original' for i in range(orig_LE_SI_df.shape[0])],
                                                'Silhouette Index': orig_LE_SI_df['Silhouette Index'].values})], ignore_index=True)
all_SI_df = pd.concat([all_SI_df, pd.DataFrame({'Technique': ['LE' for i in range(null1_LE_SI_df.shape[0])],
                                                'Data': ['Null ROI' for i in range(null1_LE_SI_df.shape[0])],
                                                'Silhouette Index': null1_LE_SI_df['Silhouette Index'].values})], ignore_index=True)
all_SI_df = pd.concat([all_SI_df, pd.DataFrame({'Technique': ['LE' for i in range(null2_LE_SI_df.shape[0])],
                                                'Data': ['Null SWC' for i in range(null2_LE_SI_df.shape[0])],
                                                'Silhouette Index': null2_LE_SI_df['Silhouette Index'].values})], ignore_index=True)
all_SI_df = pd.concat([all_SI_df, pd.DataFrame({'Technique': ['TSNE' for i in range(orig_TSNE_SI_df.shape[0])],
                                                'Data': ['Original' for i in range(orig_TSNE_SI_df.shape[0])],
                                                'Silhouette Index': orig_TSNE_SI_df['Silhouette Index'].values})], ignore_index=True)
all_SI_df = pd.concat([all_SI_df, pd.DataFrame({'Technique': ['TSNE' for i in range(null1_TSNE_SI_df.shape[0])],
                                                'Data': ['Null ROI' for i in range(null1_TSNE_SI_df.shape[0])],
                                                'Silhouette Index': null1_TSNE_SI_df['Silhouette Index'].values})], ignore_index=True)
all_SI_df = pd.concat([all_SI_df, pd.DataFrame({'Technique': ['TSNE' for i in range(null2_TSNE_SI_df.shape[0])],
                                                'Data': ['Null SWC' for i in range(null2_TSNE_SI_df.shape[0])],
                                                'Silhouette Index': null2_TSNE_SI_df['Silhouette Index'].values})], ignore_index=True)
all_SI_df = pd.concat([all_SI_df, pd.DataFrame({'Technique': ['UMAP' for i in range(orig_UMAP_SI_df.shape[0])],
                                                'Data': ['Original' for i in range(orig_UMAP_SI_df.shape[0])],
                                                'Silhouette Index': orig_UMAP_SI_df['Silhouette Index'].values})], ignore_index=True)
all_SI_df = pd.concat([all_SI_df, pd.DataFrame({'Technique': ['UMAP' for i in range(null1_UMAP_SI_df.shape[0])],
                                                'Data': ['Null ROI' for i in range(null1_UMAP_SI_df.shape[0])],
                                                'Silhouette Index': null1_UMAP_SI_df['Silhouette Index'].values})], ignore_index=True)
all_SI_df = pd.concat([all_SI_df, pd.DataFrame({'Technique': ['UMAP' for i in range(null2_UMAP_SI_df.shape[0])],
                                                'Data': ['Null SWC' for i in range(null2_UMAP_SI_df.shape[0])],
                                                'Silhouette Index': null2_UMAP_SI_df['Silhouette Index'].values})], ignore_index=True)

# ## Silhouette Index Bar Plot
# ***

# +
# Bar plot with error bars and stars
# ----------------------------------
x = 'Technique'
y = 'Silhouette Index'
hue = 'Data'
hue_order=['Null ROI', 'Null SWC', 'Original']
order = ['LE','TSNE','UMAP']
pairs=[
    (("LE", "Null ROI"), ("LE", "Original")),
    (("LE", "Null SWC"), ("LE", "Original")),
    (("TSNE", "Null ROI"), ("TSNE", "Original")),
    (("TSNE", "Null SWC"), ("TSNE", "Original")),
    (("UMAP", "Null ROI"), ("UMAP", "Original")),
    (("UMAP", "Null SWC"), ("UMAP", "Original")),
    (("LE", "Original"), ("TSNE", "Original")),
    (("TSNE", "Original"), ("UMAP", "Original")),
    (("UMAP", "Original"), ("LE", "Original")),
    ]

sns.set(rc = {'figure.figsize':(14,7)})
ax    = sns.barplot(x=x, y=y, hue=hue, data=all_SI_df, order=order, hue_order=hue_order, capsize=0.1)
annot = Annotator(ax, pairs, data=all_SI_df, x=x, y=y, hue=hue, order=order, hue_order=hue_order)
annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()
plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
# -

# ## Full F1 Accuracy Data Frame
# ***

# Full data frame of F1 values
# ----------------------------
all_F1_df = pd.DataFrame(columns=['Technique','Data','F1 Accuracy'])
all_F1_df = pd.concat([all_F1_df, pd.DataFrame({'Technique': ['LE' for i in range(orig_LE_F1_df.shape[0])],
                                                'Data': ['Original' for i in range(orig_LE_F1_df.shape[0])],
                                                'F1 Accuracy': orig_LE_F1_df['F1 Accuracy'].values})], ignore_index=True)
all_F1_df = pd.concat([all_F1_df, pd.DataFrame({'Technique': ['LE' for i in range(null1_LE_F1_df.shape[0])],
                                                'Data': ['Null ROI' for i in range(null1_LE_F1_df.shape[0])],
                                                'F1 Accuracy': null1_LE_F1_df['F1 Accuracy'].values})], ignore_index=True)
all_F1_df = pd.concat([all_F1_df, pd.DataFrame({'Technique': ['LE' for i in range(null2_LE_F1_df.shape[0])],
                                                'Data': ['Null SWC' for i in range(null2_LE_F1_df.shape[0])],
                                                'F1 Accuracy': null2_LE_F1_df['F1 Accuracy'].values})], ignore_index=True)
all_F1_df = pd.concat([all_F1_df, pd.DataFrame({'Technique': ['TSNE' for i in range(orig_TSNE_F1_df.shape[0])],
                                                'Data': ['Original' for i in range(orig_TSNE_F1_df.shape[0])],
                                                'F1 Accuracy': orig_TSNE_F1_df['F1 Accuracy'].values})], ignore_index=True)
all_F1_df = pd.concat([all_F1_df, pd.DataFrame({'Technique': ['TSNE' for i in range(null1_TSNE_F1_df.shape[0])],
                                                'Data': ['Null ROI' for i in range(null1_TSNE_F1_df.shape[0])],
                                                'F1 Accuracy': null1_TSNE_F1_df['F1 Accuracy'].values})], ignore_index=True)
all_F1_df = pd.concat([all_F1_df, pd.DataFrame({'Technique': ['TSNE' for i in range(null2_TSNE_F1_df.shape[0])],
                                                'Data': ['Null SWC' for i in range(null2_TSNE_F1_df.shape[0])],
                                                'F1 Accuracy': null2_TSNE_F1_df['F1 Accuracy'].values})], ignore_index=True)
all_F1_df = pd.concat([all_F1_df, pd.DataFrame({'Technique': ['UMAP' for i in range(orig_UMAP_F1_df.shape[0])],
                                                'Data': ['Original' for i in range(orig_UMAP_F1_df.shape[0])],
                                                'F1 Accuracy': orig_UMAP_F1_df['F1 Accuracy'].values})], ignore_index=True)
all_F1_df = pd.concat([all_F1_df, pd.DataFrame({'Technique': ['UMAP' for i in range(null1_UMAP_F1_df.shape[0])],
                                                'Data': ['Null ROI' for i in range(null1_UMAP_F1_df.shape[0])],
                                                'F1 Accuracy': null1_UMAP_F1_df['F1 Accuracy'].values})], ignore_index=True)
all_F1_df = pd.concat([all_F1_df, pd.DataFrame({'Technique': ['UMAP' for i in range(null2_UMAP_F1_df.shape[0])],
                                                'Data': ['Null SWC' for i in range(null2_UMAP_F1_df.shape[0])],
                                                'F1 Accuracy': null2_UMAP_F1_df['F1 Accuracy'].values})], ignore_index=True)

# ## F1 Accuracy Bar Plot
# ***

# +
# Bar plot with error bars and stars
# ----------------------------------
x = 'Technique'
y = 'F1 Accuracy'
hue = 'Data'
hue_order=['Null ROI', 'Null SWC', 'Original']
order = ['LE','TSNE','UMAP']
pairs=[
    (("LE", "Null ROI"), ("LE", "Original")),
    (("LE", "Null SWC"), ("LE", "Original")),
    (("TSNE", "Null ROI"), ("TSNE", "Original")),
    (("TSNE", "Null SWC"), ("TSNE", "Original")),
    (("UMAP", "Null ROI"), ("UMAP", "Original")),
    (("UMAP", "Null SWC"), ("UMAP", "Original")),
    (("LE", "Original"), ("TSNE", "Original")),
    (("TSNE", "Original"), ("UMAP", "Original")),
    (("UMAP", "Original"), ("LE", "Original")),
    ]

sns.set(rc = {'figure.figsize':(14,7)})
ax    = sns.barplot(x=x, y=y, hue=hue, data=all_F1_df, order=order, hue_order=hue_order, capsize=0.1)
annot = Annotator(ax, pairs, data=all_F1_df, x=x, y=y, hue=hue, order=order, hue_order=hue_order)
annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()
plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
# -

