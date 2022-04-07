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

# # Logistic Regression Plots
#
# This notebook is for creating the plots of Logistic Regression accuracy values for each manifold learning technique.

import pandas as pd
import numpy as np
import os.path as osp
from utils.data_info import PRJDIR, LE_k_list, p_list, UMAP_k_list, wl_sec, tr, SBJ_list, task_labels
import holoviews as hv
import panel as pn
hv.extension('bokeh')

dist_metric_list = ['correlation', 'cosine', 'euclidean']
n = 3

# ## Laplacian Eigenmap
# ***

# Load F1 accuracy values
# -----------------------
all_SBJ_acur = {}
for SBJ in SBJ_list:
    F1_acur_df = pd.DataFrame(LE_k_list, columns=['k value'])
    for metric in dist_metric_list:
        metric_acur_list = []
        for LE_k in LE_k_list:
            LE_file = SBJ+'_LE_LRclassrep_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
            LE_path = osp.join(PRJDIR,'derivatives','Log_Reg',LE_file)
            LE_class_report_df = pd.read_csv(LE_path)
            F1_acur = LE_class_report_df.loc[4,'f1-score']
            metric_acur_list.append(F1_acur)
        F1_acur_df[metric] = metric_acur_list
    all_SBJ_acur[SBJ] = F1_acur_df

# Compute average and standard error of group F1 accuracy
# --------------------------------------------------------
avg_group_acur = pd.concat([all_SBJ_acur[SBJ] for SBJ in SBJ_list]).groupby(level=0).mean() # Average
sem_group_acur = pd.concat([all_SBJ_acur[SBJ] for SBJ in SBJ_list]).groupby(level=0).sem()  # Standerd Error

# Plot data frame
# ---------------
plot_df = pd.DataFrame(LE_k_list,columns=['k-NN value'])
for metric in dist_metric_list:
    plot_df[metric] = avg_group_acur[metric].copy()
    plot_df[metric+' +SE'] = avg_group_acur[metric] + sem_group_acur[metric]
    plot_df[metric+' -SE'] = avg_group_acur[metric] - sem_group_acur[metric]

# Plot
# ----
((hv.Area((plot_df['k-NN value' ], plot_df['correlation +SE'], plot_df['correlation -SE']), vdims=['correlation +SE', 'correlation -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=['k-NN value' ,'correlation'], label='correlation'))*\
(hv.Area((plot_df['k-NN value' ], plot_df['cosine +SE'], plot_df['cosine -SE']), vdims=['cosine +SE', 'cosine -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=['k-NN value' ,'cosine'], label='cosine'))*\
(hv.Area((plot_df['k-NN value' ], plot_df['euclidean +SE'], plot_df['euclidean -SE']), vdims=['euclidean +SE', 'euclidean -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=['k-NN value' ,'euclidean'], label='euclidean')))\
.opts(width=700, height=500, xlabel='k-NN value' , ylabel='Average F1 Accuracy',fontsize={'labels':14,'xticks':12,'yticks':12,'legend':14})

# ## TSNE
# ***

# Load F1 accuracy values
# -----------------------
all_SBJ_acur = {}
for SBJ in SBJ_list:
    F1_acur_df = pd.DataFrame(p_list, columns=['k value'])
    for metric in dist_metric_list:
        metric_acur_list = []
        for p in p_list:
            TSNE_file = SBJ+'_TSNE_LRclassrep_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
            TSNE_path = osp.join(PRJDIR,'derivatives','Log_Reg',TSNE_file)
            TSNE_class_report_df = pd.read_csv(TSNE_path)
            F1_acur = TSNE_class_report_df.loc[4,'f1-score']
            metric_acur_list.append(F1_acur)
        F1_acur_df[metric] = metric_acur_list
    all_SBJ_acur[SBJ] = F1_acur_df

# Compute average and standard error of group F1 accuracy
# --------------------------------------------------------
avg_group_acur = pd.concat([all_SBJ_acur[SBJ] for SBJ in SBJ_list]).groupby(level=0).mean() # Average
sem_group_acur = pd.concat([all_SBJ_acur[SBJ] for SBJ in SBJ_list]).groupby(level=0).sem()  # Standerd Error

# Plot data frame
# ---------------
plot_df = pd.DataFrame(p_list,columns=['perplexity value'])
for metric in dist_metric_list:
    plot_df[metric] = avg_group_acur[metric].copy()
    plot_df[metric+' +SE'] = avg_group_acur[metric] + sem_group_acur[metric]
    plot_df[metric+' -SE'] = avg_group_acur[metric] - sem_group_acur[metric]

# Plot
# ----
((hv.Area((plot_df['perplexity value' ], plot_df['correlation +SE'], plot_df['correlation -SE']), vdims=['correlation +SE', 'correlation -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=['perplexity value' ,'correlation'], label='correlation'))*\
(hv.Area((plot_df['perplexity value' ], plot_df['cosine +SE'], plot_df['cosine -SE']), vdims=['cosine +SE', 'cosine -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=['perplexity value' ,'cosine'], label='cosine'))*\
(hv.Area((plot_df['perplexity value' ], plot_df['euclidean +SE'], plot_df['euclidean -SE']), vdims=['euclidean +SE', 'euclidean -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=['perplexity value' ,'euclidean'], label='euclidean')))\
.opts(width=700, height=500, xlabel='perplexity value' , ylabel='Average F1 Accuracy',fontsize={'labels':14,'xticks':12,'yticks':12,'legend':14})

# ## UMAP
# ***

# Load F1 accuracy values
# -----------------------
all_SBJ_acur = {}
for SBJ in SBJ_list:
    F1_acur_df = pd.DataFrame(UMAP_k_list, columns=['k value'])
    for metric in dist_metric_list:
        metric_acur_list = []
        for UMAP_k in UMAP_k_list:
            UMAP_file = SBJ+'_UMAP_LRclassrep_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
            UMAP_path = osp.join(PRJDIR,'derivatives','Log_Reg',UMAP_file)
            UMAP_class_report_df = pd.read_csv(UMAP_path)
            F1_acur = UMAP_class_report_df.loc[4,'f1-score']
            metric_acur_list.append(F1_acur)
        F1_acur_df[metric] = metric_acur_list
    all_SBJ_acur[SBJ] = F1_acur_df

# Compute average and standard error of group F1 accuracy
# --------------------------------------------------------
avg_group_acur = pd.concat([all_SBJ_acur[SBJ] for SBJ in SBJ_list]).groupby(level=0).mean() # Average
sem_group_acur = pd.concat([all_SBJ_acur[SBJ] for SBJ in SBJ_list]).groupby(level=0).sem()  # Standerd Error

# Plot data frame
# ---------------
plot_df = pd.DataFrame(LE_k_list,columns=['k-NN value'])
for metric in dist_metric_list:
    plot_df[metric] = avg_group_acur[metric].copy()
    plot_df[metric+' +SE'] = avg_group_acur[metric] + sem_group_acur[metric]
    plot_df[metric+' -SE'] = avg_group_acur[metric] - sem_group_acur[metric]

# Plot
# ----
((hv.Area((plot_df['k-NN value' ], plot_df['correlation +SE'], plot_df['correlation -SE']), vdims=['correlation +SE', 'correlation -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=['k-NN value' ,'correlation'], label='correlation'))*\
(hv.Area((plot_df['k-NN value' ], plot_df['cosine +SE'], plot_df['cosine -SE']), vdims=['cosine +SE', 'cosine -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=['k-NN value' ,'cosine'], label='cosine'))*\
(hv.Area((plot_df['k-NN value' ], plot_df['euclidean +SE'], plot_df['euclidean -SE']), vdims=['euclidean +SE', 'euclidean -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=['k-NN value' ,'euclidean'], label='euclidean')))\
.opts(width=700, height=500, xlabel='k-NN value' , ylabel='Average F1 Accuracy',fontsize={'labels':14,'xticks':12,'yticks':12,'legend':14})