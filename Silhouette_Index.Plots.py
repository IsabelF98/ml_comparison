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

# # Silhouette Index Plots
#
# This notebook is for creating the plots of silhouette index for each manifold learning technique.

import pandas as pd
import numpy as np
import os.path as osp
from utils.data_info import PRJDIR, LE_k_list, p_list, UMAP_k_list, wl_sec, tr, SBJ_list, task_labels
import matplotlib.pyplot as plt
import holoviews as hv
import panel as pn
hv.extension('bokeh')

# +
# Load Silhouette Index Data
# --------------------------
embedding = 'UMAP' # CHOOSE EMBEDDING ('LE', 'TSNE', or 'UMAP')
all_SBJ_SI = {}

for SBJ in SBJ_list:
    file_name = SBJ+'_Silh_Idx_'+embedding+'_wl'+str(wl_sec).zfill(3)+'.csv'
    file_path = osp.join(PRJDIR,'derivatives','Silh_Idx',file_name)
    SI_df = pd.read_csv(file_path)
    all_SBJ_SI[SBJ] = SI_df
    print('++ INFO: Data loaded for',SBJ)
# -

# Compute average and standard error of group silhouette index
# ------------------------------------------------------------
avg_group_SI = pd.concat([all_SBJ_SI[SBJ] for SBJ in SBJ_list]).groupby(level=0).mean() # Average
sem_group_SI = pd.concat([all_SBJ_SI[SBJ] for SBJ in SBJ_list]).groupby(level=0).sem()  # Standerd Error

# +
# Plot data frame
# ---------------
if embedding == 'LE':
    plot_df = pd.DataFrame(LE_k_list,columns=['k-NN value'])  
elif embedding == 'TSNE':
    plot_df = pd.DataFrame(p_list,columns=['perplexity value'])
elif embedding == 'UMAP':
    plot_df = pd.DataFrame(UMAP_k_list,columns=['k-NN value'])
    
for metric in ['correlation','cosine','euclidean']:
    plot_df[metric] = avg_group_SI[metric].copy()
    plot_df[metric+' +SE'] = avg_group_SI[metric] + sem_group_SI[metric]
    plot_df[metric+' -SE'] = avg_group_SI[metric] - sem_group_SI[metric]

# +
# Plot
# ----
if embedding == 'LE' or embedding == 'UMAP':
    x_axis = 'k-NN value' 
elif embedding == 'TSNE':
    x_axis = 'perplexity value'

((hv.Area((plot_df[x_axis], plot_df['correlation +SE'], plot_df['correlation -SE']), vdims=['correlation +SE', 'correlation -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=[x_axis,'correlation'], label='correlation'))*\
(hv.Area((plot_df[x_axis], plot_df['cosine +SE'], plot_df['cosine -SE']), vdims=['cosine +SE', 'cosine -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=[x_axis,'cosine'], label='cosine'))*\
(hv.Area((plot_df[x_axis], plot_df['euclidean +SE'], plot_df['euclidean -SE']), vdims=['euclidean +SE', 'euclidean -SE']).opts(alpha=0.3)*\
hv.Points(plot_df, kdims=[x_axis,'euclidean'], label='euclidean')))\
.opts(width=700, height=500, xlabel=x_axis, ylabel='Average Silhouette Index',fontsize={'labels':14,'xticks':12,'yticks':12,'legend':14})
# -

