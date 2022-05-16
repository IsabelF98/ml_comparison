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
# This notebook is for creating the plots of silhouette index values for each manifold learning technique.

import pandas as pd
import numpy as np
import os.path as osp
from utils.data_info import PRJDIR, LE_k_list, p_list, UMAP_k_list, wl_sec, tr, SBJ_list, task_labels
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
import panel as pn
hv.extension('bokeh')

# +
# Load Silhouette Index Data
# --------------------------
embedding = 'UMAP' # CHOOSE EMBEDDING ('LE', 'TSNE', or 'UMAP')
drop = 'FullData'
all_SBJ_SI = {}

for SBJ in SBJ_list:
    file_name = SBJ+'_Silh_Idx_'+embedding+'_wl'+str(wl_sec).zfill(3)+'_'+drop+'.csv'
    file_path = osp.join(PRJDIR,'derivatives','Silh_Idx',file_name)
    SI_df = pd.read_csv(file_path)
    all_SBJ_SI[SBJ] = SI_df
    print('++ INFO: Data loaded for',SBJ)
# -

# ## Scatter Plot 1
# Plot with holoviews

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
.opts(width=700, height=500, xlabel=x_axis, ylabel='Average Silhouette Index',fontsize={'labels':14,'xticks':12,'yticks':12,'legend':14}, legend_position='top_left')
# -
# ## Scatter Plot 2
# Using seaborn w/ t-test

# +
# Scatter plot data frame
# -----------------------
if embedding == 'LE' or embedding == 'UMAP':
    x_axis = 'k-NN value' 
elif embedding == 'TSNE':
    x_axis = 'perplexity value'

SI_pointplot_df = pd.DataFrame(columns=['SBJ', x_axis, 'Distance Metric', 'Silhouette Index'])
for SBJ in SBJ_list:
    SI_df = all_SBJ_SI[SBJ]
    for metric in ['correlation','cosine','euclidean']:
        temp_df = pd.DataFrame({'SBJ': [SBJ]*SI_df.shape[0],
                                x_axis: SI_df['Unnamed: 0'].values,
                                'Distance Metric': [metric]*SI_df.shape[0],
                                'Silhouette Index': SI_df[metric].values})
        SI_pointplot_df = SI_pointplot_df.append(temp_df).reset_index(drop=True)

# +
# Scatter plot with t-test
# ------------------------
x = x_axis
y = 'Silhouette Index'
hue = 'Distance Metric'
best_metric = 'euclidean'
best_k = 160

pairs = [((best_k, best_metric),(k, best_metric)) for k in UMAP_k_list]
pairs.remove(((best_k, best_metric),(best_k, best_metric)))

sns.set(rc = {'figure.figsize':(14,7)})
ax    = sns.pointplot(x=x, y=y, hue=hue, data=SI_pointplot_df, capsize=0.1)
annot = Annotator(ax, pairs, data=SI_pointplot_df, x=x, y=y, hue=hue)
annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()
plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
# -

# ## Box Plot
# Comparing distance metrics for best k or p value

# Embedding parameters
# --------------------
best_k = 160 # best k or p value
k_index = UMAP_k_list.index(best_k) # k or p value index (change list acording to method)

# Bok plot data frame
# -------------------
SI_boxplot_df = pd.DataFrame(columns=['SBJ', 'Distance Metric', 'Silhouette index'])
for SBJ in SBJ_list:
    SI_cor = all_SBJ_SI[SBJ].loc[k_index, 'correlation']
    SI_cos = all_SBJ_SI[SBJ].loc[k_index, 'cosine']
    SI_euc = all_SBJ_SI[SBJ].loc[k_index, 'euclidean']
    SI_boxplot_df.loc[SI_boxplot_df.shape[0]] = {'SBJ': SBJ, 'Distance Metric': 'correlation', 'Silhouette index': SI_cor}
    SI_boxplot_df.loc[SI_boxplot_df.shape[0]] = {'SBJ': SBJ, 'Distance Metric': 'cosine', 'Silhouette index': SI_cos}
    SI_boxplot_df.loc[SI_boxplot_df.shape[0]] = {'SBJ': SBJ, 'Distance Metric': 'euclidean', 'Silhouette index': SI_euc}

# +
# Box plot with t-test
# --------------------
x = 'Distance Metric'
y = 'Silhouette index'
order = ['correlation','cosine','euclidean']
pairs=[("correlation", "cosine"), ("cosine", "euclidean"), ("euclidean", "correlation")]

sns.set(rc = {'figure.figsize':(9,7)})
ax    = sns.boxplot(x=x, y=y, data=SI_boxplot_df, order=order)
annot = Annotator(ax, pairs, data=SI_boxplot_df, x=x, y=y, order=order)
annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()
