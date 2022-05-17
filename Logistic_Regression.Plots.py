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
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
import panel as pn
hv.extension('bokeh')

dist_metric_list = ['correlation', 'cosine', 'euclidean']
n = 3
drop = 'FullData'

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
            LE_file = SBJ+'_LE_LRclassrep_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
            LE_path = osp.join(PRJDIR,'derivatives','Log_Reg',LE_file)
            LE_class_report_df = pd.read_csv(LE_path)
            F1_acur = LE_class_report_df.loc[4,'f1-score']
            metric_acur_list.append(F1_acur)
        F1_acur_df[metric] = metric_acur_list
    all_SBJ_acur[SBJ] = F1_acur_df

# ## Scatter Plot 1
# Plot with holoviews

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
.opts(width=700, height=500, xlabel='k-NN value' , ylabel='Average F1 Accuracy',fontsize={'labels':16,'xticks':12,'yticks':12,'legend':16}, legend_position='bottom_right')

# ## Scatter Plot 2
# Using seaborn w/ t-test

# +
# Scatter plot data frame
# -----------------------
x_axis = 'k-NN value' 

F1_pointplot_df = pd.DataFrame(columns=['SBJ', x_axis, 'Distance Metric', 'F1 Accuracy'])
for SBJ in SBJ_list:
    F1_df = all_SBJ_acur[SBJ]
    for metric in ['correlation','cosine','euclidean']:
        temp_df = pd.DataFrame({'SBJ': [SBJ]*F1_df.shape[0],
                                x_axis: F1_df['k value'].values,
                                'Distance Metric': [metric]*F1_df.shape[0],
                                'F1 Accuracy': F1_df[metric].values})
        F1_pointplot_df = F1_pointplot_df.append(temp_df).reset_index(drop=True)

# +
# Scatter plot with t-test
# ------------------------
x = x_axis
y = 'F1 Accuracy'
hue = 'Distance Metric'
best_metric = 'correlation'
best_k = 80

pairs = [((best_k, best_metric),(k, best_metric)) for k in LE_k_list]
pairs.remove(((best_k, best_metric),(best_k, best_metric)))

sns.set(rc={'figure.figsize':(14,7)}, font_scale=2)
ax    = sns.pointplot(x=x, y=y, hue=hue, data=F1_pointplot_df, capsize=0.1)
annot = Annotator(ax, pairs, data=F1_pointplot_df, x=x, y=y, hue=hue)
annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()
plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
# -

# ## TSNE
# ***

# Load F1 accuracy values
# -----------------------
all_SBJ_acur = {}
for SBJ in SBJ_list:
    F1_acur_df = pd.DataFrame(p_list, columns=['p value'])
    for metric in dist_metric_list:
        metric_acur_list = []
        for p in p_list:
            TSNE_file = SBJ+'_TSNE_LRclassrep_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
            TSNE_path = osp.join(PRJDIR,'derivatives','Log_Reg',TSNE_file)
            TSNE_class_report_df = pd.read_csv(TSNE_path)
            F1_acur = TSNE_class_report_df.loc[4,'f1-score']
            metric_acur_list.append(F1_acur)
        F1_acur_df[metric] = metric_acur_list
    all_SBJ_acur[SBJ] = F1_acur_df

# ## Scatter Plot 1
# Plot with holoviews

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
.opts(width=700, height=500, xlabel='perplexity value' , ylabel='Average F1 Accuracy',fontsize={'labels':16,'xticks':12,'yticks':12,'legend':16}, legend_position='bottom_right')

# ## Scatter Plot 2
# Using seaborn w/ t-test

# +
# Scatter plot data frame
# -----------------------
x_axis = 'perplexity value' 

F1_pointplot_df = pd.DataFrame(columns=['SBJ', x_axis, 'Distance Metric', 'F1 Accuracy'])
for SBJ in SBJ_list:
    F1_df = all_SBJ_acur[SBJ]
    for metric in ['correlation','cosine','euclidean']:
        temp_df = pd.DataFrame({'SBJ': [SBJ]*F1_df.shape[0],
                                x_axis: F1_df['p value'].values,
                                'Distance Metric': [metric]*F1_df.shape[0],
                                'F1 Accuracy': F1_df[metric].values})
        F1_pointplot_df = F1_pointplot_df.append(temp_df).reset_index(drop=True)

# +
# Scatter plot w/ t-test
# ----------------------
x = x_axis
y = 'F1 Accuracy'
hue = 'Distance Metric'
best_metric = 'cosine'
best_p = 50

pairs = [((best_p, best_metric),(p, best_metric)) for p in p_list]
pairs.remove(((best_p, best_metric),(best_p, best_metric)))

sns.set(rc={'figure.figsize':(14,7)}, font_scale=2)
ax    = sns.pointplot(x=x, y=y, hue=hue, data=F1_pointplot_df, capsize=0.1)
annot = Annotator(ax, pairs, data=F1_pointplot_df, x=x, y=y, hue=hue)
annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()
plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
# -

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
            UMAP_file = SBJ+'_UMAP_LRclassrep_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
            UMAP_path = osp.join(PRJDIR,'derivatives','Log_Reg',UMAP_file)
            UMAP_class_report_df = pd.read_csv(UMAP_path)
            F1_acur = UMAP_class_report_df.loc[4,'f1-score']
            metric_acur_list.append(F1_acur)
        F1_acur_df[metric] = metric_acur_list
    all_SBJ_acur[SBJ] = F1_acur_df

# ## Scatter Plot 1
# Plot with holoviews

# Compute average and standard error of group F1 accuracy
# --------------------------------------------------------
avg_group_acur = pd.concat([all_SBJ_acur[SBJ] for SBJ in SBJ_list]).groupby(level=0).mean() # Average
sem_group_acur = pd.concat([all_SBJ_acur[SBJ] for SBJ in SBJ_list]).groupby(level=0).sem()  # Standerd Error

# Plot data frame
# ---------------
plot_df = pd.DataFrame(UMAP_k_list,columns=['k-NN value'])
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
.opts(width=700, height=500, xlabel='k-NN value' , ylabel='Average F1 Accuracy',fontsize={'labels':16,'xticks':12,'yticks':12,'legend':16}, legend_position='top_left')

# ## Scatter Plot 2
# Using seaborn w/ t-test

# +
# Scatter plot data frame
# -----------------------
x_axis = 'k-NN value' 

F1_pointplot_df = pd.DataFrame(columns=['SBJ', x_axis, 'Distance Metric', 'F1 Accuracy'])
for SBJ in SBJ_list:
    F1_df = all_SBJ_acur[SBJ]
    for metric in ['correlation','cosine','euclidean']:
        temp_df = pd.DataFrame({'SBJ': [SBJ]*F1_df.shape[0],
                                x_axis: F1_df['k value'].values,
                                'Distance Metric': [metric]*F1_df.shape[0],
                                'F1 Accuracy': F1_df[metric].values})
        F1_pointplot_df = F1_pointplot_df.append(temp_df).reset_index(drop=True)

# +
# Scatter plot w/ t-test
# ----------------------
x = x_axis
y = 'F1 Accuracy'
hue = 'Distance Metric'
best_metric = 'euclidean'
best_k = 200

pairs = [((best_k, best_metric),(k, best_metric)) for k in UMAP_k_list]
pairs.remove(((best_k, best_metric),(best_k, best_metric)))

sns.set(rc={'figure.figsize':(14,7)}, font_scale=2)
ax    = sns.pointplot(x=x, y=y, hue=hue, data=F1_pointplot_df, capsize=0.1)
annot = Annotator(ax, pairs, data=F1_pointplot_df, x=x, y=y, hue=hue)
annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()
plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
# -

# ## Box Plot
# ***

# Embedding parameters
# --------------------
best_k = 200 # best k or p value
k_index = UMAP_k_list.index(best_k) # k or p value index (change list acording to method)

# Blox plot data frame
# --------------------
F1_boxplot_df = pd.DataFrame(columns=['SBJ', 'Distance Metric', 'F1 Accuracy'])
for SBJ in SBJ_list:
    F1_cor = all_SBJ_acur[SBJ].loc[k_index, 'correlation']
    F1_cos = all_SBJ_acur[SBJ].loc[k_index, 'cosine']
    F1_euc = all_SBJ_acur[SBJ].loc[k_index, 'euclidean']
    F1_boxplot_df.loc[F1_boxplot_df.shape[0]] = {'SBJ': SBJ, 'Distance Metric': 'correlation', 'F1 Accuracy': F1_cor}
    F1_boxplot_df.loc[F1_boxplot_df.shape[0]] = {'SBJ': SBJ, 'Distance Metric': 'cosine', 'F1 Accuracy': F1_cos}
    F1_boxplot_df.loc[F1_boxplot_df.shape[0]] = {'SBJ': SBJ, 'Distance Metric': 'euclidean', 'F1 Accuracy': F1_euc}

# +
# Box plot w/ t-test
# ------------------
x = 'Distance Metric'
y = 'F1 Accuracy'
order = ['correlation','cosine','euclidean']
pairs=[("correlation", "cosine"), ("cosine", "euclidean"), ("euclidean", "correlation")]

sns.set(rc={'figure.figsize':(9,7)}, font_scale=2)
ax    = sns.boxplot(x=x, y=y, data=F1_boxplot_df, order=order)
annot = Annotator(ax, pairs, data=F1_boxplot_df, x=x, y=y, order=order)
annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()
# -


