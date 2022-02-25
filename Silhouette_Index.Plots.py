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
from utils.data_info import PRJDIR, wl_sec, tr, SBJ_list, task_labels
import matplotlib.pyplot as plt
import holoviews as hv
import panel as pn
hv.extension('bokeh')

# +
# Load Silhouette Index Data
# --------------------------
embedding = 'LE'
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
# Plot data
# ---------
