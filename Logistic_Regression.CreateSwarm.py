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

# # Logistic Regression
#
# This notebook creates the swarm file to compute Logistic Regression coefficients and accuracy.

import os
import os.path as osp
from utils.data_info import PRJDIR, conda_loc, conda_env, SBJ_list, wl_sec, tr

LE_k   = 50 # Best k-NN value for LE
p      = 55 # Best perplexity value for TSNE
UMAP_k = 130 # Best k-NN value for UMAP
n = 30 # Number of dimesnions for embedding
metric = 'correlation' # Best distance metric

# Create logs directory if doesnt already exist
os.system('if [ ! -d ../logs ]; then mkdir ../logs; fi')
os.system('if [ ! -d ../logs/Logistic_Regression.logs ]; then mkdir ../logs/Logistic_Regression.logs; fi')

# Create data directory if doesnt already exist
os.system('if [ ! -d ../derivatives ]; then mkdir ../derivatives; fi')
os.system('if [ ! -d ../derivatives/Log_Reg ]; then mkdir ../derivatives/Log_Reg; fi')

# Create SWARM file
os.system('echo "#swarm -f ./Logistic_Regression.SWARM.sh -g 30 -t 30 --time 8:00:00 --logdir ../logs/Logistic_Regression.logs" > ./Logistic_Regression.SWARM.sh')
for SBJ in SBJ_list:
    os.system('echo "export PRJDIR={PRJDIR} conda_loc={conda_loc} conda_env={conda_env} SBJ={SBJ} wl_sec={wl_sec} tr={tr} LE_k={LE_k} p={p} UMAP_k={UMAP_k} n={n} metric={metric}; sh ./Logistic_Regression.sh" >> ./Logistic_Regression.SWARM.sh'.format(PRJDIR=PRJDIR, conda_loc=conda_loc, conda_env=conda_env, SBJ=SBJ, wl_sec=wl_sec, tr=tr, LE_k=LE_k, p=p, UMAP_k=UMAP_k, n=n, metric=metric))


