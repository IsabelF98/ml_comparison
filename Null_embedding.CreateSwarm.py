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

# # Null Data Embeddings
#
# This notebook creates the swarm file to compute embeddings using null data.

import os
import os.path as osp
from utils.data_info import PRJDIR, conda_loc, conda_env, SBJ_list, wl_sec, tr, ws_trs

LE_k   = 130 # Best k-NN value for LE
p      = 50 # Best perplexity value for TSNE
UMAP_k = 95 # Best k-NN value for UMAP
n = 3 # Number of dimesnions to reduce
metric = 'correlation' # Best distance metric

# Create logs directory if doesnt already exist
os.system('if [ ! -d ../logs ]; then mkdir ../logs; fi')
os.system('if [ ! -d ../logs/Null_embedding.logs ]; then mkdir ../logs/Null_embedding.logs; fi')

# Create data directory if doesnt already exist
os.system('if [ ! -d ../derivatives ]; then mkdir ../derivatives; fi')
os.system('if [ ! -d ../derivatives/Null_Data ]; then mkdir ../derivatives/Null_Data; fi')

# Create SWARM file
os.system('echo "#swarm -f ./Null_embedding.SWARM.sh -g 30 -t 30 --time 8:00:00 --logdir ../logs/Null_embedding.logs" > ./Null_embedding.SWARM.sh')
for SBJ in SBJ_list:
    for data in ['ROI', 'SWC']:
        os.system('echo "export PRJDIR={PRJDIR} conda_loc={conda_loc} conda_env={conda_env} SBJ={SBJ} wl_sec={wl_sec} tr={tr} ws_trs={ws_trs} LE_k={LE_k} p={p} UMAP_k={UMAP_k} n={n} metric={metric} data={data}; sh ./Null_embedding.sh" >> ./Null_embedding.SWARM.sh'.format(PRJDIR=PRJDIR, conda_loc=conda_loc, conda_env=conda_env, SBJ=SBJ, wl_sec=wl_sec, tr=tr, ws_trs=ws_trs, LE_k=LE_k, p=p, UMAP_k=UMAP_k, n=n, metric=metric, data=data))


