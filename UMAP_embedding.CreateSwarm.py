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

# # Uniform Manifold Approximation and Projection Embedding
#
# This notebook creates the swarm file to compute the UMAP embedding for all subjects.

import os
import os.path as osp
from utils.data_info import PRJDIR, UMAP_k_list, conda_loc, conda_env, SBJ_list, wl_sec, tr

# Create logs directory if doesnt already exist
os.system('if [ ! -d ../logs ]; then mkdir ../logs; fi')
os.system('if [ ! -d ../logs/UMAP_embedding.logs ]; then mkdir ../logs/UMAP_embedding.logs; fi')

# Create data directory if doesnt already exist
os.system('if [ ! -d ../derivatives ]; then mkdir ../derivatives; fi')
os.system('if [ ! -d ../derivatives/UMAP ]; then mkdir ../derivatives/UMAP; fi')

# +
# Create SWARM file
n = 3 # number of dimensions

os.system('echo "#swarm -f ./UMAP_embedding.SWARM.sh -g 30 -t 30 --time 8:00:00 --logdir ../logs/UMAP_embedding.logs" > ./UMAP_embedding.SWARM.sh')
for SBJ in SBJ_list:
    for k in [3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24]:
        for metric in ['correlation', 'cosine', 'euclidean']:
            for drop in ['Drop5', 'Drop10', 'Drop15']:
                os.system('echo "export PRJDIR={PRJDIR} conda_loc={conda_loc} conda_env={conda_env} SBJ={SBJ} wl_sec={wl_sec} tr={tr} k={k} n={n} metric={metric} drop={drop}; sh ./UMAP_embedding.sh" >> ./UMAP_embedding.SWARM.sh'.format(PRJDIR=PRJDIR, conda_loc=conda_loc, conda_env=conda_env, SBJ=SBJ, wl_sec=wl_sec, tr=tr, k=k, n=n, metric=metric, drop=drop))
# -

