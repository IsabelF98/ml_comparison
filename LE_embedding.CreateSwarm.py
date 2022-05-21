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

# # Laplacian Eigenmap Embedding
#
# This notebook creates the swarm file to compute the LE embedding for all subjects.

import os
import os.path as osp
from utils.data_info import PRJDIR, LE_k_list, conda_loc, conda_env, SBJ_list, wl_sec, tr

# Create logs directory if doesnt already exist
os.system('if [ ! -d ../logs ]; then mkdir ../logs; fi')
os.system('if [ ! -d ../logs/LE_embedding.logs ]; then mkdir ../logs/LE_embedding.logs; fi')

# Create data directory if doesnt already exist
os.system('if [ ! -d ../derivatives ]; then mkdir ../derivatives; fi')
os.system('if [ ! -d ../derivatives/LE ]; then mkdir ../derivatives/LE; fi')

# +
# Create SWARM file
n = 3 # number of dimensions

os.system('echo "#swarm -f ./LE_embedding.SWARM.sh -g 30 -t 30 --time 8:00:00 --logdir ../logs/LE_embedding.logs" > ./LE_embedding.SWARM.sh')
for SBJ in SBJ_list:
    for k in [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34]:
        for metric in ['correlation', 'cosine', 'euclidean']:
            for drop in ['Drop5', 'Drop10', 'Drop15']:
                os.system('echo "export PRJDIR={PRJDIR} conda_loc={conda_loc} conda_env={conda_env} SBJ={SBJ} wl_sec={wl_sec} tr={tr} k={k} n={n} metric={metric} drop={drop}; sh ./LE_embedding.sh" >> ./LE_embedding.SWARM.sh'.format(PRJDIR=PRJDIR, conda_loc=conda_loc, conda_env=conda_env, SBJ=SBJ, wl_sec=wl_sec, tr=tr, k=k, n=n, metric=metric, drop=drop))
# -

