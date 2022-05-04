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

# # TSNE Embedding
#
# This notebook creates the swarm file to compute the TSNE embedding for all subjects.

import os
import os.path as osp
from utils.data_info import PRJDIR, p_list, conda_loc, conda_env, SBJ_list, wl_sec, tr

# Create logs directory if doesnt already exist
os.system('if [ ! -d ../logs ]; then mkdir ../logs; fi')
os.system('if [ ! -d ../logs/TSNE_embedding.logs ]; then mkdir ../logs/TSNE_embedding.logs; fi')

# Create data directory if doesnt already exist
os.system('if [ ! -d ../derivatives ]; then mkdir ../derivatives; fi')
os.system('if [ ! -d ../derivatives/TSNE ]; then mkdir ../derivatives/TSNE; fi')

# +
# Create SWARM file
n = 3 # number of dimensions

os.system('echo "#swarm -f ./TSNE_embedding.SWARM.sh -g 30 -t 30 --time 8:00:00 --logdir ../logs/TSNE_embedding.logs" > ./TSNE_embedding.SWARM.sh')
for SBJ in SBJ_list:
    for p in p_list:
        for metric in ['correlation', 'cosine', 'euclidean']:
            for drop in ['Drop5', 'Drop10', 'Drop15']:
                os.system('echo "export PRJDIR={PRJDIR} conda_loc={conda_loc} conda_env={conda_env} SBJ={SBJ} wl_sec={wl_sec} tr={tr} p={p} n={n} metric={metric} drop={drop}; sh ./TSNE_embedding.sh" >> ./TSNE_embedding.SWARM.sh'.format(PRJDIR=PRJDIR, conda_loc=conda_loc, conda_env=conda_env, SBJ=SBJ, wl_sec=wl_sec, tr=tr, p=p, n=n, metric=metric, drop=drop))
# -

