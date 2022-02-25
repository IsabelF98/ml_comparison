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

# # Silhouette Index
#
# This notebook creates the swarm file to compute the silhouette index for all subjects.

import os
import os.path as osp
from utils.data_info import PRJDIR, conda_loc, conda_env, SBJ_list, wl_sec, tr

# Create logs directory if doesnt already exist
os.system('if [ ! -d ../logs ]; then mkdir ../logs; fi')
os.system('if [ ! -d ../logs/Silhouette_Index.logs ]; then mkdir ../logs/Silhouette_Index.logs; fi')

# Create data directory if doesnt already exist
os.system('if [ ! -d ../derivatives ]; then mkdir ../derivatives; fi')
os.system('if [ ! -d ../derivatives/Silh_Idx ]; then mkdir ../derivatives/Silh_Idx; fi')

# Create SWARM file
os.system('echo "#swarm -f ./Silhouette_Index.SWARM.sh -g 30 -t 30 --time 8:00:00 --logdir ../logs/Silhouette_Index.logs" > ./Silhouette_Index.SWARM.sh')
for SBJ in SBJ_list:
    for embedding in ['TSNE']:
        os.system('echo "export PRJDIR={PRJDIR} conda_loc={conda_loc} conda_env={conda_env} SBJ={SBJ} wl_sec={wl_sec} tr={tr} embedding={embedding}; sh ./Silhouette_Index.sh" >> ./Silhouette_Index.SWARM.sh'.format(PRJDIR=PRJDIR, conda_loc=conda_loc, conda_env=conda_env, SBJ=SBJ, wl_sec=wl_sec, tr=tr, embedding=embedding))


