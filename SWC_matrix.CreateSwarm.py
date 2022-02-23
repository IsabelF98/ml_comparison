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

# # Sliding Window Correlation Metrix
#
# This notebook creates the swarm file to compute the SWC matrix for all subjects.

import os
import os.path as osp
from utils.data_info import PRJDIR, conda_loc, conda_env, SBJ_list, wl_sec, tr, ws_trs

# Create logs directory if doesnt already exist
os.system('if [ ! -d ../logs ]; then mkdir ../logs; fi')
os.system('if [ ! -d ../logs/SWC_matrix.logs ]; then mkdir ../logs/SWC_matrix.logs; fi')

# Create data directory if doesnt already exist
os.system('if [ ! -d ../derivatives ]; then mkdir ../derivatives; fi')
os.system('if [ ! -d ../derivatives/SWC ]; then mkdir ../derivatives/SWC; fi')

# Create SWARM file
os.system('echo "#swarm -f ./SWC_matrix.SWARM.sh -g 30 -t 30 --time 8:00:00 --logdir ../logs/SWC_matrix.logs" > ./SWC_matrix.SWARM.sh')
for SBJ in SBJ_list:
    os.system('echo "export PRJDIR={PRJDIR} conda_loc={conda_loc} conda_env={conda_env} SBJ={SBJ} wl_sec={wl_sec} tr={tr} ws_trs={ws_trs}; sh ./SWC_matrix.sh" >> ./SWC_matrix.SWARM.sh'.format(PRJDIR=PRJDIR, conda_loc=conda_loc, conda_env=conda_env, SBJ=SBJ, wl_sec=wl_sec, tr=tr, ws_trs=ws_trs))



