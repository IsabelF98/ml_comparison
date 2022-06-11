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

# # Drop windows in SWC matrix
#
# This notebook creates the swarm file to drop windows (every 5, 10, 15) in SWC matrix.

import os
import os.path as osp
from utils.data_info import PRJDIR, conda_loc, conda_env, SBJ_list, wl_sec, tr

# Create logs directory if doesnt already exist
os.system('if [ ! -d ../logs ]; then mkdir ../logs; fi')
os.system('if [ ! -d ../logs/Drop_Windows.logs ]; then mkdir ../logs/Drop_Windows.logs; fi')

# Create SWARM file
os.system('echo "#swarm -f ./Drop_Windows.SWARM.sh -g 30 -t 30 --time 8:00:00 --logdir ../logs/Drop_Windows.logs" > ./Drop_Windows.SWARM.sh')
for SBJ in SBJ_list:
    for ndrop in [5,10,15]:
        os.system('echo "export PRJDIR={PRJDIR} conda_loc={conda_loc} conda_env={conda_env} SBJ={SBJ} wl_sec={wl_sec} tr={tr} ndrop={ndrop}; sh ./Drop_Windows.sh" >> ./Drop_Windows.SWARM.sh'.format(PRJDIR=PRJDIR, conda_loc=conda_loc, conda_env=conda_env, SBJ=SBJ, wl_sec=wl_sec, tr=tr, ndrop=ndrop))


