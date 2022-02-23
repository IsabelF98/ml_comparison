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
from utils.data_info import DATADIR, PRJDIR, SBJ_list, wl_sec, tr

# Create logs directory if doesnt already exist
os.system('if [ ! -d ../logs ]; then mkdir ../logs; fi')
os.system('if [ ! -d ../logs/LE_embedding.logs ]; then mkdir ../logs/LE_embedding.logs; fi')

# Create data directory if doesnt already exist
os.system('if [ ! -d ../derivatives ]; then mkdir ../derivatives; fi')
os.system('if [ ! -d ../derivatives/LE ]; then mkdir ../derivatives/LE; fi')

# Create SWARM file
n = 3 # number of dimensions
os.system('echo "#swarm -f ./LE_embedding.SWARM.sh -g 30 -t 30 --time 8:00:00 --logdir ../logs/LE_embedding.logs" > ./LE_embedding.SWARM.sh')
for SBJ in SBJ_list:
    for k in [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,120,130,140,150,160,170,180,190,200]:
        for metric in ['correlation', 'cosine', 'euclidean']:
            os.system('echo "./LE_embedding.py -sbj {SBJ} -wl_sec {wl_sec} -tr {tr} -k {k} -n {n} -met {metric}" >> ./LE_embedding.SWARM.sh'.format(SBJ=SBJ,
                                                                                                                                          wl_sec=wl_sec,
                                                                                                                                          tr=tr,
                                                                                                                                          k=k,
                                                                                                                                          n=n,
                                                                                                                                          metric=metric))


