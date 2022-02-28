#! /usr/bin/env python
# Isabel Fernandez 2/28/2022

# This file computes the embeddings using the Uniform Manifold Approximation and Projection algorithum
# 1) Load sliding window correlation matrix
# 2) Drop inbetween task windows
# 3) Compute embedding
# 4) Save file

import argparse
import pandas as pd
import numpy as np
import os.path as osp
from scipy.spatial.distance import correlation, cosine, euclidean
from utils.embedding_functions import Uniform_Manifold_Approximation_Projection
from utils.data_info import PRJDIR, task_labels

def run(args):
    SBJ    = args.subject
    wl_sec = args.window_length_sec
    tr     = args.time_resolution
    k      = args.k
    n      = args.n
    metric = args.metric
    print(' ')
    print('++ INFO: Run information')
    print('         SBJ:   ',SBJ)
    print('         wl_sec:',wl_sec)
    print('         tr:    ',tr)
    print('         k:     ',k)
    print('         n:     ',n)
    print('         metric:',metric)
    print(' ')
    
    # Load SWC matrix
    # ---------------
    file_name = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'.csv'
    file_path = osp.join(PRJDIR,'derivatives','SWC',file_name)
    SWC_df    = pd.read_csv(file_path)  
    print('++ INFO: SWC matrix loaded')
    print('         Data shape:',SWC_df.shape)
    print(' ')
    
    # Drop inbwtween task windows
    # ---------------------------
    wl_trs = int(wl_sec/tr)
    task_df     = task_labels(wl_trs, PURE=False) # USE YOUR OWN FUNCTION TO LOAD TASK LABELS AS PD.DATAFRAME
    drop_index  = task_df.index[task_df['Task'] == 'Inbetween']
    drop_SWC_df = SWC_df.drop(drop_index).reset_index(drop=True)
    print('++ INFO: Inbetween task windows dropped')
    print('         Data shape:',drop_SWC_df.shape)
    print(' ')
    
    # Compute Embedding
    # -----------------
    UMAP_df = Uniform_Manifold_Approximation_Projection(drop_SWC_df,k=k,n=n,metric=metric)
    print('++INFO: UMAP embedding computed')
    print('        Data shape:',UMAP_df.shape)
    print(' ')
    
    # Save file to outside directory
    # ------------------------------
    out_file = SBJ+'_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','UMAP',out_file)
    UMAP_df.to_csv(out_path, index=False)
    print('++ INFO: Data saved to')
    print('       ',out_path)

def main():
    parser=argparse.ArgumentParser(description="Compute embeddings using the UMAP algorithum.")
    parser.add_argument("-sbj",help="subject name in SBJXX format", dest="subject", type=str, required=True)
    parser.add_argument("-wl_sec",help="window length in seconds", dest="window_length_sec", type=int, required=True)
    parser.add_argument("-tr",help="time resolution", dest="time_resolution", type=float, required=True)
    parser.add_argument("-k",help="Nearest Neighboor value", dest="k", type=int, required=True)
    parser.add_argument("-n",help="number of dimensions", dest="n", type=int, required=True)
    parser.add_argument("-met",help="distance metric (correlation, cosine, euclidean)", dest="metric", type=str, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()