#! /usr/bin/env python
# Isabel Fernandez 2/24/2022

# This file computes the embeddings using the T-distribution Stochastic Neighbor Embedding algorithum
# 1) Load ROI sliding window correlation matrix
# 2) Drop inbetween task windows (if not full data)
# 3) Compute embedding
# 4) Save file

import argparse
import pandas as pd
import numpy as np
import os.path as osp
from scipy.spatial.distance import correlation, cosine, euclidean
from utils.embedding_functions import T_Stochastic_Neighbor_Embedding
from utils.data_info import PRJDIR, task_labels

def run(args):
    SBJ    = args.subject
    wl_sec = args.window_length_sec
    tr     = args.time_resolution
    p      = args.p
    n      = args.n
    metric = args.metric
    drop   = args.drop
    print(' ')
    print('++ INFO: Run information')
    print('         SBJ:   ',SBJ)
    print('         wl_sec:',wl_sec)
    print('         tr:    ',tr)
    print('         p:     ',p)
    print('         n:     ',n)
    print('         metric:',metric)
    print('         drop:  ',drop)
    print(' ')
    
    # Load SWC matrix
    # ---------------
    file_name = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'.csv'
    file_path = osp.join(PRJDIR,'derivatives','SWC',file_name)
    SWC_df    = pd.read_csv(file_path)  
    print('++ INFO: SWC matrix loaded')
    print('         Data shape:',SWC_df.shape)
    print(' ')
    
    if drop == 'DropData':
        # Drop inbwtween task windows
        # ---------------------------
        wl_trs = int(wl_sec/tr)
        task_df     = task_labels(wl_trs, PURE=False) # USE YOUR OWN FUNCTION TO LOAD TASK LABELS AS PD.DATAFRAME
        drop_index  = task_df.index[task_df['Task'] == 'Inbetween']
        drop_SWC_df = SWC_df.drop(drop_index).reset_index(drop=True)
        print('++ INFO: Inbetween task windows dropped')
        print('         Data shape:',drop_SWC_df.shape)
        print(' ')
    elif drop == 'FullData':
        drop_SWC_df = SWC_df.copy()
    
    # Compute Embedding
    # -----------------
    TSNE_df = T_Stochastic_Neighbor_Embedding(drop_SWC_df,p=p,n=n,metric=metric)
    print('++ INFO: TSNE embedding computed')
    print('         Data shape:',TSNE_df.shape)
    print(' ')
    
    # Save file to outside directory
    # ------------------------------
    out_file = SBJ+'_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','TSNE',out_file)
    TSNE_df.to_csv(out_path, index=False)
    print('++ INFO: Data saved to')
    print('       ',out_path)

def main():
    parser=argparse.ArgumentParser(description="Compute embeddings using the Laplacian Eigenmap algorithum.")
    parser.add_argument("-sbj",help="subject name in SBJXX format", dest="subject", type=str, required=True)
    parser.add_argument("-wl_sec",help="window length in seconds", dest="window_length_sec", type=int, required=True)
    parser.add_argument("-tr",help="time resolution", dest="time_resolution", type=float, required=True)
    parser.add_argument("-p",help="perplexity value", dest="p", type=int, required=True)
    parser.add_argument("-n",help="number of dimensions", dest="n", type=int, required=True)
    parser.add_argument("-met",help="distance metric (correlation, cosine, euclidean)", dest="metric", type=str, required=True)
    parser.add_argument("-drop", help="Drop inbetween windows or full data (DropData or FullData)", dest="drop", type=str, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()