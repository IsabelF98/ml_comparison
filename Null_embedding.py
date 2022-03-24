#! /usr/bin/env python
# Isabel Fernandez 3/24/2022

# This file computes embeddings using each techniques (LE, TSNE, UMAP) on null data
# 1) Load sliding window correlation matrix
# 2) Drop inbetween task windows
# 3) Compute null data (randomize connections or Dan's method)
# 4) Compute embeddings
# 5) Save files

import argparse
import pandas as pd
import numpy as np
import os.path as osp
from scipy.spatial.distance import correlation, cosine, euclidean
from utils.embedding_functions import Uniform_Manifold_Approximation_Projection, T_Stochastic_Neighbor_Embedding, Laplacain_Eigenmap
from utils.data_functions import randomize_conn
from utils.data_info import PRJDIR, task_labels

def run(args):
    SBJ    = args.subject
    wl_sec = args.window_length_sec
    tr     = args.time_resolution
    LE_k   = args.LE_k
    p      = args.p
    UMAP_k = args.UMAP_k
    n      = args.n
    metric = args.metric
    null   = args.null
    print(' ')
    print('++ INFO: Run information')
    print('         SBJ:   ',SBJ)
    print('         wl_sec:',wl_sec)
    print('         tr:    ',tr)
    print('         LE_k:  ',LE_k)
    print('         p:     ',p)
    print('         UMAP_k:',UMAP_k)
    print('         n:     ',n)
    print('         metric:',metric)
    print('         null:  ',null)
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
    
    # Compute null data
    # -----------------
    if null == 'randomize_conn':
        null_SWC_df = randomize_conn(drop_SWC_df)
    print('++ INFO: Null data computed')
    print('         Data shape:',null_SWC_df.shape)
    print(' ')
    
    # Compute LE Embedding
    # --------------------
    dist_metric_dict = {'correlation':correlation, 'cosine':cosine, 'euclidean':euclidean}
    LE_df = Laplacain_Eigenmap(null_SWC_df,k=LE_k,n=n,metric=dist_metric_dict[metric])
    print('++INFO: Laplacian Eigenmap embedding computed')
    print('        Data shape:',LE_df.shape)
    print(' ')
    
    # Save LE file to outside directory
    # ---------------------------------
    out_file = SBJ+'_Null_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','Null_Data',out_file)
    LE_df.to_csv(out_path, index=False)
    print('++ INFO: LE data saved to')
    print('       ',out_path)
    print(' ')
    
    # Compute Embedding
    # -----------------
    TSNE_df = T_Stochastic_Neighbor_Embedding(null_SWC_df,p=p,n=n,metric=metric)
    print('++ INFO: TSNE embedding computed')
    print('         Data shape:',TSNE_df.shape)
    print(' ')
    
    # Save file to outside directory
    # ------------------------------
    out_file = SBJ+'_Null_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','Null_Data',out_file)
    TSNE_df.to_csv(out_path, index=False)
    print('++ INFO: TSNE data saved to')
    print('       ',out_path)
    print(' ')
    
    # Compute Embedding
    # -----------------
    UMAP_df = Uniform_Manifold_Approximation_Projection(null_SWC_df,k=UMAP_k,n=n,metric=metric)
    print('++INFO: UMAP embedding computed')
    print('        Data shape:',UMAP_df.shape)
    print(' ')
    
    # Save file to outside directory
    # ------------------------------
    out_file = SBJ+'_Null_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','Null_Data',out_file)
    UMAP_df.to_csv(out_path, index=False)
    print('++ INFO: UMAP data saved to')
    print('       ',out_path)

def main():
    parser=argparse.ArgumentParser(description="Compute embeddings with null data.")
    parser.add_argument("-sbj", help="subject name in SBJXX format", dest="subject", type=str, required=True)
    parser.add_argument("-wl_sec", help="window length in seconds", dest="window_length_sec", type=int, required=True)
    parser.add_argument("-tr", help="time resolution", dest="time_resolution", type=float, required=True)
    parser.add_argument("-LE_k", help="LE Nearest Neighboor value", dest="LE_k", type=int, required=True)
    parser.add_argument("-p", help="TSNE perplexity value", dest="p", type=int, required=True)
    parser.add_argument("-UMAP_k",help="UMAP Nearest Neighboor value", dest="UMAP_k", type=int, required=True)
    parser.add_argument("-n", help="number of dimensions", dest="n", type=int, required=True)
    parser.add_argument("-met", help="distance metric (correlation, cosine, euclidean)", dest="metric", type=str, required=True)
    parser.add_argument("-null", help="Method for comuting null data (randomize_conn or X)", dest="null", type=str, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()