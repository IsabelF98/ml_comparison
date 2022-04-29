#! /usr/bin/env python
# Isabel Fernandez 3/24/2022

# This file computes embeddings using each techniques (LE, TSNE, UMAP) on null data
# 1) Load data (either ROI time series or SWC matrix)
# 2) Compute null data (shuffle or phase)
# 3) Compute SWC matrix if ROI time series
# 4) Drop inbetween task windows
# 5) Compute embeddings
# 6) Save files

import argparse
import pandas as pd
import numpy as np
import os.path as osp
from scipy.spatial.distance import correlation, cosine, euclidean
from utils.embedding_functions import Uniform_Manifold_Approximation_Projection, T_Stochastic_Neighbor_Embedding, Laplacain_Eigenmap
from utils.data_functions import compute_SWC, randomize_ROI, randomize_conn
from utils.data_info import DATADIR, PRJDIR, load_task_ROI_TS, task_labels

def run(args):
    SBJ    = args.subject
    wl_sec = args.window_length_sec
    tr     = args.time_resolution
    ws_trs = args.window_space_tr
    LE_k   = args.LE_k
    p      = args.p
    UMAP_k = args.UMAP_k
    n      = args.n
    metric = args.metric
    data   = args.data
    drop   = args.drop
    print(' ')
    print('++ INFO: Run information')
    print('         SBJ:   ',SBJ)
    print('         wl_sec:',wl_sec)
    print('         tr:    ',tr)
    print('         ws_trs:',ws_trs)
    print('         LE_k:  ',LE_k)
    print('         p:     ',p)
    print('         UMAP_k:',UMAP_k)
    print('         n:     ',n)
    print('         metric:',metric)
    print('         data:  ',data)
    print('         drop:  ',drop)
    print(' ')
    
    if data == 'ROI':
        # Load ROI time series
        # --------------------
        ROI_ts = load_task_ROI_TS(DATADIR,SBJ, wl_sec) # USE YOUR OWN FUNCTION TO LOAD ROI TIME SERIES AS PD.DATAFRAME (TRxROI)
        print('++ INFO: ROI time series loaded')
        print('         Data shape:',ROI_ts.shape)
        print(' ')
        
        # Compute null data
        # -----------------
        null_ROI_ts_df = randomize_ROI(ROI_ts)
        print('++ INFO: ROI TS null data computed')
        print('         Data shape:',null_ROI_ts_df.shape)
        print(' ')
    
        # Compute SWC matrix
        # ------------------
        wl_trs = int(wl_sec/tr)
        window = np.ones((wl_trs,))
        swc_r, swc_Z, winInfo = compute_SWC(null_ROI_ts_df,wl_trs,ws_trs,window=window)
        SWC_df = swc_Z.reset_index(drop=True).T
        SWC_df = SWC_df.reset_index(drop=True)
        print('++ INFO: SWC matrix computed')
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
        elif drop == 'FullData':
            drop_SWC_df = SWC_df.copy()
    
    
    elif data == 'SWC':
        # Load SWC matrix
        # ---------------
        file_name = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'.csv'
        file_path = osp.join(PRJDIR,'derivatives','SWC',file_name)
        SWC_df    = pd.read_csv(file_path)  
        print('++ INFO: SWC matrix loaded')
        print('         Data shape:',SWC_df.shape)
        print(' ')
    
        # Compute null data
        # -----------------
        null_SWC_df = randomize_conn(SWC_df)
        print('++ INFO: SWC matrix null data computed')
        print('         Data shape:',null_SWC_df.shape)
        print(' ')
        
        if drop == 'DropData':
            # Drop inbwtween task windows
            # ---------------------------
            wl_trs = int(wl_sec/tr)
            task_df     = task_labels(wl_trs, PURE=False) # USE YOUR OWN FUNCTION TO LOAD TASK LABELS AS PD.DATAFRAME
            drop_index  = task_df.index[task_df['Task'] == 'Inbetween']
            drop_SWC_df = null_SWC_df.drop(drop_index).reset_index(drop=True)
            print('++ INFO: Inbetween task windows dropped')
            print('         Data shape:',drop_SWC_df.shape)
            print(' ')
        elif drop == 'FullData':
            drop_SWC_df = null_SWC_df.copy()
    
    # Compute LE Embedding
    # --------------------
    dist_metric_dict = {'correlation':correlation, 'cosine':cosine, 'euclidean':euclidean}
    LE_df = Laplacain_Eigenmap(drop_SWC_df,k=LE_k,n=n,metric=dist_metric_dict[metric])
    print('++INFO: Laplacian Eigenmap embedding computed')
    print('        Data shape:',LE_df.shape)
    print(' ')
    
    # Save LE file to outside directory
    # ---------------------------------
    out_file = SBJ+'_'+data+'_Null_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','Null_Data',out_file)
    LE_df.to_csv(out_path, index=False)
    print('++ INFO: LE data saved to')
    print('       ',out_path)
    print(' ')
    
    # Compute TSNE Embedding
    # ----------------------
    TSNE_df = T_Stochastic_Neighbor_Embedding(drop_SWC_df,p=p,n=n,metric=metric)
    print('++ INFO: TSNE embedding computed')
    print('         Data shape:',TSNE_df.shape)
    print(' ')
   
    # Save TSNE file to outside directory
    # ------------------------------
    out_file = SBJ+'_'+data+'_Null_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','Null_Data',out_file)
    TSNE_df.to_csv(out_path, index=False)
    print('++ INFO: TSNE data saved to')
    print('       ',out_path)
    print(' ')
    
    # Compute UMAP Embedding
    # -----------------
    UMAP_df = Uniform_Manifold_Approximation_Projection(drop_SWC_df,k=UMAP_k,n=n,metric=metric)
    print('++INFO: UMAP embedding computed')
    print('        Data shape:',UMAP_df.shape)
    print(' ')
    
    # Save UMAP file to outside directory
    # -----------------------------------
    out_file = SBJ+'_'+data+'_Null_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','Null_Data',out_file)
    UMAP_df.to_csv(out_path, index=False)
    print('++ INFO: UMAP data saved to')
    print('       ',out_path)

def main():
    parser=argparse.ArgumentParser(description="Compute embeddings with null data.")
    parser.add_argument("-sbj", help="subject name in SBJXX format", dest="subject", type=str, required=True)
    parser.add_argument("-wl_sec", help="window length in seconds", dest="window_length_sec", type=int, required=True)
    parser.add_argument("-tr", help="time resolution", dest="time_resolution", type=float, required=True)
    parser.add_argument("-ws_trs",help="window spaces in tr", dest="window_space_tr", type=int, required=True)
    parser.add_argument("-LE_k", help="LE Nearest Neighboor value", dest="LE_k", type=int, required=True)
    parser.add_argument("-p", help="TSNE perplexity value", dest="p", type=int, required=True)
    parser.add_argument("-UMAP_k",help="UMAP Nearest Neighboor value", dest="UMAP_k", type=int, required=True)
    parser.add_argument("-n", help="number of dimensions", dest="n", type=int, required=True)
    parser.add_argument("-met", help="distance metric (correlation, cosine, euclidean)", dest="metric", type=str, required=True)
    parser.add_argument("-data", help="Data to be randomized (ROI or SWC)", dest="data", type=str, required=True)
    parser.add_argument("-drop", help="Drop inbetween windows or full data (DropData or FullData)", dest="drop", type=str, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()