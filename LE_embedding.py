# Isabel Fernandez 2/22/2022

# This file computes the embeddings using the Laplacian Eigenmap algorithum
# 1) Load ROI time series
# 2) Compute sliding window correlation matrix
# 3) Drop inbetween task windows
# 4) Compute embedding
# 5) Save file

import argparse
import pandas as pd
import numpy as np
import os.path as osp
from scipy.spatial.distance import correlation, cosine, euclidean
from utils.embedding_functions import Laplacain_Eigenmap
from utils.data_functions import load_task_ROI_TS, compute_SWC, task_labels
from utils.data_info import DATADIR, PRJDIR, wl_sec, tr, ws_trs, SBJ_list

def run(args):
    SBJ    = args.subject
    k      = args.k
    n      = args.n
    metric = args.metric
    print('++INFO: Run information')
    print('        SBJ:   ',SBJ)
    print('        k:     ',k)
    print('        n:     ',n)
    print('        metric:',metric)
    print(' ')
    
    # Load ROI time series
    # --------------------
    ROI_ts = load_task_ROI_TS(DATADIR,SBJ, wl_sec)
    print('++INFO: ROI time series loaded')
    print('        Data shape:',ROI_ts.shape)
    print(' ')
    
    # Compute SWC matrix
    # ------------------
    wl_trs = int(wl_sec/tr)
    window = np.ones((WL_trs,))
    swc_r, swc_Z, winInfo = compute_swc(pca_df,wl_trs,ws_trs,window=window)
    SWC_df = swc_Z.reset_index(drop=True).T
    print('++INFO: SWC matrix computed')
    print('        Data shape:',SWC_df.shape)
    print(' ')
    
    # Drop inbwtween task windows
    # ---------------------------
    task_df     = task_labels(wl_trs, PURE=False)
    drop_index  = task_df.index[task_df['Task'] == 'Inbetween']
    drop_SWC_df = SWC_df.drop(['W'+str(i).zfill(4) for i in drop_index]).reset_index(drop=True)
    print('++INFO: Inbetween task windows dropped')
    print('        Data shape:',drop_SWC_df.shape)
    print(' ')
    
    # Compure Embedding
    # -----------------
    dist_metric_dict = {'correlation':correlation, 'cosine':cosine, 'euclidean':euclidean}
    LE_df = Laplacain_Eigenmap(drop_SWC_df,k=k,n=n,metric=dist_metric_dict[metric])
    print('++INFO: Laplacian Eigenmap embedding computed')
    print('        Data shape:',LE_df.shape)
    print(' ')
    
    # Save file to outside directory
    # ------------------------------
    out_file = SBJ+'_embedding_data_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(3)+'_'+metric+'.pkl'
    out_path = osp.join(PRJADIR,'derivative','Laplacian_Eigenmap',out_file)
    LE_data.to_pickle(out_path)
    print('++INFO: Data saved to')
    print('      ',out_path)

def main():
    parser=argparse.ArgumentParser(description="Compute embeddings using the Laplacian Eigenmap algorithum.")
    parser.add_argument("-sbj",help="subject name in SBJXX format", dest="subject", type=str, required=True)
    parser.add_argument("-k",help="k-Nearest Neighboor value", dest="k", type=int, required=True)
    parser.add_argument("-n",help="number of dimensions", dest="n", type=int, required=True)
    parser.add_argument("-met",help="distance metric (correlation, cosine, euclidean)", dest="metric", type=str, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()