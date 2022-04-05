#! /usr/bin/env python
# Isabel Fernandez 4/5/2022

# This file computes the Logistic Regression coefficients and accuracy for a given embedding

import argparse
import pandas as pd
import numpy as np
import os.path as osp
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
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
    print(' ')
    
    # Train and test split
    # --------------------
    train_idx = (0, 363)
    test_idx  = (364, 728)
    
    # Load task labels
    # ----------------
    wl_trs = int(wl_sec/tr)
    task_df = task_labels(wl_trs, PURE=True)
    train_task_df = task_df.loc[train_idx[0]:train_idx[1]].copy()
    test_task_df  = task_df.loc[test_idx[0]:test_idx[1]].copy() 
    print('++ INFO: Task labels loaded and split')
    print('         Train data shape:', train_task_df.shape)
    print('         Test data shape:', test_task_df.shape)
    print(' ')
    
    
    # Logistic Regression solver
    # --------------------------
    Log_Reg = LogisticRegression(solver='liblinear', penalty='l2', random_state=0)
    
    # Laplacian Eigenmap LR
    # ---------------------
    # Load embedding
    LE_file_name = SBJ+'_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    LE_file_path = osp.join(PRJDIR,'derivatives','LE',LE_file_name)
    LE_df        = pd.read_csv(LE_file_path)
    # Split data
    train_LE_df = LE_df.loc[train_idx[0]:train_idx[1]].copy()
    test_LE_df  = LE_df.loc[test_idx[0]:test_idx[1]].copy()
    # Compute LR
    LE_Log_Reg      = Log_Reg.fit(train_LE_df[[str(i)+'_norm' for i in range(1,n+1)]], train_task_df['Task'])
    LE_Log_Reg_Coef = LE_Log_Reg.coef_
    LE_Log_Reg_Clas = LE_Log_Reg.classes_
    LE_LR_df = pd.DataFrame(LE_Log_Reg_Coef.T, columns=list(LE_Log_Reg_Clas))
    # Save LR coeffs
    LE_out_file = SBJ+'_LE_LRcoef_wl'+str(wl_sec).zfill(3)+'_k'+str(LE_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    LE_out_path = osp.join(PRJDIR,'derivatives','Log_Reg',LE_out_file)
    LE_LR_df.to_csv(LE_out_path, index=False)
    print('++ INFO: LE LR coefficients saved to')
    print('       ',LE_out_path)
    print(' ')
    # Predict test data
    LE_predicted = LE_Log_Reg.predict(test_LE_df[[str(i)+'_norm' for i in range(1,n+1)]])
    # Classification Report
    LE_class_report = f"{metrics.classification_report(test_task_df['Task'], LE_predicted)}\n"
    print('++INFO: LE Logistic Regression classificanion report complete')
    print(LE_class_report)
    print(' ')


#    TSNE NOT WORKING FOR n>30
#    # TSNE LR
#    # -------
#    # Load embedding
#    TSNE_file_name = SBJ+'_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
#    TSNE_file_path = osp.join(PRJDIR,'derivatives','TSNE',TSNE_file_name)
#    TSNE_df        = pd.read_csv(TSNE_file_path)
#    # Split data
#    train_TSNE_df = TSNE_df.loc[train_idx[0]:train_idx[1]].copy()
#    test_TSNE_df  = TSNE_df.loc[test_idx[0]:test_idx[1]].copy()
#    # Compute LR
#    TSNE_Log_Reg      = Log_Reg.fit(train_TSNE_df[[str(i)+'_norm' for i in range(1,n+1)]], train_task_df['Task'])
#    TSNE_Log_Reg_Coef = TSNE_Log_Reg.coef_
#    TSNE_Log_Reg_Clas = TSNE_Log_Reg.classes_
#    TSNE_LR_df = pd.DataFrame(TSNE_Log_Reg_Coef.T, columns=list(TSNE_Log_Reg_Clas))
#    # Save LR coeffs
#    TSNE_out_file = SBJ+'_TSNE_LRcoef_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
#    TSNE_out_path = osp.join(PRJDIR,'derivatives','Log_Reg',TSNE_out_file)
#    TSNE_LR_df.to_csv(TSNE_out_path, index=False)
#    print('++ INFO: TSNE LR coefficients saved to')
#    print('       ',TSNE_out_path)
#    print(' ')
#    # Predict test data
#    TSNE_predicted = TSNE_Log_Reg.predict(test_TSNE_df[[str(i)+'_norm' for i in range(1,n+1)]])
#    # Classification Report
#    TSNE_class_report = f"{metrics.classification_report(test_task_df['Task'], TSNE_predicted)}\n"
#    print('++INFO: TSNE Logistic Regression classificanion report complete')
#    print(TSNE_class_report)
#    print(' ')

    
    # UMAP LR
    # -------
    # Load embedding
    UMAP_file_name = SBJ+'_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    UMAP_file_path = osp.join(PRJDIR,'derivatives','UMAP',UMAP_file_name)
    UMAP_df        = pd.read_csv(UMAP_file_path)
    # Split data
    train_UMAP_df = UMAP_df.loc[train_idx[0]:train_idx[1]].copy()
    test_UMAP_df  = UMAP_df.loc[test_idx[0]:test_idx[1]].copy()
    # Compute LR
    UMAP_Log_Reg      = Log_Reg.fit(train_UMAP_df[[str(i)+'_norm' for i in range(1,n+1)]], train_task_df['Task'])
    UMAP_Log_Reg_Coef = UMAP_Log_Reg.coef_
    UMAP_Log_Reg_Clas = UMAP_Log_Reg.classes_
    UMAP_LR_df = pd.DataFrame(UMAP_Log_Reg_Coef.T, columns=list(UMAP_Log_Reg_Clas))
    # Save LR coeffs
    UMAP_out_file = SBJ+'_UMAP_LRcoef_wl'+str(wl_sec).zfill(3)+'_k'+str(UMAP_k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'.csv'
    UMAP_out_path = osp.join(PRJDIR,'derivatives','Log_Reg',UMAP_out_file)
    UMAP_LR_df.to_csv(UMAP_out_path, index=False)
    print('++ INFO: UMAP LR coefficients saved to')
    print('       ',UMAP_out_path)
    print(' ')
    # Predict test data
    UMAP_predicted = UMAP_Log_Reg.predict(test_UMAP_df[[str(i)+'_norm' for i in range(1,n+1)]])
    # Classification Report
    UMAP_class_report = f"{metrics.classification_report(test_task_df['Task'], UMAP_predicted)}\n"
    print('++INFO: UMAP Logistic Regression classificanion report complete')
    print(UMAP_class_report)
    print(' ')
    
def main():
    parser=argparse.ArgumentParser(description="Compute Logistic Regression coefficients and accuracy.")
    parser.add_argument("-sbj", help="subject name in SBJXX format", dest="subject", type=str, required=True)
    parser.add_argument("-wl_sec", help="window length in seconds", dest="window_length_sec", type=int, required=True)
    parser.add_argument("-tr", help="time resolution", dest="time_resolution", type=float, required=True)
    parser.add_argument("-LE_k", help="LE Nearest Neighboor value", dest="LE_k", type=int, required=True)
    parser.add_argument("-p", help="TSNE perplexity value", dest="p", type=int, required=True)
    parser.add_argument("-UMAP_k",help="UMAP Nearest Neighboor value", dest="UMAP_k", type=int, required=True)
    parser.add_argument("-n", help="number of dimensions", dest="n", type=int, required=True)
    parser.add_argument("-met", help="distance metric (correlation, cosine, euclidean)", dest="metric", type=str, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()