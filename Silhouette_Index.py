#! /usr/bin/env python
# Isabel Fernandez 2/25/2022

# This file computes the shilouette index for a given embedding based on task labels

import argparse
import pandas as pd
import numpy as np
import os.path as osp
from sklearn.metrics import silhouette_score
from utils.data_info import PRJDIR, LE_k_list, p_list, UMAP_k_list, task_labels

def run(args):
    SBJ       = args.subject
    wl_sec    = args.window_length_sec
    tr        = args.time_resolution
    embedding = args.embedding
    drop      = args.drop
    print(' ')
    print('++ INFO: Run information')
    print('         SBJ:   ',SBJ)
    print('         wl_sec:',wl_sec)
    print('         tr:    ',tr)
    print('         embed: ',embedding)
    print('         drop:  ',drop)
    print(' ')
    
    # Load embedding and compute SI
    # -----------------------------
    wl_trs = int(wl_sec/tr)
    if drop == 'FullData':
        task_df = task_labels(wl_trs, PURE=False)
    elif drop == 'DropData':
        task_df = task_labels(wl_trs, PURE=True)
    print('++ INFO: Task labels loaded')
    print('         Data shape:',task_df.shape)
    print(' ')
    
    n = 3 # Number of dimensions
    
    dist_metric_list = ['correlation', 'cosine', 'euclidean']
    if embedding == 'LE':
        SI_df = pd.DataFrame(index=LE_k_list, columns=dist_metric_list)
        for metric in dist_metric_list:
            SI_list = []
            for k in LE_k_list:
                file_name = SBJ+'_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
                file_path = osp.join(PRJDIR,'derivatives','LE',file_name)
                embed_df  = pd.read_csv(file_path)
                if drop == 'FullData':
                    drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                    drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                    drop_task_df = task_df.drop(drop_index).reset_index(drop=True)
                    silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                elif drop == 'DropData':
                    silh_idx = silhouette_score(embed_df[['1_norm', '2_norm', '3_norm']], task_df['Task'].values)
                SI_list.append(silh_idx)
            SI_df[metric] = SI_list
            print('++ INFO: SIs computed for',metric)
                
    elif embedding == 'TSNE':
        SI_df = pd.DataFrame(index=p_list, columns=dist_metric_list)
        for metric in dist_metric_list:
            SI_list = []
            for p in p_list:
                file_name = SBJ+'_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
                file_path = osp.join(PRJDIR,'derivatives','TSNE',file_name)
                embed_df  = pd.read_csv(file_path)  
                if drop == 'FullData':
                    drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                    drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                    drop_task_df = task_df.drop(drop_index).reset_index(drop=True)
                    silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                elif drop == 'DropData':
                    silh_idx = silhouette_score(embed_df[['1_norm', '2_norm', '3_norm']], task_df['Task'].values)
                SI_list.append(silh_idx)
            SI_df[metric] = SI_list
            print('++ INFO: SIs computed for',metric)
            
    elif embedding == 'UMAP':
        SI_df = pd.DataFrame(index=UMAP_k_list, columns=dist_metric_list)
        for metric in dist_metric_list:
            SI_list = []
            for k in UMAP_k_list:
                file_name = SBJ+'_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
                file_path = osp.join(PRJDIR,'derivatives','UMAP',file_name)
                embed_df  = pd.read_csv(file_path)  
                if drop == 'FullData':
                    drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                    drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                    drop_task_df = task_df.drop(drop_index).reset_index(drop=True)
                    silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                elif drop == 'DropData':
                    silh_idx = silhouette_score(embed_df[['1_norm', '2_norm', '3_norm']], task_df['Task'].values)
                SI_list.append(silh_idx)
            SI_df[metric] = SI_list
            print('++ INFO: SIs computed for',metric)
            
    print('++ INFO: SI data fame complete')
    print('         Data shape:',SI_df.shape)
    print(' ')
    
    # Save file to outside directory
    # ------------------------------
    out_file = SBJ+'_Silh_Idx_'+embedding+'_wl'+str(wl_sec).zfill(3)+'_'+drop+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','Silh_Idx',out_file)
    SI_df.to_csv(out_path, index=True)
    print('++ INFO: Data saved to')
    print('       ',out_path)

def main():
    parser=argparse.ArgumentParser(description="Compute silhouette index for a given subject 3D embedding.")
    parser.add_argument("-sbj",help="subject name in SBJXX format", dest="subject", type=str, required=True)
    parser.add_argument("-wl_sec",help="window length in seconds", dest="window_length_sec", type=int, required=True)
    parser.add_argument("-tr",help="time resolution", dest="time_resolution", type=float, required=True)
    parser.add_argument("-emb",help="embedding type (LE, TSNE, UMAP)", dest="embedding", type=str, required=True)
    parser.add_argument("-drop", help="Drop inbetween windows or full data (DropData or FullData)", dest="drop", type=str, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()