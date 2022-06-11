#! /usr/bin/env python
# Isabel Fernandez 2/25/2022

# This file computes the shilouette index (SI) for a given embedding based on task labels
# 1. If using DropX data add extra k or perplexity values
# 2. Creat task label data frame
# For a given embedding:
# 3. Load embedding for a given distance metric and k or perplexity value
# 4. Drop any inbetween task windows
# 5. Compute SI of the embedding
# 6. Save it to the subject SI data frame
# 7. Do steps 3 to 6 for reach distence metric and k or perplexity value combination
# 8. Save subject SI data frame (distace metric X k or perplexity value) as csv file

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
    
    # Extra hyperparameter values for dropped data
    # --------------------------------------------
    if drop == 'Drop5' or drop == 'Drop10' or drop == 'Drop15':
        global LE_k_list
        global p_list
        global UMAP_k_list
        LE_k_list   += [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34]
        p_list      += [4,6,8,12,13,14,16,17,18,19,21,22,23,24]
        UMAP_k_list += [3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24]
        LE_k_list.sort()
        p_list.sort()
        UMAP_k_list.sort()
    
    # Load embedding and compute SI
    # -----------------------------
    wl_trs = int(wl_sec/tr)
    if drop == 'DropData':
        task_df = task_labels(wl_trs, PURE=True)
    elif drop == 'FullData':
        task_df = task_labels(wl_trs, PURE=False)

    elif drop == 'Drop5':
        task_df = task_labels(wl_trs, PURE=False)
        task_df = task_df.loc[range(0, task_df.shape[0], 5)]
    elif drop == 'Drop10':
        task_df = task_labels(wl_trs, PURE=False)
        task_df = task_df.loc[range(0, task_df.shape[0], 10)]
    elif drop == 'Drop15':
        task_df = task_labels(wl_trs, PURE=False)
        task_df = task_df.loc[range(0, task_df.shape[0], 15)]
    
    n = 3 # Number of dimensions is always 3 for SI
    
    dist_metric_list = ['correlation', 'cosine', 'euclidean']
    if embedding == 'LE':
        SI_df = pd.DataFrame(index=LE_k_list, columns=dist_metric_list)
        for metric in dist_metric_list:
            SI_list = []
            for k in LE_k_list:
                try:
                    file_name = SBJ+'_LE_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
                    file_path = osp.join(PRJDIR,'derivatives','LE',file_name)
                    embed_df  = pd.read_csv(file_path)
                    print('++ INFO Data shape:', embed_df.shape)
                    print(' ')
                    if drop == 'FullData':
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'Drop5':
                        embed_df.index = task_df.index
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'Drop10':
                        embed_df.index = task_df.index
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'Drop15':
                        embed_df.index = task_df.index
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'DropData':
                        silh_idx = silhouette_score(embed_df[['1_norm', '2_norm', '3_norm']], task_df['Task'].values)
                    SI_list.append(silh_idx)
                except:
                    print('++ ERROR: This embedding does not exist for k', k)
                    print(' ')
                    SI_list.append(np.nan) # add nan value for non existant embedding (might be too kigh a k values)
            SI_df[metric] = SI_list
            print('++ INFO: SIs computed for',metric)
                
    elif embedding == 'TSNE':
        SI_df = pd.DataFrame(index=p_list, columns=dist_metric_list)
        for metric in dist_metric_list:
            SI_list = []
            for p in p_list:
                try:
                    file_name = SBJ+'_TSNE_embedding_wl'+str(wl_sec).zfill(3)+'_p'+str(p).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
                    file_path = osp.join(PRJDIR,'derivatives','TSNE',file_name)
                    embed_df  = pd.read_csv(file_path)
                    print('++ INFO Data shape:', embed_df.shape)
                    print(' ')
                    if drop == 'FullData':
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'Drop5':
                        embed_df.index = task_df.index
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'Drop10':
                        embed_df.index = task_df.index
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'Drop15':
                        embed_df.index = task_df.index
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'DropData':
                        silh_idx = silhouette_score(embed_df[['1_norm', '2_norm', '3_norm']], task_df['Task'].values)
                    SI_list.append(silh_idx)
                except:
                    print('++ ERROR: This embedding does not exist for p', p)
                    print(' ')
                    SI_list.append(np.nan) # add nan value for non existant embedding (might be too kigh a perplexity values)
            SI_df[metric] = SI_list
            print('++ INFO: SIs computed for',metric)
            
    elif embedding == 'UMAP':
        SI_df = pd.DataFrame(index=UMAP_k_list, columns=dist_metric_list)
        for metric in dist_metric_list:
            SI_list = []
            for k in UMAP_k_list:
                try:
                    file_name = SBJ+'_UMAP_embedding_wl'+str(wl_sec).zfill(3)+'_k'+str(k).zfill(3)+'_n'+str(n).zfill(2)+'_'+metric+'_'+drop+'.csv'
                    file_path = osp.join(PRJDIR,'derivatives','UMAP',file_name)
                    embed_df  = pd.read_csv(file_path)
                    print('++ INFO Data shape:', embed_df.shape)
                    print(' ')
                    if drop == 'FullData':
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'Drop5':
                        embed_df.index = task_df.index
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'Drop10':
                        embed_df.index = task_df.index
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'Drop15':
                        embed_df.index = task_df.index
                        drop_index    = task_df.index[task_df['Task'] == 'Inbetween']
                        drop_embed_df = embed_df.drop(drop_index).reset_index(drop=True)
                        drop_task_df  = task_df.drop(drop_index).reset_index(drop=True)
                        silh_idx = silhouette_score(drop_embed_df[['1_norm', '2_norm', '3_norm']], drop_task_df['Task'].values)
                    elif drop == 'DropData':
                        silh_idx = silhouette_score(embed_df[['1_norm', '2_norm', '3_norm']], task_df['Task'].values)
                    SI_list.append(silh_idx)
                except:
                    print('++ ERROR: This embedding does not exist for k', k)
                    print(' ')
                    SI_list.append(np.nan) # add nan value for non existant embedding (might be too kigh a k values)
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