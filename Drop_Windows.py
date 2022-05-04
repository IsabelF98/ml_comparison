#! /usr/bin/env python
# Isabel Fernandez 5/4/2022

# This file computes the embeddings using the Laplacain Eigenmap algorithum
# 1) Load sliding window correlation matrix
# 2) Drop every n windows
# 3) Save new SWC matrix

import argparse
import pandas as pd
import numpy as np
import os.path as osp
from utils.data_info import PRJDIR

def run(args):
    SBJ    = args.subject
    wl_sec = args.window_length_sec
    tr     = args.time_resolution
    n_drop = args.ndrop
    print(' ')
    print('++ INFO: Run information')
    print('         SBJ:   ',SBJ)
    print('         wl_sec:',wl_sec)
    print('         tr:    ',tr)
    print('         n drop:',n_drop)
    print(' ')
    
    # Load SWC matrix
    # ---------------
    file_name = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'_FullData.csv'
    file_path = osp.join(PRJDIR,'derivatives','SWC',file_name)
    SWC_df    = pd.read_csv(file_path)  
    print('++ INFO: SWC matrix loaded')
    print('         Data shape:',SWC_df.shape)
    print(' ')
    
    # Drop every n windows
    # --------------------
    SWC_df = SWC_df.loc[range(0, SWC_df.shape[0], n_drop)].copy()
    print('++ INFO: Windows dropped')
    print('         Data shape:',SWC_df.shape)
    print(' ')
    
    # Save file to outside directory
    # ------------------------------
    out_file = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'_Drop'+str(n_drop)+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','SWC',out_file)
    SWC_df.to_csv(out_path, index=False)
    print('++ INFO: Data saved to')
    print('       ',out_path)
    
def main():
    parser=argparse.ArgumentParser(description="Compute sliding window correlation matrix.")
    parser.add_argument("-sbj",help="subject name in SBJXX format", dest="subject", type=str, required=True)
    parser.add_argument("-wl_sec",help="window length in seconds", dest="window_length_sec", type=int, required=True)
    parser.add_argument("-tr",help="time resolution", dest="time_resolution", type=float, required=True)
    parser.add_argument("-ndrop",help="number of windows to drop", dest="ndrop", type=int, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()