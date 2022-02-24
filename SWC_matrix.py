#! /usr/bin/env python
# Isabel Fernandez 2/23/2022

# This file computes the sliding window correlation (SWC) matrix which we use as
# input for the different manifold learning techniques.
# 1) Load ROI time series
# 2) Compute sliding window correlation matrix
# 3) Save SWC matrix as csv file

import argparse
import pandas as pd
import numpy as np
import os.path as osp
from utils.data_functions import compute_SWC
from utils.data_info import DATADIR, PRJDIR, load_task_ROI_TS, conda_loc, conda_env

def run(args):
    SBJ    = args.subject
    wl_sec = args.window_length_sec
    tr     = args.time_resolution
    ws_trs = args.window_space_tr
    print(' ')
    print('++ INFO: Run information')
    print('         SBJ:   ',SBJ)
    print('         wl_sec:',wl_sec)
    print('         tr:    ',tr)
    print('         ws_trs:',ws_trs)
    print(' ')
    
    # Load ROI time series
    # --------------------
    ROI_ts = load_task_ROI_TS(DATADIR,SBJ, wl_sec) # USE YOUR OWN FUNCTION TO LOAD ROI TIME SERIES AS PD.DATAFRAME (TRxROI)
    print('++ INFO: ROI time series loaded')
    print('         Data shape:',ROI_ts.shape)
    print(' ')
    
    # Compute SWC matrix
    # ------------------
    wl_trs = int(wl_sec/tr)
    window = np.ones((wl_trs,))
    swc_r, swc_Z, winInfo = compute_SWC(ROI_ts,wl_trs,ws_trs,window=window)
    SWC_df = swc_Z.reset_index(drop=True).T
    print('++ INFO: SWC matrix computed')
    print('         Data shape:',SWC_df.shape)
    print(' ')
    
    # Save file to outside directory
    # ------------------------------
    out_file = SBJ+'_SWC_matrix_wl'+str(wl_sec).zfill(3)+'.csv'
    out_path = osp.join(PRJDIR,'derivatives','SWC',out_file)
    SWC_df.to_csv(out_path, index=False)
    print('++ INFO: Data saved to')
    print('       ',out_path)
    
def main():
    parser=argparse.ArgumentParser(description="Compute sliding window correlation matrix.")
    parser.add_argument("-sbj",help="subject name in SBJXX format", dest="subject", type=str, required=True)
    parser.add_argument("-wl_sec",help="window length in seconds", dest="window_length_sec", type=int, required=True)
    parser.add_argument("-tr",help="time resolution", dest="time_resolution", type=float, required=True)
    parser.add_argument("-ws_trs",help="window spaces in tr", dest="window_space_tr", type=int, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()