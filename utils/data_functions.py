#! /usr/bin/env python
# Isabel Fernandez 02/22/2022

# This file contains functional that computes the sliding window correlation matrix.

import pandas as pd
import numpy as np
import os.path as osp


# Compute Sliding Window Correlation
# ----------------------------------
def compute_SWC(ts,wl_trs,ws_trs,win_names=None,window=None):
    """
    This function will perform the following actions:
    1) Generate windows based on length, step and TR. This means computing window onsets and offsets
    2) Generate window names if those are not provided
    3) For each sliding window:
       * extract time series for all ROIs
       * multiply by the provided window shape
       * compute connectivity matrix
       * extract top triangle
       * apply fisher-transform
       
    INPUTS
    ------
    ts: (array) ROI timeseries in the form of a pd.DataFrame
    wl_trs: (int) window length in number of TRs
    ws_trs: (int) window step in number of TRs
    win_names: window labels as string array. If empty, labels will be generated automatically
    window: (np.array of length equal to wl_trs) window shape to apply
    
    OUTPUTS
    -------
    swc_r: (pd.Dataframe) sliding window connectivity matrix as Pearson's correlation
    swc_Z: (pd.Dataframe) sliding window connectivity matrix as Fisher's transform
    winInfo: (dict) containing window onsets, offsets, and labels.
    """

    [Nacq,Nrois] = ts.shape
    winInfo             = {'durInTR':int(wl_trs),'stepInTR':int(ws_trs)} # Create Window Information
    winInfo['numWins']  = int(np.ceil((Nacq-(winInfo['durInTR']-1))/winInfo['stepInTR'])) # Computer Number of Windows
    winInfo['onsetTRs'] = np.linspace(0,winInfo['numWins'],winInfo['numWins']+1, dtype='int')[0:winInfo['numWins']] # Compute Window Onsets
    winInfo['offsetTRs']= winInfo['onsetTRs'] + winInfo['durInTR']
    
    # Create Window Names
    if win_names is None:
        winInfo['winNames'] = ['W'+str(i).zfill(4) for i in range(winInfo['numWins'])]
    else:
        winInfo['winNames'] = win_names
    
    # Create boxcar window (if none provided)
    if window is None:
        window=np.ones((wl_trs,))
    
    # Compute SWC Matrix
    for w in range(winInfo['numWins']):
        aux_ts          = ts[winInfo['onsetTRs'][w]:winInfo['offsetTRs'][w]]
        aux_ts_windowed = aux_ts.mul(window,axis=0)
        aux_fc          = aux_ts_windowed.corr()
        sel             = np.triu(np.ones(aux_fc.shape),1).astype(np.bool)
        aux_fc_v        = aux_fc.where(sel)

        if w == 0:
            swc_r  = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
        else:
            new_df = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
            swc_r  = pd.concat([swc_r,new_df],axis=1)
    swc_Z = swc_r.apply(np.arctanh)
    
    return swc_r, swc_Z, winInfo

# Randomize Connections for Null Data
# -----------------------------------
def randomize_conn(SWC_df):
    """
    This function randomwizes the connections in each window for a sliding window connectivity matrix.
    This function uses the numpy function np.random.shuffle() to randomize the connection order.
    Note that since the np.random.shuffle() is random the function will output a different SWC even with 
    the same SWC matrix as input.
       
    INPUTS
    ------
    SWC_df: (pd.DataFrame) sliding window connectivity
    
    OUTPUTS
    -------
    null_SWC_df: (pd.DataFrame) Null sliding window connectivity with randomized connection order
    """
    
    Nwin, Nconn = SWC_df.shape # Save number of windows and connections in SWC matrix
    null_SWC_df = pd.DataFrame() # Empty data frame for null data
    for col in range(0,Nconn): # For each column in the data
        col_arr                = SWC_df[str(col)].copy().values # Copy the connection values in a given colum as a numpy array col_arr
        np.random.shuffle(col_arr) # Shuffle the connection values in the array
        null_SWC_df[str(col)] = col_arr # Save the shuffled array in null data data frame
    
    return null_SWC_df