#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 19:53:54 2021

@author: aurelien
"""

#%%
import os
import mne
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#import scipy.io as sio
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from meegkit.utils.matrix import sliding_window
import pandas as pd
import pdc_dtf, pdc_dtf2
import math

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%%

np.seterr(all='raise')

sfreq = 250
# resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'
resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results_shortEpochs/'

# srs = np.load(resultsLoc+'srs_file.npy', allow_pickle='TRUE').item()
# frqs = np.load(resultsLoc+'frqs_file.npy', allow_pickle='TRUE').item() # frequency ranges
evoked_all = np.load(resultsLoc+'evoked_all.npy', allow_pickle='TRUE').item()
psds_all = np.load(resultsLoc+'psds_all.npy', allow_pickle='TRUE').item()
evoked_by_N = np.load(resultsLoc+'evoked_by_N.npy',allow_pickle='TRUE').item()
epochs_by_N = np.load(resultsLoc+'epochs_by_N.npy',allow_pickle='TRUE').item()
freqs = np.load(resultsLoc+'freqs.npy',allow_pickle='TRUE')

n_participants = evoked_all['a1'].shape[0]
Ns = list(epochs_by_N.keys())
group_name = ['a1_aav1','a1_v1','v1_vav1','aav1_vav1']
conds = ['a1','v1','aav1','vav1']

fname = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/data_preprocessed_asr10/sub-Pcb0805/sub-Pcb0805_ses-S001_task-Default_run-001_preprocessed_rtMax2_eeg_raw.fif'
EEG = mne.io.read_raw_fif(fname,verbose=False)
ch_names = EEG.ch_names
del EEG
cluster = ['F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','Oz']
ch_idx = []
for ch in cluster:
    index = ch_names.index(ch)
    ch_idx.append(index)
ch_idx = np.array(ch_idx)

for N in Ns:
    for key in conds:
        for i,e in enumerate(epochs_by_N[N][key]):
            mean,std = e.mean(axis=1),e.std(axis=1)
            mean,std = mean[:,np.newaxis],std[:,np.newaxis]
            epochs_by_N[N][key][i] = (e-mean)/std 
        mean,std = epochs_by_N[N][key].mean(axis=0),epochs_by_N[N][key].std(axis=0)
        for i,e in enumerate(epochs_by_N[N][key]):
            epochs_by_N[N][key][i] = (e-mean)/std

freq_bands = {'delta':[1,4],'theta':[4,8],'alpha':[8,13],'beta':[13,30],'gamma':[30,45]}

DTF_all = {}
for key in conds:
    DTF_all[key] = []
    print('Condition: ',key)
    for N_idx,N in enumerate(Ns):
        if N_idx == 15:
            continue
        print('Participant {}/{}'.format(N_idx+1,n_participants))
        epochs_data = epochs_by_N[N][key]
        n_epochs = epochs_data.shape[0]
        for i_e,e in enumerate(epochs_data):
            bArray = np.zeros((5,16,16))
            epch = e[ch_idx[:,None],:][:,0,:]
            try:
                # p,bic = pdc_dtf2.compute_order(epch, p_max=7)
                A_est, sigma = pdc_dtf2.mvar_fit(epch, 4)
            except FloatingPointError:
                print("ERROR")
                continue
            
            sigma = np.diag(sigma)  # DTF + PDC support diagonal noise
            # sigma = None
            # compute DTF
            try:
                D, freqs = pdc_dtf2.DTF(A_est, sigma)
            except FloatingPointError:
                print("ERROR")
                continue
            F = freqs * sfreq
            clu_idx = 0 # clu_idx <= nbCluster
            for band in freq_bands.values():
                lower_f = band[0]
                upper_f = band[1]
                f_range = (F >= lower_f) & (F < upper_f)
                D_integrated = D[np.ix_(f_range)].mean(axis=0)
                np.fill_diagonal(D_integrated,0)
                bArray[clu_idx,:,:] = D_integrated
                clu_idx += 1
            # print('Epoch {}/{}'.format(i_e+1,n_epochs))
            DTF_all[key].append(bArray)

for key in conds:
    DTF_all[key]=np.array(DTF_all[key])
    
np.save(resultsLoc+'DTF_all_epochs_noCluster.npy',DTF_all)
 

# DTF_all2 = {}  
# for key in conds:
#     DTF_all2[key] = {}
#     for clu_idx in range(len(sensorsByCluster)):
#         clu_name = 'cluster'+str(clu_idx+1)
#         n_sensors = len(sensorsByCluster[clu_name])
#         DTF_all2[key][clu_name] = np.zeros((n_participants,len(freqs),n_sensors,n_sensors))
#         for N_idx in range(n_participants):
#             DTF_all2[key][clu_name][N_idx] = DTF_all[key][N_idx][clu_name]
# DTF_all = DTF_all2
        

# np.save(resultsLoc+'DTFbig_all.npy',DTFbig_all)
# np.save(resultsLoc+'DTF_all.npy',DTF_all)
# np.save(resultsLoc+'sensorsByCluster.npy',sensorsByCluster)
# np.save(resultsLoc+'freqs_DTF.npy',F)



    
#%%

# DTFbig_all = np.load(resultsLoc+'DTFbig_all.npy',allow_pickle=True).item()


    
    
    
    
    
    
    
    
    
    
    