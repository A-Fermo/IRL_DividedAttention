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

sfreq = 250
resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

srs = np.load(resultsLoc+'srs_file.npy', allow_pickle='TRUE').item()
frqs = np.load(resultsLoc+'frqs_file.npy', allow_pickle='TRUE').item() # frequency ranges
evoked_all = np.load(resultsLoc+'evoked_all.npy', allow_pickle='TRUE').item()
psds_all = np.load(resultsLoc+'psds_all.npy', allow_pickle='TRUE').item()
evoked_by_N = np.load(resultsLoc+'evoked_by_N.npy',allow_pickle='TRUE').item()
epochs_by_N = np.load(resultsLoc+'epochs_by_N.npy',allow_pickle='TRUE').item()
freqs = np.load(resultsLoc+'freqs.npy',allow_pickle='TRUE')

n_participants = evoked_all['a1'].shape[0]
Ns = list(epochs_by_N.keys())
group_name = ['a1_aav1','a1_v1','v1_vav1','aav1_vav1']
conds = ['a1','v1','aav1','vav1']

sensorsByCluster = {}
clu_idx = 1
for gp in group_name:
    for clu in srs[gp]:
        sensorsByCluster['cluster'+str(clu_idx)]=clu
        clu_idx+=1
        

    

##################
    
    # for key in conds:
    #     evoked_all[key] = np.transpose(evoked_all[key],(0,2,1))
    
    # DTF Per participant per evoked (all the epochs)
    # evoked_clu = {} # evoked_clu['aav1_vav1'] = list(clusters1 (#participants x #sensors x #samples), cluster2)
    # for key in conds:
    #     l = []
    #     for gp in group_name:
    #         for clu in srs[gp]:
    #             epchs = epochs_by_N[N]['a1'][:,clu[:,None],:][:,:,0,:]
    #             for e in epchs:
    #                 p,bic = pdc_dtf2.compute_order(e, p_max=10)
    #                 A_est, sigma = pdc_dtf2.mvar_fit(e, p)
    #                 sigma = np.diag(sigma)  # DTF + PDC support diagonal noise
    #                 # sigma = None
    #                 # compute DTF
    #                 D, freqs = pdc_dtf2.DTF(A_est, sigma)

nbClusters = 0
for gp in group_name:
    nbClusters += len(srs[gp])

# evoked_clu = {}
# for key in conds:
#     print('Condition: ',key)
#     bigArray = np.zeros((n_participants,nbClusters,64,64))
#     for N_idx in range(n_participants):
#         clu_idx = 0 # clu_idx <= nbCluster
#         for gp in group_name:
#             for i,clu in enumerate(srs[gp]):
#                 evk = evoked_all['a1'][N_idx,clu[:,None],:][:,0,:]
#                 p,bic = pdc_dtf2.compute_order(evk, p_max=5)
#                 A_est, sigma = pdc_dtf2.mvar_fit(evk, p)
#                 sigma = np.diag(sigma)  # DTF + PDC support diagonal noise
#                 # sigma = None
#                 # compute DTF
#                 D, freqs = pdc_dtf2.DTF(A_est, sigma)
#                 matrix = np.zeros((64,64))
#                 F = freqs * sfreq
#                 lower_f = frqs[gp][i][0]
#                 upper_f = frqs[gp][i][-1]
#                 f_range = (F >= lower_f) & (F <= upper_f)
#                 D_integrated = D[np.ix_(f_range)].mean(axis=0)
                
#                 for s1,ch_row in enumerate(clu):
#                     for s2,ch_col in enumerate(clu):
#                         matrix[ch_row,ch_col] = D_integrated[s1,s2]   
#                 bigArray[N_idx,clu_idx,:,:] = matrix # 64x64 matrix including DTF
#                 clu_idx += 1
#         print('Participant {}/{}'.format(N_idx+1,n_participants))
#     evoked_clu[key] = bigArray

for N in Ns:
    for key in conds:
        for i,e in enumerate(epochs_by_N[N][key]):
            mean,std = e.mean(axis=1),e.std(axis=1)
            mean,std = mean[:,np.newaxis],std[:,np.newaxis]
            epochs_by_N[N][key][i] = (e-mean)/std 
        mean,std = epochs_by_N[N][key].mean(axis=0),epochs_by_N[N][key].std(axis=0)
        for i,e in enumerate(epochs_by_N[N][key]):
            epochs_by_N[N][key][i] = (e-mean)/std

key = 'vav1'
print('Condition: ',key)
A_coeff = {}
Sigmas = {}
for N_idx,N in enumerate(Ns):
    A_list = []
    S_list = []
    print('Participant {}/{}'.format(N_idx+1,n_participants))
    epochs_data = epochs_by_N[N][key]
    n_epochs = epochs_data.shape[0]
    for i_e,e in enumerate(epochs_data):
        bArray = np.zeros((nbClusters,64,64))
        clu_idx = 0 # clu_idx <= nbCluster
        # for gp in group_name:
        for i,clu in enumerate(srs['a1_aav1']):
            if i == 3:
                epch = e[clu[:,None],:][:,0,:]
                # p,bic = pdc_dtf2.compute_order(epch, p_max=5)
                A_est, sigma = pdc_dtf2.mvar_fit(epch, 4)
                sigma = np.diag(sigma)  # DTF + PDC support diagonal noise
                # sigma = None
                # compute DTF
                A_list.append(A_est)
                S_list.append(sigma)
                A_coeff[N] = np.array(A_list)
                Sigmas[N] = np.array(S_list)
      
        
      
# A_coeff_a1 = A_coeff
# A_coeff_v1 = A_coeff

for n in Ns:
    n_epochs = len(A_coeff[n])
    arr = np.zeros((n_epochs,4,9,9))
    arr_s = np.zeros((n_epochs,9))
    for i,e in enumerate(A_coeff[n]):
        arr[i] = e
    A_coeff[n] = arr
    for i,e in enumerate(Sigmas[n]):
        arr_s[i] = e
    Sigmas[n] = arr_s

A_coeff_averaged = np.zeros((20,4,9,9))
for i,n in enumerate(Ns):     
    mean = A_coeff[n].mean(axis=0)
    A_coeff_averaged[i] = mean
    mean = Sigmas[n].mean(axis=0)
    
DTF_array = np.zeros((20,64,64))
DTF_big = np.zeros((20,512,9,9))
clu =  np.array([ 6, 11, 22, 23, 28, 39, 52, 56, 63])
for i in range(20):
    D, freqs = pdc_dtf2.DTF(A_coeff_averaged[i], sigma)
    matrix = np.zeros((64,64))
    F = freqs * sfreq
    lower_f = frqs['a1_aav1'][3][0]
    upper_f = frqs['a1_aav1'][3][-1]
    f_range = (F >= lower_f) & (F <= upper_f)
    D_integrated = D[np.ix_(f_range)].mean(axis=0)
    
    for s1,ch_row in enumerate(clu):
        for s2,ch_col in enumerate(clu):
            if ch_row != ch_col:
                matrix[ch_row,ch_col] = D_integrated[s1,s2]
    DTF_array[i] = matrix
    DTF_big[i] = D
  
# np.save(resultsLoc+'DTFbiga1_averagedcoeff.npy',DTF_big)
# np.save(resultsLoc+'DTFbigaav1_averagedcoeff.npy',DTF_big)
# np.save(resultsLoc+'DTFbigv1_averagedcoeff.npy',DTF_big)
np.save(resultsLoc+'DTFbigvav1_averagedcoeff.npy',DTF_big)
    
# np.save(resultsLoc+'DTFa1_averagedcoeff.npy',DTF_array)
# np.save(resultsLoc+'DTFaav1_averagedcoeff.npy',DTF_array)
# np.save(resultsLoc+'DTFv1_averagedcoeff.npy',DTF_array)
np.save(resultsLoc+'DTFvav1_averagedcoeff.npy',DTF_array)
# np.save(resultsLoc+'DTF_all_epochs_a1_averaged.npy',DTF_all_byCluster)
 

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


    
    
    
    
    
    
    
    
    
    
    