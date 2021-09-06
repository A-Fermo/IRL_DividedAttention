#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 12:30:05 2021

@author: aurelien
"""

#%%
import os
import mne
import numpy as np

#%%

def find(path,fname):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if fname == file:
                files.append(os.path.join(r, file))
    if len(files) == 0:
        print('This file does not exist or is not in ../GitHub_DCAS/data/.. \n')
        return False, None
    elif len(files) > 1:
        print('Several files corresponding to this name have been found. \n')
        return False, None
    elif len(files) == 1:
        file = files[0]
        return True,file
    
def f(x,y):
    if x-y<0:
        return 0
    else:
        return x-y
    
def PSDS_computing(N,psds_by_N,epochs,tmin,tmax,evoked_by_N=None,epochs_by_N=None,):
    conds = ['a1','v1','aav1','vav1']
    sfreq = epochs['a1'].info['sfreq']
    # n_samples = epochs['a1'].get_data().shape[2]
    # n_fft = int(sfreq*3)
    n_fft = int(1.5*sfreq)
    n_overlap = n_fft/2
    # n_overlap = 0
    
    psds = {}
    psds_mean = {} # average over epochs (with epochs being of shape = (n_epoch,n_chan,n_samples))
    psds_std = {} # standard deviation over epochs
    evoked = {}
    for key,elt in epochs.items():
        psds[key], freqs = mne.time_frequency.psd_welch(elt,n_fft=n_fft,n_overlap=n_overlap,
                                                   n_per_seg=None,tmin=tmin, tmax=tmax,fmin=3, fmax=100,average='mean',verbose=False)
        psds[key] = 10*np.log10(psds[key])
        psds_mean[key] = psds[key].mean(0)
        psds_std[key] = psds[key].std(0)
        evoked[key] = elt.average().data
    psds_by_N[N] = psds_mean
    if evoked_by_N != None and epochs_by_N != None:
        evoked_by_N[N] = evoked
        epochs_data = {}
        for c in conds:
            epochs_data[c] = epochs[c].get_data()
        epochs_by_N[N] = epochs_data
        return freqs,psds_by_N,evoked_by_N,epochs_by_N
    else:
        return freqs,psds_by_N