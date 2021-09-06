#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 08:17:54 2021

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

resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

# DTFbig_all = np.load(resultsLoc+'DTFbig_all.npy',allow_pickle='TRUE').item()
# DTF_all = np.load(resultsLoc+'DTF_all.npy',allow_pickle='TRUE').item()
sensorsByCluster = np.load(resultsLoc+'sensorsByCluster.npy',allow_pickle='TRUE').item()
freqs_DTF = np.load(resultsLoc+'freqs_DTF.npy',allow_pickle='TRUE')
ch_names = np.load(resultsLoc+'channelsName.npy',allow_pickle='TRUE')
sfreq = np.load(resultsLoc+'sfreq.npy',allow_pickle='TRUE')
EEG_info = np.load(resultsLoc+'EEG_info.npy',allow_pickle='TRUE').item()

ch_idx = sensorsByCluster['cluster4']
N_ch = ch_names[ch_idx]

# DTFa1mean = DTF_all['a1']['cluster4'].mean(axis=0)
# DTFaav1mean = DTF_all['aav1']['cluster4'].mean(axis=0)
# DTFvav1mean = DTF_all['vav1']['cluster4'].mean(axis=0)
# DTFv1mean = DTF_all['v1']['cluster4'].mean(axis=0)


DTFa1mean = np.load(resultsLoc+'DTFbiga1_averagedcoeff.npy',allow_pickle=True).mean(axis=0)
DTFaav1mean = np.load(resultsLoc+'DTFbigaav1_averagedcoeff.npy',allow_pickle=True).mean(axis=0)
DTFvav1mean = np.load(resultsLoc+'DTFbigvav1_averagedcoeff.npy',allow_pickle=True).mean(axis=0)
DTFv1mean = np.load(resultsLoc+'DTFbigv1_averagedcoeff.npy',allow_pickle=True).mean(axis=0)

pdc_dtf2.plot_all(freqs_DTF, DTFa1mean, 'Cluster n°4 - DTF (across participants) - Audio',N_ch)
plt.savefig(resultsLoc+'DTFa1meanCluster4.eps',format='eps',dpi=300)

pdc_dtf2.plot_all(freqs_DTF, DTFaav1mean, 'Cluster n°4 - DTF (across participants) - Audiovisual',N_ch)
plt.savefig(resultsLoc+'DTFaav1meanCluster4.eps',format='eps',dpi=300)

pdc_dtf2.plot_all(freqs_DTF, DTFvav1mean, 'Cluster n°4 - DTF (across participants) - Visioauditory',N_ch)
plt.savefig(resultsLoc+'DTFvav1meanCluster4.eps',format='eps',dpi=300)

pdc_dtf2.plot_all(freqs_DTF, DTFv1mean, 'Cluster n°4 - DTF (across participants) - Visual',N_ch)
plt.savefig(resultsLoc+'DTFv1meanCluster4.eps',format='eps',dpi=300)

DTF_3da1 = np.load(resultsLoc+'DTFa1_averagedcoeff.npy',allow_pickle=True)
DTF_3dv1 = np.load(resultsLoc+'DTFv1_averagedcoeff.npy',allow_pickle=True)
DTF_3daav1 = np.load(resultsLoc+'DTFaav1_averagedcoeff.npy',allow_pickle=True)
DTF_3dvav1 = np.load(resultsLoc+'DTFvav1_averagedcoeff.npy',allow_pickle=True)

# plot3d_a1mean_c4 = DTFbig_all['v1'].mean(axis=0)[3]
# plot3d = DTF_3da1.mean(axis=0)
# plot3d = DTF_3dv1.mean(axis=0)
# plot3d = DTF_3daav1.mean(axis=0)
plot3d = DTF_3dvav1.mean(axis=0)
# a1_c4 = pd.DataFrame(plot3d_a1mean_c4)
# a1_c4.to_csv(resultsLoc+'a1_c4df.csv',index=True)

D_1 = plot3d.copy()
D_2 = plot3d.copy()
for i in range(64):
    D_1[i:,i] = 0
for i in range(64):
    D_2[i,i:] = 0
    
# a1_c4D1 = pd.DataFrame(D_1)
# a1_c4D1.to_csv(resultsLoc+'a1_c4D1df.csv',index=True)
# a1_c4D2 = pd.DataFrame(D_2)
# a1_c4D2.to_csv(resultsLoc+'a1_c4D2df.csv',index=True)

mne.viz.plot_sensors_connectivity(EEG_info, D_1)
mne.viz.plot_sensors_connectivity(EEG_info, D_2)

# mne.viz.plot_sensors(EEG_info)

