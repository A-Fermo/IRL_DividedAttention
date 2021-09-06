#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 09:31:58 2021

@author: aurelien
"""

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

#%%
resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

fname = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/data_preprocessed_asr10/sub-PAI0912/sub-PAI0912_ses-S001_task-Default_run-001_preprocessed_rtMax2_eeg_raw.fif'
EEG = mne.io.read_raw_fif(fname,verbose=False)
montage = EEG.set_montage('standard_1020')

ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
ten_twenty_montage.plot()
montage.plot()
ch_names = EEG.ch_names
np.save(resultsLoc+'channelsName.npy',ch_names)