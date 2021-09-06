#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 16:03:43 2021

@author: aurelien
"""

import mne
import numpy as np


fname = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/data_preprocessed_asr10/sub-Pcb0805/sub-Pcb0805_ses-S001_task-Default_run-001_preprocessed_rtMax2_eeg_raw.fif'
EEG = mne.io.read_raw_fif(fname,verbose=False)

ch_names = EEG.ch_names
del EEG

cluster = ['F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','Oz']

idx = []
for ch in cluster:
    index = ch_names.index(ch)
    idx.append(index)

idx = np.array(idx)


mont = mne.channels.make_standard_montage('standard_1020')
mont.plot(kind='topomap', show_names=True)











