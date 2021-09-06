#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:19:30 2021

@author: aurelien
"""

#%%
import os,sys,logging,shutil
from utils import find
import mne
import matplotlib.pyplot as plt
import numpy as np
# from meegkit import ress
from meegkit.dss import dss_line
from meegkit.asr import ASR
from meegkit.utils.matrix import sliding_window
import math
# from meegkit.utils import matmul3d

#%%

nchan = 64    

os.system('cls' if os.name == 'nt' else 'clear')
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_folder = os.path.join(root_dir,'data')
data_prepro_folder = os.path.join(root_dir,'data_preprocessed_asr10','')
resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

message = \
"""
####################################################################
PREPROCESSING: READING EEG FILE AND FILTERING...
####################################################################
"""

asr_cleanWdw = {'sub-Pcb0805':[100,160],'sub-PEF0102':[100,160],'sub-PEJ0511':[900,960],'sub-PJFS0801':[100,160],'sub-PLD2501':[100,160],
                'sub-PMF0711':[1250,1310],'sub-PMFH0507':[490,550],'sub-PSL2804':[500,560],'sub-PJE0610':[1000,1060],'sub-PAI0912':[120,180],
                'sub-PND1607':[110,170],'sub-PSV2801':[100,160],'sub-PVB2304':[1280,1340],'sub-PZG0801':[100,160],'sub-PZG2305':[800,860],
                'sub-PAS0505':[100,160],'sub-PBJ0703':[900,960],'sub-PJJTT0601':[100,160],'sub-PMR1501':[1220,1280],'sub-POL3007':[270,330],
                'sub-PTH1804':[500,560]}
bad_chans = {'sub-Pcb0805':['T8'],'sub-PEF0102':[],'sub-PEJ0511':[],'sub-PJFS0801':['AF7','Fp1'],'sub-PLD2501':[],
                'sub-PMF0711':['C1,T7'],'sub-PMFH0507':['Fp2'],'sub-PSL2804':['CP2','F3'],'sub-PJE0610':[],'sub-PAI0912':[],
                'sub-PND1607':['T7'],'sub-PSV2801':[],'sub-PVB2304':[],'sub-PZG0801':['CP1', 'Fp2'],'sub-PZG2305':['C3'],'sub-PAS0505':['O2'],
                'sub-PBJ0703':[],'sub-PJJTT0601':[],'sub-PMR1501':['Fp1'],'sub-POL3007':['AF8'],'sub-PTH1804':[]}

while True:
    fname = input(' Please indicate the name of the EEG file (*_eeg.set) to preprocess.\n File: ')
    folder_name = fname.split('_')[0]
    saving_to = os.path.join(data_prepro_folder,folder_name,'')
    fname_to_save = fname.split('eeg.set')[0]+'preprocessed_eeg_raw.fif'
    log_fname = saving_to+folder_name+'.log'
    found,file = find(path=data_folder,fname=fname)
    if found == True:
        try:
            os.mkdir(saving_to)
            break
        except FileExistsError:
            response = input("""\n WARNING! this file has already been preprocessed and saved in the '..GitHub/data_preprocessed/' folder. Do you want to preprocess again and overwrite the existing file [y/n]? """)
            if response.lower() == 'y':
                shutil.rmtree(saving_to)
                os.mkdir(saving_to)
                break
            elif response.lower() == 'n':
                sys.exit('Preprocessing aborted')
print('')

logging.basicConfig(level=logging.INFO,format="[%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_fname),logging.StreamHandler()])
# mne.set_log_level(verbose='INFO')
mne.set_log_file(fname=log_fname, output_format='[%(levelname)s] %(message)s', overwrite=False)
logging.info(message)
EEG = mne.io.read_raw_eeglab(file,preload=True)
info = EEG.info
sfreq = info['sfreq']
if folder_name == 'sub-PLD2501':
    EEG.crop(tmax=4100)
EEG.set_montage('standard_1020')
event_id = {'[]':0,'5':5,'10':10,'11':11,'12':12,'13':13,'14':14,'15':15,'20':20,'40':40,'50':50,'52':52,'60':60,'70':70,'60_vtp':116011,
            '60_vfp':116001,'60_vtn':116010,'60_vfn':116000,'60_avtp':146011,'60_avfp':146001,'60_avtn':146010,'60_avfn':146000,
            '60_afp':126001,'60_atn':126010,'70_atp':127011,'70_afp':127001,'70_atn':127010,'70_afn':127000,'70_avtp':147011,'70_avfp':147001,
            '70_avtn':147010,'70_avfn':147000,'70_vfp':117001,'70_vtn':117010,'101':101,'102':102,'103':103,'104':104,'108':108,'109':109,
            '110':110,'112':112,'114':114,'115':115,'116':116,'119':119,'120':120,'121':121}
(events,event_dict) = mne.events_from_annotations(EEG,event_id=event_id,verbose=False)
meas_date = EEG.info['meas_date']
interTrials_annot = EEG.annotations

#%%
# Annoting bad inter-trial for ICA

tmin = math.floor(events[0][0]/500)
blocks = mne.pick_events(events,include=[40,10])
onsets = blocks[np.arange(0,blocks.shape[0],2)]/sfreq
offsets = blocks[np.arange(1,blocks.shape[0],2)]/sfreq
onsets = onsets[:,0]
onsets = onsets-0.25
onsets = np.append(onsets,EEG[:,:][1][-1])
offsets = offsets[:,0]
offsets = offsets+0.25
offsets = np.append(0,offsets)

durations = onsets-offsets
for i,elt in enumerate(durations):
    if elt < 0:
        durations[i] = 0

descriptions = ['bad_interTrials']*len(durations)
interTrials_annot = mne.Annotations(offsets, durations, descriptions,
                              orig_time=meas_date)

l_freq = 0.1
h_freq = 100
n_channels = info['nchan']
EEG.filter(l_freq=l_freq, h_freq=h_freq)
# EEG_filt = EEG_filt.notch_filter(freqs=(50),picks=mne.pick_types(info,eeg=True))
# EEG_filt.plot_psd(fmax=250,average=True)
if nchan == 62:
    EEG.drop_channels(['Fp1','Fp2'])
info = EEG.info
n_channels = info['nchan']

EEG_data = EEG.get_data().T

EEG_data, artifiacts = dss_line(EEG_data,50,sfreq)

EEG_data = EEG_data.T
EEG = mne.io.RawArray(EEG_data,info)
EEG.set_annotations(interTrials_annot,emit_warning=False)
# EEG.plot(duration=40,block=True)
EEG.info['bads'].extend(bad_chans[folder_name])
logging.info('Channels marked as bad: {}'.format(EEG.info['bads']))
EEG.set_eeg_reference(ref_channels=['TP9','TP10'])
# EEG_reref.set_eeg_reference(ref_channels='average')
logging.info('Channels used for reference: TP9, TP10 (mastoids)')
EEG.interpolate_bads(verbose=True)
# EEG.plot(duration=10,block=True)
del EEG

#%% Artifact Subspace Reconstruction

# Train on a clean portion of data
message = \
"""
####################################################################
PREPROCESSING: ARTIFACT SUBSPACE RECONSTRUCTION (ASR)...
####################################################################
"""
logging.info(message)

start_window, end_window = asr_cleanWdw[folder_name][0],asr_cleanWdw[folder_name][1]

plt.close()
print('')
logging.info('Selected clean window of data for ASR: {}s-{}s'.format(start_window,end_window))
asr = ASR(sfreq=sfreq,cutoff=10,method='euclid',estimator='lwf')
train_idx = np.arange(start_window * sfreq, end_window * sfreq, dtype=int)
_, sample_mask = asr.fit(EEG_data[:, train_idx])

# Apply filter using sliding (non-overlapping) windows
X = sliding_window(EEG_data, window=int(sfreq), step=int(sfreq))
Y = np.zeros_like(X)
for i in range(X.shape[1]):
    Y[:, i, :] = asr.transform(X[:, i, :])

EEG_filt_data = X.reshape(n_channels, -1)  # reshape to (n_chans, n_times)
clean = Y.reshape(n_channels, -1)
EEG_postASR = mne.io.RawArray(clean, info)
EEG_postASR.set_annotations(interTrials_annot,emit_warning=False)


#%%
# ICA on the continuous data
# If ASR applied later on, then we need to work on continuous data

message = \
"""
####################################################################
PREPROCESSING: INDEPENDANT COMPONENT ANALYSIS (ICA)...
####################################################################
"""   
# print(message)
logging.info(message)

ica = mne.preprocessing.ICA(n_components=0.99, random_state=97, max_iter='auto')
ica.fit(EEG_postASR,reject_by_annotation=True)
EEG_temporary = EEG_postASR.copy().set_annotations(None)
EEG_temporary.load_data()
plt.ion()
figures = ica.plot_components()
for idx,f in enumerate(figures):
    f.savefig(saving_to+'ica_components'+str(idx+1)+'.pdf',format='pdf')
plt.show()

while True:
    response = input('Which ICA component(s) do you want to plot [c1,c2,..]? ')
    if len (response) == 0:
        break
    else:
        try:
            l = []
            for elt in response.split(','):
                l.append(int(elt))
            l = sorted(l)
            figures = ica.plot_properties(EEG_temporary, picks=l)
            for idx,f in enumerate(figures):
                f.savefig(saving_to+'ica_property'+str(l[idx])+'.pdf',format='pdf')
            break   
        except:
            print('Wrong response. Please retry.')
        
while True:
    response = input('Do you want to plot ICA sources [y/n]? ')       
    if response.lower() == 'y':
        ica.plot_sources(EEG_temporary,start=0,stop=20)
        break
    elif response.lower() == 'n':
        break
    else:
        print('Wrong response.')

del EEG_temporary   
while True:
    response = input('Do you want to plot EEG signals post ASR [y/n]? ')       
    if response.lower() == 'y':
        EEG_postASR.plot(duration=10)
        break
    elif response.lower() == 'n':
        break
    else:
        print('Wrong response.')
        
while True:
    exclude = input('Which component(s) do you want to exclude [c1,c2,...]? ')
    try:
        l = []
        for elt in exclude.split(','):
            l.append(int(elt))
        l = sorted(l)
        ica.exclude = l
        break
    except:
        print('You did not give a list of integers or at least one integer is out of range. Please retry.')
        
plt.close('all')
logging.info('ICA component(s) excluded: {}'.format(l))
ica.apply(EEG_postASR)

#%%
# Annotating EEG data

message = \
"""
####################################################################
PREPROCESSING: ANNOTATING DATA...
####################################################################
""" 

rt_max = 2
vTarget_detected = mne.Annotations(None, None, None)
conditions = {}
rt_auditory = []
rt_visual = []
rt_audiovisual = []
rt_visioauditory = []
hits_auditory = 0
hits_visual = 0
hits_audiovisual = 0
hits_visioauditory = 0

for condition,name in [(11,"visual"),(12,"audio"),(13,"visual_audio"),(14,"audio_visual")]:
    indexes = []
    blocks = mne.pick_events(events,include=[10,condition,40])
    for i,elt in enumerate(blocks):
        if elt[2] == condition:
            if blocks[i-1][2] == 10 and blocks[i+1][2] == 40:
                indexes+=[i-1,i+1]
    blocks = blocks[indexes]
    start = []
    end = []
    trials = np.empty((0, 3), int)
    for i,elt in enumerate(blocks):
        if i%2 == 0:
            start.append(np.where(np.all(events == elt,axis=1))[0][0])
        else:
            end.append(np.where(np.all(events == elt,axis=1))[0][0])
    for i in range(len(end)):
        trials = np.concatenate((trials,events[start[i]:end[i]+1,:]),axis=0)
    indexes = []
    for i,elt in enumerate(trials):
        if elt[2] in [10,condition,50,52,60,70,40]:
            indexes.append(i)
    trials = trials[indexes]
    conditions[name] = trials
    
    rows = (trials[:,2] == 52)|(trials[:,2] == 60)
    visualCues = trials[np.ix_(rows,[0,1,2])]
    rows = (trials[:,2] == 50)|(trials[:,2] == 70)
    audioCues = trials[np.ix_(rows,[0,1,2])]
    if condition in [13,14]:
        rows = (trials[:,2] == 52)|(trials[:,2] == 60)|(trials[:,2] == 50)|(trials[:,2] == 70)
        visualaudioCues = trials[np.ix_(rows,[0,1,2])]

    rows = (trials[:,2] == 10)|(trials[:,2] == 40)
    start_end = np.ix_(rows)[0]

    if condition in [11,12]:
        idx = 0 if condition == 11 else 1
        for i,elt in enumerate(visualCues):
            if elt[2] == 60:
                try:
                    rt = (visualCues[i+1][0] - elt[0])/sfreq
                    if visualCues[i+1][2] == 52 and rt < rt_max:
                        vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description=['60_vtp','60_afp'][idx])
                        if condition == 11:
                            rt_visual.append(rt)
                            hits_visual += 1
                    else:
                        vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description=['60_vfn','60_atn'][idx])
                except IndexError:
                    vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description=['60_vfn','60_atn'][idx])
        
        for i,elt in enumerate(audioCues):
            if elt[2] == 70:
                try:
                    rt = (audioCues[i+1][0] - elt[0])/sfreq
                    if audioCues[i+1][2] == 50 and rt < rt_max:
                        vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description=['70_vfp','70_atp'][idx])
                        if condition == 12:
                            rt_auditory.append(rt)
                            hits_auditory += 1
                    else:
                        vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description=['70_vtn','70_afn'][idx])
                except IndexError:
                    vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description=['70_vtn','70_afn'][idx])
        
    elif condition in [13,14]:
        for ii in np.arange(0,10,2):
            try:
                visualaudioCues = trials[start_end[ii]+2:start_end[ii+2]-1]
            except IndexError:
                visualaudioCues = trials[start_end[ii]+2:len(trials)-1]
            # print(visualaudioCues)
            for i,elt in enumerate(visualaudioCues):
                idx=1
                index=1
                while visualaudioCues[f(i,idx)][2] not in [50,52]:
                    if f(i,idx) == 0:
                        break
                    else:
                        idx+=1
                try:
                    while visualaudioCues[i+index][2] not in [50,52]:
                        index+=1   
                except IndexError:
                    index=-1
                    
                index2 = i+index+1
                try:
                    while visualaudioCues[index2][2] not in [50,52]:
                        index2+=1   
                except IndexError:
                    index2=-1
                if i > 0 and i < len(visualaudioCues)-1:
                    rt = (visualaudioCues[i+index][0] - elt[0])/sfreq
                    if elt[2] == 60:
                        if visualaudioCues[i+index][2] == 52 and visualaudioCues[index2][2] == 50 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avtp')
                            rt_visioauditory.append(rt)
                            hits_visioauditory += 1
                        elif visualaudioCues[i+index][2] == 52 and visualaudioCues[index2][2] == 52 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avfp')
                        elif (visualaudioCues[f(i,idx)][2] == 52 or visualaudioCues[f(i,idx)][2] not in [50,52]) and (visualaudioCues[i+index][2] == 50 or visualaudioCues[i+index][2] not in [50,52]):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avtn')
                        elif (visualaudioCues[f(i,idx)][2] == 50 or visualaudioCues[f(i,idx)][2] not in [50,52]) and (visualaudioCues[i+index][2] == 50 or (visualaudioCues[i+index][2] == 52 and rt >= rt_max) or visualaudioCues[i+index][2] not in [50,52]):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avfn')
                        elif (visualaudioCues[f(i,idx)][2] == 52 or visualaudioCues[f(i,idx)][2] not in [50,52]) and ((visualaudioCues[i+index][2] == 52 and rt >= rt_max) or visualaudioCues[i+index][2] not in [50,52]):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avtn')
                        else:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60')
                    elif elt[2] == 70:   
                        if visualaudioCues[i+index][2] == 50 and visualaudioCues[index2][2] == 52 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avtp')
                            rt_audiovisual.append(rt)
                            hits_audiovisual += 1
                        elif visualaudioCues[i+index][2] == 50 and visualaudioCues[index2][2] == 50 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avfp')
                        elif (visualaudioCues[f(i,idx)][2] == 50 or visualaudioCues[f(i,idx)][2] not in [50,52]) and (visualaudioCues[i+index][2] == 52 or visualaudioCues[i+index][2] not in [50,52]):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avtn')
                        elif (visualaudioCues[f(i,idx)][2] == 52 or visualaudioCues[f(i,idx)][2] not in [50,52]) and (visualaudioCues[i+index][2] == 52 or (visualaudioCues[i+index][2] == 50 and rt >= rt_max) or visualaudioCues[i+index][2] not in [50,52]):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avfn')
                        elif (visualaudioCues[f(i,idx)][2] == 50 or visualaudioCues[f(i,idx)][2] not in [50,52]) and ((visualaudioCues[i+index][2] == 50 and rt >= rt_max) or visualaudioCues[i+index][2] not in [50,52]):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avtn')
                        else:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70')
    
                elif i == 0:
                    rt = (visualaudioCues[i+index][0] - elt[0])/sfreq
                    if condition == 13 and elt[2] == 60:
                        if visualaudioCues[i+index][2] == 52 and visualaudioCues[index2][2] == 50 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avtp')
                            rt_visioauditory.append(rt)
                            hits_visioauditory += 1
                        elif visualaudioCues[i+index][2] == 52 and visualaudioCues[index2][2] == 52 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avfp')
                        elif visualaudioCues[i+index][2] == 50 or (visualaudioCues[i+index][2] == 52 and rt >= rt_max):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avfn')
                        else:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60')
                    elif condition == 13 and elt[2] == 70:
                        if visualaudioCues[i+index][2] == 50 and visualaudioCues[index2][2] == 52 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avtp')
                            rt_audiovisual.append(rt)
                            hits_audiovisual += 1
                        elif visualaudioCues[i+index][2] == 50 and visualaudioCues[index2][2] == 50 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avfp')
                        elif visualaudioCues[i+index][2] == 52 or (visualaudioCues[i+index][2] == 50 and rt >= rt_max):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avtn')
                        else:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70')
                    elif condition == 14 and elt[2] == 60:
                        if visualaudioCues[i+index][2] == 52 and visualaudioCues[index2][2] == 50 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avtp')
                            rt_visioauditory.append(rt)
                            hits_visioauditory += 1
                        elif visualaudioCues[i+index][2] == 52 and visualaudioCues[index2][2] == 52 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avfp')    
                        elif visualaudioCues[i+index][2] == 50 or (visualaudioCues[i+index][2] == 52 and rt >= rt_max):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avtn')
                        else:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60')
                    elif condition == 14 and elt[2] == 70:
                        if visualaudioCues[i+index][2] == 50 and visualaudioCues[index2][2] == 52 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avtp')
                            rt_audiovisual.append(rt)
                            hits_audiovisual += 1
                        elif visualaudioCues[i+index][2] == 50 and visualaudioCues[index2][2] == 50 and rt < rt_max:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avfp')
                        elif visualaudioCues[i+index][2] == 52 or (visualaudioCues[i+index][2] == 50 and rt >= rt_max):
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avfn')
                        else:
                            vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70')
                else:
                    if elt[2] == 60 and visualaudioCues[f(i,idx)][2] == 50:
                        vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avfn')
                    elif elt[2] == 60 and visualaudioCues[f(i,idx)][2] == 52:
                        vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='60_avtn')
                    elif elt[2] == 70 and visualaudioCues[f(i,idx)][2] == 52:
                        vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avfn')
                    elif elt[2] == 70 and visualaudioCues[f(i,idx)][2] == 50:
                        vTarget_detected = vTarget_detected + mne.Annotations(onset=elt[0]/sfreq, duration=0.002, description='70_avtn')    

vTarget_detected.delete(0)
mapping = {5:'5',10:'10',11:'11',12:'12',13:'13',14:'14',40:'40',50:'50',52:'52'}
events = mne.pick_events(events,include=[5,10,11,12,13,14,40,50,52])
main_events = mne.annotations_from_events(events=events, event_desc=mapping, sfreq=sfreq)
EEG_postASR.set_annotations(interTrials_annot + main_events + vTarget_detected)

stats = {'rt_visual':rt_visual,'rt_auditory':rt_auditory,'rt_visioauditory':rt_visioauditory,'rt_audiovisual':rt_audiovisual}
np.save(resultsLoc+'npyFiles','/Acc_RTs/rt_'+folder_name,stats)

#%%
# Saving preprocessed data

message = \
"""
####################################################################
PREPROCESSING: SAVING FILE TO '..GitHub/data_preprocessed/..'
####################################################################
""" 
# print(message)
logging.info(message)
precision = 'double'

print('')

EEG_postASR.save(saving_to+fname_to_save,fmt=precision,overwrite=True)
del EEG_postASR