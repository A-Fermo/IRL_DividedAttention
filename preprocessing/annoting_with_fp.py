#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:48:46 2021

@author: a.fermo
"""

nchan = 64
rt_max = 2

import os
import mne
import numpy as np
import math
import pandas as pd
#%%

root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_folder = os.path.join(root_dir,'data_preprocessed_asr10','')
raw_data = os.path.join(root_dir,'data','')
EEG_folder = os.listdir(data_folder)
EEG_files = []
response = input('Specific EEG file (*_eeg_raw.fif) (if none, all files are taken): ')
resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

def f(x,y):
    if x-y<0:
        return 0
    else:
        return x-y
    
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
        return file

root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_folder = os.path.join(root_dir,'data_preprocessed_asr10','')
raw_data = os.path.join(root_dir,'data','')
EEG_folder = os.listdir(data_folder)
EEG_files = []
response = input('Specific EEG file (*_eeg_raw.fif) (if none, all files are taken): ')
if len(response) != 0:
    EEG_files.append(find(path=data_folder,fname=response))
else:
    for folder in EEG_folder:
        folder = os.path.join(data_folder,folder)
        files = os.listdir(folder)
        for file in files:
            if file.split('.')[-1] == 'fif':
                if file.split('_')[-3] == 'preprocessed':
                    file = os.path.join(folder,file)
                    EEG_files.append(file)
modality_rt,modality_acc = [],[]
task_rt,task_acc = [],[]
participant_rt,participant_acc = [],[]
var_rt,var_acc = [],[]
RTs = []
Acc = []
for index, fname in enumerate(EEG_files):
    N = fname.split('/')[-1].split('_')[0] # ID of each participant
    raw_file = fname.split('/')[-1].split('preprocessed_eeg')[0]+'eeg.set'
    raw_file = find(path=raw_data,fname=raw_file)
    EEG_raw = mne.io.read_raw_eeglab(raw_file)
    event_id = {'[]':0,'5':5,'10':10,'11':11,'12':12,'13':13,'14':14,'15':15,'20':20,'40':40,'50':50,'52':52,'60':60,'70':70,'60_vtp':116011,
            '60_vfp':116001,'60_vtn':116010,'60_vfn':116000,'60_avtp':146011,'60_avfp':146001,'60_avtn':146010,'60_avfn':146000,
            '60_afp':126001,'60_atn':126010,'70_atp':127011,'70_afp':127001,'70_atn':127010,'70_afn':127000,'70_avtp':147011,'70_avfp':147001,
            '70_avtn':147010,'70_avfn':147000,'70_vfp':117001,'70_vtn':117010,'101':101,'102':102,'103':103,'104':104,'108':108,'109':109,
            '110':110,'112':112,'114':114,'115':115,'116':116,'119':119,'120':120,'121':121}
    (events,event_dict) = mne.events_from_annotations(EEG_raw,event_id=event_id,verbose=False)
    EEG = mne.io.read_raw_fif(fname,verbose=False)
    nchan = len(EEG.ch_names)
    ch_names = EEG.ch_names
    sfreq = EEG.info['sfreq']
    meas_date = EEG_raw.info['meas_date']
    del EEG_raw

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
    interTrials_annot = EEG.annotations
    EEG.set_annotations(interTrials_annot + main_events + vTarget_detected)
    (events,event_dict) = mne.events_from_annotations(EEG,event_id=event_id,verbose=False)


    for elt in rt_visual:
        RTs.append(elt)
        var_rt.append(1)
        modality_rt.append('VISUAL')
        task_rt.append('SINGLE')
        participant_rt.append(N)

    for elt in rt_auditory:
        RTs.append(elt)
        var_rt.append(2)
        modality_rt.append('AUDIO')
        task_rt.append('SINGLE')
        participant_rt.append(N)

    for elt in rt_visioauditory:
        RTs.append(elt)
        var_rt.append(3)
        modality_rt.append('VISUAL')
        task_rt.append('SWITCH')
        participant_rt.append(N)

    for elt in rt_audiovisual:
        RTs.append(elt)
        var_rt.append(4)
        modality_rt.append('AUDIO')
        task_rt.append('SWITCH')
        participant_rt.append(N)
    
    i = 0
    while i < 4:
        participant_acc.append(N)
        i+=1
    Acc.append(len(rt_visual))
    Acc.append(len(rt_auditory))
    Acc.append(len(rt_visioauditory))
    Acc.append(len(rt_audiovisual))
    var_acc.append(1)
    var_acc.append(2)
    var_acc.append(3)
    var_acc.append(4)
    modality_acc.append('VISUAL')
    modality_acc.append('AUDIO')
    modality_acc.append('VISUAL')
    modality_acc.append('AUDIO')
    task_acc.append('SINGLE')
    task_acc.append('SINGLE')
    task_acc.append('SWITCH')
    task_acc.append('SWITCH')
    stats = {'rt_visual':rt_visual,'rt_auditory':rt_auditory,'rt_visioauditory':rt_visioauditory,'rt_audiovisual':rt_audiovisual}
    np.save(resultsLoc+'/RTs/rt_'+N,stats)
    #%%
    
    # fname_tosave = fname.split('_eeg_raw.fif')[0]+'_rtMax{}_eeg_raw.fif'.format(rt_max)
    # EEG.save(fname_tosave,fmt='double',overwrite=True)
    del EEG
    

# dfRT = {'RT':RTs,'id':participant_rt,'var':var_rt,'modality':modality_rt,'task':task_rt}
# dfRT = pd.DataFrame(dfRT)
# dfRT.to_csv(resultsLoc+'RT_All.csv',index=False)

# dfAcc = {'Acc':Acc,'id':participant_acc,'var':var_acc,'modality':modality_acc,'task':task_acc}
# dfAcc = pd.DataFrame(dfAcc)
# dfAcc.to_csv(resultsLoc+'Acc_All.csv',index=False)