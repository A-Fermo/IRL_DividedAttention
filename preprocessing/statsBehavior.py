#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 14:35:30 2021

@author: aurelien
"""

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pingouin as pg

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%%

resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

root_dir = os.path.dirname(os.getcwd())
data_folder = os.path.join(root_dir,'analysis','results','npyFiles','Acc_RTs','')
RTs_all = os.listdir(data_folder)
              
modality_rt,modality_acc = [],[]
task_rt,task_acc = [],[]
participant_rt,participant_acc = [],[]
var_rt,var_acc = [],[]
RTs,Acc = [],[]

for rt in RTs_all:
    if rt.split('.')[1] != 'npy':
        continue
    N = rt.split('_')[1].split('.')[0]
    stats = np.load(data_folder+rt,allow_pickle=True).item()
    rt_visual = stats['rt_visual']
    rt_auditory = stats['rt_auditory']
    rt_visioauditory = stats['rt_visioauditory']
    rt_audiovisual = stats['rt_audiovisual']

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
    
dfRT = {'RT':RTs,'id':participant_rt,'var':var_rt,'modality':modality_rt,'task':task_rt}
dfRT = pd.DataFrame(dfRT)
dfRT.to_csv(resultsLoc+'RT_All.csv',index=False)

dfAcc = {'Acc':Acc,'id':participant_acc,'var':var_acc,'modality':modality_acc,'task':task_acc}
dfAcc = pd.DataFrame(dfAcc)
dfAcc.to_csv(resultsLoc+'Acc_All.csv',index=False)

#BOXPLOTS
fig, axes = plt.subplots(1,2,figsize=(12,5))
pltAcc = sns.boxplot(ax = axes[0], x='var', y = 'Acc', data = dfAcc, palette = "Set2")
pltAcc.set_xticklabels(['Auditory', 'Visual', 'Audiovisual', 'Visioauditory'])
pltAcc.set_xlabel('Task')
pltAcc.set_ylabel('Accuracy (%)')
pltAcc.set_title('Accuracy across participants')
pltRT =  sns.boxplot(ax = axes[1], x='var', y = 'RT', data = dfRT, palette = "Set2")
pltRT.set_xticklabels(['Auditory', 'Visual', 'Audiovisual', 'Visioauditory'])
pltRT.set_xlabel('Task')
pltRT.set_ylabel('Reaction time (s)')
pltRT.set_title('Reaction times across participants')
plt.savefig(resultsLoc+'Acc_RTs/boxplotAccRT.eps', format='eps',dpi=300)

# ANOVA
# pg.normality(data = df, dv = 'Behavior_subj', group = 'var')
anovaAcc = pg.rm_anova(dv='Acc', within=['task', 'modality'], subject='id',data=dfAcc, detailed=True)
anovaRT = pg.rm_anova(dv='RT', within=['task', 'modality'], subject='id',data=dfRT, detailed=True)
# anovaRT.to_csv(resultsLoc+'anovaRT.csv',index=False)
# anovaAcc.to_csv(resultsLoc+'anovaAcc.csv',index=False)
# print(resAcc)
# print(resRT)

# POST-HOCS
postHocAcc = pg.pairwise_ttests(dv = 'Acc', within = ['task'], subject = 'id', padjust = 'bonf', effsize = 'eta-square', data = dfAcc, return_desc = True, interaction = True)
postHocRT = pg.pairwise_ttests(dv = 'RT', within = ['task'], subject = 'id', padjust = 'bonf', effsize = 'eta-square', data = dfRT, return_desc = True, interaction = True)
# postHocAcc.to_csv(resultsLoc+'postHocAcc.csv',index=False)
# postHocRT.to_csv(resultsLoc+'postHocRT.csv',index=False) 
# print(postHocAcc)
# print(postHocRT)