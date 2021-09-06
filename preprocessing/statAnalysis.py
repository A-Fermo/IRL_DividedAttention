#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 18:21:24 2021

@author: aurelien
"""
#%%
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pingouin as pg

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

#%%
# Behavioural analysis

resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

dfRT = pd.read_csv(resultsLoc+'/Acc_RTs/RT_All.csv')
dfAcc = pd.read_csv(resultsLoc+'/Acc_RTs/Acc_All.csv')

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
plt.savefig(resultsLoc+'boxplotAccRT.eps', format='eps',dpi=300)

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

# Frequency analysis

dfSNR_40 = pd.read_csv(resultsLoc+'SNR_40.csv')
dfSNR_48 = pd.read_csv(resultsLoc+'SNR_48.csv')

fig, axes = plt.subplots(1,2,figsize=(12,5))
pltSNR40 = sns.boxplot(ax = axes[0], x='var', y = 'SNR40', data = dfSNR_40, palette = "Set2")
pltSNR40.set_xticklabels(['Auditory', 'Visual', 'Audiovisual', 'Visioauditory'])
pltSNR40.set_xlabel('Task')
pltSNR40.set_ylabel('SNR (a.u.)')
pltSNR40.set_title('40Hz filtered SNR across participants')
pltSNR48 = sns.boxplot(ax = axes[1], x='var', y = 'SNR48', data = dfSNR_48, palette = "Set2")
pltSNR48.set_xticklabels(['Auditory', 'Visual', 'Audiovisual', 'Visioauditory'])
pltSNR48.set_xlabel('Task')
pltSNR48.set_ylabel('SNR (a.u.)')
pltSNR48.set_title('48Hz filtered SNR across participants')
plt.savefig(resultsLoc+'boxplotSNR.eps', format='eps',dpi=300)

# ANOVA
anovaSNR40 = pg.rm_anova(dv='SNR40', within=['task', 'modality'], subject='id',data=dfSNR_40, detailed=True)
anovaSNR48 = pg.rm_anova(dv='SNR48', within=['task', 'modality'], subject='id',data=dfSNR_48, detailed=True)
# anovaSNR40.to_csv(resultsLoc+'anovaSNR40.csv',index=False)
# anovaSNR48.to_csv(resultsLoc+'anovaSNR48.csv',index=False)

# POST-HOCS
postHocSNR40_tm = pg.pairwise_ttests(dv = 'SNR40', within = ['task','modality'], subject = 'id', padjust = 'bonf', effsize = 'eta-square', data = dfSNR_40, return_desc = True, interaction = True)
postHocSNR40_mt = pg.pairwise_ttests(dv = 'SNR40', within = ['modality','task'], subject = 'id', padjust = 'bonf', effsize = 'eta-square', data = dfSNR_40, return_desc = True, interaction = True)
postHocSNR40 = pd.concat([postHocSNR40_tm,postHocSNR40_mt])
# postHocSNR40.to_csv(resultsLoc+'postHocSNR40.csv',index=False)

