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
from meegkit.utils import fold, matmul3d, rms, snr_spectrum, unfold
from meegkit import ress
import pandas as pd
import pdc_dtf, pdc_dtf2
import math

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#%%

nchan = 64
rt_max = 2
tmin, tmax = -3, 3
downsampling = False
    
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
data_folder = os.path.join(root_dir,'data_preprocessed_asr10','')
EEG_folder = os.listdir(data_folder)
clusters_folder = os.path.join(os.getcwd(),'clusters','')
EEG_files = []
for folder in EEG_folder:
    folder = os.path.join(data_folder,folder)
    files = os.listdir(folder)
    for file in files:
        if file.split('.')[-1] == 'fif':
            file = os.path.join(folder,file)
            EEG_files.append(file)
#%%
# Epoching data

# all_events = mne.pick_events(events,include=[50,52,60,70,102,108,114,119])
# all_events = mne.pick_events(events,include=[6000,6001,6010,6011,7000,7001,7010,7011])
psds_by_N = {}
evoked_by_N = {}
epochs_by_N = {}
a0,a1,v0,v1,aav0,aav1,vav0,vav1 = [],[],[],[],[],[],[],[]
n_idx = []
conds = ['a1','v1','aav1','vav1']
for index, fname in enumerate(EEG_files):
    EEG = mne.io.read_raw_fif(fname,verbose=False)
    event_id = {'[]':0,'5':5,'10':10,'11':11,'12':12,'13':13,'14':14,'15':15,'20':20,'40':40,'50':50,'52':52,'60':60,'70':70,'60_vtp':116011,
            '60_vfp':116001,'60_vtn':116010,'60_vfn':116000,'60_avtp':146011,'60_avfp':146001,'60_avtn':146010,'60_avfn':146000,
            '60_afp':126001,'60_atn':126010,'70_atp':127011,'70_afp':127001,'70_atn':127010,'70_afn':127000,'70_avtp':147011,'70_avfp':147001,
            '70_avtn':147010,'70_avfn':147000,'70_vfp':117001,'70_vtn':117010,'101':101,'102':102,'103':103,'104':104,'108':108,'109':109,
            '110':110,'112':112,'114':114,'115':115,'116':116,'119':119,'120':120,'121':121}
    (events,event_dict) = mne.events_from_annotations(EEG,event_id=event_id,verbose=False)
    N = fname.split('/')[-1].split('_')[0] # ID of each participant
    n_idx.append(N)
    
    nchan = len(EEG.ch_names)
    ch_names = EEG.ch_names
    sfreq = EEG.info['sfreq']
    meas_date = EEG.info['meas_date']
   
    evts = {}
    evts0 = {}
    epochs0 = {}
    try:
        evts0['a0'] = mne.pick_events(events,include=[127000])
        epochs0['a0'] = mne.Epochs(EEG,evts['a0'],tmin=tmin,tmax=tmax,detrend=0,preload=True)
    except:
        epochs0['a0'] = []
    try:
        evts0['v0'] = mne.pick_events(events,include=[116000])
        epochs0['v0'] = mne.Epochs(EEG,evts['v0'],tmin=tmin,tmax=tmax,detrend=0,preload=True)
    except:
        epochs0['v0'] = []
    try:
        evts0['aav0'] = mne.pick_events(events,include=[147000])
        epochs0['aav0'] = mne.Epochs(EEG,evts['aav0'],tmin=tmin,tmax=tmax,detrend=0,preload=True)
    except:
        epochs0['aav0'] = []
    try:
        evts0['vav0'] = mne.pick_events(events,include=[146000])
        epochs0['vav0'] = mne.Epochs(EEG,evts['vav0'],tmin=tmin,tmax=tmax,detrend=0,preload=True)
    except:
        epochs0['vav0'] = []
        
    evts['a1'] = mne.pick_events(events,include=[127011])
    evts['v1'] = mne.pick_events(events,include=[116011])
    evts['aav1'] = mne.pick_events(events,include=[147011])
    evts['vav1'] = mne.pick_events(events,include=[146011])
    
    epochs = {}
    for key,elt in evts.items():
        epochs[key] = mne.Epochs(EEG,elt,tmin=tmin,tmax=tmax,detrend=0,preload=True)
    # mne.epochs.equalize_epoch_counts([epochs['a1'],epochs['v1'],epochs['aav1'],epochs['vav1']])
    
    epochs_info = epochs['a1'].info
    n_samples = epochs['a1'].get_data().shape[2]
    # epochs.plot()
    # epochs_a0.plot()
    
    # Simulating data of one audio-visual block for 'sub-PVB2304'
    
    if N == 'sub-PVB2304':
        epochsData = {}
        epochsData['aav1'] = epochs['aav1'].get_data()
        epochsData['vav1'] = epochs['vav1'].get_data()
        for ntrials,cond in ([6,'aav1'],[7,'vav1']): # 6 and 7 because it is about the number of 70_avtp and 60_avtp per block for this participant
            mean = epochsData[cond].mean(axis=0)
            std = epochsData[cond].std(axis=0)
            mean = mean[np.newaxis,:,:]
            std = std[np.newaxis,:,:]
            np.random.seed(0)
            new_data = np.random.randn(ntrials,64,n_samples)*std+mean
            epochsData[cond] = np.concatenate((epochsData[cond],new_data),axis=0)
            epochs[cond] = mne.EpochsArray(epochsData[cond],epochs[cond].info)
    
    a0.append(len(epochs0['a0']))
    v0.append(len(epochs0['v0']))
    aav0.append(len(epochs0['aav0']))
    vav0.append(len(epochs0['vav0']))

    a1.append(len(epochs['a1']))
    v1.append(len(epochs['v1']))
    aav1.append(len(epochs['aav1']))
    vav1.append(len(epochs['vav1']))
                  
    # print("""\n Number of events\t
    #   -----------------
    #   # audio miss:\t\t\t\t{}
    #   # visual miss:\t\t\t\t{}
    #   # audio-visual audio miss:\t{}
    #   # audio-visual visual miss:\t{}
     
    #   audio hit:\t\t\t\t\t{}
    #   visual hit:\t\t\t\t\t{}
    #   audio-visual audio hit:\t\t{}
    #   audio-visual visual hit:\t{}\n""".format(len(evts0['a0']),len(evts0['v0']),len(evts0['aav0']),len(evts0['vav0']),len(evts['a1']),
    #   len(evts['v1']),len(evts['aav1']),len(evts['vav1'])))
        
    #%%
    # RESS
    
    epochsData = {}
    epochsInfo = {}
    ressData = {}
    ressEpochs = {}
    for key,elt in epochs.items():
        epochsData[key] = np.transpose(elt.get_data(), (2, 1, 0))
        epochsInfo[key] = elt.info
    for key in epochsData.keys():
        key_40 = key+'_40'
        key_48 = key+'_48'
        ressData[key_40], fromRESS, _ = ress.RESS(epochsData[key], sfreq=sfreq, peak_freq=40,return_maps=True)
        ressData[key_40] = matmul3d(ressData[key_40], fromRESS)
        ressData[key_40] = np.transpose(ressData[key_40],(2,1,0))
        ressEpochs[key_40] = mne.EpochsArray(ressData[key_40],epochsInfo[key])
        
        ressData[key_48], fromRESS, _ = ress.RESS(epochsData[key], sfreq=sfreq, peak_freq=48,return_maps=True)
        ressData[key_48] = matmul3d(ressData[key_48], fromRESS)
        ressData[key_48] = np.transpose(ressData[key_48],(2,1,0))
        ressEpochs[key_48] = mne.EpochsArray(ressData[key_48],epochsInfo[key])
        
    
    #%%
    # Defining regions of interest
    # freqs = np.logspace(*np.log10([1, 40]), num=40)
    # # freqs = np.arange(1,41)
    # n_cycles = freqs/3.
    # power, itc = mne.time_frequency.tfr_morlet(epochs_tpv, freqs=freqs, n_cycles=n_cycles, use_fft=True,
    #                         return_itc=True, decim=3, n_jobs=1)
    # power.plot_topo(baseline=(-0.5, 0), mode='logratio', title='Average power')
    # power.data.shape
    
    ###############################################"
    #%%
    if downsampling == True:
        for key,elt in epochs.items():
            epochs[key].resample(sfreq=250)
    sfreq = epochs['a1'].info['sfreq']
    n_samples = epochs['a1'].get_data().shape[2]
    n_fft = int(sfreq*3)
    # n_fft = int(1.5*sfreq)
    # n_overlap = n_fft/2
    n_overlap = 0
    
    psds = {}
    psds_mean = {} # average over epochs (with epochs being of shape = (n_epoch,n_chan,n_samples))
    psds_std = {} # standard deviation over epochs
    evoked = {}
    for key,elt in ressEpochs.items():
        psds[key], freqs = mne.time_frequency.psd_welch(elt,n_fft=n_fft,n_overlap=n_overlap,
                                                   n_per_seg=None,window='hamming',tmin=tmin, tmax=tmax,fmin=1, fmax=100,average='mean',
                                                   verbose=False)
        # psds[key] = 10*np.log10(psds[key])
        psds_mean[key] = psds[key].mean(0)
        psds_std[key] = psds[key].std(0)
        evoked[key] = elt.average().data
    psds_by_N[N] = psds_mean
    evoked_by_N[N] = evoked
    epochs_data = {}
    for c in conds:
        epochs_data[c] = epochs[c].get_data()
    epochs_by_N[N] = epochs_data
    # nfreqs = len(freqs)
    # idx = 21
    # channel = EEG.ch_names[idx]
    # cond_mean, cond_std = psds_mean['vav1'], psds_std['vav1']
    # f, ax = plt.subplots()
    # ax.plot(freqs, cond_mean[idx,:], color='k')
    # ax.fill_between(freqs, cond_mean[idx,:] - cond_std[idx,:], cond_mean[idx,:] + cond_std[idx,:],
    #                 color='k', alpha=.5)
    # ax.set(title='Welch PSD (EEG {}, averaged across epochs)'.format(channel), xlabel='Frequency (Hz)',
    #         ylabel='Power Spectral Density (dB)')
    # plt.show()
    
    del EEG

#%%

resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

psds_all = {} # Contains the psds for all participants for each condition 'a0','a1','v0',... For instance psds_all['a0'].shape = (n_participants,n_chan,n_freqs)
# evoked_all = {} 
for key in ressEpochs.keys():
    psds_all[key] = np.zeros((len(psds_by_N),nchan,len(freqs)))
    # evoked_all[key] = np.zeros((len(psds_by_N),nchan,n_samples))
    idx=0
    for elt in psds_by_N.values():
        psds_all[key][idx] = elt[key]
        idx+=1
    # idx=0
    # for elt in evoked_by_N.values():
    #     evoked_all[cond][idx] = elt[cond]
    #     idx+=1

# Replacing missing PSDS (=0) to the mean. That is for channels with 0 entries, replace by the mean over all the other channels.
# This is necessary for computing snr (for each individual participants) otherwise give 'nan'
# Needed for 2 channels only in total (one for 'v1_48' in 'PMFH0711' and one for 'aav1_40' in 'PJFS0801')
def PSDSsim(psds):
    idx = np.where(psds.mean(axis=1)==0)[0]
    if len(idx) != 0:
        print('PSDS simulation...')
        mean = np.sum(psds,axis=0)/len(idx)
        psds[idx] = mean
    return psds

for i in range(20):
    for key in psds_all.keys():
        psds_all[key][i] = PSDSsim(psds_all[key][i])

psdsMean = {} # psds averaged over participants (all channels x freqs)
snr_all = {} # snr for all participants for all channels (all participants x all channels x freqs)
snrMean = {} # snr averaged over participants (all channels x freqs)
snr_allMean = {} # snr for all participants averaged over channels (all participants x freqs)
for key in psds_all.keys():
    snr_all[key] = np.zeros((len(EEG_files),len(freqs),64))
    for i in range(len(EEG_files)):
        snr_all[key][i] = snr_spectrum(psds_all[key][i].T, freqs, skipbins=1, n_avg=3)
    # snr_all[key] = snr_spectrum(np.transpose(psds_all[key],(2,1,0)), freqs, skipbins=1, n_avg=3)
    # snr_all[key] = np.transpose(snr_all[key],(2,1,0))
    snr_all[key] = np.transpose(snr_all[key],(0,2,1))
    snr_allMean[key] = snr_all[key].mean(axis=1)
    psdsMean[key] = psds_all[key].mean(axis=0).T
    # snrMean[key] = snr_spectrum(psdsMean[key], freqs, skipbins=1, n_avg=3)
    # snrMean[key] = snrMean[key].T
    snrMean[key] = snr_all[key].mean(axis=0)
    psdsMean[key] = psdsMean[key].T
    psdsMean[key] = 10 * np.log10(psdsMean[key])

snr_mean = {}
snr_std = {}
psds_mean = {}
psds_std = {}
for key in snrMean.keys():
    snr_mean[key] = snrMean[key].mean(axis=0)
    snr_std[key] = snrMean[key].std(axis=0)
    psds_mean[key] = psdsMean[key].mean(axis=0)
    psds_std[key] = psdsMean[key].std(axis=0)

def plotSpectrum(axis,freqs,spectrum_mean,color,label=None,title=None):
    axis.plot(freqs,spectrum_mean,label=label,color=color)
    if title != None:
        axis.set_title(title,fontsize=12)
    if label != None:
        axis.legend(fontsize=9)

fig,axs = plt.subplots(4,2,figsize=(8,8))
plotSpectrum(axis=axs[0,0],freqs=freqs,spectrum_mean=snr_mean['a1_40'],title='40Hz filtered',color='m')
plotSpectrum(axis=axs[0,1],freqs=freqs,spectrum_mean=snr_mean['a1_48'],title='48Hz filtered',label='Audio',color='m')
plotSpectrum(axis=axs[1,0],freqs=freqs,spectrum_mean=snr_mean['v1_40'],color='g')
plotSpectrum(axis=axs[1,1],freqs=freqs,spectrum_mean=snr_mean['v1_48'],label='Visual',color='g')
plotSpectrum(axis=axs[2,0],freqs=freqs,spectrum_mean=snr_mean['aav1_40'],color='r')
plotSpectrum(axis=axs[2,1],freqs=freqs,spectrum_mean=snr_mean['aav1_48'],label='Audiovisual',color='r')
plotSpectrum(axis=axs[3,0],freqs=freqs,spectrum_mean=snr_mean['vav1_40'],color='c')
plotSpectrum(axis=axs[3,1],freqs=freqs,spectrum_mean=snr_mean['vav1_48'],label='Visioauditory',color='c')
for ax in axs.flat:
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('SNR (a.u.)',fontsize=10)
    ax.label_outer()
    ax.set_ylim([0,10])
plt.savefig(resultsLoc+'ressSNR.eps',format='eps',dpi=300)

fig,axs = plt.subplots(4,2,figsize=(8,8))
plotSpectrum(axis=axs[0,0],freqs=freqs,spectrum_mean=psds_mean['a1_40'],title='40Hz filtered',color='m')
plotSpectrum(axis=axs[0,1],freqs=freqs,spectrum_mean=psds_mean['a1_48'],title='48Hz filtered',label='Audio',color='m')
plotSpectrum(axis=axs[1,0],freqs=freqs,spectrum_mean=psds_mean['v1_40'],color='g')
plotSpectrum(axis=axs[1,1],freqs=freqs,spectrum_mean=psds_mean['v1_48'],label='Visual',color='g')
plotSpectrum(axis=axs[2,0],freqs=freqs,spectrum_mean=psds_mean['aav1_40'],color='r')
plotSpectrum(axis=axs[2,1],freqs=freqs,spectrum_mean=psds_mean['aav1_48'],label='Audiovisual',color='r')
plotSpectrum(axis=axs[3,0],freqs=freqs,spectrum_mean=psds_mean['vav1_40'],color='c')
plotSpectrum(axis=axs[3,1],freqs=freqs,spectrum_mean=psds_mean['vav1_48'],label='Visioauditory',color='c')
for ax in axs.flat:
    ax.set_xlabel('Frequency (Hz)', fontsize=10)
    ax.set_ylabel('PSD (dB)',fontsize=10)
    ax.label_outer()
    ax.set_ylim([-150,-110])
plt.savefig(resultsLoc+'ressPSDS.eps',format='eps',dpi=300)

# Putting everything in a CSV file for ANOVA
snr_Peak40v1 = []
snr_Peak48v1 = []
snr_Peak40a1 = []
snr_Peak48a1 = []
snr_Peak40aav1 = []
snr_Peak48aav1 = []
snr_Peak40vav1 = []
snr_Peak48vav1 = []

for i in range(len(EEG_files)):
    snr_Peak40v1.append(snr_allMean['v1_40'][i][np.ix_(freqs==40)[0][0]])
    snr_Peak48v1.append(snr_allMean['v1_48'][i][np.ix_(freqs==48)[0][0]])
    snr_Peak40a1.append(snr_allMean['a1_40'][i][np.ix_(freqs==40)[0][0]])
    snr_Peak48a1.append(snr_allMean['a1_48'][i][np.ix_(freqs==48)[0][0]])
    snr_Peak40aav1.append(snr_allMean['aav1_40'][i][np.ix_(freqs==40)[0][0]])
    snr_Peak48aav1.append(snr_allMean['aav1_48'][i][np.ix_(freqs==48)[0][0]])
    snr_Peak40vav1.append(snr_allMean['vav1_40'][i][np.ix_(freqs==40)[0][0]])
    snr_Peak48vav1.append(snr_allMean['vav1_48'][i][np.ix_(freqs==48)[0][0]])

snr_Peak40,snr_Peak48 = [],[]
var_40,var_48 = [],[]
modality_40,modality_48 = [],[]
task_40,task_48 = [],[]
participant_40,participant_48 = [],[]
for i in range(len(EEG_files)):
    snr_Peak40.append(snr_Peak40v1[i])
    var_40.append(1)
    modality_40.append('VISUAL')
    task_40.append('SINGLE')
    participant_40.append(n_idx[i])
    
    snr_Peak48.append(snr_Peak48v1[i])
    var_48.append(1)
    modality_48.append('VISUAL')
    task_48.append('SINGLE')
    participant_48.append(n_idx[i])

    snr_Peak40.append(snr_Peak40a1[i])
    var_40.append(2)
    modality_40.append('AUDIO')
    task_40.append('SINGLE')
    participant_40.append(n_idx[i])
    
    snr_Peak48.append(snr_Peak48a1[i])
    var_48.append(2)
    modality_48.append('AUDIO')
    task_48.append('SINGLE')
    participant_48.append(n_idx[i])
    
    snr_Peak40.append(snr_Peak40vav1[i])
    var_40.append(3)
    modality_40.append('VISUAL')
    task_40.append('SWITCH')
    participant_40.append(n_idx[i])
    
    snr_Peak48.append(snr_Peak48vav1[i])
    var_48.append(3)
    modality_48.append('VISUAL')
    task_48.append('SWITCH')
    participant_48.append(n_idx[i])

    snr_Peak40.append(snr_Peak40aav1[i])
    var_40.append(4)
    modality_40.append('AUDIO')
    task_40.append('SWITCH')
    participant_40.append(n_idx[i])
    
    snr_Peak48.append(snr_Peak48aav1[i])
    var_48.append(4)
    modality_48.append('AUDIO')
    task_48.append('SWITCH')
    participant_48.append(n_idx[i])

dfSNR40 = {'SNR40':snr_Peak40,'id':participant_40,'var':var_40,'modality':modality_40,'task':task_40}
dfSNR40 = pd.DataFrame(dfSNR40)
dfSNR40.to_csv(resultsLoc+'SNR_40.csv',index=False)

dfSNR48 = {'SNR48':snr_Peak48,'id':participant_48,'var':var_48,'modality':modality_48,'task':task_48}
dfSNR48 = pd.DataFrame(dfSNR48)
dfSNR48.to_csv(resultsLoc+'SNR_48.csv',index=False)



