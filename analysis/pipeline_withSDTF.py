#%%
import os
import mne
import matplotlib.pyplot as plt
import numpy as np
#import scipy.io as sio
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from meegkit.utils.matrix import sliding_window
import pandas as pd
import pdc_dtf, pdc_dtf2
import math

#%%

nchan = 64
rt_max = 2
tmin, tmax = -3, 3
downsampling = False

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
EEG_folder = os.listdir(data_folder)
clusters_folder = os.path.join(os.getcwd(),'clusters','')
EEG_files = []
for folder in EEG_folder:
    folder = os.path.join(data_folder,folder)
    files = os.listdir(folder)
    for file in files:
        if file.split('.')[-1] == 'fif':
            if file.split('_')[-3] == 'rtMax{}'.format(rt_max):
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
    
    # epochsData = {}
    # epochsInfo = {}
    # for key,elt in epochs.items():
    #     epochsData[key] = np.transpose(elt.get_data(), (2, 1, 0))
    #     epochsInfo[key] = elt.info
    # for key in epochsData.keys():
    #     out, fromRESS, _ = ress.RESS(epochsData[key], sfreq=sfreq, peak_freq=40,return_maps=True)
    #     epochsData[key] = matmul3d(out, fromRESS)
    #     epochsData[key] = np.transpose(epochsData[key],(2,1,0))
    #     epochs[key] = mne.EpochsArray(epochsData[key],epochsInfo[key]) 
    
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
            epochs[key].resample(sfreq=150)
    sfreq = epochs['a1'].info['sfreq']
    n_samples = epochs['a1'].get_data().shape[2]
    n_fft = int(sfreq*3)
    # n_fft = int(1.5*sfreq)
    # n_fft = 256
    # n_overlap = n_fft/2
    n_overlap = 0
    
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

dataframe = {'participant':n_idx,'a0':a0,'v0':v0,'aav0':aav0,'vav0':vav0,'a1':a1,'v1':v1,'aav1':aav1,'vav1':vav1}
dataframe = pd.DataFrame(dataframe)
dataframe.to_csv('responses_rtMax{}.csv'.format(rt_max),index=False)
#%%
psds_all = {} # Contains the psds for all participants for each condition 'a0','a1','v0',... For instance psds_all['a0'].shape = (n_participants,n_chan,n_freqs)
evoked_all = {} 
for cond in conds:
    psds_all[cond] = np.zeros((len(psds_by_N),nchan,len(freqs)))
    evoked_all[cond] = np.zeros((len(psds_by_N),nchan,n_samples))
    idx=0
    for elt in psds_by_N.values():
        psds_all[cond][idx] = elt[cond]
        idx+=1
    idx=0
    for elt in evoked_by_N.values():
        evoked_all[cond][idx] = elt[cond]
        idx+=1

    # Defining the Regions of Interest (ROI) for the connectivity at the sensors level
    # We use spatio-temporal (-spectral) permutation F-test

for key in psds_all.keys():
    psds_all[key] = np.transpose(psds_all[key], (0, 2, 1)) # In order to use the permutation cluster test function. Now psds_all['a0'].shape = (n_participants,n_freqs,n_chan)  
    # evoked_all[key] = np.transpose(evoked_all[key], (0, 2, 1))
    # adjacency, ch_names = mne.channels.find_ch_adjacency(epochs_a1.info, ch_type='eeg')
    # print(type(adjacency))
    # plt.imshow(adjacency.toarray(), cmap='gray', origin='lower',
    #            interpolation='nearest')
    # plt.xlabel('{} electrods'.format(len(ch_names)))
    # plt.ylabel('{} electrods'.format(len(ch_names)))
    # plt.title('Between-sensor adjacency')
    
    # set cluster threshold
    # threshold = 50.0  # very high, but the test is quite sensitive on this data
    # set family-wise p-value
 
#%%
adjacency, ch_names = mne.channels.find_ch_adjacency(epochs_info, ch_type='eeg') # adjacency matrices are the same regardless of the participants and epoch, so we just pick whatever epoch info.
p_threshold = 0.001
t_threshold = -scipy.stats.distributions.t.ppf(p_threshold / 2., len(EEG_files) - 1)
# t_threshold = None
p_accept = 0.05

# conditions = [psds_a0,psds_a1,psds_v0,psds_v1,psds_aav0,psds_aav1,psds_vav0,psds_vav1]

cond_a1_aav1 = psds_all['a1']-psds_all['aav1']
cond_a1_v1 = psds_all['a1']-psds_all['v1']
cond_v1_vav1 = psds_all['v1']-psds_all['vav1']
cond_aav1_vav1 = psds_all['aav1']-psds_all['vav1']

groups = [cond_a1_aav1,cond_a1_v1,cond_v1_vav1,cond_aav1_vav1]
group_name = ['a1_aav1','a1_v1','v1_vav1','aav1_vav1']
good_clusters, good_clusters_neg, good_clusters_pos = {}, {}, {}
T_obs, T_obs_neg, T_obs_pos = {}, {}, {}
clusters, clusters_neg, clusters_pos = {}, {}, {}
p_values, p_values_neg, p_values_pos = {}, {}, {}

for i,gp in enumerate(groups):
    n_gp = group_name[i]
    T, c, p, _ = mne.stats.spatio_temporal_cluster_1samp_test(gp, adjacency=adjacency, n_jobs=1, 
                                                                               threshold=t_threshold,max_step=1,n_permutations=1024)
    # T_neg, c_neg, p_neg, _ = mne.stats.spatio_temporal_cluster_1samp_test(gp, adjacency=adjacency, n_jobs=1, 
    #                                                                            threshold=t_threshold, buffer_size=None,tail=-1)
    # T_pos, c_pos, p_pos, _ = mne.stats.spatio_temporal_cluster_1samp_test(gp, adjacency=adjacency, n_jobs=1, 
    #                                                                            threshold=t_threshold, buffer_size=None,tail=1)
    
    good_cluster_idx = np.where(p < p_accept)[0]
    good_clusters[n_gp] = good_cluster_idx
    T_obs[n_gp] = T
    clusters[n_gp] = c
    p_values[n_gp] = p
    
    # good_cluster_idx = np.where(p_neg < p_accept)[0]
    # good_clusters_neg[n_gp] = good_cluster_idx
    # T_obs_neg[n_gp] = T_neg
    # clusters_neg[n_gp] = c_neg
    # p_values_neg[n_gp] = p_neg
    
    # good_cluster_idx = np.where(p_pos < p_accept)[0]
    # good_clusters_pos[n_gp] = good_cluster_idx
    # T_obs_pos[n_gp] = T_pos
    # clusters_pos[n_gp] = c_pos
    # p_values_pos[n_gp] = p_pos

for key,value in good_clusters.items():
    print('')
    print('{}: {}'.format(key,value))
    
# for key,value in p_values_by_group.items():
#     print('')
#     print('{}: {}'.format(key,value))


#%%

colors = {"Aud": "crimson", "Vis": 'steelblue'}
linestyles = {"L": '-', "R": '--'}

# fmap_types = ['diff','tobs']
SIG_CLUST = []
SIG_SENS = []
SIG_FREQ = []
srs = {}
frqs = {}
fmap_types = ['diff']
for key in good_clusters.keys():
    l_sr = []
    l_frqs = []
    for idx,clu in enumerate(good_clusters[key]):
        for tp in fmap_types:
            psds_idx,ch_idx = np.squeeze(clusters[key][clu])
            psds_idx = np.unique(psds_idx)
            ch_idx = np.unique(ch_idx)
            
            if key == 'a1_aav1':
                if idx == 3:
                    pick_ch = ch_idx
            
            SIG_CLUST.append(key)
            SIG_SENS.append(ch_idx)
            SIG_FREQ.append(freqs[psds_idx])
            l_sr.append(ch_idx)
            l_frqs.append(freqs[psds_idx])
            srs[key] = l_sr
            frqs[key] = l_frqs
            
            
            print('Cluster #{}'.format(clu))
            print('----------------------')
            print('Sensors: ',ch_idx)
            # print(psds_idx)
            print('Frequencies: ',freqs[psds_idx])
            print('')
        
            # evoked_aav1 = evoked_all['aav1'].mean(axis=0)
            # evoked_aav1 = np.transpose(evoked_aav1,(1,0))
            # evoked_aav1 = mne.EvokedArray(evoked_aav1,epochs['aav1'].info,tmin=-2,baseline=(-2,0))
            
            # evoked_vav1 = evoked_all['vav1'].mean(axis=0)
            # evoked_vav1 = np.transpose(evoked_vav1,(1,0))
            # evoked_vav1 = mne.EvokedArray(evoked_vav1,epochs['aav1'].info,tmin=-2,baseline=(-2,0))
            
            # evokeds = {'audio->visual':evoked_aav1,'visual->audio':evoked_vav1}
            # ch_names = np.array(ch_names)
            # picks = list(ch_names[ch_idx])
            # mne.viz.plot_compare_evokeds(evokeds, picks=picks, combine='mean')
            
        
            # get topography for F stat
            f_map = T_obs[key][psds_idx, ...].mean(axis=0)
            cond1 = key.split('_')[0]
            cond2 = key.split('_')[1]
            psds_diff = psds_all[cond1].mean(axis=0)-psds_all[cond2].mean(axis=0)
            psds_diff = psds_diff[psds_idx,:].mean(axis=0)
            
            # get signals at the sensors contributing to the cluster
            # sig_times = epochs.times[time_inds]
            
            # create spatial mask
            mask = np.zeros((f_map.shape[0], 1), dtype=bool)
            mask[ch_idx, :] = True
            
            # initialize figure
            fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
            
            # plot average test statistic and mark significant sensors
            if tp == 'tobs':
                mne.viz.plot_topomap(f_map,pos=epochs_info,mask=mask, axes=ax_topo, cmap='Reds',
                                      vmin=np.min, vmax=np.max, show=False, mask_params=dict(markersize=6),sensors='k.')
            elif tp == 'diff':
                mne.viz.plot_topomap(psds_diff,pos=epochs_info,mask=mask, axes=ax_topo, cmap='RdBu_r',
                                      vmin=np.min, vmax=np.max, show=False, mask_params=dict(markersize=6),sensors='k.')
            
            divider = make_axes_locatable(ax_topo)
            image = ax_topo.images[0]
            ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(image, cax=ax_colorbar)
            # create additional axes (for ERF and colorbar)
        
            
            # add axes for colorbar
        
            # ax_topo.set_xlabel(
            #     'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))
            
            # add new axis for time courses and plot time courses
            ax_signals = divider.append_axes('right', size='300%', pad=1.2)
            title = 'Cluster #{}, {} sensor'.format(idx + 1, len(ch_idx))
            if len(ch_idx) > 1:
                title += "s (mean)"
                
            psds_cond1 = psds_all[cond1].mean(axis=0)
            # psds_cond1 = psds_cond1[psds_idx,:]
            psds_cond1 = psds_cond1.mean(axis=1)
            psds_cond2 = psds_all[cond2].mean(axis=0)
            # psds_cond2 = psds_cond2[psds_idx,:]
            psds_cond2 = psds_cond2.mean(axis=1)
            
            n_cond = {'a1':'audio','v1':'visual','aav1':'audiovisual','vav1':'visuoauditory'}
            cond1 = n_cond[cond1]
            cond2 = n_cond[cond2]
            plt.plot(freqs,psds_cond1,label=cond1)
            plt.plot(freqs,psds_cond2,label=cond2)
            # plt.title(tp+' '+key+" Cluster#{}, {} - {}Hz".format(idx+1,round(freqs[psds_idx][0],1),round(freqs[psds_idx][-1],1)))
            plt.title('Cluster n°{} significant at {} - {}Hz'.format(idx+1,round(freqs[psds_idx][0],1),round(freqs[psds_idx][-1],1)))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (db)')
            plt.legend()
            # .spines['right'].set_visible[False]
            # .spines['top'].set_visible[False]
            # plot_compare_evokeds(evokeds, title=title, picks=ch_inds, axes=ax_signals,
            #                       colors=colors, linestyles=linestyles, show=False,
            #                       split_legend=True, truncate_yaxis='auto')
            
            # plot temporal cluster extent
            ymin, ymax = ax_signals.get_ylim()
            ax_signals.fill_betweenx((ymin, ymax), freqs[psds_idx][0], freqs[psds_idx][-1],
                                      color='orange', alpha=0.3)
            
            # clean up viz
            mne.viz.tight_layout(fig=fig)
            fig.subplots_adjust(bottom=.05)
            plt.show()
            figname = key+'_{}_cluster{}'.format(tp,idx+1)
            # fig.savefig(clusters_folder+figname,format='png')

resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

dfClusters = {'clusters':SIG_CLUST,'sensors':SIG_SENS,'frequencies':SIG_FREQ}
dfClusters = pd.DataFrame(dfClusters)
dfClusters.to_csv(resultsLoc+'clusters.csv',index=False)

np.save(resultsLoc+'srs_file.npy',srs)
np.save(resultsLoc+'frqs_file.npy',frqs)


    #%%
    # Splitting data into relevant functional networks in order to find a set of functional states.
    # We want that functional networks that are similar to each other be marked as instances of the same state.
    # So the number of states will be (hopefully) less than the number of functional networks.
    
    # Test of MVAR modelling on the first epoch in 'epochs' created above.

# np.seterr(all='raise')
# sdtf_by_N = {}
# print(N)
# print('--------')
# s_by_c = {}

srs = np.load(resultsLoc+'srs_file.npy', allow_pickle='TRUE').item()
frqs = np.load(resultsLoc+'frqs_file.npy', allow_pickle='TRUE').item()

pick_ch = np.array([6,23,28,39,56,63])
def shortWindowing(N,epochs_by_N,wsize,overlap,sfreq,z_scoring2='axis0'):
    swindows_by_c = {}
    window = sfreq*wsize/1000
    step = sfreq*(wsize-overlap)/1000
    for c in ['a1','v1','aav1','vav1']:
        # print('**condition: {}**'.format(c))
        epochs_data = epochs_by_N[N][c][:,pick_ch[:,None],:][:,:,0,:]
        # epochs_data = epochs_by_N[N][c][:,:5,:]
        # z-scoring across samples (within each epoch)
        for i,e in enumerate(epochs_data):
            mean,std = e.mean(axis=1),e.std(axis=1)
            mean,std = mean[:,np.newaxis],std[:,np.newaxis]
            epochs_data[i] = (e-mean)/std    
        # z-scoring across trials   
        if z_scoring2 == 'axis0_2':
            mean,std = epochs_data.mean(axis=(0,2)),epochs_data.std(axis=(0,2))
            mean,std = mean[:,np.newaxis],std[:,np.newaxis]
        elif z_scoring2 == 'axis0':
            mean,std = epochs_data.mean(axis=0),epochs_data.std(axis=0)
        for i,e in enumerate(epochs_data):
            epochs_data[i] = (e-mean)/std
        
        # splitting each trial (i.e. epoch) into overlapping 
        swindows = sliding_window(epochs_data,window=int(window),step=int(step))
        swindows = np.transpose(swindows,(0,2,1,3)) # 4D array of shape (n_epochs,n_swindows,n_chan,n_samples_by_swindow)
        swindows_by_c[c] = swindows
        
    return swindows_by_c

def estimatingR(N,swindows_by_c,p_order):
    R_by_c = {}
    for c in ['a1','v1','aav1','vav1']:
        n_swindows = swindows_by_c[c].shape[1] # number of short windows per epoch
        n_trials = swindows_by_c[c].shape[0]
        swindows = swindows_by_c[c]
        R = []
        for w_idx in range(n_swindows):
            Rs = []
            for t_idx in range(n_trials):
                X = swindows[t_idx,w_idx,:,:]
                Rs.append(pdc_dtf.cov(X,p = p_order))
            Rs = np.array(Rs)
            R_hat = Rs.sum(axis=0)/n_trials
            R.append(R_hat)
        R = np.array(R)
        R_by_c[c] = R
    
    return R_by_c

def sdtf(N,R_by_c):
    sdtf_by_c = {}
    errors = 0
    for c in ['a1','v1','aav1','vav1']:
        n_swindows = R_by_c[c].shape[0]
        p_order = R_by_c[c].shape[1]-1
        nchan = R_by_c[c].shape[2]
        # print('*condition: {}*'.format(c))
        sdtf_by_w = [] # shape = (n_swindows,n_fft,n_chan,n_chan)
        # errors = 0
        for w_idx in range(n_swindows):
            R = R_by_c[c][w_idx,:,:,:]
            A,sigma = pdc_dtf.mvar_fit(R=R, nchan=nchan, p=p_order)
            sigma = np.diag(sigma)
            with np.errstate(all='raise'):
                try:
                    D,F = pdc_dtf.DTF(A=A,sigma=sigma,n_fft=n_fft)
                    sdtf_by_w.append(D)
                except FloatingPointError:
                    # print(sigma)
                    errors +=1   
        sdtf_by_w = np.array(sdtf_by_w)
        sdtf_by_c[c] = sdtf_by_w
        # print('errors: {}'.format(errors))
        
    return sdtf_by_c, F, errors


wsize = 60
overlap = 10
err_tot1 = 0
err_tot2 = 0
# for N in n_idx:
#     print('--------')
#     print(N)
#     swindows_by_c = shortWindowing(N,epochs_by_N,wsize,overlap,sfreq,z_scoring2='axis0')
#     R_by_c = estimatingR(N,swindows_by_c,p_order=3)
#     sdtf_by_c, F, errors1 = sdtf(N,R_by_c)
#     print('errors: {}'.format(errors1))
#     swindows_by_c = shortWindowing(N,epochs_by_N,wsize,overlap,sfreq,z_scoring2='axis0_2')
#     R_by_c = estimatingR(N,swindows_by_c,p_order=3)
#     sdtf_by_c, errors2 = sdtf(N,R_by_c)
#     print('errors: {}'.format(errors2))
#     err_tot1 += errors1
#     err_tot2 += errors2
# print('{} vs {}'.format(err_tot1,err_tot2))

swindows_by_c = shortWindowing('sub-PEF0102',epochs_by_N,wsize,overlap,sfreq,z_scoring2='axis0')
R_by_c = estimatingR('sub-PEF0102',swindows_by_c,p_order=3)
sdtf_by_c, F, errors1 = sdtf('sub-PEF0102',R_by_c)


   
#%%
F = F*sfreq
for i in range(3):
    pdc_dtf.plot_all(F, sdtf_by_c['a1'][i] , "audio (window {}/{})".format(i+1,swindows_by_c['a1'].shape[1]))
    plt.savefig(clusters_folder+'DTF_audio{}'.format(i+1),format='png')
    pdc_dtf.plot_all(F, sdtf_by_c['aav1'][i] , "audiovisual (window {}/{})".format(i+1,swindows_by_c['a1'].shape[1]))
    plt.savefig(clusters_folder+'DTF_audiovisual{}'.format(i+1),format='png')

n_swindows = swindows_by_c['a1'].shape[1]
sdtf_theta = {}
sdtf_alpha = {}
sdtf_beta = {}
sdtf_gamma = {}
for c in conds:
    sdtf_w_t, sdtf_w_a, sdtf_w_b, sdtf_w_g  = [], [], [], []
    for w_idx in range(n_swindows-2):
        f_theta = (F >= 4) & (F <= 8)
        f_alpha = (F > 8) & (F <= 13)
        f_beta = (F > 13) & (F <= 30)
        f_gamma = (F > 30) & (F <= 45)
        sdtf_t = sdtf_by_c[c][w_idx][np.ix_(f_theta)].mean(axis=0)
        sdtf_a = sdtf_by_c[c][w_idx][np.ix_(f_alpha)].mean(axis=0)
        sdtf_b = sdtf_by_c[c][w_idx][np.ix_(f_beta)].mean(axis=0)
        sdtf_g = sdtf_by_c[c][w_idx][np.ix_(f_gamma)].mean(axis=0)
        sdtf_w_t.append(sdtf_t)
        sdtf_w_a.append(sdtf_a)
        sdtf_w_b.append(sdtf_b)
        sdtf_w_g.append(sdtf_g)
    sdtf_w_t, sdtf_w_a, sdtf_w_b, sdtf_w_g = np.array(sdtf_w_t), np.array(sdtf_w_a), np.array(sdtf_w_b), np.array(sdtf_w_g)
    sdtf_theta[c] = sdtf_w_t
    sdtf_alpha[c] = sdtf_w_a
    sdtf_beta[c] = sdtf_w_b
    sdtf_gamma[c] = sdtf_w_g

Z = np.zeros((64,64))
for i,ch_row in enumerate(pick_ch):
    for i2,ch_col in enumerate(pick_ch):
        Z[ch_row,ch_col] = sdtf_beta['a1'][0][i,i2]
Z2 = np.zeros((64,64))
for i,ch_row in enumerate(pick_ch):
    for i2,ch_col in enumerate(pick_ch):
        Z2[ch_row,ch_col] = sdtf_beta['aav1'][0][i,i2]

mne.viz.plot_sensors_connectivity(epochs['a1'].info, Z)
mne.viz.plot_sensors_connectivity(epochs['aav1'].info, Z2)



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

evoked_clu = {}
for key in conds:
    print('Condition: ',key)
    bigArray = np.zeros((len(EEG_files),nbClusters,64,64))
    for N_idx in range(len(EEG_files)):
        clu_idx = 0 # clu_idx <= nbCluster
        for gp in group_name:
            for i,clu in enumerate(srs[gp]):
                evk = evoked_all['a1'][N_idx,clu[:,None],:][:,0,:]
                p,bic = pdc_dtf2.compute_order(evk, p_max=10)
                A_est, sigma = pdc_dtf2.mvar_fit(evk, p)
                sigma = np.diag(sigma)  # DTF + PDC support diagonal noise
                # sigma = None
                # compute DTF
                D, freqs = pdc_dtf2.DTF(A_est, sigma)
                matrix = np.zeros((64,64))
                F = freqs * sfreq
                lower_f = frqs[gp][i][0]
                upper_f = frqs[gp][i][-1]
                f_range = (F >= lower_f) & (F <= upper_f)
                D_integrated = D[np.ix_(f_range)].mean(axis=0)
                
                for s1,ch_row in enumerate(clu):
                    for s2,ch_col in enumerate(clu):
                        matrix[ch_row,ch_col] = D_integrated[s1,s2]   
                bigArray[N_idx,clu_idx,:,:] = matrix # 64x64 matrix including DTF
                clu_idx += 1
        print('Participant {}/{}'.format(N_idx+1,len(EEG_files)))
    evoked_clu[key] = bigArray

                




















