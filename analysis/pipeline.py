#%%
import os
import mne
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
#import scipy.io as sio
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from meegkit import ress
from meegkit.utils import matmul3d
from utils import PSDS_computing

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#%%

resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'

nchan = 64
rt_max = 2
tmin, tmax = -3, 3
downsampling = True
  
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
psdsRESS_by_N = {}
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
    EEG_info = EEG.info
    nchan = len(EEG.ch_names)
    ch_names = EEG.ch_names
    sfreq = EEG.info['sfreq']
    meas_date = EEG.info['meas_date']
   
    evts = {}
    evts0 = {}
    epochs0 = {}
    try:
        evts0['a0'] = mne.pick_events(events,include=[127000])
        epochs0['a0'] = mne.Epochs(EEG,evts0['a0'],tmin=tmin,tmax=tmax,detrend=0,preload=True)
    except:
        epochs0['a0'] = []
    try:
        evts0['v0'] = mne.pick_events(events,include=[116000])
        epochs0['v0'] = mne.Epochs(EEG,evts0['v0'],tmin=tmin,tmax=tmax,detrend=0,preload=True)
    except:
        epochs0['v0'] = []
    try:
        evts0['aav0'] = mne.pick_events(events,include=[147000])
        epochs0['aav0'] = mne.Epochs(EEG,evts0['aav0'],tmin=tmin,tmax=tmax,detrend=0,preload=True)
    except:
        epochs0['aav0'] = []
    try:
        evts0['vav0'] = mne.pick_events(events,include=[146000])
        epochs0['vav0'] = mne.Epochs(EEG,evts0['vav0'],tmin=tmin,tmax=tmax,detrend=0,preload=True)
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
    if downsampling == True:
        for key,elt in epochs.items():
            epochs[key].resample(sfreq=250)
            
    freqs,psds_by_N,evoked_by_N,epochs_by_N = PSDS_computing(N, psds_by_N, evoked_by_N, epochs_by_N, epochs, tmin, tmax)
    _,psdsRESS_by_N = PSDS_computing(N, psdsRESS_by_N, ressEpochs, tmin, tmax)


    del EEG

dataframe = {'participant':n_idx,'a0':a0,'v0':v0,'aav0':aav0,'vav0':vav0,'a1':a1,'v1':v1,'aav1':aav1,'vav1':vav1}
dataframe = pd.DataFrame(dataframe)
dataframe.to_csv(resultsLoc+'responses.csv',index=False)
#%%

psds_all = {} # Contains the psds for all participants for each condition 'a0','a1','v0',... For instance psds_all['a0'].shape = (n_participants,n_chan,n_freqs)
evoked_all = {} 
psdsRESS_all = {}
for cond in conds:
    psds_all[cond] = np.zeros((len(psds_by_N),nchan,len(freqs)))
    evoked_all[cond] = np.zeros((len(psds_by_N),nchan,n_samples))
    psdsRESS_all[cond] = np.zeros((len(psdsRESS_by_N),nchan,len(freqs)))
    idx=0
    for elt in psds_by_N.values():
        psds_all[cond][idx] = elt[cond]
        idx+=1
    idx=0
    for elt in evoked_by_N.values():
        evoked_all[cond][idx] = elt[cond]
        idx+=1
    for elt in psdsRESS_by_N.values():
        psdsRESS_all[key][idx] = elt[key]
        idx+=1

#%%
# Defining the Regions of Interest (ROI) for the connectivity at the sensors level
# We use spatio-temporal (-spectral) permutation F-test
if downsampling == False:
    for key in psds_all.keys():
        psds_all[key] = np.transpose(psds_all[key], (0, 2, 1)) # In order to use the permutation cluster test function. Now psds_all['a0'].shape = (n_participants,n_freqs,n_chan)  

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
if downsampling == False:
    colors = {"Aud": "crimson", "Vis": 'steelblue'}
    linestyles = {"L": '-', "R": '--'}
    
    # fmap_types = ['diff','tobs']
    SIG_CLUST = []
    SIG_SENS = []
    SIG_FREQ = []
    srs = {}
    frqs = {}
    fmap_types = ['tobs','diff']
    # fmap_types = ['diff']
    i_clu = 1
    Tobs_clusters = {}
    psdsDiff_clusters = {}
    for key in good_clusters.keys():
        l_sr = []
        l_frqs = []
        for idx,clu in enumerate(good_clusters[key]):
            psds_idx,ch_idx = np.squeeze(clusters[key][clu])
            psds_idx = np.unique(psds_idx)
            ch_idx = np.unique(ch_idx)

            
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
            for tp in fmap_types:

            
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
                
                n_cond = {'a1':'audio','v1':'visual','aav1':'audiovisual','vav1':'visioauditory'}
                cond1 = n_cond[cond1]
                cond2 = n_cond[cond2]
                plt.plot(freqs,psds_cond1,label=cond1)
                plt.plot(freqs,psds_cond2,label=cond2)
                # plt.title(tp+' '+key+" Cluster#{}, {} - {}Hz".format(idx+1,round(freqs[psds_idx][0],1),round(freqs[psds_idx][-1],1)))
                plt.title('Cluster n°{} significant at {} - {}Hz'.format(i_clu,round(freqs[psds_idx][0],1),round(freqs[psds_idx][-1],1)))
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
                figname = key+'_{}_cluster{}'.format(tp,i_clu)
                fig.savefig(clusters_folder+figname+'eps',format='eps',dpi=300)
                
                Tobs_clusters['cluster'+str(i_clu)] = f_map[ch_idx]
                psdsDiff_clusters['cluster'+str(i_clu)] = psds_diff[ch_idx]
            i_clu += 1
    
    dfClusters = {'clusters':SIG_CLUST,'sensors':SIG_SENS,'frequencies':SIG_FREQ}
    dfClusters = pd.DataFrame(dfClusters)
    dfClusters.to_csv(resultsLoc+'clusters.csv',index=False)
    
    
    np.save(resultsLoc+'srs_file.npy',srs)
    np.save(resultsLoc+'frqs_file.npy',frqs)
    np.save(resultsLoc+'Tobs_clusters.npy',Tobs_clusters)
    np.save(resultsLoc+'psdsDiff_clusters.npy',psdsDiff_clusters)

if downsampling == True:
    np.save(resultsLoc+'evoked_all.npy',evoked_all)
    np.save(resultsLoc+'psds_all.npy',psds_all)
    np.save(resultsLoc+'evoked_by_N.npy',evoked_by_N)
    np.save(resultsLoc+'epochs_by_N.npy',epochs_by_N)
    np.save(resultsLoc+'freqs.npy',freqs)
    np.save(resultsLoc+'sfreq.npy',sfreq)
np.save(resultsLoc+'channelsName.npy',ch_names)
np.save(resultsLoc+'EEG_info.npy',EEG_info)

    #%%
    # Splitting data into relevant functional networks in order to find a set of functional states.
    # We want that functional networks that are similar to each other be marked as instances of the same state.
    # So the number of states will be (hopefully) less than the number of functional networks.
    
    # Test of MVAR modelling on the first epoch in 'epochs' created above.



                




















