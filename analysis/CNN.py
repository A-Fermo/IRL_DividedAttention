#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:44:16 2021

@author: aurelien
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# inputs = np.concatenate((train_images, test_images), axis=0)
# targets = np.concatenate((train_labels, test_labels), axis=0)

# resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'
resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results_shortEpochs/'
DTF_all_epochs = np.load(resultsLoc+'DTF_all_epochs_noCluster.npy',allow_pickle='TRUE').item()

DTF_audio = np.transpose(DTF_all_epochs['a1'],[0,2,3,1])[:,:,:,:]
DTF_visual = np.transpose(DTF_all_epochs['v1'],[0,2,3,1])[:,:,:,:]
DTF_audiovisual = np.transpose(DTF_all_epochs['aav1'],[0,2,3,1])[:,:,:,:]
DTF_visioauditory = np.transpose(DTF_all_epochs['vav1'],[0,2,3,1])[:,:,:,:]

r1 = np.random.choice(DTF_audio.shape[0],size=round(DTF_audio.shape[0]*20/100),replace=False)
r2 = np.random.choice(DTF_visual.shape[0],size=round(DTF_visual.shape[0]*20/100),replace=False)
r3 = np.random.choice(DTF_audiovisual.shape[0],size=round(DTF_audiovisual.shape[0]*20/100),replace=False)
r4 = np.random.choice(DTF_visioauditory.shape[0],size=round(DTF_visioauditory.shape[0]*20/100),replace=False)

DTF_audio_test = DTF_audio[r1]
DTF_visual_test = DTF_visual[r2]
DTF_audiovisual_test = DTF_audiovisual[r3]
DTF_visioauditory_test = DTF_visioauditory[r4]

DTF_audio_train = np.delete(DTF_audio,r1,axis=0)
DTF_visual_train = np.delete(DTF_visual,r2,axis=0)
DTF_audiovisual_train = np.delete(DTF_audiovisual,r3,axis=0)
DTF_visioauditory_train = np.delete(DTF_visioauditory,r4,axis=0)


label_audio_train = np.ones((DTF_audio.shape[0]-len(r1),1))
label_visual_train = np.ones((DTF_visual.shape[0]-len(r2),1))*2
label_audiovisual_train = np.ones((DTF_audiovisual.shape[0]-len(r3),1))*3
label_visioauditory_train = np.ones((DTF_visioauditory.shape[0]-len(r4),1))*4

label_audio_test = np.ones((len(r1),1),dtype='uint8')
label_visual_test = np.ones((len(r2),1),dtype='uint8')*2
label_audiovisual_test = np.ones((len(r3),1),dtype='uint8')*3
label_visioauditory_test = np.ones((len(r4),1),dtype='uint8')*4

DTF_train = np.concatenate((DTF_audio_train,DTF_visual_train,DTF_audiovisual_train,DTF_visioauditory_train),axis=0)
DTF_test = np.concatenate((DTF_audio_test,DTF_visual_test,DTF_audiovisual_test,DTF_visioauditory_test),axis=0)

labels_train = np.concatenate((label_audio_train,label_visual_train,label_audiovisual_train,label_visioauditory_train),axis=0)
labels_test = np.concatenate((label_audio_test,label_visual_test,label_audiovisual_test,label_visioauditory_test),axis=0)

labels_train = tf.keras.utils.to_categorical(labels_train)[:,1:]
labels_test = tf.keras.utils.to_categorical(labels_test)[:,1:]
# DTF_input = np.concatenate((DTF_audio,DTF_visual,DTF_audiovisual,DTF_visioauditory),axis=0)
# labels = np.concatenate((label_audio,label_visual,label_audiovisual,label_visioauditory),axis=0)

input_shape = DTF_train[0].shape




#%%

# Define the model architecture
opt = optimizers.Adam(learning_rate=0.001)
model = models.Sequential()
model.add(layers.Conv2D(8, kernel_size=(2,2), activation='relu', input_shape=input_shape))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=opt,
              metrics=['accuracy'])


# Fit data to model
history = model.fit(DTF_train,labels_train,epochs=500,validation_data=(DTF_test, labels_test))

# Generate generalization metrics
test_loss, test_acc = model.evaluate(DTF_test,labels_test, verbose=2)

