#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:46:37 2021

@author: aurelien
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam,schedules
from sklearn.model_selection import KFold

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# inputs = np.concatenate((train_images, test_images), axis=0)
# targets = np.concatenate((train_labels, test_labels), axis=0)

resultsLoc = '/home/aurelien/Desktop/AURELE/M2_DCAS/GitHub_DCAS/IRL_DividedAttention/analysis/results/'
DTFbig_all = np.load(resultsLoc+'DTFbig_all.npy',allow_pickle='TRUE').item()

DTF_audio = np.transpose(DTFbig_all['a1'],[0,2,3,1])
DTF_visual = np.transpose(DTFbig_all['v1'],[0,2,3,1])
DTF_audiovisual = np.transpose(DTFbig_all['aav1'],[0,2,3,1])
DTF_visioauditory = np.transpose(DTFbig_all['vav1'],[0,2,3,1])

r1 = np.random.randint(20,size=4)
r2 = np.random.randint(20,size=4)
r3 = np.random.randint(20,size=4)
r4 = np.random.randint(20,size=4)

DTF_audio_test = DTF_audio[r1]
DTF_visual_test = DTF_visual[r2]
DTF_audiovisual_test = DTF_audiovisual[r3]
DTF_visioauditory_test = DTF_visioauditory[r4]

DTF_audio_train = np.delete(DTF_audio,r1,axis=0)
DTF_visual_train = np.delete(DTF_audio,r1,axis=0)
DTF_audiovisual_train = np.delete(DTF_audio,r1,axis=0)
DTF_visioauditory_train = np.delete(DTF_audio,r1,axis=0)


label_audio = np.ones((20,1))
label_visual = np.ones((20,1))*2
label_audiovisual = np.ones((20,1))*3
label_visioauditory = np.ones((20,1))*4

DTF_input = np.concatenate((DTF_audio,DTF_visual,DTF_audiovisual,DTF_visioauditory),axis=0)
labels = np.concatenate((label_audio,label_visual,label_audiovisual,label_visioauditory),axis=0)

input_shape = DTF_input[0].shape




#%%
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=500,
    decay_rate=1e-6)
opt = Adam(learning_rate=lr_schedule)

# opt = Adam(learning_rate=0.0001)
# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
acc_per_fold = []
loss_per_fold = []
for train, test in kfold.split(DTF_input, labels):

  # Define the model architecture
  model = Sequential()
  model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=input_shape))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  # model.add(Dense(128, activation='relu'))
  model.add(Dense(4, activation='softmax'))

  # Compile the model
  model.compile(loss=sparse_categorical_crossentropy(from_logits=True),
                optimizer=opt,
                metrics=['accuracy'])


  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(DTF_input[train], labels[train],
              epochs=500,
              verbose=True)

  # Generate generalization metrics
  scores = model.evaluate(DTF_input[test], labels[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

