#
# 2D Aorta Segmentation Project
#
# 2. U-net
#
# created by
# Shuai Chen
# PhD student
# Medical Informatics
#
# P.O. Box 2040, 3000 CA Rotterdam, The Netherlands, internal postal address Na-2603
# Visiting address: office 2616, Wytemaweg 80, 3015 CN Rotterdam, The Netherlands
# Email s.chen.2@erasmusmc.nl | Telephone +31 6 334 516 99
# www.erasmusmc.nl
#
# created on 11/01/2018
# Last update: 11/01/2018
########################################################################################

from __future__ import print_function

import Modules.Common_modules as cm
import Networks.UNet_2D as UNet_2D
import Modules.Callbacks as cb
import Modules.Visualization as vs
import numpy as np
import datetime
from keras import callbacks
import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model
import sys
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler

img_rows = 512
img_cols = 512

smooth = 1

def train_and_predict(use_existing):

  cm.mkdir(cm.workingPath.model_path)
  cm.mkdir(cm.workingPath.best_model_path)
  cm.mkdir(cm.workingPath.visual_path)

  class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
      self.losses = [1, 1]
      self.val_losses = []
      self.sd = []

    def on_epoch_end(self, epoch, logs={}):
      self.losses.append(logs.get('loss'))
      loss_file = list(self.losses)
      np.savetxt(cm.workingPath.model_path + 'loss.txt', loss_file, newline='\r\n')

      self.val_losses.append(logs.get('val_loss'))
      val_loss_file = list(self.val_losses)
      np.savetxt(cm.workingPath.model_path + 'val_loss.txt', val_loss_file, newline='\r\n')

      self.sd.append(step_decay(len(self.losses)))
      print('\nlr:', step_decay(len(self.losses)))
      lrate_file = list(self.sd)
      np.savetxt(cm.workingPath.model_path + 'lrate.txt', lrate_file, newline='\r\n')

  learning_rate = 0.0001
  decay_rate = 5e-6
  momentum = 0.9

  # sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
  adam = Adam(lr=learning_rate, decay=decay_rate)

  opti = adam
  def step_decay(losses):
    if float(2 * np.sqrt(np.array(history.losses[-1]))) < 1.0:
      lrate = 0.0001 * 1 / (1 + 0.1 * len(history.losses))
      # lrate = 0.0001
      momentum = 0.8
      decay_rate = 2e-6
      return lrate
    else:
      lrate = 0.0001
      return lrate

  history = LossHistory()
  lrate = LearningRateScheduler(step_decay)

  print('-' * 30)
  print('Loading and preprocessing train data...')
  print('-' * 30)

  # Choose which subset you would like to use:
  i = 0

  imgs_train = np.load(cm.workingPath.trainingSet_path + 'trainImages_%04d.npy' % (i)).astype(np.float32)
  imgs_mask_train = np.load(cm.workingPath.trainingSet_path + 'trainMasks_%04d.npy' % (i)).astype(np.float32)

  # imgs_test = np.load(cm.workingPath.trainingSet_path + 'testImages_%04d.npy'%(i)).astype(np.float32)
  # imgs_mask_test_true = np.load(cm.workingPath.trainingSet_path + 'testMasks_%04d.npy'%(i)).astype(np.float32)

  # Mean for data centering:
  # mean= np.mean(imgs_train)
  # Std for data normalization:
  # std = np.std(imgs_train)

  # imgs_train -= mean
  # imgs_train /= std

  print('_' * 30)
  print('Creating and compiling model...')
  print('_' * 30)

  model = UNet_2D.get_unet(opti)
  # model = nw.get_shallow_unet(sgd)
  # model = nw.get_unet(opti)
  # model = nw.get_dropout_unet()
  # model = nw.get_unet_less_feature()
  # model = nw.get_unet_dilated_conv_4()
  # model = nw.get_unet_dilated_conv_7()
  # model = nw.get_2D_Deeply_supervised_network()

  modelname = 'model.png'
  plot_model(model, show_shapes=True, to_file=cm.workingPath.model_path + modelname)
  model.summary()
  # config = model.get_config()
  # print(config)

  # Callbacks:

  filepath = cm.workingPath.model_path + 'weights.{epoch:02d}-{loss:.5f}.hdf5'
  bestfilepath = cm.workingPath.model_path + 'Best_weights.{epoch:02d}-{loss:.5f}.hdf5'

  model_checkpoint = callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True)
  model_best_checkpoint = callbacks.ModelCheckpoint(bestfilepath, monitor='val_loss', verbose=0, save_best_only=True)

  # history = cm.LossHistory_Gerda(cm.workingPath.working_path)
  history = LossHistory()
  # model_history = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True,
  #  							embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata= None)
  # gradients = cb.recordGradients_Florian(imgs_train, cm.workingPath.model_path, model, True)
  visual = vs.visualize_activation_in_layer(model, imgs_train[100])

  callbacks_list = [history, lrate, visual, model_checkpoint, model_best_checkpoint]

  # Should we load existing weights?
  # Set argument for call to train_and_predict to true at end of script
  if use_existing:
    model.load_weights('./unet.hdf5')

  print('-' * 30)
  print('Fitting model...')
  print('-' * 30)

  temp_weights = model.get_weights()
  vs.plot_conv_weights(temp_weights[2], cm.workingPath.visual_path, 'conv_1')

  model.fit(imgs_train, imgs_mask_train, batch_size=8, epochs=1, verbose=1, shuffle=True,
            validation_split=0.1, callbacks=callbacks_list)

  print('training finished')


if __name__ == '__main__':
  # Choose whether to train based on the last model:
  # Show runtime:
  starttime = datetime.datetime.now()

  train_and_predict(False)

  endtime = datetime.datetime.now()
  print(endtime - starttime)


  sys.exit(0)
