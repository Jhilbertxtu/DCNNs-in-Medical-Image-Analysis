########################################################################################
# 3D Aorta Segmentation Project                                                        #
#                                                                                      #
# 2. 3D U-net                                                                          #
#                                                                                      #
# created by                                                                           #
# Shuai Chen                                                                           #
# PhD student                                                                          #
# Medical Informatics                                                                  #
#                                                                                      #
# P.O. Box 2040, 3000 CA Rotterdam, The Netherlands, internal postal address Na-2603   #
# Visiting address: office 2616, Wytemaweg 80, 3015 CN Rotterdam, The Netherlands      #
# Email s.chen.2@erasmusmc.nl | Telephone +31 6 334 516 99                             #
# www.erasmusmc.nl                                                                     #
#                                                                                      #
# created on 25/12/2017                                                                #
# Last update: 25/12/2017                                                              #
########################################################################################

from __future__ import print_function

import Modules.Common_modules as cm
import Modules.Callbacks as cb
import Networks.DenseUNet_3D as DenseUNet_3D
import Networks.RSUNet_3D as RSUNet_3D
import Networks.UNet_3D as UNet_3D
import numpy as np
import datetime
from keras import callbacks
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import sys
from Prototypes.ManyFilesBatchGenerator import *
import random

# cm.set_limit_gpu_memory_usage()

random.seed(0)

img_rows = cm.img_rows_3d
img_cols = cm.img_cols_3d
slices = cm.slices_3d
smooth = 1


def train_and_predict(use_existing):

  cm.mkdir(cm.workingPath.model_path)
  cm.mkdir(cm.workingPath.best_model_path)

  class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
      self.losses = []
      self.val_losses = []
      self.sd = []

    def on_epoch_end(self, epoch, logs={}):
      self.losses.append(logs.get('loss'))

      self.val_losses.append(logs.get('val_loss'))

      self.sd.append(step_decay(len(self.losses)))
      print('\nlr:', step_decay(len(self.losses)))
      lrate_file = list(self.sd)
      np.savetxt(cm.workingPath.model_path + 'lrate.txt', lrate_file, newline='\r\n')

  learning_rate = 0.00001

  adam = Adam(lr=learning_rate)

  opti = adam

  def step_decay(losses):
    if len(history.losses)==0:
      lrate = 0.00001
      return lrate
    elif float(2 * np.sqrt(np.array(history.losses[-1]))) < 1.0:
      lrate = 0.00001 * 1.0 / (1.0 + 0.1 * len(history.losses))
      return lrate
    else:
      lrate = 0.00001
      return lrate

  history = LossHistory()
  lrate = callbacks.LearningRateScheduler(step_decay)

  print('-' * 30)
  print('Loading and preprocessing train data...')
  print('-' * 30)

  # Choose which subset you would like to use:

  imgs_train = np.load(cm.workingPath.home_path + 'trainImages3D16.npy')
  imgs_mask_train = np.load(cm.workingPath.home_path + 'trainMasks3D16.npy')

  # imgs_train = np.load(cm.workingPath.home_path + 'vesselImages.npy')
  # imgs_mask_train = np.load(cm.workingPath.home_path + 'vesselMasks.npy')

  # imgs_train = np.load(cm.workingPath.trainingPatchesSet_path + 'img_0000.npy')
  # imgs_mask_train = np.load(cm.workingPath.trainingPatchesSet_path + 'mask_0000.npy')
  # x_val = np.load(cm.workingPath.home_path + 'vesselValImages.npy')
  # y_val = np.load(cm.workingPath.home_path + 'vesselValMasks.npy')

  # x_val = np.load(cm.workingPath.validationSet_path + 'valImages.npy')
  # y_val = np.load(cm.workingPath.validationSet_path + 'valMasks.npy')


  print('_' * 30)
  print('Creating and compiling model...')
  print('_' * 30)

  # model = nw.get_3D_unet()
  # model = nw.get_3D_Eunet()
  # model = DenseUNet_3D.get_3d_denseunet()
  model = UNet_3D.get_3d_unet()
  # model = UNet_3D.get_3d_wnet(opti)
  # model = RSUNet_3D.get_3d_rsunet(opti)

  modelname = 'model.png'
  plot_model(model, show_shapes=True, to_file=cm.workingPath.model_path + modelname)
  model.summary()

  # Callbacks:
  filepath = cm.workingPath.model_path + 'weights.{epoch:02d}-{loss:.5f}-{val_loss:.5f}.hdf5'
  bestfilepath = cm.workingPath.model_path + 'Best_weights.{epoch:02d}-{loss:.5f}-{val_loss:.5f}.hdf5'

  model_checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False)
  model_best_checkpoint = callbacks.ModelCheckpoint(bestfilepath, monitor='val_loss', verbose=0, save_best_only=True)

  # history = cm.LossHistory_Gerda(cm.workingPath.working_path)
  record_history = cb.RecordLossHistory()
  # model_history = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True,
  #  							embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata= None)

  callbacks_list = [record_history, model_best_checkpoint]

  # Should we load existing weights?
  # Set argument for call to train_and_predict to true at end of script
  if use_existing:
    model.load_weights('./unet.hdf5')

  print('-' * 30)
  print('Fitting model...')
  print('-' * 30)

  print(imgs_train.shape)
  print(imgs_mask_train.shape)

  model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=4000, verbose=1, shuffle=True,
            validation_split=0.1, callbacks=callbacks_list)

  print('training finished')


if __name__ == '__main__':
  # Choose whether to train based on the last model:
  # Show runtime:
  starttime = datetime.datetime.now()

  # train_and_predict(True)
  train_and_predict(False)

  endtime = datetime.datetime.now()
  print(endtime - starttime)


  sys.exit(0)