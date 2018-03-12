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
# created on 12/01/2018
# Last update: 12/01/2018
########################################################################################

from __future__ import print_function

import Modules.Common_modules as cm
import Modules.Network as nw
import numpy as np
import datetime
from keras import callbacks
from keras.utils import plot_model
import sys
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from glob import glob
import gc

img_rows = 512
img_cols = 512

smooth = 1


def train_and_predict(use_existing, x_train, x_val, y_train, y_val, cross, pre_lrate):
  cm.mkdir(cm.workingPath.model_path)
  cm.mkdir(cm.workingPath.best_model_path)

  class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
      self.losses = [1]
      self.val_losses = []
      self.lr = []

    def on_epoch_end(self, epoch, logs={}):
      self.losses.append(logs.get('loss'))
      loss_file = (list(self.losses))
      np.savetxt(cm.workingPath.model_path + 'loss_' + 'Val.%02d.txt' % (cross), loss_file[1:], newline='\r\n')

      self.val_losses.append(logs.get('val_loss'))
      val_loss_file = (list(self.val_losses))
      np.savetxt(cm.workingPath.model_path + 'val_loss_' + 'Val.%02d.txt' % (cross), val_loss_file, newline='\r\n')

      self.lr.append(step_decay(len(self.losses)))
      print('\nLearning rate:', step_decay(len(self.losses)))
      lrate_file = (list(self.lr))
      np.savetxt(cm.workingPath.model_path + 'lrate_' + 'Val.%02d.txt' % (cross), lrate_file, newline='\r\n')

  if cross == 0:
    learning_rate = 0.0001
    decay_rate = 5e-6
    momentum = 0.9
  else:
    learning_rate = pre_lrate
    decay_rate = 5e-6
    momentum = 0.9

  # sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
  adam = Adam(lr=learning_rate, decay=decay_rate)

  opti = adam

  def step_decay(losses):
    if float(2 * np.sqrt(np.array(history.losses[-1]))) < 100:
      if cross == 0:
        lrate = 0.0001 * 1 / (1 + 0.1 * len(history.losses))
        # lrate = 0.0001
        momentum = 0.8
        decay_rate = 2e-6
      else:
        lrate = pre_lrate * 1 / (1 + 0.1 * len(history.losses))
        # lrate = 0.0001
        momentum = 0.8
        decay_rate = 2e-6
      return lrate
    else:
      if cross == 0:
        lrate = 0.0001
      else:
        lrate = pre_lrate
      return lrate

  history = LossHistory()
  lrate = LearningRateScheduler(step_decay)

  print('_' * 30)
  print('Creating and compiling model...')
  print('_' * 30)

  # model = nw.get_simple_unet(opti)
  # model = nw.get_shallow_unet(sgd)
  model = nw.get_unet(opti)
  # model = nw.get_dropout_unet()
  # model = nw.get_unet_less_feature()
  # model = nw.get_unet_dilated_conv_4()
  # model = nw.get_unet_dilated_conv_7()
  # model = nw.get_2D_Deeply_supervised_network()

  modelname = 'model.png'
  # plot_model(model, show_shapes=True, to_file=cm.workingPath.model_path + modelname)
  model.summary()

  # Callbacks:

  filepath = cm.workingPath.model_path + 'Val.%02d_' % (cross) + 'weights.{epoch:02d}-{loss:.5f}.hdf5'
  bestfilepath = cm.workingPath.best_model_path + 'Val.%02d_' % (cross) + 'Best_weights.{epoch:02d}-{loss:.5f}.hdf5'
  unet_hdf5_path = cm.workingPath.working_path + 'unet.hdf5'

  model_checkpoint = callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True)
  model_best_checkpoint = callbacks.ModelCheckpoint(bestfilepath, monitor='val_loss', verbose=0, save_best_only=True)
  model_best_unet_hdf5 = callbacks.ModelCheckpoint(unet_hdf5_path, monitor='val_loss', verbose=0, save_best_only=True)

  history = LossHistory()

  callbacks_list = [history, lrate, model_checkpoint, model_best_checkpoint, model_best_unet_hdf5]

  # Should we load existing weights?
  # Set argument for call to train_and_predict to true at end of script
  if use_existing:
    model.load_weights('./unet.hdf5')

  print('-' * 30)
  print('Fitting model...')
  print('-' * 30)

  model.fit(x_train, y_train, batch_size=4, epochs=50, verbose=1, shuffle=True,
            validation_data=(x_val, y_val), callbacks=callbacks_list)

  print('training finished')


if __name__ == '__main__':
  # Choose whether to train based on the last model:
  # Show runtime:
  starttime = datetime.datetime.now()

  print('-' * 30)
  print('Loading and preprocessing train data...')
  print('-' * 30)

  # Choose which subset you would like to use:
  i = 0
  n_folds = 10

  for i in range(0, n_folds):

    imgs_train = np.load(cm.workingPath.trainingSet_path + 'trainImages_0000.npy')
    imgs_mask_train = np.load(cm.workingPath.trainingSet_path + 'trainMasks_0000.npy')

    n_slices = int(len(imgs_train) / n_folds)

    start_slice = int(i * n_slices)
    end_slice = int((i + 1) * n_slices)
    x_val = imgs_train[start_slice:end_slice].copy()
    x_train = np.concatenate((imgs_train[:start_slice], imgs_train[end_slice:]), axis=0)



    y_val = imgs_mask_train[start_slice:end_slice].copy()
    y_train = np.concatenate((imgs_mask_train[:start_slice], imgs_mask_train[end_slice:]), axis=0)

    del imgs_train
    del imgs_mask_train

    gc.collect()

    fileloss = []
    filevalloss = []
    filelrate = []

    if i == 0:
      pre_lrate = 0.0001
      train_and_predict(False, x_train, x_val, y_train, y_val, i, pre_lrate)
    else:

      # pre_lrate = float(filelrate[-1])
      pre_lrate = 0.0001
      train_and_predict(True, x_train, x_val, y_train, y_val, i, pre_lrate)

      loss_list = sorted(glob(cm.workingPath.model_path + 'loss_Val.*.txt'))
      val_loss_list = sorted(glob(cm.workingPath.model_path + 'val_loss_Val.*.txt'))
      lrate_list = sorted(glob(cm.workingPath.model_path + 'lrate_Val.*.txt'))

      for k in range(len(loss_list)):
        for line in open(loss_list[k]):
          fileloss.append(line[:-1])

      for k in range(len(val_loss_list)):
        for line in open(val_loss_list[k]):
          filevalloss.append(line[:-1])

      for k in range(len(lrate_list)):
        for line in open(lrate_list[k]):
          filelrate.append(line[:-1])

      thefile = open(cm.workingPath.model_path + 'loss.txt', 'w')
      for item in fileloss:
        thefile.write('%s\n' % item)
        thefile.flush()

      thefile = open(cm.workingPath.model_path + 'val_loss.txt', 'w')
      for item in filevalloss:
        thefile.write('%s\n' % item)
        thefile.flush()

      thefile = open(cm.workingPath.model_path + 'lrate.txt', 'w')
      for item in filelrate:
        thefile.write('%s\n' % item)
        thefile.flush()


  endtime = datetime.datetime.now()
  print(endtime - starttime)

  sys.exit(0)
