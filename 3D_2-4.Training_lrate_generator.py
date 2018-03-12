#
# 3D Aorta and Pulmonary Segmentation
#
# 2. 3D training
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
# created on 09/03/2018
# Last update: 09/03/2018
########################################################################################

from __future__ import print_function

from glob import glob
import Modules.Common_modules as cm
import Modules.Callbacks as cb
import Modules.DataProcess as dp
import Networks.DenseUNet_3D as DenseUNet_3D
import Networks.RSUNet_3D as RSUNet_3D
import Networks.UNet_3D as UNet_3D
import Networks.RSUNet_3D_Gerda as RSUNet_3D_Gerda
import numpy as np
import datetime
from keras import callbacks
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import sys
from Prototypes.BatchGeneratorClass import *

np.random.seed(0)

def train_and_predict(use_existing):

  cm.mkdir(cm.workingPath.model_path)
  cm.mkdir(cm.workingPath.best_model_path)
  cm.mkdir(cm.workingPath.visual_path)

  learning_rate = 0.00001

  adam = Adam(lr=learning_rate)

  opti = adam

  lrate = callbacks.LearningRateScheduler(cb.step_decay)

  print('-' * 30)
  print('Loading and preprocessing train data...')
  print('-' * 30)

  # Scanning training data list:
  originFile_list = sorted(glob(cm.workingPath.training3DSet_path + 'img_*.npy'))
  mask_list = sorted(glob(cm.workingPath.training3DSet_path + 'mask_*.npy'))

  xDatatest = np.load(originFile_list[0])
  yDatatest = np.load(mask_list[0])

  # Scanning validation data list:
  originValFile_list = sorted(glob(cm.workingPath.validationSet_path + 'valImages.npy'))
  maskVal_list = sorted(glob(cm.workingPath.validationSet_path + 'valMasks.npy'))

  x_val = np.load(originValFile_list[0])
  y_val = np.load(maskVal_list[0])

  # Calculate the total amount of training sets:
  nb_file = int(len(originFile_list))
  nb_val_file = int(len(originValFile_list))

  print('_' * 30)
  print('Creating and compiling model...')
  print('_' * 30)

  # Select the model you want to train:
  # model = nw.get_3D_unet()
  # model = nw.get_3D_Eunet()
  # model = DenseUNet_3D.get_3d_denseunet()
  # model = UNet_3D.get_3d_unet(opti)
  # model = RSUNet_3D.get_3d_rsunet(opti)
  model = RSUNet_3D_Gerda.get_3d_rsunet_Gerdafeature(opti)

  # Plot the model:
  modelname = 'model.png'
  plot_model(model, show_shapes=True, to_file=cm.workingPath.model_path + modelname)
  model.summary()

  # Should we load existing weights?
  if use_existing:
    model.load_weights(cm.workingPath.model_path + './unet.hdf5')

  print('-' * 30)
  print('Fitting model...')
  print('-' * 30)

  # Callbacks:
  filepath = cm.workingPath.model_path + 'weights.{epoch:02d}-{loss:.5f}.hdf5'
  bestfilepath = cm.workingPath.model_path + 'Best_weights.{epoch:02d}-{loss:.5f}.hdf5'

  model_checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False)
  model_best_checkpoint = callbacks.ModelCheckpoint(bestfilepath, monitor='val_loss', verbose=0, save_best_only=True)

  record_history = cb.RecordLossHistory()
  # model_history = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True,
  #  							embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata= None)

  callbacks_list = [record_history, lrate, model_best_checkpoint]

  model_info = model.fit_generator(BatchGenerator(originFile_list,
                                                  mask_list,
                                                  batch_size=1),  # BATCH_SIZE
                                   nb_epoch=4000,
                                   verbose=1,
                                   shuffle=False,
                                   validation_data=(x_val, y_val),
                                   callbacks=callbacks_list)

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