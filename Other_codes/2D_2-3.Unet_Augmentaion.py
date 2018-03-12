########################################################################################
# 2D Aorta Segmentation Project                                                        #
#                                                                                      #
# 2. U-net with Validation and Augmentation                                            #
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
# created on 23/12/2017                                                                #
# Last update: 25/12/2017                                                              #
########################################################################################

from __future__ import print_function

import Modules.Common_modules as cm
import Modules.Network as nw
import numpy as np
import datetime
from keras import callbacks
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import sys

img_rows = 512
img_cols = 512

smooth = 1

datagen = ImageDataGenerator(rotation_range=10)

def train_and_predict(use_existing):
  print('-' * 30)
  print('Loading and preprocessing train data...')
  print('-' * 30)

  # Choose which subset you would like to use:
  i = 0

  imgs_train = np.load(cm.workingPath.trainingSet_path + 'trainImages_%04d.npy' % (i)).astype(np.float32)
  imgs_mask_train = np.load(cm.workingPath.trainingSet_path + 'trainMasks_%04d.npy' % (i)).astype(np.float32)

  # imgs_val = np.load(cm.workingPath.validationSet_path + 'valImages_%04d.npy' % (i)).astype(np.float32)
  # imgs_mask_val = np.load(cm.workingPath.validationSet_path + 'valMasks_%04d.npy' % (i)).astype(np.float32)
  # imgs_test = np.load(cm.workingPath.trainingSet_path + 'testImages_%04d.npy'%(i)).astype(np.float32)
  # imgs_mask_test_true = np.load(cm.workingPath.trainingSet_path + 'testMasks_%04d.npy'%(i)).astype(np.float32)

  datagen.fit(imgs_train)

  # Mean for data centering:
  # mean= np.mean(imgs_train)
  # Std for data normalization:
  # std = np.std(imgs_train)

  # imgs_train -= mean
  # imgs_train /= std

  print('_' * 30)
  print('Creating and compiling model...')
  print('_' * 30)
  # model = nw.get_simple_unet()
  model = nw.get_unet()
  # model = nw.get_unet_less_feature()
  # model = nw.get_unet_dilated_conv_4()
  # model = nw.get_unet_dilated_conv_7()

  modelname = 'model.png'
  plot_model(model, show_shapes=False, to_file=cm.workingPath.model_path + modelname)
  model.summary()
  # config = model.get_config()
  # print(config)

  # Callbacks:

  filepath = cm.workingPath.model_path + 'weights.{epoch:02d}-{loss:.5f}.hdf5'
  bestfilepath = cm.workingPath.model_path + 'Best_weights.{epoch:02d}-{loss:.5f}.hdf5'

  model_checkpoint = callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False)
  model_best_checkpoint = callbacks.ModelCheckpoint(bestfilepath, monitor='loss', verbose=0, save_best_only=True)

  # history = cm.LossHistory_Gerda(cm.workingPath.working_path)
  history = nw.LossHistory()
  # model_history = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True,
  #  							embeddings_freq=1, embeddings_layer_names=None, embeddings_metadata= None)

  callbacks_list = [history, model_best_checkpoint]

  # Should we load existing weights?
  # Set argument for call to train_and_predict to true at end of script
  if use_existing:
    model.load_weights('./unet.hdf5')

  print('-' * 30)
  print('Fitting model...')
  print('-' * 30)

  # model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=400, verbose=1, shuffle=True,
  #           validation_data=(imgs_val, imgs_mask_val), callbacks=callbacks_list)

  model.fit_generator(datagen.flow(imgs_train, imgs_mask_train, batch_size=4),
                      steps_per_epoch=len(imgs_train) / 4, epochs=500, verbose=1, shuffle=True,
                      callbacks=callbacks_list)

  print('training finished')


# # loading best weights from training session:
# print('-' * 30)
# print('Loading saved weights...')
# print('-' * 30)
# model.load_weights('./unet.hdf5')
#
# print('-' * 30)
# print('Predicting masks on test data...')
# print('-' * 30)
#
# num_test = len(imgs_test)
# imgs_mask_test = np.ndarray([num_test,1,512,512], dtype=np.float32)
#
# for i in range(num_test):
# 	imgs_mask_test[i] = resize((model.predict([imgs_test[i:i + 1]], verbose=0)[0]), [1, 512, 512])
# np.save(workingPath.trainingSet_path + 'masksTestPredicted.npy', imgs_mask_test)
# mean = 0.0
#
#
# for i in range(num_test):
# 	mean += dice_coef_np(imgs_mask_test_true[i,0], imgs_mask_test[i,0])
# mean /= num_test
# print('Mean Dice Coeff', mean)

if __name__ == '__main__':
  # Choose whether to train based on the last model:
  # Show runtime:
  starttime = datetime.datetime.now()

  train_and_predict(False)

  endtime = datetime.datetime.now()
  print(endtime - starttime)


  sys.exit(0)