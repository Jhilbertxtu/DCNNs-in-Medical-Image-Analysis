

from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, AtrousConv2D, Dropout, Deconvolution2D
from keras.layers import Input, merge, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation
from keras.optimizers import Adam, Adadelta
import Modules.LossFunction as lf
import Modules.Common_modules as cm
import numpy as np

#######################################################
# Getting 3D U-net:


def get_3d_unet_bn():

  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(inputs)
  bn1 = BatchNormalization(axis=-1)(conv1)
  act1 = Activation('relu')(bn1)
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act1)
  bn1 = BatchNormalization(axis=-1)(conv1)
  act1 = Activation('relu')(bn1)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2))(act1)

  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(pool1)
  bn2 = BatchNormalization(axis=-1)(conv2)
  act2 = Activation('relu')(bn2)
  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act2)
  bn2 = BatchNormalization(axis=-1)(conv2)
  act2 = Activation('relu')(bn2)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2))(act2)

  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(pool2)
  bn3 = BatchNormalization(axis=-1)(conv3)
  act3 = Activation('relu')(bn3)
  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act3)
  bn3 = BatchNormalization(axis=-1)(conv3)
  act3 = Activation('relu')(bn3)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(act3)

  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(pool3)
  bn4 = BatchNormalization(axis=-1)(conv4)
  act4 = Activation('relu')(bn4)
  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act4)
  bn4 = BatchNormalization(axis=-1)(conv4)
  act4 = Activation('relu')(bn4)
  pool4 = MaxPooling3D(pool_size=(2, 2, 2))(act4)

  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(pool4)
  bn5 = BatchNormalization(axis=-1)(conv5)
  act5 = Activation('relu')(bn5)
  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act5)
  bn5 = BatchNormalization(axis=-1)(conv5)
  act5 = Activation('relu')(bn5)

  up6 = merge([UpSampling3D(size=(2, 2, 2))(act5), act4], mode='concat', concat_axis=-1)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(up6)
  bn6 = BatchNormalization(axis=-1)(conv6)
  act6 = Activation('relu')(bn6)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act6)
  bn6 = BatchNormalization(axis=-1)(conv6)
  act6 = Activation('relu')(bn6)

  up7 = merge([UpSampling3D(size=(2, 2, 2))(act6), act3], mode='concat', concat_axis=-1)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(up7)
  bn7 = BatchNormalization(axis=-1)(conv7)
  act7 = Activation('relu')(bn7)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act7)
  bn7 = BatchNormalization(axis=-1)(conv7)
  act7 = Activation('relu')(bn7)

  up8 = merge([UpSampling3D(size=(2, 2, 2))(act7), act2], mode='concat', concat_axis=-1)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(up8)
  bn8 = BatchNormalization(axis=-1)(conv8)
  act8 = Activation('relu')(bn8)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act8)
  bn8 = BatchNormalization(axis=-1)(conv8)
  act8 = Activation('relu')(bn8)

  up9 = merge([UpSampling3D(size=(2, 2, 2))(act8), act1], mode='concat', concat_axis=-1)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(up9)
  bn9 = BatchNormalization(axis=-1)(conv9)
  act9 = Activation('relu')(bn9)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), border_mode='same')(act9)
  bn9 = BatchNormalization(axis=-1)(conv9)
  act9 = Activation('relu')(bn9)

  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(act9)

  model = Model(input=inputs, output=conv10)

  model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model


def get_3d_unet(opti):

  inputs = Input((cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1))
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(inputs)
  conv1 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv1)
  pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool1)
  conv2 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv2)
  pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool2)
  conv3 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv3)
  pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool3)
  conv4 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv4)
  pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(pool4)
  conv5 = Conv3D(filters=512, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv5)

  up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv4], mode='concat', concat_axis=-1)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up6)
  conv6 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv6)

  up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv3], mode='concat', concat_axis=-1)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up7)
  conv7 = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv7)

  up8 = merge([UpSampling3D(size=(2, 2, 2))(conv7), conv2], mode='concat', concat_axis=-1)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up8)
  conv8 = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv8)

  up9 = merge([UpSampling3D(size=(2, 2, 2))(conv8), conv1], mode='concat', concat_axis=-1)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(up9)
  conv9 = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', border_mode='same')(conv9)

  conv10 = Conv3D(filters=3, kernel_size=(1, 1, 1), strides=(1, 1, 1), activation='sigmoid')(conv9)

  model = Model(input=inputs, output=conv10)

  weights = np.array([0.1, 10, 10])
  loss = lf.weighted_categorical_crossentropy_loss(weights)
  # model.compile(optimizer=Adam(lr=1.0e-5), loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  model.compile(optimizer=opti, loss=loss, metrics=["categorical_accuracy"])
  # model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=["categorical_accuracy"])

  return model