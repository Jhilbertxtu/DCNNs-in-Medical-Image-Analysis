########################################################################################
# 2D Aorta Segmentation                                                                #
#                                                                                      #
# 4. Test Single Image Aniso                                                           #
#                                                                                      #
# created by                                                                           #
# Shuai Chen                                                                           #
# PhD student                                                                          #
# Radiology and Medical Informatics                                                    #
#                                                                                      #
# P.O. Box 2040, 3000 CA Rotterdam, The Netherlands, internal postal address Na-2603   #
# Visiting address: office 2616, Wytemaweg 80, 3015 CN Rotterdam, The Netherlands      #
# Email s.chen.2@erasmusmc.nl                                                          #
# www.erasmusmc.nl                                                                     #
#                                                                                      #
# created on 12/11/2017                                                                #
# Last update: 15/12/2017                                                              #
########################################################################################

from __future__ import print_function

import Modules.Common_modules as cm
import Modules.Network as nw
import datetime
import numpy as np
import keras.losses
from keras.optimizers import SGD, Adam
import dicom
import re
import SimpleITK as sitk
from glob import glob
from skimage.transform import resize
from skimage import morphology
import matplotlib.pyplot as plt
import logging
import sys

stdout_backup = sys.stdout
log_file = open(cm.workingPath.testingSet_path + "logs.txt", "w")
sys.stdout = log_file

# Show runtime:
starttime = datetime.datetime.now()


# load dcm file:
def loadFile(filename):
  ds = sitk.ReadImage(filename)
  img_array = sitk.GetArrayFromImage(ds)
  frame_num, width, height = img_array.shape
  return img_array, frame_num, width, height


# load dcm file imformation:
def loadFileInformation(filename):
  information = {}
  ds = dicom.read_file(filename)
  information['PatientID'] = ds.PatientID
  information['PatientName'] = ds.PatientName
  information['PatientBirthDate'] = ds.PatientBirthDate
  information['PatientSex'] = ds.PatientSex
  information['StudyID'] = ds.StudyID
  # information['StudyTime'] = ds.Studytime
  information['InstitutionName'] = ds.InstitutionName
  information['Manufacturer'] = ds.Manufacturer
  information['NumberOfFrames'] = ds.NumberOfFrames
  return information


def model_test(use_existing):
  print('-' * 30)
  print('Loading test data...')
  print('-' * 30)

  # Loading test data:
  filename = cm.filename
  modelname = cm.modellist[0]
  originFile_list = sorted(glob(cm.workingPath.originTestingSet_path + filename))
  maskFile_list = sorted(glob(cm.workingPath.maskTestingSet_path + filename))
  file = str(originFile_list[0])[23:]

  out_test_images = []
  out_test_masks = []
  for i in range(len(originFile_list)):
    # originTestVolInfo = loadFileInformation(originFile_list[i])
    # maskTestVolInfo = loadFileInformation(maskFile_list[i])

    originTestVol, originTestVol_num, originTestVolwidth, originTestVolheight = loadFile(originFile_list[i])
    maskTestVol, maskTestVol_num, maskTestVolwidth, maskTestVolheight = loadFile(maskFile_list[i])

    for j in range(len(maskTestVol)):
      maskTestVol[j] = np.where(maskTestVol[j] != 0, 1, 0)
    for img in originTestVol:
      # new_img = resize(img, [512, 512])
      out_test_images.append(img)
    for img in maskTestVol:
      # new_mask = resize(img, [512, 512])
      out_test_masks.append(img)

  num_test_images = len(out_test_images)
  # num_test_images = 1
  # test_num = 241

  final_test_images = np.ndarray([num_test_images, 512, 512], dtype=np.int16)
  final_test_masks = np.ndarray([num_test_images, 512, 512], dtype=np.int8)
  # import pdb; pdb.set_trace()
  for i in range(num_test_images):
    final_test_images[i] = out_test_images[i]
    final_test_masks[i] = out_test_masks[i]
  final_test_images = np.expand_dims(final_test_images, axis=-1)
  final_test_masks = np.expand_dims(final_test_masks, axis=-1)


  learning_rate = 0.0001
  decay_rate = 5e-6
  momentum = 0.9

  # sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
  adam = Adam(lr=learning_rate, decay=decay_rate)

  opti = adam
  print('_' * 30)
  print('calculating model...')
  print('_' * 30)
  # model = nw.get_unet_more_feature()
  # model = nw.get_unet_less_feature()
  # model = nw.get_unet_dilated_conv_4()
  model = nw.get_unet(opti)
  # model = nw.get_shallow_unet()
  # model = nw.get_dropout_unet()

  using_start_end = 0
  start_slice = 121
  end_slice = 283

  if use_existing:
    model.load_weights(modelname)

  imgs_mask_test = np.ndarray([num_test_images, 512, 512, 1], dtype=np.float32)

  # logs = []
  for i in range(num_test_images):
    imgs_mask_test[i] = resize((model.predict([final_test_images[i:i + 1]], verbose=0)[0]), [512, 512, 1])

    # imgs_mask_test[i] = np.where(imgs_mask_test[i] < 0.5, 0, 1)

    # imgs_mask_test[i] = morphology.erosion(imgs_mask_test[i], np.ones([2, 2, 1]))
    # imgs_mask_test[i] = morphology.dilation(imgs_mask_test[i], np.ones([5, 5, 1]))

    dice_coef_slice = nw.dice_coef_np(final_test_masks[i], imgs_mask_test[i])
    total_progress = i / num_test_images * 100

    print('predicting slice: %03d' % (i), '  total progress: %5.2f%%' % (total_progress),
          '  dice coef: %5.3f' % (dice_coef_slice))

  final_test_images = final_test_images + 4000

  np.save(cm.workingPath.testingSet_path + 'testImages.npy', final_test_images)
  np.save(cm.workingPath.testingSet_path + 'testMasks.npy', final_test_masks)

  np.save(cm.workingPath.testingSet_path + 'masksTestPredicted.npy', imgs_mask_test)

  # Load data:

  imgs_origin = np.load(cm.workingPath.testingSet_path + 'testImages.npy').astype(np.int16)
  imgs_true = np.load(cm.workingPath.testingSet_path + 'testMasks.npy').astype(np.int8)
  imgs_predict = np.load(cm.workingPath.testingSet_path + 'masksTestPredicted.npy').astype(np.float32)
  imgs_predict_threshold = np.load(cm.workingPath.testingSet_path + 'masksTestPredicted.npy').astype(np.float32)

  imgs_origin = np.squeeze(imgs_origin, axis=-1)
  imgs_true = np.squeeze(imgs_true, axis=-1)
  imgs_predict = np.squeeze(imgs_predict, axis=-1)
  imgs_predict_threshold = np.squeeze(imgs_predict_threshold, axis=-1)

  imgs_predict_threshold = np.where(imgs_predict_threshold < 0.5, 0, 1)

  if using_start_end == 1:
    mean = nw.dice_coef_np(imgs_predict[start_slice:end_slice], imgs_true[start_slice:end_slice])
  else:
    mean = nw.dice_coef_np(imgs_predict, imgs_true)

  np.savetxt(cm.workingPath.testingSet_path + 'dicemean.txt', np.array(mean).reshape(1, ), fmt='%.5f')
  print('-' * 30)
  print('Model file:', modelname)
  print('Total Dice Coeff', mean)
  print('-' * 30)

  # Draw the subplots of figures:

  color1 = 'gray'  # ***
  color2 = 'viridis'  # ******
  # color = 'plasma'  # **
  # color = 'magma'  # ***
  # color2 = 'RdPu'  # ***
  # color = 'gray'  # ***
  # color = 'gray'  # ***

  transparent1 = 1.0
  transparent2 = 0.5

  # Slice parameters:

  #############################################
  # Automatically:

  steps = 40
  slice = range(0, len(imgs_true), steps)
  plt_row = 3
  plt_col = int(len(imgs_true) / steps)

  plt.figure(1, figsize=(25, 12))

  for i in slice:
    if i == 0:
      plt_num = int(i / steps) + 1
    else:
      plt_num = int(i / steps)

    if plt_num <= plt_col:

      plt.figure(1)

      ax1 = plt.subplot(plt_row, plt_col, plt_num)
      title = 'slice=' + str(i)
      plt.title(title)
      ax1.imshow(imgs_origin[i, :, :], cmap=color1, alpha=transparent1)
      ax1.imshow(imgs_true[i, :, :], cmap=color2, alpha=transparent2)

      ax2 = plt.subplot(plt_row, plt_col, plt_num + plt_col)
      title = 'slice=' + str(i)
      plt.title(title)
      ax2.imshow(imgs_origin[i, :, :], cmap=color1, alpha=transparent1)
      ax2.imshow(imgs_predict[i, :, :], cmap=color2, alpha=transparent2)

      ax3 = plt.subplot(plt_row, plt_col, plt_num + 2 * plt_col)
      title = 'slice=' + str(i)
      plt.title(title)
      ax3.imshow(imgs_origin[i, :, :], cmap=color1, alpha=transparent1)
      ax3.imshow(imgs_predict_threshold[i, :, :], cmap=color2, alpha=transparent2)
    else:
      pass

  modelname = cm.modellist[0]

  imageName = re.findall(r'\d+\.?\d*', modelname)
  epoch_num = int(imageName[0]) + 1
  accuracy = float(np.loadtxt(cm.workingPath.testingSet_path + 'dicemean.txt', float))

  # saveName = 'epoch_' + str(epoch_num) + '_dice_' +str(accuracy) + '.png'
  saveName = 'epoch_%02d_dice_%.3f.png' % (epoch_num-1, accuracy)

  plt.subplots_adjust(left=0.0, bottom=0.05, right=1.0, top=0.95, hspace=0.3, wspace=0.3)
  plt.savefig(cm.workingPath.testingSet_path + saveName)
  # plt.show()

  ###################################
  # Manually:
  #
  # slice = (100,150,230,250)
  #
  # plt.figure(2)
  #
  # ax1 = plt.subplot(2,4,1)
  # title = 'slice=' + str(slice[0])
  # plt.title(title)
  # ax1.imshow(new_imgs_origin[slice[0],0,:,:],cmap=color1,alpha=transparent1)
  # ax1.imshow(new_imgs_true[slice[0],0,:,:],cmap=color2,alpha=transparent2)
  #
  # ax2 = plt.subplot(2,4,5)
  # title = 'slice=' + str(slice[0])
  # plt.title(title)
  # ax2.imshow(new_imgs_origin[slice[0],0,:,:],cmap=color1,alpha=transparent1)
  # ax2.imshow(new_imgs_predict[slice[0],0,:,:],cmap=color2,alpha=transparent2)
  #
  # ax3 = plt.subplot(2,4,2)
  # title = 'slice=' + str(slice[1])
  # plt.title(title)
  # ax3.imshow(new_imgs_origin[slice[1],0,:,:],cmap=color1,alpha=transparent1)
  # ax3.imshow(new_imgs_true[slice[1],0,:,:],cmap=color2,alpha=transparent2)
  #
  # ax4 = plt.subplot(2,4,6)
  # title = 'slice=' + str(slice[1])
  # plt.title(title)
  # ax4.imshow(new_imgs_origin[slice[1],0,:,:],cmap=color1,alpha=transparent1)
  # ax4.imshow(new_imgs_predict[slice[1],0,:,:],cmap=color2,alpha=transparent2)
  #
  # ax5 = plt.subplot(2,4,3)
  # title = 'slice=' + str(slice[2])
  # plt.title(title)
  # ax5.imshow(new_imgs_origin[slice[2],0,:,:],cmap=color1,alpha=transparent1)
  # ax5.imshow(new_imgs_true[slice[2],0,:,:],cmap=color2,alpha=transparent2)
  #
  # ax6 = plt.subplot(2,4,7)
  # title = 'slice=' + str(slice[2])
  # plt.title(title)
  # ax6.imshow(new_imgs_origin[slice[2],0,:,:],cmap=color1,alpha=transparent1)
  # ax6.imshow(new_imgs_predict[slice[2],0,:,:],cmap=color2,alpha=transparent2)
  #
  # ax7 = plt.subplot(2,4,4)
  # title = 'slice=' + str(slice[3])
  # plt.title(title)
  # ax7.imshow(new_imgs_origin[slice[3],0,:,:],cmap=color1,alpha=transparent1)
  # ax7.imshow(new_imgs_true[slice[3],0,:,:],cmap=color2,alpha=transparent2)
  #
  # ax8 = plt.subplot(2,4,8)
  # title = 'slice=' + str(slice[3])
  # plt.title(title)
  # ax8.imshow(new_imgs_origin[slice[3],0,:,:],cmap=color1,alpha=transparent1)
  # ax8.imshow(new_imgs_predict[slice[3],0,:,:],cmap=color2,alpha=transparent2)
  #
  # plt.show()

  #############################################

  print('Images saved')

  # Save npy as dcm files:

  imgs_origin = np.uint16(imgs_origin)
  imgs_true = np.uint16(imgs_true)
  imgs_predict_threshold = np.uint16(imgs_predict_threshold)

  new_imgs_origin_dcm = sitk.GetImageFromArray(imgs_origin)
  new_imgs_true_dcm = sitk.GetImageFromArray(imgs_true)
  new_imgs_predict_dcm = sitk.GetImageFromArray(imgs_predict_threshold)

  sitk.WriteImage(new_imgs_predict_dcm, cm.workingPath.testingSet_path + "Predicted_" + file)

  ds1 = dicom.read_file(maskFile_list[0])
  ds2 = dicom.read_file(cm.workingPath.testingSet_path + "Predicted_" + file)
  ds1.PixelData = ds2.PixelData
  ds1.save_as(cm.workingPath.testingSet_path + "Predicted_" + file)

  print('DICOM saved')


if __name__ == '__main__':
  # Choose whether to train based on the last model:
  model_test(True)
  endtime = datetime.datetime.now()
  print('-' * 30)
  print('running time:', endtime - starttime)

  log_file.close()
  sys.stdout = stdout_backup

  sys.exit(0)
