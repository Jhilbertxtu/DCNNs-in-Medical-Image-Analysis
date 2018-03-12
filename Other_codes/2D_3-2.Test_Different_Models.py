########################################################################################
# Aorta Segmentation Project                                                           #
#                                                                                      #
# 4. Test Single Image                                                                 #
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
# created on 12/11/2017                                                                #
# Last update: 12/11/2017                                                              #
########################################################################################

from __future__ import print_function

import Modules.Common_modules as cm
import Modules.Network as nw
import datetime
import numpy as np
import keras.losses
import dicom
import re
import SimpleITK as sitk
from glob import glob
from skimage.transform import resize
import matplotlib.pyplot as plt

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


def model_test(use_existing, ii):
  print('-' * 30)
  print('Loading test data...')
  print('-' * 30)

  # Loading test data:
  filename = cm.filename
  modelname = cm.modellist[ii]
  originFile_list = glob(cm.workingPath.originTestingSet_path + filename)
  maskFile_list = glob(cm.workingPath.maskTestingSet_path + filename)

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

  # num_test_images = len(out_test_images)
  num_test_images = 1
  test_num = 140

  final_test_images = np.ndarray([num_test_images, 512, 512], dtype=np.float32)
  final_test_masks = np.ndarray([num_test_images, 512, 512], dtype=np.float32)
  # import pdb; pdb.set_trace()
  final_test_images[0] = out_test_images[test_num]
  final_test_masks[0] = out_test_masks[test_num]
  final_test_images = np.expand_dims(final_test_images, axis=-1)
  final_test_masks = np.expand_dims(final_test_masks, axis=-1)

  print('_' * 30)
  print('calculating model...')
  print('_' * 30)
  model = nw.get_unet()

  if use_existing:
    model.load_weights(modelname)

  imgs_mask_test = np.ndarray([num_test_images, 512, 512, 1], dtype=np.float32)

  imgs_mask_test[0] = resize((model.predict([final_test_images[0:1]], verbose=0)[0]), [512, 512, 1])

  np.save(cm.workingPath.testingSet_path + 'testImages.npy', final_test_images)
  np.save(cm.workingPath.testingSet_path + 'testMasks.npy', final_test_masks)

  np.save(cm.workingPath.testingSet_path + 'masksTestPredicted.npy', imgs_mask_test)

  # Save npy as dcm files:
  imgs_origin = np.load(cm.workingPath.testingSet_path + 'testImages.npy').astype(np.uint16)

  imgs_true = np.load(cm.workingPath.testingSet_path + 'testMasks.npy').astype(np.uint16)
  imgs_predict = np.load(cm.workingPath.testingSet_path + 'masksTestPredicted.npy').astype(np.uint16)

  # for j in range(len(imgs_true)):
  # 	imgs_true[j] = np.where(imgs_true[j]  < 0.5, 0, 1)
  # for j in range(len(imgs_predict)):
  # 	imgs_predict[j] = np.where(imgs_predict[j]  < 0.5, 0, 1)

  imgs_origin = np.squeeze(imgs_origin, axis=-1)
  imgs_true = np.squeeze(imgs_true, axis=-1)
  imgs_predict = np.squeeze(imgs_predict, axis=-1)

  new_image_origin = sitk.GetImageFromArray(imgs_origin)
  new_image_true = sitk.GetImageFromArray(imgs_true)
  new_image_predict = sitk.GetImageFromArray(imgs_predict)

  # sitk.WriteImage(new_image_origin, cm.workingPath.testingSet_path + 'testImages.dcm')
  # sitk.WriteImage(new_image_true, cm.workingPath.testingSet_path + 'testMasks.dcm')
  # sitk.WriteImage(new_image_predict, cm.workingPath.testingSet_path + 'masksTestPredicted.dcm')

  mean = 0.0

  for i in range(num_test_images):
    mean += nw.dice_coef_np(final_test_masks[i], imgs_mask_test[i])
  mean /= num_test_images
  np.savetxt(cm.workingPath.testingSet_path + 'dicemean.txt', np.array(mean).reshape(1, ), fmt='%.3f')

  print('model file:', modelname)
  print('Mean Dice Coeff', mean)

  # Load data:

  imgs_origin = np.load(cm.workingPath.testingSet_path + 'testImages.npy').astype(np.float32)

  imgs_true = np.load(cm.workingPath.testingSet_path + 'testMasks.npy').astype(np.float32)
  imgs_predict = np.load(cm.workingPath.testingSet_path + 'masksTestPredicted.npy').astype(np.float32)

  # Turn images into binary images from (0,1):

  for i in range(len(imgs_true)):
    imgs_true[i] = np.where(imgs_true[i] < 0.5, 0, 1)

  for j in range(len(imgs_predict)):
    imgs_predict[j] = np.where(imgs_predict[j] < 0.5, 0, 1)

  # Prepare to do some operations on images, or not:
  new_imgs_origin = imgs_origin
  new_imgs_true = imgs_true
  new_imgs_predict = imgs_predict

  # for i in range(len(imgs_true)):
  # 	new_imgs_true[len(imgs_true)-i-1] = imgs_true[i]
  #
  # for i in range(len(imgs_predict)):
  # 	new_imgs_predict[len(imgs_predict)-i-1] = imgs_predict[i]

  # Draw the subplots of figures:

  color1 = 'gray'  # ***
  color2 = 'viridis'  # ******
  # color = 'plasma'  # **
  # color = 'magma'  # ***
  # color2 = 'RdPu'  # ***
  # color = 'gray'  # ***
  # color = 'gray'  # ***
  # color = 'gray'  # ***
  # color = 'gray'  # ***
  # color = 'gray'  # ***
  # color = 'gray'  # ***
  # color = 'gray'  # ***
  # color = 'gray'  # ***
  # color = 'gray'  # ***

  transparent1 = 1.0
  transparent2 = 0.5

  # Slice parameters:

  #############################################
  # Automatically:

  steps = 1
  slice = range(0, len(new_imgs_true), steps)
  plt_row = 2
  plt_col = int(len(new_imgs_true) / steps)

  plt.figure(1, figsize=(20, 10))

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
      ax1.imshow(new_imgs_origin[i, :, :, 0], cmap=color1, alpha=transparent1)
      ax1.imshow(new_imgs_true[i, :, :, 0], cmap=color2, alpha=transparent2)

      ax2 = plt.subplot(plt_row, plt_col, plt_num + plt_col)
      title = 'slice=' + str(i)
      plt.title(title)
      ax2.imshow(new_imgs_origin[i, :, :, 0], cmap=color1, alpha=transparent1)
      ax2.imshow(new_imgs_predict[i, :, :, 0], cmap=color2, alpha=transparent2)
    else:
      pass

  modelname = cm.modellist[ii]

  imageName = re.findall(r'\d+\.?\d*', modelname)
  epoch_num = int(imageName[0]) + 1
  accuracy = float(np.loadtxt(cm.workingPath.testingSet_path + 'dicemean.txt', float))

  # saveName = 'epoch_' + str(epoch_num) + '_dice_' +str(accuracy) + '.png'
  saveName = 'epoch_%02d_dice_%.3f.png' % (epoch_num, accuracy)

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

  print('Images showing')


if __name__ == '__main__':
  # Choose whether to train based on the last model:
  for i in range(1, 78):
    model_test(True, i)
  endtime = datetime.datetime.now()
  print(endtime - starttime)
