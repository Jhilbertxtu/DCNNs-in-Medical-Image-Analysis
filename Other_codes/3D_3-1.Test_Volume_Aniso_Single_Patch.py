########################################################################################
# 3D Aorta Segmentation                                                                #
#                                                                                      #
# 3. Test Volume Aniso Single Patch                                                    #
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
# created on 25/12/2017                                                                #
# Last update: 28/12/2017                                                              #
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
      out_test_images.append(img)
    for img in maskTestVol:
      out_test_masks.append(img)

  num_test_images = len(out_test_images)

  final_test_images = np.ndarray([num_test_images, 512, 512], dtype=np.int16)
  final_test_masks = np.ndarray([num_test_images, 512, 512], dtype=np.int8)

  # import pdb; pdb.set_trace()
  for i in range(num_test_images):
    final_test_images[i] = out_test_images[i]
    final_test_masks[i] = out_test_masks[i]
  final_test_images = np.expand_dims(final_test_images, axis=-1)
  final_test_masks = np.expand_dims(final_test_masks, axis=-1)

  row = nw.img_rows_3d
  col = nw.img_cols_3d
  slices = nw.slices_3d
  row_1 = int((512-row)/2)
  row_2 = int(512- (512-row)/2)
  col_1 = int((512-col)/2)
  col_2 = int(512- (512-col)/2)
  final_test_images_crop = final_test_images[:, row_1:row_2, col_1:col_2, :]
  final_test_masks_crop = final_test_masks[:, row_1:row_2, col_1:col_2, :]

  imgs_image_test = np.ndarray([0, row, col, 1], dtype=np.int16)
  imgs_mask_test = np.ndarray([0, row, col, 1], dtype=np.float32)

  num_patches = int(final_test_images_crop.shape[0] / slices)
  for num_patch in range(0, num_patches):
    num_rows = 1
    for num_row in range(0, num_rows):
      num_coles = 1
      for num_col in range(0, num_coles):
        imgs_image_test = np.concatenate(
          (imgs_image_test, final_test_images_crop[num_patch * slices:(num_patch * slices + slices),
                            num_row * row:(num_row * row + row), num_col * col:(num_col * col + col), :]), axis=0)

        imgs_mask_test = np.concatenate(
          (imgs_mask_test, final_test_masks_crop[num_patch * slices:(num_patch * slices + slices),
                           num_row * row:(num_row * row + row), num_col * col:(num_col * col + col), :]), axis=0)

  tubes = int(imgs_image_test.shape[0] / slices)
  imgs_image_test_last = np.ndarray([tubes, slices, row, col, 1], dtype=np.int16)
  imgs_mask_test_last = np.ndarray([tubes, slices, row, col, 1], dtype=np.int8)
  imgs_predict_last = np.ndarray([tubes, slices, row, col, 1], dtype=np.float32)

  for i in range(tubes):
    imgs_image_test_last[i, :, :, :, :] = imgs_image_test[i * slices:(i + 1) * slices, :, :, :]
    imgs_mask_test_last[i, :, :, :, :] = imgs_mask_test[i * slices:(i + 1) * slices, :, :, :]

  print('_' * 30)
  print('calculating model...')
  print('_' * 30)

  model = nw.get_3D_unet()
  # model = nw.get_less_feature_3D_unet()


  using_start_end = 1
  start_slice = cm.start_slice
  end_slice = -1


  if use_existing:
    model.load_weights(modelname)

  # logs = []
  for i in range(imgs_image_test_last.shape[0]):
    imgs_predict_last[i] = model.predict([imgs_image_test_last[i:i + 1]])

    # imgs_mask_test[i] = np.where(imgs_mask_test[i] < 0.5, 0, 1)

    # imgs_mask_test[i] = morphology.erosion(imgs_mask_test[i], np.ones([2, 2, 1]))
    # imgs_mask_test[i] = morphology.dilation(imgs_mask_test[i], np.ones([5, 5, 1]))

    dice_coef_slice = nw.dice_coef_np(imgs_predict_last[i], imgs_mask_test_last[i])
    total_progress = i / imgs_image_test_last.shape[0] * 100

    print('predicting slice: %03d' % (i), '  total progress: %5.2f%%' % (total_progress),
          '  dice coef: %5.3f' % (dice_coef_slice))

  imgs_image_test_last = imgs_image_test_last + 4000

  np.save(cm.workingPath.testingSet_path + 'testImages.npy', imgs_image_test_last)
  np.save(cm.workingPath.testingSet_path + 'testMasks.npy', imgs_mask_test_last)

  np.save(cm.workingPath.testingSet_path + 'masksTestPredicted.npy', imgs_predict_last)

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


  imgs_image_sized = np.ndarray([num_patches * slices, row, col], dtype=np.int16)
  imgs_mask_sized = np.ndarray([num_patches * slices, row, col], dtype=np.int8)
  imgs_predict_sized = np.ndarray([num_patches * slices, row, col], dtype=np.float32)
  imgs_predict_threshold_sized = np.ndarray([num_patches * slices, row, col], dtype=np.int8)

  num_patches = int(final_test_images_crop.shape[0] / slices)
  for num_patch in range(0, num_patches):
    num_rows = 1
    for num_row in range(0, num_rows):
      num_coles = 1
      for num_col in range(0, num_coles):
        count = num_patch * num_rows * num_coles + num_row * num_coles + num_col
        imgs_image_sized[num_patch * slices:(num_patch + 1) * slices, num_row * row:(num_row + 1) * row,
        num_col * col:(num_col + 1) * col] = imgs_origin[count, :, :, :]
        imgs_mask_sized[num_patch * slices:(num_patch + 1) * slices, num_row * row:(num_row + 1) * row,
        num_col * col:(num_col + 1) * col] = imgs_true[count, :, :, :]
        imgs_predict_sized[num_patch * slices:(num_patch + 1) * slices, num_row * row:(num_row + 1) * row,
        num_col * col:(num_col + 1) * col] = imgs_predict[count, :, :, :]
        imgs_predict_threshold_sized[num_patch * slices:(num_patch + 1) * slices, num_row * row:(num_row + 1) * row,
        num_col * col:(num_col + 1) * col] = imgs_predict_threshold[count, :, :, :]

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
  slice = range(0, len(imgs_mask_sized), steps)
  plt_row = 3
  plt_col = int(len(imgs_mask_sized) / steps)

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
      ax1.imshow(imgs_image_sized[i, :, :], cmap=color1, alpha=transparent1)
      ax1.imshow(imgs_mask_sized[i, :, :], cmap=color2, alpha=transparent2)

      ax2 = plt.subplot(plt_row, plt_col, plt_num + plt_col)
      title = 'slice=' + str(i)
      plt.title(title)
      ax2.imshow(imgs_image_sized[i, :, :], cmap=color1, alpha=transparent1)
      ax2.imshow(imgs_predict_sized[i, :, :], cmap=color2, alpha=transparent2)

      ax3 = plt.subplot(plt_row, plt_col, plt_num + 2 * plt_col)
      title = 'slice=' + str(i)
      plt.title(title)
      ax3.imshow(imgs_image_sized[i, :, :], cmap=color1, alpha=transparent1)
      ax3.imshow(imgs_predict_threshold_sized[i, :, :], cmap=color2, alpha=transparent2)
    else:
      pass

  modelname = cm.modellist[0]

  imageName = re.findall(r'\d+\.?\d*', modelname)
  epoch_num = int(imageName[0]) + 1


  if using_start_end == 1:
    mean = nw.dice_coef_np(imgs_predict_threshold_sized[start_slice:end_slice], imgs_mask_sized[start_slice:end_slice])
  else:
    mean = nw.dice_coef_np(imgs_predict_threshold_sized, imgs_true)

  np.savetxt(cm.workingPath.testingSet_path + 'dicemean.txt', np.array(mean).reshape(1, ), fmt='%.5f')
  print('-' * 30)
  print('Model file:', modelname)
  print('Total Dice Coeff', mean)
  print('-' * 30)



  accuracy = float(np.loadtxt(cm.workingPath.testingSet_path + 'dicemean.txt', float))

  # saveName = 'epoch_' + str(epoch_num) + '_dice_' +str(accuracy) + '.png'
  saveName = 'epoch_%02d_dice_%.3f.png' % (epoch_num, accuracy)

  plt.subplots_adjust(left=0.0, bottom=0.05, right=1.0, top=0.95, hspace=0.3, wspace=0.3)
  plt.savefig(cm.workingPath.testingSet_path + saveName)
  # plt.show()

  print('Images saved')

  # Save npy as dcm files:

  # final_test_predicted_threshold = np.ndarray([num_test_images, 512, 512], dtype=np.int8)

  final_test_images = np.squeeze(final_test_images + 4000, axis=-1)
  final_test_masks = np.squeeze(final_test_masks, axis=-1)

  final_test_images[0:num_patches * slices, row_1:row_2, col_1:col_2,] = imgs_image_sized[:, :, :]
  final_test_masks[0:num_patches * slices:, row_1:row_2, col_1:col_2,] = imgs_mask_sized[:, :, :]
  final_test_predicted_threshold = final_test_masks
  final_test_predicted_threshold[0:num_patches * slices, row_1:row_2, col_1:col_2,] = imgs_predict_threshold_sized[:, :, :]

  final_test_predicted_threshold = np.uint16(final_test_predicted_threshold)

  new_imgs_predict_dcm = sitk.GetImageFromArray(final_test_predicted_threshold)

  sitk.WriteImage(new_imgs_predict_dcm, cm.workingPath.testingSet_path + 'masksTestPredicted.dcm')

  ds1 = dicom.read_file(maskFile_list[0])
  ds2 = dicom.read_file(cm.workingPath.testingSet_path + 'masksTestPredicted.dcm')
  ds1.PixelData = ds2.PixelData
  ds1.save_as(cm.workingPath.testingSet_path + 'masksTestPredicted.dcm')

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
