########################################################################################
# 3D Aorta Segmentation                                                                #
#                                                                                      #
# 3. Test Volume Aniso Scan Patch                                                    #
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
# created on 25/01/2018                                                                #
# Last update: 25/01/2018                                                              #
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
  preFile_list = sorted(glob(cm.workingPath.testingSet_path + 'mask*.dcm'))

  out_test_images = []
  out_test_masks = []
  out_pre_masks = []

  for i in range(len(originFile_list)):
    # originTestVolInfo = loadFileInformation(originFile_list[i])
    # maskTestVolInfo = loadFileInformation(maskFile_list[i])

    originTestVol, originTestVol_num, originTestVolwidth, originTestVolheight = loadFile(originFile_list[i])
    maskTestVol, maskTestVol_num, maskTestVolwidth, maskTestVolheight = loadFile(maskFile_list[i])
    preTestVol, preTestVol_num, preTestVolwidth, preTestVolheight = loadFile(preFile_list[i])

    for j in range(len(maskTestVol)):
      maskTestVol[j] = np.where(maskTestVol[j] != 0, 1, 0)
    for img in originTestVol:
      out_test_images.append(img)
    for img in maskTestVol:
      out_test_masks.append(img)
    for img in preTestVol:
      out_pre_masks.append(img)

  num_test_images = len(out_test_images)

  final_test_images = np.ndarray([num_test_images, 512, 512], dtype=np.int16)
  final_test_masks = np.ndarray([num_test_images, 512, 512], dtype=np.int8)
  final_pre_masks = np.ndarray([num_test_images, 512, 512], dtype=np.int8)

  for i in range(num_test_images):
    final_test_images[i] = out_test_images[i]
    final_test_masks[i] = out_test_masks[i]
    final_pre_masks[i] = out_pre_masks[i]

  final_test_images = np.expand_dims(final_test_images, axis=-1)
  final_test_masks = np.expand_dims(final_test_masks, axis=-1)
  final_pre_masks = np.expand_dims(final_pre_masks, axis=-1)


  for i in range(num_test_images):
    dice_coef_slice = nw.dice_coef_np(final_test_masks[i], final_pre_masks[i])
    total_progress = i / num_test_images * 100

    print('predicting slice: %03d' % (i), '  total progress: %5.2f%%' % (total_progress),
        '  dice coef: %5.3f' % (dice_coef_slice))


if __name__ == '__main__':
  # Choose whether to train based on the last model:
  model_test(True)
  endtime = datetime.datetime.now()
  print('-' * 30)
  print('running time:', endtime - starttime)

  log_file.close()
  sys.stdout = stdout_backup

  sys.exit(0)
