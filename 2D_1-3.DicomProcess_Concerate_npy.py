#
# Aorta Segmentation Project
#
# 1. Dicom Process Concerate npy
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
from glob import glob
import Modules.Common_modules as cm
import SimpleITK as sitk
import numpy as np
from PIL import Image
import dicom
import cv2

try:
  from tqdm import tqdm  # long waits are not fun
except:
  print('TQDM does make much nicer wait bars...')
  tqdm = lambda i: i


###########################################
########### Main Program Begin ############
###########################################

# Import dcm file and turn it into an image array:
# Get the file list:
originFile_list = sorted(glob(cm.workingPath.validationPatchesSet_path + 'img_*.npy'))
maskFile_list = sorted(glob(cm.workingPath.validationPatchesSet_path + 'mask_*.npy'))

# Load file:
out_images = []
out_masks = []
# out_test_images = []
# out_test_masks = []
axis_process = "Axial"  ## axis you want to process
# axis_process = "Sagittal"
# axis_process = "Coronal"

# Training set:
print('-' * 30)
print('Loading files...')
print('-' * 30)

if axis_process == "Axial":
  for i in range(len(originFile_list)):

    # Read information from dcm file:
    # originVolInfo = loadFileInformation(originFile_list[i])
    # maskVolInfo = loadFileInformation(maskFile_list[i])

    originVol= np.load(originFile_list[i])
    maskVol= np.load(maskFile_list[i])

    # Turn the mask images to binary images:
    for i in range(originVol.shape[0]):
      img = originVol[i, :, :, :]
      out_images.append(img)
    for i in range(maskVol.shape[0]):
      img = maskVol[i, :, :, :]
      out_masks.append(img)

  num_images = len(out_images)

  # Writing out images and masks as (1 channel) arrays for input into network
  final_images = np.ndarray([num_images, cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 1], dtype=np.int16)
  final_masks = np.ndarray([num_images, cm.slices_3d, cm.img_rows_3d, cm.img_cols_3d, 3], dtype=np.int8)
  for i in range(num_images):
    final_images[i] = out_images[i]
    final_masks[i] = out_masks[i]

  print('Saving Images...')
  print('-' * 30)

  np.save(cm.workingPath.validationSet_path + 'valImages.npy', final_images)
  np.save(cm.workingPath.validationSet_path + 'valMasks.npy', final_masks)

else:
  pass

print('Training Images Saved')
print('-' * 30)
print('Finished')
