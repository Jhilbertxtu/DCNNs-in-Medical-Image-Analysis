#
# 3D Aorta Segmentation Project
#
# 1. Dicom Process with large data set from hard disk
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

from glob import glob
import Modules.Common_modules as cm
import Modules.DataProcess as dp
from keras.utils import to_categorical
import numpy as np

try:
  from tqdm import tqdm  # long waits are not fun
except:
  print('TQDM does make much nicer wait bars...')
  tqdm = lambda i: i

# Import dcm file and turn it into an image array:

# Get the file list:
# originFile_list = sorted(glob(cm.workingPath.originTrainingSet_path + 'vol*.npy'))
# maskFile_list = sorted(glob(cm.workingPath.maskTrainingSet_path + 'vol*.npy'))
originFile_list = sorted(glob(cm.workingPath.originValidationSet_path + 'vol*.dcm'))
maskAortaFile_list = sorted(glob(cm.workingPath.aortaValidationSet_path + 'vol*.dcm'))
maskPulFile_list = sorted(glob(cm.workingPath.pulValidationSet_path + 'vol*.dcm'))

filename = str(originFile_list[0])[24:-4]

filelist = open(cm.workingPath.validationSet_path + "file.txt", 'w')
filelist.write(str(len(originFile_list)))
filelist.write(" datasets involved")
filelist.write("\n")
for file in originFile_list:
  filelist.write(file)
  filelist.write('\n')
filelist.close()

# Load file:

axis_process = "Axial"  ## axis you want to process
# axis_process = "Sagittal"
# axis_process = "Coronal"

# Training set:
print('-' * 30)
print('Loading files...')
print('-' * 30)

for nb_file in range(len(originFile_list)):

  out_images = []
  out_masks = []

  # Read information from dcm file:
  originVol, originVol_num, originVolwidth, originVolheight = dp.loadFile(originFile_list[nb_file])
  maskAortaVol, maskAortaVol_num, maskAortaVolwidth, maskAortaVolheight = dp.loadFile(maskAortaFile_list[nb_file])
  maskPulVol, maskPulVol_num, maskPulVolwidth, maskPulVolheight = dp.loadFile(maskPulFile_list[nb_file])
  maskVol = maskAortaVol

  # Turn the mask images to binary images:
  for j in range(len(maskAortaVol)):
    maskAortaVol[j] = np.where(maskAortaVol[j] != 0, 1, 0)
  for j in range(len(maskPulVol)):
    maskPulVol[j] = np.where(maskPulVol[j] != 0, 2, 0)

  maskVol = maskVol + maskPulVol

  for j in range(len(maskVol)):
    maskVol[j] = np.where(maskVol[j] > 2, 0, maskVol[j])
  for j in range(originVol.shape[0]):
    img = originVol[j, :, :]
    out_images.append(img)
  for j in range(maskVol.shape[0]):
    img = maskVol[j, :, :]
    out_masks.append(img)

  vol_cur_slices = originVol.shape[0]
  outmasks_onehot = to_categorical(out_masks, num_classes=3)
  final_images = np.ndarray([vol_cur_slices, originVol.shape[1], originVol.shape[2], 1], dtype=np.int16)
  final_masks = np.ndarray([vol_cur_slices, originVol.shape[1], originVol.shape[2], 3], dtype=np.int8)

  for j in range(len(out_images)):
    final_images[j, :, :, 0] = out_images[j]
    final_masks[j] = outmasks_onehot[j]

  row = cm.img_rows_3d
  col = cm.img_cols_3d
  num_rowes = 1
  num_coles = 1
  row_1 = int((512 - row) / 2)
  row_2 = int(512 - (512 - row) / 2)
  col_1 = int((512 - col) / 2)
  col_2 = int(512 - (512 - col) / 2)
  slices = cm.slices_3d
  gaps = cm.gaps_3d
  thi_1 = int((512 - slices) / 2)
  thi_2 = int(512 - (512 - slices) / 2)
  final_images_crop = final_images[:, row_1:row_2, col_1:col_2, :]
  final_masks_crop = final_masks[:, row_1:row_2, col_1:col_2, :]

  print('Images to files')
  print('-' * 30)

  print('Saving Images...%04d' % nb_file)
  print('-' * 30)

  np.save(cm.workingPath.validationPatchesSet_path + 'img_%04d.npy' %nb_file, final_images_crop)
  np.save(cm.workingPath.validationPatchesSet_path + 'mask_%04d.npy' %nb_file, final_masks_crop)

print('Training Images Saved')
print('-' * 30)

print("Finished")
