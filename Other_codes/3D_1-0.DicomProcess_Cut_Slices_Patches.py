########################################################################################
# 3D Aorta Segmentation Project                                                        #
#                                                                                      #
# 1. Dicom Process Cut Slices with 3D Patches                                          #
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
# created on 25/12/2017                                                                #
# Last update: 25/12/2017                                                              #
########################################################################################


from glob import glob
import Modules.Common_modules as cm
import Modules.Network as nw

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


# extract specific image array from dcm file:
def showImage(img_array, frame_num=0):
  img_bitmap = Image.fromarray(img_array[frame_num])
  return img_bitmap


# optimize image using Constrast Limit Adaptive Histogram Equalization (CLAHE):
def limitedEqualize(img_array, limit=2.0):
  img_array_list = []
  for img in img_array:
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
    img_array_list.append(clahe.apply(img))
  img_array_limited_equalized = np.array(img_array_list)
  return img_array_limited_equalized


# def writeVideo(img_array):
# 	frame_num, width, height = img_array.shape
# 	filename_output = filename.split('.')[0]+'.avi'
# 	video = cv2.VideoWriter(filename_output, -1, 16, (width, height))
# 	for img in img_array:
# 		video.write(img)
# 	video.release()

###########################################
########### Main Program Begin ############
###########################################


# Import dcm file and turn it into an image array:

# Get the file list:
originFile_list = sorted(glob(cm.workingPath.originTrainingSet_path + 'vol*.dcm'))
maskFile_list = sorted(glob(cm.workingPath.maskTrainingSet_path + 'vol*.dcm'))
# originTestFile_list = glob(cm.workingPath.originTestingSet_path + 'vol*.dcm')
# maskTestFile_list = glob(cm.workingPath.maskTestingSet_path + 'vol*.dcm')

# Load file:

# out_test_images = []
# out_test_masks = []
axis_process = "Axial"  ## axis you want to process
# axis_process = "Sagittal"
# axis_process = "Coronal"

# Training set:
print('-' * 30)
print('Loading files...')
print('-' * 30)

patch_slices = []
vol_slices = []
for i in range(len(originFile_list)):

  # Read information from dcm file:
  # originVolInfo = loadFileInformation(originFile_list[i])
  # maskVolInfo = loadFileInformation(maskFile_list[i])

  originVol, originVol_num, originVolwidth, originVolheight = loadFile(originFile_list[i])
  maskVol, maskVol_num, maskVolwidth, maskVolheight = loadFile(maskFile_list[i])

  # Turn the mask images to binary images:
  for j in range(len(maskVol)):
    maskVol[j] = np.where(maskVol[j] != 0, 1, 0)

  cut = -1
  atls = -1
  for img in maskVol:
    atls = atls + 1
    if 1 in img or (atls >= 150):  # want to save the upper part
      # if 1 in img:  # cut them all
      cut = cut + 1
    else:
      pass

  num_images_cut = cut + 1
  patch_slices.append(num_images_cut)
  vol_slices.append(originVol.shape[0])

final_images_cut_pre = []
final_masks_cut_pre = []

final_images_cut = np.ndarray([sum(patch_slices), 512, 512], dtype=np.int16)
final_masks_cut = np.ndarray([sum(patch_slices), 512, 512], dtype=np.int8)

for i in range(len(originFile_list)):
  out_images = []
  out_masks = []
  # Read information from dcm file:
  # originVolInfo = loadFileInformation(originFile_list[i])
  # maskVolInfo = loadFileInformation(maskFile_list[i])

  originVol, originVol_num, originVolwidth, originVolheight = loadFile(originFile_list[i])
  maskVol, maskVol_num, maskVolwidth, maskVolheight = loadFile(maskFile_list[i])

  # Turn the mask images to binary images:
  for j in range(len(maskVol)):
    maskVol[j] = np.where(maskVol[j] != 0, 1, 0)
  for i in range(originVol.shape[0]):
    img = originVol[i, :, :]
    # new_img = resize(img, [512, 512])
    out_images.append(img)
  for i in range(maskVol.shape[0]):
    # new_mask = resize(img, [512, 512])
    img = maskVol[i, :, :]
    out_masks.append(img)

  # Writing Cut Images:
  atls_1 = -1
  for mask in out_masks:
    atls_1 = atls_1 + 1
    if 1 in mask or (atls_1 >= 150):  # want to save the upper part
      # if 1 in mask:  # cut them all
      final_images_cut_pre.append(out_images[atls_1])
      final_masks_cut_pre.append(mask)
    else:
      pass

for i in range(len(final_images_cut_pre)):
  final_images_cut[i, :, :] = final_images_cut_pre[i]
  final_masks_cut[i, :, :] = final_masks_cut_pre[i]

final_images_cut = np.expand_dims(final_images_cut, axis=-1)
final_masks_cut = np.expand_dims(final_masks_cut, axis=-1)

num_file = range(0, len(patch_slices))

row = nw.img_rows_3d
col = nw.img_cols_3d
num_rowes = 3
num_coles = 3
row_1 = int((512 - row) / 2)
row_2 = int(512 - (512 - row) / 2)
col_1 = int((512 - col) / 2)
col_2 = int(512 - (512 - col) / 2)
slices = nw.slices_3d
final_images_cut_crop = final_images_cut[:, row_1:row_2, col_1:col_2, :]
final_masks_cut_crop = final_masks_cut[:, row_1:row_2, col_1:col_2, :]

num1 = 0

final_images_cut_crop_save = np.ndarray([num1, row, col, 1], dtype=np.int16)
final_masks_cut_crop_save = np.ndarray([num1, row, col, 1], dtype=np.int8)

print('Turning Images to Patches')
print('-' * 30)

count = 0
for num_vol in range(0, len(patch_slices)):
  num_patches = int(patch_slices[num_vol] / slices)
  if num_vol>0:
    count = count + patch_slices[num_vol-1]
  else:
    pass
  for num_patch in range(0, num_patches):
    for num_row in range(0, num_rowes):
      for num_col in range(0, num_coles):
        count1 = count + num_patch*slices
        count2 = count + (num_patch+1)*slices

        row_count1 = num_row * row
        row_count2 = (num_row * row + row)

        col_count1 = num_col * col
        col_count2 = (num_col * col + col)

        final_images_cut_crop_save = np.concatenate(
          (final_images_cut_crop_save, final_images_cut_crop[count1:count2,
                                       row_count1:row_count2, col_count1:col_count2, :]), axis=0)

        final_masks_cut_crop_save = np.concatenate(
          (final_masks_cut_crop_save, final_masks_cut_crop[count1:count2,
                                      row_count1:row_count2, col_count1:col_count2, :]), axis=0)


tubes = int(final_images_cut_crop_save.shape[0]/slices)
final_images_cut_crop_save_last = np.ndarray([tubes, slices, row, col, 1], dtype=np.int16)
final_masks_cut_crop_save_last = np.ndarray([tubes, slices, row, col, 1], dtype=np.int8)


for i in range(tubes):
  final_images_cut_crop_save_last[i,:,:,:,:] = final_images_cut_crop_save[i*slices:(i+1)*slices, :,:,:]
  final_masks_cut_crop_save_last[i,:,:,:,:] = final_masks_cut_crop_save[i*slices:(i+1)*slices, :,:,:]



print('Saving Images...')
print('-' * 30)

i = 0
np.save(cm.workingPath.training3DSet_path + 'trainImages_%04d.npy' % (i), final_images_cut_crop_save_last)
np.save(cm.workingPath.training3DSet_path + 'trainMasks_%04d.npy' % (i), final_masks_cut_crop_save_last)

print('Training Images Saved')
print('-' * 30)

print("Finished")