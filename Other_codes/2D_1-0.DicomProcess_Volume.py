########################################################################################
# Aorta Segmentation Project                                                           #
#                                                                                      #
# 1. Dicom Process                                                                     #
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
# created on 19/10/2017                                                                #
# Last update: 12/11/2017                                                              #
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

###########################################
########### Main Program Begin ############
###########################################


# Import dcm file and turn it into an image array:

# Get the file list:
originFile_list = sorted(glob(cm.workingPath.originTrainingSet_path + 'vol*.dcm'))
maskFile_list = sorted(glob(cm.workingPath.maskTrainingSet_path + 'vol*.dcm'))

filelist = open(cm.workingPath.trainingSet_path + "file.txt", 'w')
filelist.write(str(len(originFile_list)))
filelist.write(" datasets involved")
filelist.write("\n")
for file in originFile_list:
  filelist.write(file)
  filelist.write('\n')
filelist.close()

# Load file:
out_images = []
out_masks = []



# Training set:
print('-' * 30)
print('Loading files...')
print('-' * 30)
for i in range(len(originFile_list)):


  originVol, originVol_num, originVolwidth, originVolheight = loadFile(originFile_list[i])
  maskVol, maskVol_num, maskVolwidth, maskVolheight = loadFile(maskFile_list[i])

  # Turn the mask images to binary images:
  for j in range(len(maskVol)):
    maskVol[j] = np.where(maskVol[j] != 0, 1, 0)
  for img in originVol:
    # new_img = resize(img, [512, 512])
    out_images.append(img)
  for img in maskVol:
    # new_mask = resize(img, [512, 512])
    out_masks.append(img)

num_images = len(out_images)
# num_test_images = len(out_test_images)

# Writing out images and masks as 1 channel arrays for input into network
final_images = np.ndarray([num_images, 512, 512], dtype=np.int16)
final_masks = np.ndarray([num_images, 512, 512], dtype=np.int8)
for i in range(num_images):
  final_images[i] = out_images[i]
  final_masks[i] = out_masks[i]

final_images = np.expand_dims(final_images, axis=-1)
final_masks = np.expand_dims(final_masks, axis=-1)


num_file = range(0, 1)

rand_i = np.random.choice(range(num_images), size=num_images, replace=False)

print('Saving Images...')
print('-' * 30)
for i in num_file:


  np.save(cm.workingPath.trainingSet_path + 'trainImages_%04d.npy' % (i), final_images[rand_i[:]])
  np.save(cm.workingPath.trainingSet_path + 'trainMasks_%04d.npy' % (i), final_masks[rand_i[:]])


print('Training Images Saved')
print('-' * 30)
print('Finished')
