#
# Aorta Segmentation Project
#
# 1. Dicom Process Custom Cut
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

filename = str(originFile_list[0])[24:-4]

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

    originVol, originVol_num, originVolwidth, originVolheight = loadFile(originFile_list[i])
    maskVol, maskVol_num, maskVolwidth, maskVolheight = loadFile(maskFile_list[i])

    # Turn the mask images to binary images:
    for j in range(len(maskVol)):
      maskVol[j] = np.where(maskVol[j] != 0, 1, 0)
    for i in range(originVol.shape[0]):
      img = originVol[i, :, :]
      out_images.append(img)
    for i in range(maskVol.shape[0]):
      img = maskVol[i, :, :]
      out_masks.append(img)

  num_images = len(out_images)

  # Writing out images and masks as (1 channel) arrays for input into network
  final_images = np.ndarray([num_images, 512, 512], dtype=np.int16)
  final_masks = np.ndarray([num_images, 512, 512], dtype=np.int8)
  for i in range(num_images):
    final_images[i] = out_images[i]
    final_masks[i] = out_masks[i]

  cut_slice = 113
  num_images_cut =num_images - cut_slice

  # Writing Cut Images:
  final_images_cut = np.ndarray([num_images_cut, 512, 512], dtype=np.int16)
  final_masks_cut = np.ndarray([num_images_cut, 512, 512], dtype=np.int8)

  final_images_cut = final_images[cut_slice:]
  final_masks_cut = final_masks[cut_slice:]

  final_images_cut = np.expand_dims(final_images_cut, axis=-1)
  final_masks_cut = np.expand_dims(final_masks_cut, axis=-1)

  num_file = range(0, 1)

  print('Saving Images...')
  print('-' * 30)
  for i in num_file:

    np.save(cm.workingPath.trainingSet_path + filename + '_origin.npy', final_images_cut)
    np.save(cm.workingPath.trainingSet_path + filename + '_mask.npy', final_masks_cut)

else:
  pass
print('Training Images Saved')
print('-' * 30)
print('Finished')
