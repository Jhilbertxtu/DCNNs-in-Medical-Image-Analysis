########################################################################################
# 3D Aorta Segmentation Project                                                        #
#                                                                                      #
# 1. Dicom Process with 3D Overlap Patch                                                #
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
# created on 24/01/2018                                                                #
# Last update: 24/01/2018                                                              #
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
originFile_list = sorted(glob(cm.workingPath.training3DSet_path + 'trainImages3D*.npy'))
maskFile_list = sorted(glob(cm.workingPath.training3DSet_path + 'trainMasks3D*.npy'))
# originFile_list = sorted(glob(cm.workingPath.originTrainingSet_path + 'vol*.dcm'))
# maskFile_list = sorted(glob(cm.workingPath.maskTrainingSet_path + 'vol*.dcm'))


filename = str(originFile_list[0])[24:-4]

filelist = open(cm.workingPath.trainingSet_path + "file.txt", 'w')
filelist.write(str(len(originFile_list)))
filelist.write(" datasets involved")
filelist.write("\n")
for file in originFile_list:
  filelist.write(file)
  filelist.write('\n')
filelist.close()

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

# final_images = np.ndarray([0, slices, row, col, 1], dtype=np.int16)
final_masks = np.ndarray([0, slices, row, col, 3], dtype=np.int8)

for i in range(len(originFile_list)):

  # Read information from dcm file:

  # originVol = np.load(originFile_list[i])
  maskVol = np.load(maskFile_list[i])

  # final_images = np.concatenate((final_images, originVol), axis=0)
  final_masks = np.concatenate((final_masks, maskVol), axis=0)


print('Saving Images...')
print('-' * 30)

# random_scale = int(len(final_images))
# random_scale = int(len(final_masks))

# rand_i = np.random.choice(range(random_scale), size=random_scale, replace=False)

# np.save(cm.workingPath.home_path + 'trainImages3D.npy', final_images)
np.save(cm.workingPath.home_path + 'trainMasks3D.npy', final_masks)

print('Training Images Saved')
print('-' * 30)

print("Finished")
