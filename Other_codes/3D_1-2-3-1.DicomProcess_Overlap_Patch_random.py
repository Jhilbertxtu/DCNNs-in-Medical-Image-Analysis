########################################################################################
# 3D Aorta Segmentation Project                                                        #
#                                                                                      #
# 1. Dicom Process with 3D Overlap Patch                                               #
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
import Modules.Network as nw

import SimpleITK as sitk

import numpy as np

from PIL import Image
import dicom
import cv2

# Import dcm file and turn it into an image array:

# Get the file list:
originVol = np.load(cm.workingPath.home_path + 'trainImages3D.npy')
maskVol = np.load(cm.workingPath.home_path + 'trainMasks3D.npy')

print('Saving Images...')
print('-' * 30)

random_scale = int(len(originVol))
# random_scale = int(len(maskVol))

rand_i = np.random.choice(range(random_scale), size=random_scale, replace=False)

np.save(cm.workingPath.home_path + 'trainImages3D16.npy', originVol[rand_i[:]])
np.save(cm.workingPath.home_path + 'trainMasks3D16.npy', maskVol[rand_i[:]])

print('Training Images Saved')
print('-' * 30)

print("Finished")
