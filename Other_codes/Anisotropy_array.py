# =======================================================================================
# Aorta Segmentation Project                                                           #
#                                                                                      #
# 0. Test 1                                                                            #
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
# created on 26/10/2017                                                                #
# Last update: 26/10/2017                                                              #
# =======================================================================================

from Modules import Common_modules as common
import dicom
import os, time, sys
import numpy
from matplotlib import pyplot, cm
import SimpleITK as sitk


# if len(sys.argv) < 3:
#   print("Usage: python " + __file__ + " <input_directory_with_DICOM_series> <output_directory>")
#   sys.exit(1)




original_filename = common.workingPath.originTestingSet_path + "vol3885_20090217-D-1.dcm"
mask_filename = common.workingPath.maskTestingSet_path + "vol3885_Manual_3DMask.dcm"
predict_filename = common.workingPath.testingSet_path + "masksTestPredicted.dcm"
predict_new_filename = common.workingPath.testingSet_path + "masksTestPredictedRescale.dcm"

ds3 = dicom.read_file("dicom.dcm")

ds1 = dicom.read_file(mask_filename)
ds2 = dicom.read_file(predict_filename)
ds1.PixelData = ds2.PixelData
ds1.save_as("dicom.dcm")
# ds2 = dicom.read_file(mask_filename)
# ds3 = dicom.read_file(predict_filename)
# ds4 = dicom.read_file(predict_new_filename)
sys.exit(0)
image_reader = sitk.ImageFileReader()

image_reader.SetFileName(predict_filename)
image_reader.LoadPrivateTagsOn()

predict3D = image_reader.Execute()


image_reader.SetFileName(original_filename)
image_reader.LoadPrivateTagsOn()

image3D = image_reader.Execute()







image3D1 = sitk.GetArrayFromImage(image3D)



image11 = image3D.GetPixelIDValue()










writer = sitk.ImageFileWriter()
writer.KeepOriginalImageUIDOn()
modification_time = time.strftime("%H%M%S")
modification_date = time.strftime("%Y%m%d")


image_slice = image3D
predict_slice = predict3D

for k in image_slice.GetMetaDataKeys():
  imageData = image_slice.GetMetaData(k)
  predict_slice.SetMetaData(k, imageData)
  predictData = predict_slice.GetMetaData(k)

predict_slice.SetMetaData("0008|0031", modification_time)
predict_slice.SetMetaData("0008|0021", modification_date)

writer.SetFileName(common.workingPath.testingSet_path + "masksTestPredictedRescale.dcm")
writer.Execute(predict_slice)

print("finished")

sys.exit(0)












#
# # PathDicom = "./MyHead/"
# PathDicom = "./Vol/"
# lstFilesDCM = []  # create an empty list
# for dirName, subdirList, fileList in os.walk(PathDicom):
#   for filename in fileList:
#     if ".dcm" in filename.lower():  # check whether the file's DICOM
#       lstFilesDCM.append(os.path.join(dirName, filename))
#
# # Get ref file
# RefDs = dicom.read_file(lstFilesDCM[0])
#
# # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
# ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), int(RefDs.NumberofFrames))
#
# # Load spacing values (in mm)
# ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
#
# x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
# y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
# z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
#
# # The array is sized based on 'ConstPixelDims'
# ArrayDicom = numpy.zeros(ConstPixelDims)
#
# # loop through all the DICOM files
# ds = sitk.ReadImage(lstFilesDCM)
#
#
#
# pyplot.figure(dpi=300)
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.set_cmap(pyplot.gray())
# pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 80]))
