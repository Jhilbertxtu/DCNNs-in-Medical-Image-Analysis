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

#
# from numpy import argmax
# # define input string
# data = 'hello world'
# print(data)
# # define universe of possible input values
# alphabet = 'abcdefghijklmnopqrstuvwxyz '
# # define a mapping of chars to integers
# char_to_int = dict((c, i) for i, c in enumerate(alphabet))
# int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# # integer encode input data
# integer_encoded = [char_to_int[char] for char in data]
# print(integer_encoded)
# # one hot encode
# onehot_encoded = list()
# for value in integer_encoded:
# 	letter = [0 for _ in range(len(alphabet))]
# 	letter[value] = 1
# 	onehot_encoded.append(letter)
# print(onehot_encoded)
# # invert encoding
# inverted = int_to_char[argmax(onehot_encoded[0])]
# print(inverted)

# from numpy import array
# from numpy import argmax
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# # define example
# data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
# values = array(data)
# print(values)
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
# # invert first example
# inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
# print(inverted)
#

from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)
# invert encoding
inverted = argmax(encoded[0])
print(inverted)



#
# img = tf.constant(value=[
#
# [1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]
#
#
#
# ],dtype=tf.float32)
#
#
#
# img = tf.expand_dims(img, 0)
#
# img = tf.expand_dims(img, -1)
#
# img2 = tf.concat(values=[img,img],axis=3)
#
# filter = tf.constant(value=1, shape=[3,3,2,3], dtype=tf.float32)
#
# out_img1 = tf.nn.atrous_conv2d(value=img2, filters=filter, rate=1, padding='SAME')
#
# out_img11 = tf.nn.conv2d(input=img2, filter=filter, strides=[1,1,1,1], padding='SAME')
#
# out_img2 = tf.nn.atrous_conv2d(value=img2, filters=filter, rate=1, padding='VALID')
#
# out_img3 = tf.nn.atrous_conv2d(value=img2, filters=filter, rate=2, padding='SAME')
#
#
#
# #error
#
# #out_img4 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=2, padding='VALID')
#
#
#
# with tf.Session() as sess:
#
# 	print(sess.run((img)))
# 	print(sess.run((img2)))
# 	print(sess.run((filter)))
# 	print('rate=1, SAME mode result:')
#
# 	print(sess.run(out_img1))
#
#
#
# 	print('strides=1, SAME mode result:')
#
# 	print(sess.run(out_img11))
#
#
#
# 	print('rate=1, VALID mode result:')
#
# 	print(sess.run(out_img2))
#
#
#
# 	print('rate=2, SAME mode result:')
#
# 	print(sess.run(out_img3))
#
#
#
# 	print(sess.run(img))
#
# # error
#
# #print 'rate=2, VALID mode result:'
#
# #print(sess.run(out_img4))


#
# from __future__ import print_function
#
# import numpy as np
# import Modules.Common_modules as cm
# import SimpleITK as sitk
# import sys, time
# import re
# import dicom
#
# def loadFile(filename):
# 	ds = sitk.ReadImage(filename)
# 	img_array = sitk.GetArrayFromImage(ds)
# 	frame_num, width, height = img_array.shape
# 	return img_array, frame_num, width, height
#
# def loadFileInformation(filename):
# 	information = {}
# 	ds= dicom.read_file(filename)
# 	information['PatientID'] = ds.PatientID
# 	information['PatientName'] = ds.PatientName
# 	information['PatientBirthDate'] = ds.PatientBirthDate
# 	information['PatientSex'] = ds.PatientSex
# 	information['StudyID'] = ds.StudyID
# 	# information['StudyTime'] = ds.Studytime
# 	information['InstitutionName'] = ds.InstitutionName
# 	information['Manufacturer'] = ds.Manufacturer
# 	information['NumberOfFrames'] = ds.NumberOfFrames
# 	return information
#
# # image = sitk.ReadImage(cm.workingPath.originTestingSet_path + 'vol3885_20090217-D-1.dcm')
# #
# # img_array = sitk.GetArrayFromImage(image)
# #
# # img_array = img_array + 4000
# #
# # img_array = np.uint16(img_array)
# #
# # new_image = sitk.GetImageFromArray(img_array)
# #
# # sitk.WriteImage(new_image, '1.dcm')
#
#
#
# # imgs_origin = np.load(cm.workingPath.testingSet_path + 'testImages.npy').astype(np.uint16)
# #
# # imgs_true = np.load(cm.workingPath.testingSet_path + 'testMasks.npy').astype(np.uint16)
# # imgs_predict = np.load(cm.workingPath.testingSet_path + 'masksTestPredicted.npy').astype(np.uint16)
# #
# # # for j in range(len(imgs_true)):
# # # 	imgs_true[j] = np.where(imgs_true[j] != 0, 1, 0)
# # # for j in range(len(imgs_predict)):
# # # 	imgs_predict[j] = np.where(imgs_predict[j] != 0, 1, 0)
# #
# # # imgs_origin = np.squeeze(imgs_origin, axis=-1)
# # # imgs_true = np.squeeze(imgs_true, axis=-1)
# # # imgs_predict = np.squeeze(imgs_predict, axis=-1)
# #
# # # imgs_origin[:,:,:] = imgs_origin[:,:,:,0]
# # # imgs_true[:,:,:] = imgs_true[:,:,:,0]
# # # imgs_predict[:,:,:] = imgs_predict[:,:,:,0]
# #
# #
# # new_image_origin = sitk.GetImageFromArray(imgs_origin)
# # new_image_true = sitk.GetImageFromArray(imgs_true)
# # new_image_predict = sitk.GetImageFromArray(imgs_predict)
#
#
# # sitk.WriteImage(new_image_origin,cm.workingPath.testingSet_path + 'testImages.dcm')
# # sitk.WriteImage(new_image_true,cm.workingPath.testingSet_path + 'testMasks.dcm')
# # sitk.WriteImage(new_image_predict,cm.workingPath.testingSet_path + 'masksTestPredicted.dcm')
# #
#
# imgs_origin = np.load(cm.workingPath.trainingSet_path + 'trainImages_0000.npy').astype(np.uint16)
#
# imgs_true = np.load(cm.workingPath.trainingSet_path + 'trainMasks_0000.npy').astype(np.uint16)
#
# imgs_origin = imgs_origin + 4000
#
#
# new_image_origin = sitk.GetImageFromArray(imgs_origin)
# new_image_true = sitk.GetImageFromArray(imgs_true)
#
# sitk.WriteImage(new_image_origin,cm.workingPath.trainingSet_path + 'trainImages.dcm')
# sitk.WriteImage(new_image_true,cm.workingPath.trainingSet_path + 'trainMasks.dcm')
#
# # # modelname = 'Best_weights.03-0.00808.hdf5'
# # #
# # # imageName = re.findall(r'\d+\.?\d*',modelname)
# # # epoch_num = imageName[0]
# # # accuracy = imageName[1]
# #
# # # mean = 0.1471246781264
# # # np.savetxt(cm.workingPath.testingSet_path + 'dicemean.txt', np.array(mean).reshape(1,), fmt='%.3f')
# # # print('Mean Dice Coeff', mean)
#
#
#
# print('finished')
#
# pass
