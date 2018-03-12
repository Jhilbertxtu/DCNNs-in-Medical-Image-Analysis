########################################################################################
# Aorta Segmentation Project                                                           #
#                                                                                      #
# 5. Show Results                                                                      #
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
# created on 27/10/2017                                                                #
# Last update: 30/10/2017                                                              #
########################################################################################

import Modules.Common_modules as cm
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import re



# Show runtime:
starttime = datetime.datetime.now()


# Load data:

imgs_origin = np.load(cm.workingPath.testingSet_path + 'testImages.npy').astype(np.float32)
imgs_true = np.load(cm.workingPath.testingSet_path + 'testMasks.npy').astype(np.float32)
imgs_predict = np.load(cm.workingPath.testingSet_path + 'masksTestPredicted.npy').astype(np.float32)
imgs_predict_threshold = np.load(cm.workingPath.testingSet_path + 'masksTestPredicted.npy').astype(np.float32)

imgs_predict_threshold = np.where(imgs_predict_threshold < 0.5, 0, 1)

# Turn images into binary images from (0,1):

# Prepare to do some operations on images, or not:

# for i in range(len(imgs_true)):
# 	new_imgs_true[len(imgs_true)-i-1] = imgs_true[i]
#
# for i in range(len(imgs_predict)):
# 	new_imgs_predict[len(imgs_predict)-i-1] = imgs_predict[i]




# Draw the subplots of figures:

color1 = 'gray'  # ***
color2 = 'viridis' # ******
# color = 'plasma'  # **
# color = 'magma'  # ***
# color2 = 'RdPu'  # ***
# color = 'gray'  # ***
# color = 'gray'  # ***
# color = 'gray'  # ***
# color = 'gray'  # ***
# color = 'gray'  # ***
# color = 'gray'  # ***
# color = 'gray'  # ***
# color = 'gray'  # ***
# color = 'gray'  # ***

transparent1 = 1.0
transparent2 = 0.5

font = {'size' : 16}
# Slice parameters:
xmajorLocater = MultipleLocator(50)
xmajorFormatter = FormatStrFormatter('%d')
xminorLocater = MultipleLocator(10)

ymajorLocater = MultipleLocator(50)
ymajorFormatter = FormatStrFormatter('%d')
yminorLocater = MultipleLocator(10)

#############################################
# Automatically:

steps = 1
slice = range(0, len(imgs_true), steps)
plt_row = 3
plt_col = int(len(imgs_true)/steps)

# plt.figure(1,figsize=(25,8))
plt.figure('Single Slice', figsize = (4, 12))


for i in slice:
	if i==0:
		plt_num = int(i / steps) + 1
	else:
		plt_num = int(i / steps)

	if plt_num <= plt_col:

		ax1 = plt.subplot(plt_row,plt_col,plt_num)
		title = 'slice=' + str(i)
		plt.title(title)
		ax1.imshow(imgs_origin[i,:,:,0],cmap=color1,alpha=transparent1)
		ax1.imshow(imgs_true[i,:,:,0],cmap=color2,alpha=transparent2)

		ax2 = plt.subplot(plt_row,plt_col,plt_num + plt_col)
		title = 'slice=' + str(i)
		plt.title(title)
		ax2.imshow(imgs_origin[i,:,:,0],cmap=color1,alpha=transparent1)
		ax2.imshow(imgs_predict[i,:,:,0],cmap=color2,alpha=transparent2)

		ax3 = plt.subplot(plt_row, plt_col, plt_num + 2 * plt_col)
		title = 'slice=' + str(i)
		plt.title(title)
		ax3.imshow(imgs_origin[i,:,:,0], cmap=color1, alpha=transparent1)
		ax3.imshow(imgs_predict_threshold[i,:,:,0], cmap=color2, alpha=transparent2)
	else:
		pass

modelname = cm.modelname

imageName = re.findall(r'\d+\.?\d*',modelname)
epoch_num = int(imageName[0])+1
accuracy = float(np.loadtxt(cm.workingPath.testingSet_path + 'dicemean.txt',float))

# saveName = 'epoch_' + str(epoch_num) + '_dice_' +str(accuracy) + '.png'
saveName = 'epoch_%02d_dice_%.3f.png' %(epoch_num,accuracy)

plt.subplots_adjust(left = 0.0, bottom = 0.05, right = 1.0, top = 0.95, hspace = 0.3, wspace = 0.3)
plt.savefig(cm.workingPath.testingSet_path + saveName)
plt.show()


###################################
# Manually:
#
# slice = (100,150,230,250)
#
# plt.figure(2)
#
# ax1 = plt.subplot(2,4,1)
# title = 'slice=' + str(slice[0])
# plt.title(title)
# ax1.imshow(new_imgs_origin[slice[0],0,:,:],cmap=color1,alpha=transparent1)
# ax1.imshow(new_imgs_true[slice[0],0,:,:],cmap=color2,alpha=transparent2)
#
# ax2 = plt.subplot(2,4,5)
# title = 'slice=' + str(slice[0])
# plt.title(title)
# ax2.imshow(new_imgs_origin[slice[0],0,:,:],cmap=color1,alpha=transparent1)
# ax2.imshow(new_imgs_predict[slice[0],0,:,:],cmap=color2,alpha=transparent2)
#
# ax3 = plt.subplot(2,4,2)
# title = 'slice=' + str(slice[1])
# plt.title(title)
# ax3.imshow(new_imgs_origin[slice[1],0,:,:],cmap=color1,alpha=transparent1)
# ax3.imshow(new_imgs_true[slice[1],0,:,:],cmap=color2,alpha=transparent2)
#
# ax4 = plt.subplot(2,4,6)
# title = 'slice=' + str(slice[1])
# plt.title(title)
# ax4.imshow(new_imgs_origin[slice[1],0,:,:],cmap=color1,alpha=transparent1)
# ax4.imshow(new_imgs_predict[slice[1],0,:,:],cmap=color2,alpha=transparent2)
#
# ax5 = plt.subplot(2,4,3)
# title = 'slice=' + str(slice[2])
# plt.title(title)
# ax5.imshow(new_imgs_origin[slice[2],0,:,:],cmap=color1,alpha=transparent1)
# ax5.imshow(new_imgs_true[slice[2],0,:,:],cmap=color2,alpha=transparent2)
#
# ax6 = plt.subplot(2,4,7)
# title = 'slice=' + str(slice[2])
# plt.title(title)
# ax6.imshow(new_imgs_origin[slice[2],0,:,:],cmap=color1,alpha=transparent1)
# ax6.imshow(new_imgs_predict[slice[2],0,:,:],cmap=color2,alpha=transparent2)
#
# ax7 = plt.subplot(2,4,4)
# title = 'slice=' + str(slice[3])
# plt.title(title)
# ax7.imshow(new_imgs_origin[slice[3],0,:,:],cmap=color1,alpha=transparent1)
# ax7.imshow(new_imgs_true[slice[3],0,:,:],cmap=color2,alpha=transparent2)
#
# ax8 = plt.subplot(2,4,8)
# title = 'slice=' + str(slice[3])
# plt.title(title)
# ax8.imshow(new_imgs_origin[slice[3],0,:,:],cmap=color1,alpha=transparent1)
# ax8.imshow(new_imgs_predict[slice[3],0,:,:],cmap=color2,alpha=transparent2)
#
# plt.show()



#############################################

print('Images showing')

endtime = datetime.datetime.now()
print(endtime - starttime)
































