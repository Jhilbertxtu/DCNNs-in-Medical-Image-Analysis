from __future__ import print_function

import Modules.Common_modules as cm
import numpy as np
import os
from keras import backend as K
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt
import matplotlib.cm as mpcm
from PIL import Image
import Modules.utils as utils
import cv2


#########################################################
# Visualization:


def visualize_activation_in_layer(model, input_image):
  """
  Visualize activations in layers(By Florian)
  """
  # reformat input image
  input_image = [input_image.tolist()]

  # iterate across layers
  for layer_index in range(len(model.layers)):

    # create save folder
    if not os.path.exists(os.path.join(cm.workingPath.model_path, 'activations', 'layer_%d' % layer_index)):
      os.makedirs(os.path.join(cm.workingPath.model_path, 'activations', 'layer_%d' % layer_index))

      # define function
    convout1 = model.layers[layer_index]
    convout1_f = K.function([model.input], convout1.output)

    # compute the activation map
    C1 = convout1_f(input_image)
    C1 = np.squeeze(C1)
    print(layer_index, ' ', C1.shape)

    # save results
    if len(C1.shape) == 3:  # if only a single filter in the layer (basically to save the input)
      C1 = C1[:, :, int(C1.shape[2] * 0.4)]
      imsave(os.path.join(cm.workingPath.model_path, 'activations', 'layer_%d' % layer_index,
                          '%d_filter_%d.png' % (layer_index, 0)),
             C1)

    elif len(C1.shape) == 4:  # if several filters in the layer
      C1 = C1[:, :, :, int(C1.shape[3] * 0.4)]
      for filter_index in range(C1.shape[-1]):
        img = C1[filter_index]
        imsave(os.path.join(cm.workingPath.model_path, 'activations', 'layer_%d' % layer_index,
                            '%d_filter_%d.png' % (layer_index, filter_index)), img)

    else:
      print('error: wrong dimensions for C1')


def plot_conv_weights(weights, plot_dir, name, channels_all=True, filters_all=True, channels=[0], filters=[0]):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """

    w_min = np.min(weights)
    w_max = np.max(weights)

    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    if filters_all:
        num_filters = weights.shape[3]
        filters = range(weights.shape[3])
    else:
        num_filters = len(filters)

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel_ID in channels:
        # iterate filters inside every channel
        if num_filters == 1:
            img = weights[:, :, channel_ID, filters[0]]
            axes.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            axes.set_xticks([])
            axes.set_yticks([])
        else:
            for l, ax in enumerate(axes.flat):
                # get a single filter
                img = weights[:, :, channel_ID, filters[l]]
                # put it on the gridvisual_pathvisual_path
                ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
                # remove any labels from the axes
                ax.set_xticks([])
                ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel_ID)), bbox_inches='tight')


def plot_conv_output(conv_img, plot_dir, name, filters_all=True, filters=[0]):
  w_min = np.min(conv_img)
  w_max = np.max(conv_img)

  # get number of convolutional filters
  if filters_all:
    num_filters = conv_img.shape[3]
    filters = range(conv_img.shape[3])
  else:
    num_filters = len(filters)

  # get number of grid rows and columns
  grid_r, grid_c = utils.get_grid_dim(num_filters)

  # create figure and axes
  fig, axes = plt.subplots(min([grid_r, grid_c]),
                           max([grid_r, grid_c]))

  # iterate filters
  if num_filters == 1:
    img = conv_img[0, :, :, filters[0]]
    axes.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap=cm.hot)
    # remove any labels from the axes
    axes.set_xticks([])
    axes.set_yticks([])
  else:
    for l, ax in enumerate(axes.flat):
      # get a single image
      img = conv_img[0, :, :, filters[l]]
      # put it on the grid
      ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap=cm.hot)
      # remove any labels from the axes
      ax.set_xticks([])
      ax.set_yticks([])
  # save figure
  plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


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


def writeVideo(img_array, filename):
  frame_num, width, height = img_array.shape
  filename_output = filename.split('.')[0] + '.avi'
  video = cv2.VideoWriter(filename_output, -1, 16, (width, height))
  for img in img_array:
    video.write(img)
  video.release()
