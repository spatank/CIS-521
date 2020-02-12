############################################################
# CIS 521: Individual Functions for CNN
############################################################

student_name = "Shubhankar Patankar"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import numpy as np


############################################################
# Individual Functions
############################################################

def convolve_greyscale(image, kernel):
	kernel = np.fliplr(np.flipud(kernel)) # flip kernel vertically and horizontally 
	image_shape = image.shape
	image_height = image_shape[0] # number of rows in image
	image_width = image_shape[1] # number of columns in image
	kernel_shape = kernel.shape
	kernel_height = kernel_shape[0]
	kernel_width = kernel_shape[1]
	height_pad = int(kernel_height/2)
	width_pad = int(kernel_width/2)
	# pad image with zeros
	padded_input = np.pad(image, ((height_pad, height_pad),(width_pad, width_pad)), 'constant')
	output = np.empty(image_shape) # empty array of the size of the input
	for i in range(image_height):
		for j in range(image_width):
			submatrix = padded_input[i:i+kernel_height, j:j+kernel_width]
			output[i][j] = np.multiply(submatrix, kernel).sum()
	return output

def convolve_rgb(image, kernel):
	image_shape = image.shape # image should be a 3D array
	image_height = image_shape[0]
	image_width = image_shape[1]
	image_depth = image_shape[2]
	output = np.empty(image_shape) # empty array of the size of the input
	for color in range(image_depth): # RGB depthwise
		channel = image[:, :, color]
		output[:, :, color] = convolve_greyscale(channel, kernel)
	return output


def max_pooling(image, kernel_size, stride):
	image_shape = image.shape
	image_height = image_shape[0] # number of rows in image
	image_width = image_shape[1] # number of columns in image
	kernel_height = kernel_size[0]
	kernel_width = kernel_size[1]
	ver_stride = stride[0]
	hor_stride = stride[1]
	output_width = int((image_width - kernel_width) / hor_stride + 1)
	output_height = int((image_height - kernel_height) / ver_stride + 1)
	output = np.empty((output_height, output_width))
	row_idx = 0
	col_idx = 0
	for i in range(output_height):
		for j in range(output_width):
			submatrix = image[row_idx:row_idx+kernel_height, col_idx:col_idx+kernel_width]
			output[i][j] = np.max(submatrix)
			col_idx += hor_stride
		col_idx = 0
		row_idx += ver_stride
	return output

def average_pooling(image, kernel_size, stride):
	image_shape = image.shape
	image_height = image_shape[0] # number of rows in image
	image_width = image_shape[1] # number of columns in image
	kernel_height = kernel_size[0]
	kernel_width = kernel_size[1]
	ver_stride = stride[0]
	hor_stride = stride[1]
	output_width = int((image_width - kernel_width) / hor_stride + 1)
	output_height = int((image_height - kernel_height) / ver_stride + 1)
	output = np.empty((output_height, output_width))
	row_idx = 0
	col_idx = 0
	for i in range(output_height):
		for j in range(output_width):
			submatrix = image[row_idx:row_idx+kernel_height, col_idx:col_idx+kernel_width]
			output[i][j] = np.mean(submatrix)
			col_idx += hor_stride
		col_idx = 0
		row_idx += ver_stride
	return output


def sigmoid(x):
	return 1/(1 + np.exp(-x))
