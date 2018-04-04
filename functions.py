import numpy as np 


def reshape(x_train, x_test):
	x_train = x_train.reshape(60000, 784)
	x_test = x_test.reshape(10000, 784)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	return x_train, x_test


def normalise(x_train, x_test):
	x_train /= 255
	x_test /= 255
	return x_train, x_test


def resize(x_train, x_test):
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	input_shape = (28, 28, 1)
	return x_train, x_test