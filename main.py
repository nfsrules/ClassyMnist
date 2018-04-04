import sys
from keras.datasets import mnist
import functions
import architecture
from keras.utils import np_utils


# Import Mnist digits data set from Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Data pre-processing using custom functions
x_train, x_test = functions.reshape(x_train, x_test)

x_train, x_test = functions.normalise(x_train, x_test)

x_train, x_test = functions.resize(x_train, x_test)

y_train = np_utils.to_categorical(y_train, 10)

y_test = np_utils.to_categorical(y_test, 10)


# Import ConvNets architecture from a custom Class
model = architecture.ConvNets()

# Calling .train() method from ConvNets class
model.train(x_train, y_train)