import sys
from keras.datasets import mnist
import functions
import architecture
from keras.utils import np_utils
import arc_iron


# Import Mnist digits data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = functions.reshape(x_train, x_test)

x_train, x_test = functions.normalise(x_train, x_test)

x_train, x_test = functions.resize(x_train, x_test)

y_train = np_utils.to_categorical(y_train, 10)

y_test = np_utils.to_categorical(y_test, 10)

model = architecture.ConvNets()

model.train(x_train, y_train)