from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json



class ConvNets(Sequential):

	def __init__(self,input_shape = (28, 28, 1), learning_rate=0.5, optimizers='sgd'):
		super(ConvNets, self).__init__()

		if optimizers == 'sgd':
			# Setting network parameters
			sgd = SGD(learning_rate)
		else:
			print('Optimizer not recognized')
		# Building architecture
		# Layer 1 (conv + non-linearity + pooling)
		self.add(Conv2D(32,kernel_size=(5, 5),
		                 activation='sigmoid',
		                 input_shape=(28, 28, 1),
		                 padding='same',
		                 name='conv1'))

		self.add(MaxPooling2D(pool_size=(2, 2)))

		# Layer 2 (conv, non-linearity + pooling)
		self.add(Conv2D(64,kernel_size=(5, 5),
		                 activation='sigmoid',
		                 input_shape=(28, 28, 1),
		                 padding='same',
		                 name='conv2'))

		self.add(MaxPooling2D(pool_size=(2, 2)))

		self.add(Flatten())  # flatten output as vector

		# Fully connected layer 1
		self.add(Dense(100, name='fc1')) # is not necessary to add input_dim
		self.add(Activation('sigmoid'))   #non linearity type sigmoid

		# Fully connected layer 2
		self.add(Dense(10, name='fc2')) # is not necessary to add input_dim
		self.add(Activation('softmax'))   #non linearity type sigmoid

		# Compile model
		self.compile(loss='categorical_crossentropy',optimizer=sgd,
              metrics=['accuracy'])

		# Print model summary
		self.summary()

	def train(self, x_train, y_train, batch_size=300, nb_epoch=10):
		self.fit(x_train, y_train,batch_size=batch_size, epochs=nb_epoch,verbose=1)


