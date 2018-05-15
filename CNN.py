# Using high level keras library APIs for creating NN and evaluating MNIST.

import tensorflow as tf
from keras import backend as k
from keras.layers import Dense, Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Dropout, Activation, Flatten
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
#from tensorflow.examples.tutorials.mnist import input_data
from keras import Sequential
from keras.datasets import mnist as mnist2
from keras.utils import np_utils

#mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)
epoch = 100

sess = tf.Session()
k.set_session(sess)

img = tf.placeholder(tf.float32, shape=(None, 784))
labels = tf.placeholder(tf.float32, shape=(None, 10))


#Using Keras to make a CNN

(X_train, y_train), (X_test, y_test) = mnist2.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


#print(X_train.shape)

#CNN
model = Sequential()
model.add(Conv2D( 32, (3, 3), activation = 'relu', input_shape = (28,28, 1)))
model.add(Conv2D( 32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))    #MaxPooling2D is a way to reduce the number of parameters in our model
model.add(Dropout(0.25))       #Using Dropout is a method for regularizing our model in order to prevent overfitting
model.add(Flatten())            # Weights made 1-D
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Fitting model
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)


#Evaluating
score = model.evaluate(X_test, Y_test, verbose=0)

print(score)
