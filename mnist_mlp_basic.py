'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Lambda, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import pdb


batch_size = 128
nb_classes = 10
nb_epoch = 400
nb_retrain = 20;

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


model = Sequential()
model.add(Dense(200, input_shape=(784,),bias=False))
model.add(Activation('relu'))
model.add(Lambda(lambda x: (2*x - 1.0)))
model.add(Dense(200, bias=False))
model.add(Activation('tanh'))
model.add(Dense(10, bias=False))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

model.summary()

sgd = SGD(lr = 0.01, momentum = 0.9, decay = 0.0005)
# model.compile(loss='categorical_crossentropy',
              # optimizer=sgd,
              # metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# for i_retrain in range(nb_retrain):
    # print ('--------------------------------------------------------------')
    # # history = model.fit(X_train, Y_train,
                        # # batch_size=batch_size, nb_epoch=nb_epoch,
                        # # verbose=0, validation_data=(X_test, Y_test))
    # for layer in model.layers:
        # if type(layer) is  Dense:
            # weights = np.array(layer.get_weights()) # list of numpy arrays
            # std = np.std(weights)
            # # pdb.set_trace()
            # nearZeroIdx = weights  < 0.2 * std
            # weights[nearZeroIdx] = 0
            # layer.set_weights(weights)
            # print ("{0}'s nearZero Value: {1}".format(i_retrain, np.count_nonzero(nearZeroIdx)))
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Final Test score:', score[0])
print('Final Test accuracy:', score[1])
model.save_weights('rms_400_weights.h5')
