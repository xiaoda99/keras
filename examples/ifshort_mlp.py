from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

#from keras.datasets import mnist
from keras.models_xd import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.initializations import uniform
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

batch_size = 512
nb_epoch = 100

# the data, shuffled and split between tran and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#
#X_train = X_train.reshape(60000, 784)
#X_test = X_test.reshape(10000, 784)
#X_train = X_train.astype("float32")
#X_test = X_test.astype("float32")
#X_train /= 255
#X_test /= 255
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
#
## convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

from pylearn2.datasets.if_monthly import IFMonthlyLong
train = IFMonthlyLong(target_type='ASV', gain_range=[0, 10], which_set='train')
X_train = train.X
y_train = train.y
test = IFMonthlyLong(target_type='ASV', gain_range=[0, 10], which_set='test')
X_test = test.X
y_test = test.y

model = Sequential()
model.add(Dense(80, input_shape=(79,), 
                init='uniform',
#                init=lambda shape: uniform(shape, scale=0.05), 
                W_regularizer=l2(0.0005),
                activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(40,  
                init='uniform',
#                init=lambda shape: uniform(shape, scale=0.05), 
                W_regularizer=l2(0.0005),
                activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(1,  
                init='uniform',
#                init=lambda shape: uniform(shape, scale=0.005), 
                W_regularizer=l2(0.0005)))

rms = RMSprop()
sgd = SGD(lr=0.01, decay=1e-4, momentum=0.6, nesterov=False)
model.compile(loss='mse', optimizer=sgd)
#model.get_config(verbose=1)
yaml_string = model.to_yaml()
with open('ifshort_mlp.yaml', 'w') as f:
    f.write(yaml_string)
    
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpointer = ModelCheckpoint(filepath="/tmp/ifshort_mlp_weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, y_train, 
          batch_size=batch_size, 
          nb_epoch=nb_epoch, 
          show_accuracy=False, 
          verbose=2, 
          validation_data=(X_test, y_test), 
          callbacks=[early_stopping, checkpointer])
