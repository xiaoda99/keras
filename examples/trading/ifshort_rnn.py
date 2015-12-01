from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

#from keras.datasets import mnist
from keras.models_xd import Sequential
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.initializations import uniform
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

def build_rnn(in_dim, out_dim, h0_dim, h1_dim=None, layer_type=LSTM, return_sequences=False):
    model = Sequential()  
    model.add(layer_type(h0_dim, input_shape=(None, in_dim), return_sequences=(h1_dim is not None or return_sequences)))  
    if h1_dim is not None:
        model.add(layer_type(h1_dim, return_sequences=return_sequences))
    if return_sequences:
        model.add(TimeDistributedDense(out_dim))
    else:
        model.add(Dense(out_dim))  
    model.add(Activation("linear"))  
    model.compile(loss="mse", optimizer="rmsprop")  
    return model

def dot_product_error(y_true, y_pred):
    return -(y_pred * y_true).mean(axis=-1)

def build_mlp(in_dim, out_dim, h0_dim, h1_dim, learning_rate=.01):
    model = Sequential()
    model.add(Dense(h0_dim, input_shape=(in_dim,), 
                    init='uniform', 
                    W_regularizer=l2(0.0005),
                    activation='relu'))
    model.add(Dense(h1_dim,  
                    init='uniform', 
                    W_regularizer=l2(0.0005),
                    activation='relu'))
    model.add(Dense(out_dim,  
                    init='uniform', 
                    W_regularizer=l2(0.0005)))
    
#    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.6, nesterov=False)
    sgd = SGD(lr=learning_rate, decay=1e-14, momentum=0.6, nesterov=False)
    model.compile(loss='mse', optimizer=sgd)
    return model

def train(X_train, y_train, X_test, y_test, model, batch_size=128, nb_epoch=100):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath="/tmp/ifshort_rnn_weights.hdf5", verbose=1, save_best_only=True)
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              nb_epoch=nb_epoch, 
              show_accuracy=False, 
              verbose=2, 
              validation_data=(X_test, y_test), 
              callbacks=[early_stopping, checkpointer])
    
def train_model(dataset, h0_dim, h1_dim, out_dim):
    X_train, y_train, X_test, y_test = dataset
    batch_size = 128
    nb_epoch = 100
    
    model = Sequential()  
    model.add(RNN(h0_dim, input_shape=(None, X_train.shape[-1]), return_sequences=True))  
    model.add(TimeDistributedDense(out_dim))  
    model.add(Activation("linear"))  
    model.compile(loss="mse", optimizer="rmsprop")  
    #model.get_config(verbose=1)
    #yaml_string = model.to_yaml()
    #with open('ifshort_mlp.yaml', 'w') as f:
    #    f.write(yaml_string)
        
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath="/tmp/ifshort_rnn_weights.hdf5", verbose=1, save_best_only=True)
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              nb_epoch=nb_epoch, 
              show_accuracy=False, 
              verbose=2, 
              validation_data=(X_test, y_test), 
              callbacks=[early_stopping, checkpointer])
