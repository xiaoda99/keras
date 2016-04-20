from __future__ import absolute_import
from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

#from keras.datasets import mnist
from keras.models_xd import Sequential, model_from_yaml
from keras.layers.core import Dense, TimeDistributedMaxoutDense, TimeDistributedDense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers.recurrent_xd import RLSTM, ReducedLSTM, ReducedLSTMA, LSTM, SimpleRNN, GRU
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.initializations import uniform
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from examples.pm25.dataset import normalize #XD
from examples.pm25.config import model_savedir  #XD

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
    model.layers[0].add_init_models(14)
    
    model.compile(loss="mse", optimizer="rmsprop")  
    return model

def build_rlstm2(input_dim, h0_dim, h1_dim, output_dim=1, 
                       lstm_init='zero', lr=.001, base_name='rlstm',
                       add_input_noise=True, add_target_noise=False):
    model = Sequential()  
    if add_input_noise:
        model.add(GaussianNoise(.1, input_shape=(None, input_dim)))
    model.add(RLSTM(input_dim, h0_dim, h1_dim, output_dim, init=lstm_init,
                    W_h0_regularizer=l2(0.0005), W_h1_regularizer=l2(0.0005), return_sequences=True))
#    if add_target_noise:
#        model.add(GaussianNoise(5.))
    model.compile(loss="mse", optimizer=RMSprop(lr=lr))  
    
    model.base_name = base_name
    yaml_string = model.to_yaml()
#    print(yaml_string)
    with open(model_savedir + model.base_name+'.yaml', 'w') as f:
        f.write(yaml_string)
    return model

def build_model(input_dim, h0_dim=20, h1_dim=20, output_dim=1, 
                       rec_layer_type=ReducedLSTMA, rec_layer_init='zero',
                       layer_type=TimeDistributedDense, lr=.001, base_name='rlstm',
                       add_input_noise=True, add_target_noise=True):
    model = Sequential()  
    if add_input_noise:
        model.add(GaussianNoise(.1, input_shape=(None, input_dim)))
    model.add(LSTM(h0_dim, input_dim=input_dim, 
                    init='uniform_small',
                    activation='tanh'))
    model.add(Dropout(.4))
    model.add(LSTM(h1_dim, 
                init='uniform_small',
                activation='tanh'))
    model.add(Dropout(.4))    
    model.add(rec_layer_type(output_dim, init=rec_layer_init, return_sequences=True))
    if add_target_noise:
        model.add(GaussianNoise(5.))
    model.compile(loss="mse", optimizer=RMSprop(lr=lr))  
    
    model.base_name = base_name
    yaml_string = model.to_yaml()
#    print(yaml_string)
    with open(model_savedir + model.base_name+'.yaml', 'w') as f:
        f.write(yaml_string)
    return model

def build_rlstm(input_dim, h0_dim=40, h1_dim=None, output_dim=1, 
                       rec_layer_type=ReducedLSTMA, rec_layer_init='zero', fix_b_f=False,
                       layer_type=TimeDistributedDense, lr=.001, base_name='rlstm',
                       add_input_noise=True, add_target_noise=True):
    model = Sequential()  
    if add_input_noise:
        model.add(GaussianNoise(.1, input_shape=(None, input_dim)))
    model.add(layer_type(h0_dim, input_dim=input_dim, 
                    init='uniform_small', 
                    W_regularizer=l2(0.0005),
                    activation='tanh'))
    if h1_dim is not None:
        model.add(layer_type(h1_dim, 
                    init='uniform_small', 
                    W_regularizer=l2(0.0005),
                    activation='tanh'))
        
    model.add(rec_layer_type(output_dim, init=rec_layer_init, fix_b_f=fix_b_f, return_sequences=True))
    if add_target_noise:
        model.add(GaussianNoise(5.))
    model.compile(loss="mse", optimizer=RMSprop(lr=lr))  
    
    model.base_name = base_name
    yaml_string = model.to_yaml()
#    print(yaml_string)
    with open(model_savedir + model.base_name+'.yaml', 'w') as f:
        f.write(yaml_string)
    return model

def build_reduced_lstm(input_dim, h0_dim=40, h1_dim=None, output_dim=1, 
                       rec_layer_type=ReducedLSTMA, rec_layer_init='uniform',
                       layer_type=TimeDistributedDense, lr=.001, base_name='rlstm'):
    model = Sequential()  
    model.add(layer_type(h0_dim, input_dim=input_dim, 
                    init='uniform', 
                    W_regularizer=l2(0.0005),
                    activation='relu'))
    if h1_dim is not None:
        model.add(layer_type(h1_dim, 
                    init='uniform', 
                    W_regularizer=l2(0.0005),
                    activation='relu'))
#    model.add(LSTM(h0_dim, 
#                   input_dim=input_dim,
#                   init='uniform',
#                   inner_activation='sigmoid',
#                   return_sequences=True))
#    model.add(Dropout(0.4))
#    if h1_dim is not None:
#        model.add(LSTM(h1_dim,
#                       init='uniform',
#                       inner_activation='sigmoid',
#                       return_sequences=True))
#        model.add(Dropout(0.4))
        
    model.add(rec_layer_type(output_dim, init=rec_layer_init, return_sequences=True))
    model.compile(loss="mse", optimizer=RMSprop(lr=lr))  
    
    model.base_name = base_name
    yaml_string = model.to_yaml()
#    print(yaml_string)
    with open(model_savedir + model.base_name+'.yaml', 'w') as f:
        f.write(yaml_string)
    return model

def dot_product_error(y_true, y_pred):
    return -(y_pred * y_true).mean(axis=-1)

def build_mlp(in_dim, out_dim, h0_dim, h1_dim, optimizer='rmsprop'):
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
                    W_regularizer=l2(0.0005)
                    ))
    
#    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.6, nesterov=False)
#    sgd = SGD(lr=learning_rate, decay=1e-24, momentum=0.6, nesterov=False)
    model.compile(loss='mse', optimizer=optimizer)
    
#    model.get_config(verbose=1)
    yaml_string = model.to_yaml()
    with open(model_savedir + 'mlp.yaml', 'w') as f:
        f.write(yaml_string)
    return model

def train(X_train, y_train, X_valid, y_valid, model, batch_size=128, nb_epoch=300, patience=20):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    filepath = model_savedir + model.name + '_weights.hdf5'
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              nb_epoch=nb_epoch, 
              show_accuracy=False, 
              verbose=2, 
              validation_data=(X_valid, y_valid), 
              callbacks=[early_stopping, checkpointer])

def load_mlp():
    mlp = model_from_yaml(open(model_savedir + 'mlp40-40_batch4096.yaml').read())
    mlp.load_weights(model_savedir + 'mlp40-40_batch4096_weights.hdf5')
    return mlp

def load_rlstm(name='rlstm'):
    rlstm = model_from_yaml(open(model_savedir + 'rlstm_test' + '.yaml').read())
    rlstm.load_weights(model_savedir + name + '_weights.hdf5')
    rlstm.name = name
    return rlstm

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
