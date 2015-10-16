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


#from pylearn2.datasets.if_monthly import IFMonthlyLong, IFMonthly2
#train = IFMonthly2(which_set='train', short_ts=[5, 10], use_long=False, target_type='ASV', gain_range=[0, 10], hist_len=3)
#test = IFMonthly2(which_set='test', short_ts=[5, 10], use_long=False, target_type='ASV', gain_range=[0, 10], hist_len=3)
#train = IFMonthlyLong(which_set='train', target_type='ASV', gain_range=[0, 10])
#test = IFMonthlyLong(which_set='test', target_type='ASV', gain_range=[0, 10])
#X_train = train.X
#y_train = train.y
#X_test = test.X
#y_test = test.y

def train_model(dataset, h0_dim, h1_dim):
    X_train, y_train, X_test, y_test = dataset
    batch_size = 512
    nb_epoch = 100
    model = Sequential()
    model.add(Dense(h0_dim, input_shape=(X_train.shape[1],), 
                    init='uniform', 
                    W_regularizer=l2(0.0005),
                    activation='relu'))
    model.add(Dense(h1_dim,  
                    init='uniform', 
                    W_regularizer=l2(0.0005),
                    activation='relu'))
    model.add(Dense(1,  
                    init='uniform', 
                    W_regularizer=l2(0.0005)))
    
    rms = RMSprop()
    sgd = SGD(lr=0.01, decay=1e-4, momentum=0.6, nesterov=False)
    model.compile(loss='mse', optimizer=sgd)
    #model.get_config(verbose=1)
    #yaml_string = model.to_yaml()
    #with open('ifshort_mlp.yaml', 'w') as f:
    #    f.write(yaml_string)
        
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath="/tmp/ifshort_mlp_weights.hdf5", verbose=1, save_best_only=True)
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              nb_epoch=nb_epoch, 
              show_accuracy=False, 
              verbose=2, 
              validation_data=(X_test, y_test), 
              callbacks=[early_stopping, checkpointer])
