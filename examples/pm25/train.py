#!/usr/bin/env python
import os
import sys
import shutil
import datetime
import numpy as np
import cPickle
import gzip 
#from profilehooks import profile
from keras.layers.recurrent_xd import RLSTM, ReducedLSTM, ReducedLSTMA, ReducedLSTMB
from keras.optimizers import RMSprop
from keras.utils.train_utils import *
from data import load_data, load_data2, load_data4, segment_data
from errors import *
from dataset import *
from config import model_savedir

try: 
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict
    
from constants import *

def get_train_result(model):
    datetime_str = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    return '%s  %2d   %7.1f %7.1f   %4.2f %4.2f %4.2f   %4.2f %4.2f %4.2f   %4.2f %4.2f %4.2f   %4.2f %4.2f   %4.2f %4.1f %4.1f' % \
        ((datetime_str, model.epoch, model.train_result[0]) + tuple(model.valid_result)) 
#                                              model.valid_result[0],  # mse
#                                              model.valid_result[1], model.valid_result[2], model.valid_result[3],
#                                              model.valid_result[4], model.valid_result[5], model.valid_result[6],
#                                              model.valid_result[7], model.valid_result[8], model.valid_result[9],
                                              

def estimate_early_stop_epoch(name, mov_avg_len=3):
    log_file = name + '.log'
    with open(model_savedir + log_file) as f:
        epochs = [int(line.split()[2]) for line in f.readlines()]
    print 'epochs =', epochs
    if len(epochs) < mov_avg_len:
        epoch = sum(epochs) * 1. / len(epochs)
    else:
        epoch = sum(epochs[-mov_avg_len:]) * 1. / mov_avg_len
    print 'epoch =', epoch
    return int(round(epoch))
    
def train_model(name, is_city=False, latest=True):
    if is_city:
        if name == 'shanghai':
            train_data, valid_data = load_data4(stations=city2stations[name],
                                                heating=False,
                                                filter=False)
        else:
            train_data, valid_data = load_data4(stations=city2stations[name],
                                                heating=True,
                                                filter=False)
    else:
        train_data, valid_data = load_data4(lon_range=area2lonlat[name][0], lat_range=area2lonlat[name][1],
                                            filter=True)
    X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, pred_range=pred_range, hist_len=3)
    print 'X_train[0].shape =', X_train[0].shape
    rlstm = build_rlstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, 
                               rec_layer_init='zero', fix_b_f=is_city, base_name=name,
                               add_input_noise=is_city, add_target_noise=False)
    rlstm.name = name + '_valid'
    rlstm.data = [train_data, valid_data]
    
    rlstm.X_mask = np.ones((X_train[0].shape[-1],), dtype='int')
    rlstm.X_mask[-1:] = not is_city  # pm25 mean
    
    print '\ntraining', rlstm.name
    X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0], rlstm)
    rlstm.save_normalization_info(model_savedir + rlstm.name + '_norm_info.pkl')
    batch_size = (1 + int(not is_city)) * 64
    patience = 10 + int(is_city) * 10
    train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=batch_size, patience=patience, nb_epoch=300)
    
    result_str = get_train_result(rlstm)
    print 'result_str =', result_str
    with open(model_savedir + name + '.log', 'a') as f:
        f.write(result_str + '\n')
    # epoch = estimate_early_stop_epoch(name)
    # 
    # X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data2, valid_data, pred_range=pred_range, hist_len=3)
    # print 'X_train[0].shape =', X_train[0].shape
    # rlstm = build_rlstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, 
    #                            rec_layer_init='zero', fix_b_f=is_city, base_name=name,
    #                            add_input_noise=is_city, add_target_noise=False)
    # rlstm.name = name
    # rlstm.data = [train_data, valid_data]
    # 
    # rlstm.X_mask = np.ones((X_train[0].shape[-1],), dtype='int')
    # rlstm.X_mask[-1:] = not is_city  # pm25 mean
    # 
    # print '\ntraining', rlstm.name
    # X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0], rlstm)
    # rlstm.save_normalization_info(model_savedir + rlstm.name + '_norm_info.pkl')
    # batch_size = (1 + int(not is_city)) * 64
    # patience = 10 + int(is_city) * 10
    # train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=batch_size, patience=patience, nb_epoch=epoch)
    

if __name__ == '__main__':
#    train_model('beijing', is_city=True)
    train_model('dongbei')
    train_model('huabei')
    train_model('xibei')
    train_model('huadong')
    train_model('huaxi')
    train_model('huanan')
    train_model('tianjin', is_city=True)
    train_model('shanghai', is_city=True)

    #rsync
    machine_list = [
            "10.144.246.254", 
            "inner.wrapper2.api.caiyunapp.com", 
            "10.174.213.150", "10.251.17.17", 
            "10.165.41.213"
            ]

    print "rsync start"
    for machine in machine_list :
        os.system('rsync -av /ldata/pm25data/pm25model/rlstm/* caiyun@'+machine+':/ldata/pm25data/pm25model/rlstm/')

    print "rsync finished"
