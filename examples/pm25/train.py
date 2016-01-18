import datetime
import numpy as np
import cPickle
import gzip 
#from profilehooks import profile
from keras.layers.recurrent_xd import RLSTM, ReducedLSTM, ReducedLSTMA, ReducedLSTMB
from keras.optimizers import RMSprop
from keras.utils.train_utils import *
from data import load_data, load_data2, load_data3, segment_data
from errors import *
from dataset import *

try: 
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict
    
from constants import *

def get_train_result(model):
    datetime_str = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    return '%s   %d   %7.1f %7.1f   %4.2f %4.2f %4.2f   %4.2f %4.2f %4.2f   %4.2f %4.2f %4.2f   %4.2f %4.2f   %4.2f %4.1f %4.1f' % \
        ((datetime_str, model.epoch, model.train_result[0]) + tuple(model.valid_result)) 
#                                              model.valid_result[0],  # mse
#                                              model.valid_result[1], model.valid_result[2], model.valid_result[3],
#                                              model.valid_result[4], model.valid_result[5], model.valid_result[6],
#                                              model.valid_result[7], model.valid_result[8], model.valid_result[9],
                                              

def estimate_early_stop_epoch(name, mov_avg_len=3):
    log_file = name + '.log'
    with open(log_file) as f:
        epochs = [int(line.split()[2]) for line in f.readlines()]
    print 'epochs =', epochs
    if len(epochs) < mov_avg_len:
        epoch = sum(epochs) * 1. / len(epochs)
    else:
        epoch = sum(epochs[-mov_avg_len:]) * 1. / mov_avg_len
    print 'epoch =', epoch
    return int(round(epoch))
    
def train_model(name, is_city=False, latest=False):
    if is_city:
        if name == 'beijing':
            train_data, valid_data, train_data2 = load_data3(stations=city2stations[name],
                                                starttime='20150901', endtime='20160116',
                                                train_stop=630, valid_start=680,
                                                latest=latest,
                                                filter=False)
        else:
            train_data, valid_data, train_data2 = load_data3(stations=city2stations[name],
                                                latest=latest,
                                                filter=False)
    else:
        train_data, valid_data, train_data2 = load_data3(lon_range=area2lonlat[name][0], lat_range=area2lonlat[name][1],
                                            latest=latest,
                                            filter=True)
    X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, pred_range=pred_range, hist_len=3)
    print 'X_train[0].shape =', X_train[0].shape
    rlstm = build_rlstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, 
                               rec_layer_init='zero', fix_b_f=is_city, base_name=name,
                               add_input_noise=is_city, add_target_noise=False)
    rlstm.name = name + '_test'
    rlstm.data = [train_data, valid_data]
    
    rlstm.X_mask = np.ones((X_train[0].shape[-1],), dtype='int')
    rlstm.X_mask[-1:] = not is_city  # pm25 mean
    
    print '\ntraining', rlstm.name
    X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0], rlstm)
    rlstm.save_normalization_info(rlstm.name + '_norm_info.pkl')
    batch_size = (1 + int(not is_city)) * 64
    patience = 20
    train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=batch_size, patience=patience, nb_epoch=300)
#    result_str = get_train_result(rlstm)
#    print 'result_str =', result_str
#    log_file = name + '.log'
#    with open(log_file, 'a') as f:
#        f.write(result_str + '\n')
#    epoch = estimate_early_stop_epoch(name)
#    
#    X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data2, valid_data, pred_range=pred_range, hist_len=3)
#    print 'X_train[0].shape =', X_train[0].shape
#    rlstm = build_rlstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, 
#                               rec_layer_init='zero', fix_b_f=is_city, base_name=name,
#                               add_input_noise=is_city, add_target_noise=False)
#    rlstm.name = name
#    rlstm.data = [train_data, valid_data]
#    
#    rlstm.X_mask = np.ones((X_train[0].shape[-1],), dtype='int')
#    rlstm.X_mask[-1:] = not is_city  # pm25 mean
#    
#    print '\ntraining', rlstm.name
#    X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0], rlstm)
#    rlstm.save_normalization_info(rlstm.name + '_norm_info.pkl')
#    batch_size = (1 + int(not is_city)) * 64
#    patience = 10
#    train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=batch_size, patience=patience, nb_epoch=epoch)

if __name__ == '__main__':
    train_model('beijing', is_city=True)