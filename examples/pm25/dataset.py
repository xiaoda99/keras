import numpy as np
import cPickle
import gzip
from errors import *

def decompose_sequences(gfs, date_time, pm25_mean, pm25, pred_range):
    X = []
    y = []
    for i in range(pred_range[0], pred_range[1]):
        recent_gfs = gfs[:,i-2:i+1,:].reshape((gfs.shape[0], -1))
        current_date_time = date_time[:,i,:]
        current_pm25_mean = pm25_mean[:,i,:]
        init_pm25 = pm25[:,pred_range[0]-1,:]
        step = np.ones((pm25.shape[0],1)) * (i - pred_range[0] + 1)
        Xi = np.hstack([recent_gfs, current_date_time, current_pm25_mean, init_pm25, step])
        yi = pm25[:,i,:]
        X.append(Xi)
        y.append(yi)
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y

def delta(gfs):
    delta_gfs = np.zeros_like(gfs)
    for i in range(gfs.shape[1] - 1):
        delta_gfs[:,i,:] = gfs[:,i+1,:] - gfs[:,i,:]
    delta_gfs[:,-1,:] = gfs[:,-1,:]
    return delta_gfs

def compute_wind_direction(u, v, one_hot=True):
    if not one_hot:
        return np.arctan2(u, v)
    else:
        return np.hstack([(u >= 0) & (v >= 0), (u >= 0) & (v < 0), (u < 0) & (v >= 0), (u < 0) & (v < 0)])  
    
def transform_sequences(gfs, date_time, lonlat, pm25_mean, pm25, pred_range, hist_len=3):
#    print 'In transform_sequences.', gfs.shape, date_time.shape, lonlat.shape, pm25_mean.shape, pm25.shape
#    gfs = np.copy(gfs)
#    gfs[:,:,0] /= 10. # replace temperature with wind speed 0~20
#    gfs[:,:,1] /= 100. # humidity 20~100
#    gfs[:,:,2] /= 10. # wind_x -15~15
#    gfs[:,:,3] /= 10. # wind_y -15~15
#    gfs[:,:,4] /= 0.001  # rain 0~0.0018
#    gfs[:,:,5] /= 100.  # cloud 0~100
#    lonlat = np.copy(lonlat)
#    lonlat[:,:,:] /= 100.  # lon, lat 20~120
#    print 'pred_range =', pred_range
    X = []
    y = []
    wind = []
    for i in range(pred_range[0], pred_range[1]):
        if i - hist_len + 1 >= 0:
            recent_gfs = gfs[:,i-hist_len+1:i+1,0:]  # exclude temperature feature 
        else:
            assert False
            print 'shapes:', np.zeros((gfs.shape[0], hist_len-i-1, gfs.shape[2])).shape, gfs[:,0:i+1,:].shape
            recent_gfs = np.concatenate((np.zeros((gfs.shape[0], hist_len-i-1, gfs.shape[2])), gfs[:,0:i+1,:]), axis=1)
#        recent_gfs = delta(recent_gfs)
#        recent_gfs = recent_gfs.reshape((recent_gfs.shape[0], -1))
        recent_pm25_mean = pm25_mean[:,i-hist_len+1:i+1,0]
        u = recent_gfs[:,:,2]
        v = recent_gfs[:,:,3]
        recent_wind_speed = np.sqrt(u**2 + v**2)
#        recent_wind_direction = np.arctan2(v, u)
        recent_wind_direction = np.hstack([u, v])
#        recent_wind_direction = np.hstack([u.mean(axis=1, keepdims=True), v.mean(axis=1, keepdims=True)])
#        recent_wind_direction = np.hstack([u[:,-1:], v[:,-1:]])
#        recent_wind_direction = compute_wind_direction(u.mean(axis=1, keepdims=True), 
#                                                       v.mean(axis=1, keepdims=True), 
#                                                       one_hot=True)
        recent_temperature = recent_gfs[:,:,0]
        recent_humidity = recent_gfs[:,:,1]
        recent_rain = recent_gfs[:,:,4]
        recent_cloud = recent_gfs[:,:,5]
        
        Xi = np.hstack([
#                        recent_gfs.reshape((recent_gfs.shape[0], -1)),
                        recent_wind_direction,
                        recent_wind_speed,
                        recent_humidity,
                        recent_rain,
                        recent_cloud, 
                        date_time[:,i,0:2], # exclude day of year feature
                        lonlat[:,i,:], 
                        pm25_mean[:,i,:],
#                        recent_pm25_mean
                        ])
#        print 'Xi.shape =', Xi.shape
        yi = pm25[:,i,:]
        X.append(Xi)
        y.append(yi)
        wind.append(recent_wind_speed)
    init_pm25 = pm25[:,pred_range[0]-1,:]
    init_gfs = gfs[:,:pred_range[0],:].reshape((gfs.shape[0], -1))
#    X_init = np.hstack([init_pm25, init_gfs])
    hidden_init = init_pm25 
    cell_init = init_pm25 + pm25_mean[:,pred_range[0]-1,:]
    cell_mean = pm25_mean[:,pred_range[0]:pred_range[1],:]
    X = np.dstack(X).transpose((0, 2, 1)) #.astype('float32')
#    X = np.concatenate([gfs, date_time[:,:,:-1]], axis=2)[:,pred_range[0]:pred_range[1],:]
#    print 'In transform_sequences: X.shape =', X.shape
    y = np.dstack(y).transpose((0, 2, 1)) #.astype('float32')
    wind = np.dstack(wind).transpose((0, 2, 1))
#    print 'In transform_sequences. X.shape, y.shape =', X.shape, y.shape
    return [X, hidden_init, cell_init, cell_mean], y
#    return [X, hidden_init, hidden_init], y

def normalize(X_train, X_valid, model):
    reshaped = False
    if X_train.ndim == 3:
        n_steps = X_train.shape[1]
        X_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
        X_valid = X_valid.reshape((X_valid.shape[0] * X_valid.shape[1], X_valid.shape[2]))
        reshaped = True
    X_all = np.vstack([X_train, X_valid])
    X_mean = X_all.mean(axis=0)
    X_all = X_all - X_mean
    X_stdev = np.sqrt(X_all.var(axis=0))
    
    X_train -= X_mean
    X_train /= X_stdev
    X_valid -= X_mean
    X_valid /= X_stdev
    
    model.X_mean = X_mean
    model.X_stdev = X_stdev
#    print 'In normalize. X_mean.shape =', X_mean.shape
    if reshaped:
        model.X_mean = X_mean
        model.X_stdev = X_stdev
        X_train = X_train.reshape((X_train.shape[0] / n_steps, n_steps, X_train.shape[1]))
        X_valid = X_valid.reshape((X_valid.shape[0] / n_steps, n_steps, X_valid.shape[1]))
    if hasattr(model, 'X_mask'):
        X_train = X_train * model.X_mask
        X_valid = X_valid * model.X_mask
    return X_train, X_valid
     
def normalize_batch(Xb, model):
    X_mean = model.X_mean
    X_stdev = model.X_stdev
#    print 'In normalize_batch. X_mean.shape =', X_mean.shape
    reshaped = False
    if Xb.ndim == 3:
        n_steps = Xb.shape[1]
        Xb = Xb.reshape((Xb.shape[0] * Xb.shape[1], Xb.shape[2]))
        reshaped = True
    Xb -= X_mean
    Xb /= X_stdev
    if reshaped:
        Xb = Xb.reshape((Xb.shape[0] / n_steps, n_steps, Xb.shape[1]))
    if hasattr(model, 'X_mask'):
        Xb = Xb * model.X_mask
    return Xb
    
def parse_data(data):
    assert data.shape[2] == 6 + 3 + 2 + 1 + 1
    gfs = data[:, :, :6]
    date_time = data[:, :, 6:9]
    lonlat = data[:, :, 9:-2]
    pm25_mean = data[:, :, -2:-1]
    pm25 = data[:, :, -1:]
#    print 'In parse_data.', gfs.shape, date_time.shape, lonlat.shape, pm25_mean.shape, pm25.shape
    return gfs, date_time, lonlat, pm25_mean, pm25
    
def build_mlp_dataset(data, pred_range, valid_pct=1./4):
#    data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target 
    train_pct = 1. - valid_pct
    train_data = data[:data.shape[0]*train_pct]
    valid_data = data[data.shape[0]*train_pct:]
    print 'trainset.shape, testset.shape =', train_data.shape, valid_data.shape
#    X_train, y_train = seq2point(trainset, pred_range)
#    X_valid, y_valid = seq2point(validset, pred_range)
    X_train, y_train = decompose_sequences(*(parse_data(train_data) + (pred_range,)))
    X_valid, y_valid = decompose_sequences(*(parse_data(valid_data) + (pred_range,)))
                                           
    X_train, X_valid = normalize(X_train, X_valid)
    print 'X_train.shape, y_train.shape =', X_train.shape, y_train.shape
    return X_train, y_train, X_valid, y_valid

def split_data(data):
    train_stop = 200 * 8 * 75
    valid_stop = 200 * 8 * 97
    train_data = data[:train_stop]
    valid_data = data[train_stop:valid_stop]
    test_data = data[valid_stop:]
    return train_data, valid_data, test_data

def build_lstm_dataset(train_data, valid_data, pred_range, split_fn=split_data, hist_len=3):
#    data = np.copy(data)
#    train_pct = 1. - valid_pct
#    train_data = data[:data.shape[0]*train_pct]
#    valid_data = data[data.shape[0]*train_pct:data.shape[0]]
#    train_data, valid_data, test_data = split_fn(data)
#    print 'trainset.shape, testset.shape =', train_data.shape, valid_data.shape
    X_train, y_train = transform_sequences(*(parse_data(train_data) + (pred_range, hist_len)))
    X_valid, y_valid = transform_sequences(*(parse_data(valid_data) + (pred_range, hist_len)))
#    X_test, y_test = transform_sequences(*(parse_data(test_data) + (pred_range, hist_len)))
    assert type(X_train) == list and type(X_valid) == list
#    if not external_normalize:
#        X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0])
#    else:
#        X_train[0] = normalize_batch(X_train[0])
#        X_valid[0] = normalize_batch(X_valid[0])
#    print 'X_train.shape, y_train.shape =', X_train.shape, y_train.shape
    return X_train, y_train, X_valid, y_valid

def clear_init(X):
#    for i in range(1, len(X)):
#        X[i] *= 0.
    return [X[0], X[1]*0., X[2]*0.]
#    i = np.random.randint(data.shape[0]); plot_example(data[i], [pred[i]], ['rlstm'], model_states=[fgts[i], dxs[i], dhs[i]], state_labels=['a', 'dx', 'dh'], pred_range=[2,42])