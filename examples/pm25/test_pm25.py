import numpy as np
import cPickle
import gzip 
from profilehooks import profile
#from keras.layers.recurrent_xd import LSTM, SimpleRNN
from keras.optimizers import RMSprop
from keras.utils.train_utils import *

def pm25_mean_predict(pm25, gfs, date_time, pm25_mean, pred_range, downsample=1):
    return pm25_mean[pred_range[0]:pred_range[1]]

#@profile
def mlp_predict(pm25, gfs, date_time, pm25_mean, pred_range, downsample=1):
    assert pm25.ndim == 1
    assert gfs.ndim == 2
    assert date_time.ndim == 2
    assert pm25_mean.ndim == 1
    pm25 = pm25.reshape((1, pm25.shape[0], 1))
    gfs = gfs.reshape((1, gfs.shape[0], gfs.shape[1]))
    date_time = date_time.reshape((1, date_time.shape[0], date_time.shape[1]))
    pm25_mean = pm25_mean.reshape((1, pm25_mean.shape[0], 1))
    X, y = decompose_sequences(gfs, date_time, pm25_mean, pm25, pred_range)
    n_steps = pred_range[1] - pred_range[0]
    assert X.shape == (n_steps, 24)
    assert y.shape == (n_steps, 1)
    global model
    if model is None:
        print 'loading mlp...'
        model = load_mlp()
        print 'done.'
    print 'predicting...'
    yp = model.predict_on_batch(normalize_batch(X))
    print 'done.'
    assert yp.shape == (n_steps, 1)
    return yp.flatten()

#@profile
def mlp_predict_batch(pm25, gfs, date_time, pm25_mean, pred_range, downsample=1):
    X, y = decompose_sequences(gfs, date_time, pm25_mean, pm25, pred_range)
    n_steps = pred_range[1] - pred_range[0]
    if mlp_predict_batch.model is None:
        print 'loading mlp...'
        mlp_predict_batch.model = load_mlp()
        print 'done.'
    print 'predicting...'
    yp = mlp_predict_batch.model.predict_on_batch(normalize_batch(X))
    print 'done.'
    pred_pm25 = yp.reshape((n_steps, pm25.shape[0])).T
    return pred_pm25
mlp_predict_batch.model = None

def rlstm_predict_batch(pm25, gfs, date_time, pm25_mean, pred_range, downsample=1):
#    pm25 = pm25 - pm25_mean
    X, y = transform_sequences(gfs, date_time, pm25_mean, pm25, pred_range)
    n_steps = pred_range[1] - pred_range[0]
    if rlstm_predict_batch.model is None:
        print 'loading rlstm...'
        rlstm_predict_batch.model = load_rlstm()
        print 'done.'
#    print 'predicting...'
    X[0] = normalize_batch(X[0])
    yp = rlstm_predict_batch.model.predict_on_batch(X)
#    print 'done.'
    assert yp.ndim == 3 and yp.shape[2] == 1
    pred_pm25 = yp.reshape((yp.shape[0], yp.shape[1]))
#    pred_pm25 += pm25_mean[:, pred_range[0]:pred_range[1], 0]
    return pred_pm25
rlstm_predict_batch.model = None

def predict_all(data, predict_fn, pred_range=[2, 42]):
    predictions = []
    for i in range(data.shape[0]):
        pm25 = data[i, :, -1]
        gfs = data[i, :, :6]
        date_time = data[i, :, 6:-2]
        pm25_mean = data[i, :, -2]
        pred_pm25 = predict_fn(pm25, gfs, date_time, pm25_mean, pred_range)
        predictions.append(pred_pm25)
    predictions = np.array(predictions)
    return predictions

def predict_all_batch(data, predict_fn, pred_range=[2, 42], batch_size=1024):
    predictions = []
    for i in range(0, data.shape[0], batch_size):
        start = i
        stop = min(data.shape[0], i + batch_size)
        pm25 = data[start:stop, :, -1:]
        gfs = data[start:stop, :, :6]
        date_time = data[start:stop, :, 6:-2]
        pm25_mean = data[start:stop, :, -2:-1]
        pred_pm25 = predict_fn(pm25, gfs, date_time, pm25_mean, pred_range)
        predictions.append(pred_pm25)
    predictions = np.vstack(predictions)
    return predictions

def mean_square_error(predictions, targets):
    return np.square(predictions - targets).mean(axis=0)

def absolute_percent_error(predictions, targets, targets_mean):
    return (np.abs(predictions - targets) / np.abs(targets_mean)).mean(axis=0)
        
def absolute_error(predictions, targets):
    return np.abs(predictions - targets).mean(axis=0)

threshold = 80
    
def misclass_error(predictions, targets):
    return ((predictions >= threshold) != (targets >= threshold)).mean(axis=0)

def downsample(sequences, pool_size):
    assert sequences.ndim == 2
    assert sequences.shape[1] % pool_size == 0
    return sequences.reshape((sequences.shape[0], sequences.shape[1] / pool_size, pool_size)).max(axis=2) 

def detection_error(predictions, targets, targets_mean=None, pool_size=1):
    if targets_mean is not None:
        predictions = predictions + targets_mean
        targets = targets + targets_mean
    if pool_size != 1:
        predictions = downsample(predictions, pool_size)
        targets = downsample(targets, pool_size)
    alarm = (predictions >= threshold).mean(axis=0)
    occur = (targets >= threshold).mean(axis=0)
    hit = ((predictions >= threshold) & (targets >= threshold)).mean(axis=0)
    pod = hit / occur
    far = 1. - hit / alarm
    csi = hit / (occur + alarm - hit)
    return pod, far, csi

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

def transform_sequences(gfs, date_time, pm25_mean, pm25, pred_range, hist_len=3):
    X = []
    y = []
    for i in range(pred_range[0], pred_range[1]):
        if i - hist_len + 1 >= 0:
            recent_gfs = gfs[:,i-hist_len+1:i+1,:]
        else:
            assert False
            print 'shapes:', np.zeros((gfs.shape[0], hist_len-i-1, gfs.shape[2])).shape, gfs[:,0:i+1,:].shape
            recent_gfs = np.concatenate((np.zeros((gfs.shape[0], hist_len-i-1, gfs.shape[2])), gfs[:,0:i+1,:]), axis=1)
#        recent_gfs = delta(recent_gfs)
        recent_gfs = recent_gfs.reshape((recent_gfs.shape[0], -1))
        current_date_time = date_time[:,i,:]
        current_pm25_mean = pm25_mean[:,i,:]
        step = np.ones((pm25.shape[0],1)) * (i - pred_range[0] + 1)
        Xi = np.hstack([recent_gfs, current_date_time, current_pm25_mean])
#        Xi = np.hstack([recent_gfs, current_pm25_mean])
        yi = pm25[:,i,:]
        X.append(Xi)
        y.append(yi)
    init_pm25 = pm25[:,pred_range[0]-1,:]
    init_gfs = gfs[:,:pred_range[0],:].reshape((gfs.shape[0], -1))
#    X_init = np.hstack([init_pm25, init_gfs])
    X_init = init_pm25
    X = np.dstack(X).transpose((0, 2, 1))
    y = np.dstack(y).transpose((0, 2, 1))
#    print 'X.shape, X_init.shape, y.shape =', X.shape, X_init.shape, y.shape
    return [X, X_init, X_init], y

def normalize(X_train, X_test):
    reshaped = False
    if X_train.ndim == 3:
        n_steps = X_train.shape[1]
        X_train = X_train.reshape((X_train.shape[0] * X_train.shape[1], X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1], X_test.shape[2]))
        reshaped = True
    X_mean = X_train.mean(axis=0)
#    print 'X_mean =', X_mean
    X_train -= X_mean
    X_stdev = np.sqrt(X_train.var(axis=0))
    X_train /= X_stdev
    X_test -= X_mean
    X_test /= X_stdev
    if not reshaped:
        np.save('mlp_X_mean.npy', X_mean)
        np.save('mlp_X_stdev.npy', X_stdev)
    else:
        np.save('rlstm_X_mean.npy', X_mean)
        np.save('rlstm_X_stdev.npy', X_stdev)
        X_train = X_train.reshape((X_train.shape[0] / n_steps, n_steps, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0] / n_steps, n_steps, X_test.shape[1]))
    return X_train.astype('float32'), X_test.astype('float32')
     
def normalize_batch(Xb):
    reshaped = False
    if Xb.ndim == 3:
        X_mean = np.load('rlstm_X_mean.npy')
        X_stdev = np.load('rlstm_X_stdev.npy')
        n_steps = Xb.shape[1]
        Xb = Xb.reshape((Xb.shape[0] * Xb.shape[1], Xb.shape[2]))
        reshaped = True
    else:
        X_mean = np.load('mlp_X_mean.npy')
        X_stdev = np.load('mlp_X_stdev.npy')
    Xb -= X_mean
    Xb /= X_stdev
    if reshaped:
        Xb = Xb.reshape((Xb.shape[0] / n_steps, n_steps, Xb.shape[1]))
    return Xb
    
def parse_data(data):
    gfs = data[:, :, :6]
    date_time = data[:, :, 6:-2]
    pm25_mean = data[:, :, -2:-1]
    pm25 = data[:, :, -1:]
    return gfs, date_time, pm25_mean, pm25
    
def build_mlp_dataset(data, pred_range=[2,42], valid_pct=1./4):
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

def build_lstm_dataset(data, pred_range=[2,42], valid_pct=1./4, hist_len=3):
#    data = np.copy(data)
    train_pct = 1. - valid_pct
    train_data = data[:data.shape[0]*train_pct]
    valid_data = data[data.shape[0]*train_pct:]
    print 'trainset.shape, testset.shape =', train_data.shape, valid_data.shape
    X_train, y_train = transform_sequences(*(parse_data(train_data) + (pred_range, hist_len)))
    X_valid, y_valid = transform_sequences(*(parse_data(valid_data) + (pred_range, hist_len)))
    assert type(X_train) == list and type(X_valid) == list                        
    X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0])
#    print 'X_train.shape, y_train.shape =', X_train.shape, y_train.shape
    return X_train, y_train, X_valid, y_valid

def clear_init(X):
#    for i in range(1, len(X)):
#        X[i] *= 0.
    return [X[0], X[1]*0., X[2]*0.]

def test_model(model):
    rlstm_predict_batch.model = model
    targets = data[data.shape[0]*3./4:, 2:42, -1]
    targets_mean = data[data.shape[0]*3./4:, 2:42, -2]
    pred = predict_all_batch(data[data.shape[0]*3./4:], rlstm_predict_batch)
    mse = mean_square_error(pred, targets).mean()
    res = detection_error(pred, targets, targets_mean=targets_mean, pool_size=1)
    pod1 = res[0].mean()
    far1 = res[1].mean()
    csi1 = res[2].mean()
    res = detection_error(pred, targets, targets_mean=targets_mean, pool_size=4)
    pod4 = res[0].mean()
    far4 = res[1].mean()
    csi4 = res[2].mean()
    res = detection_error(pred, targets, targets_mean=targets_mean, pool_size=8)
    pod8 = res[0].mean()
    far8 = res[1].mean()
    csi8 = res[2].mean()
    print '%20s%8.1f%10.2f%6.2f%6.2f%10.2f%6.2f%6.2f%10.2f%6.2f%6.2f' % (model.name, mse, 
                                                                  pod1, far1, csi1, 
                                                                  pod4, far4, csi4,
                                                                  pod8, far8, csi8)
#    return mse, pod1, far1, csi1, pod4, far4, csi4, pod8, far8, csi8
    
f = gzip.open('/home/xd/data/pm25data/forXiaodaDataset20151022_t100p100.pkl.gz', 'rb')   
data = cPickle.load(f)
data[:,:,-2:] -= 80
data[:,:,2] = np.sqrt(data[:,:,2]**2 + data[:,:,3]**2)
data[:,:,3] = data[:,:,2]
data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target 
f.close()

#X_train, y_train, X_valid, y_valid = build_mlp_dataset(data)
#mlp = build_mlp(X_train.shape[-1], y_train.shape[-1], 40, 40)
#mlp.name = 'mlp'
#train(X_train, y_train, X_valid, y_valid, mlp, batch_size=4096)

X_train, y_train, X_valid, y_valid = build_lstm_dataset(data, hist_len=3)
for i in range(10):
    name = 'rlstm2h' + str(i)
    rlstm = build_reduced_lstm(X_train[0].shape[-1], h0_dim=80, name=name)
    rlstm.name = name
    batch_size = 128
    print '\ntraining', rlstm.name
    train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=batch_size)

#rlstms = []
#for i in range(10):
#    rlstm = load_rlstm('rlstm'+str(i))
#    rlstms.append(rlstm)
#    
#for rlstm in rlstms:
#    test_model(rlstm)
#    print rlstm.layers[-1].U_c.get_value(), rlstm.layers[-1].U_f.get_value(), rlstm.layers[-1].b_f.get_value()
#pred_mlp = predict_all_batch(data[data.shape[0]*3./4:], mlp_predict_batch)  
#pred = predict_all_batch(data[data.shape[0]*3./4:], rlstm_predict_batch)