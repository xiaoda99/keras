import numpy as np
import pylab as plt
import matplotlib.gridspec as gridspec
import cPickle
import gzip 
from profilehooks import profile
from keras.layers.recurrent_xd import ReducedLSTM, ReducedLSTM2, ReducedLSTM3, ReducedLSTMA, ReducedLSTMB
from keras.optimizers import RMSprop
from keras.utils.train_utils import *
from data import load_data, load_data2, segment_data
from errors import *

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
#    print 'X[0].shape =', X[0].shape
    yp = rlstm_predict_batch.model.predict_on_batch(X)
    forgets, increments, delta, delta_x, delta_h = rlstm_predict_batch.model.monitor_on_batch(X)
#    print 'done.'
    assert yp.ndim == 3 and yp.shape[2] == 1
    pred_pm25 = yp.reshape((yp.shape[0], yp.shape[1]))
#    pred_pm25 += pm25_mean[:, pred_range[0]:pred_range[1], 0]
    forgets = forgets.reshape((forgets.shape[0], forgets.shape[1]))
    increments = increments.reshape((increments.shape[0], increments.shape[1]))
    delta = delta.reshape((delta.shape[0], delta.shape[1]))
    delta_x = delta_x.reshape((delta_x.shape[0], delta_x.shape[1]))
    delta_h = delta_h.reshape((delta_h.shape[0], delta_h.shape[1]))
    return pred_pm25, forgets, increments, delta, delta_x, delta_h

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
    fgts = []
    incs = []
    ds = []
    dxs = []
    dhs = []
    for i in range(0, data.shape[0], batch_size):
        start = i
        stop = min(data.shape[0], i + batch_size)
        pm25 = data[start:stop, :, -1:]
        gfs = data[start:stop, :, :6]
        date_time = data[start:stop, :, 6:-2]
        pm25_mean = data[start:stop, :, -2:-1]
        pred_pm25, forgets, increments, delta, delta_x, delta_h = predict_fn(pm25, gfs, date_time, pm25_mean, pred_range)
        predictions.append(pred_pm25)
        fgts.append(forgets)
        incs.append(increments)
        ds.append(delta)
        dxs.append(delta_x)
        dhs.append(delta_h)
    predictions = np.vstack(predictions)
    fgts = np.vstack(fgts)
    incs = np.vstack(incs)
    ds = np.vstack(ds)
    dxs = np.vstack(dxs)
    dhs = np.vstack(dhs)
    return predictions, fgts, incs, ds, dxs, dhs

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
#    gfs = np.copy(gfs)
#    gfs[:,:,0] /= 10. # replace temperature with wind speed 0~20
#    gfs[:,:,1] /= 100. # humidity 20~100
#    gfs[:,:,2] /= 10. # wind_x -15~15
#    gfs[:,:,3] /= 10. # wind_y -15~15
#    gfs[:,:,4] /= 0.001  # rain 0~0.0018
#    gfs[:,:,5] /= 100.  # cloud 0~100
    X = []
    y = []
    for i in range(pred_range[0], pred_range[1]):
        if i - hist_len + 1 >= 0:
            recent_gfs = gfs[:,i-hist_len+1:i+1,0:]  # remove temperature feature 
        else:
            assert False
            print 'shapes:', np.zeros((gfs.shape[0], hist_len-i-1, gfs.shape[2])).shape, gfs[:,0:i+1,:].shape
            recent_gfs = np.concatenate((np.zeros((gfs.shape[0], hist_len-i-1, gfs.shape[2])), gfs[:,0:i+1,:]), axis=1)
#        recent_gfs = delta(recent_gfs)
        recent_gfs = recent_gfs.reshape((recent_gfs.shape[0], -1))
        current_date_time = date_time[:,i,:-1] # remove day of year feature
        current_pm25_mean = pm25_mean[:,i,:]
        step = np.ones((pm25.shape[0],1)) * (i - pred_range[0] + 1)
#        Xi = np.hstack([recent_gfs, current_date_time])
        Xi = np.hstack([recent_gfs, current_date_time, current_pm25_mean])
        yi = pm25[:,i,:]
        X.append(Xi)
        y.append(yi)
    init_pm25 = pm25[:,pred_range[0]-1,:]
    init_gfs = gfs[:,:pred_range[0],:].reshape((gfs.shape[0], -1))
#    X_init = np.hstack([init_pm25, init_gfs])
    hidden_init = init_pm25 
    cell_init = init_pm25 + pm25_mean[:,pred_range[0]-1,:]
    cell_mean = pm25_mean[:,pred_range[0]:pred_range[1],:]
    X = np.dstack(X).transpose((0, 2, 1)) #.astype('float32')
    y = np.dstack(y).transpose((0, 2, 1)) #.astype('float32')
#    print 'X.shape, X_init.shape, y.shape =', X.shape, X_init.shape, y.shape
    return [X, hidden_init, cell_init, cell_mean], y
#    return [X, hidden_init, hidden_init], y

def transform_sequences_old(gfs, date_time, pm25_mean, pm25, pred_range, hist_len=3):
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
    hidden_init = init_pm25 
    cell_init = init_pm25 + pm25_mean[:,pred_range[0]-1,:]
    cell_mean = pm25_mean[:,pred_range[0]:pred_range[1],:]
    X = np.dstack(X).transpose((0, 2, 1)) #.astype('float32')
    y = np.dstack(y).transpose((0, 2, 1)) #.astype('float32')
#    print 'X.shape, X_init.shape, y.shape =', X.shape, X_init.shape, y.shape
    return [X, hidden_init, cell_init, cell_mean], y
#    return [X, hidden_init, hidden_init], y

def normalize(X_train, X_valid):
#    return X_train, X_valid
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
    if not reshaped:
        np.save('mlp_X_mean.npy', X_mean)
        np.save('mlp_X_stdev.npy', X_stdev)
    else:
        np.save('X_mean.npy', X_mean)
        np.save('X_stdev.npy', X_stdev)
        X_train = X_train.reshape((X_train.shape[0] / n_steps, n_steps, X_train.shape[1]))
        X_valid = X_valid.reshape((X_valid.shape[0] / n_steps, n_steps, X_valid.shape[1]))
    return X_train, X_valid
     
def normalize_batch(Xb, base_dir='/home/xd/projects/keras/examples/pm25/'):
#    return Xb
    reshaped = False
    if Xb.ndim == 3:
        X_mean = np.load(base_dir + 'X_mean.npy')
        X_stdev = np.load(base_dir + 'X_stdev.npy')
        n_steps = Xb.shape[1]
        Xb = Xb.reshape((Xb.shape[0] * Xb.shape[1], Xb.shape[2]))
        reshaped = True
    else:
        X_mean = np.load(base_dir + 'mlp_X_mean.npy')
        X_stdev = np.load(base_dir + 'mlp_X_stdev.npy')
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

def split_data(data):
    train_stop = 200 * 8 * 75
    valid_stop = 200 * 8 * 97
    train_data = data[:train_stop]
    valid_data = data[train_stop:valid_stop]
    test_data = data[valid_stop:]
    return train_data, valid_data, test_data

def build_lstm_dataset(train_data, valid_data, pred_range=[2,42], split_fn=split_data, hist_len=3, external_normalize=False):
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
    if not external_normalize:
        X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0])
    else:
        X_train[0] = normalize_batch(X_train[0])
        X_valid[0] = normalize_batch(X_valid[0])
#    print 'X_train.shape, y_train.shape =', X_train.shape, y_train.shape
    return X_train, y_train, X_valid, y_valid

def clear_init(X):
#    for i in range(1, len(X)):
#        X[i] *= 0.
    return [X[0], X[1]*0., X[2]*0.]

def test_model(model, dataset='test', split_fn=split_data, show_details=True):
#    print dataset
    i = {'train':0, 'valid':1, 'test':2}[dataset]
    data = model.data[i]
    targets = data[:, 2:42, -1]
    targets_mean = data[:, 2:42, -2]
    
    if targets.min() >= 0:
        targets_mean = None
        
    rlstm_predict_batch.model = model
    pred, fgts, incs, ds, dxs, dhs = predict_all_batch(data, rlstm_predict_batch)
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
    if show_details:
#        print 'abs_err ='
#        print absolute_error(pred, targets)
        print 'forget =', fgts.mean()
#        print fgts.mean(axis=0)
#        print 'pred ='
#        print np.abs(pred).mean(axis=0)
#        print 'delta ='
#        print np.abs(ds).mean(axis=0)
        print 'delta_x =', np.abs(dxs).mean()
#        print np.abs(dxs).mean(axis=0)
        print 'delta_h =', np.abs(dhs).mean()
#        print np.abs(dhs).mean(axis=0)
#        print 'fgts.mean() =', fgts.mean(), 'fgts.min() =', fgts.min()
        print 'delta mean, abs_mean, abs_mean+, abs_mean-:', dxs.mean(), np.abs(dxs).mean(), np.abs(dxs[dxs>0]).mean(), np.abs(dxs[dxs<0]).mean()
        print 'U_c =', model.layers[-1].U_c.get_value(), 'U_f =', model.layers[-1].U_f.get_value(), 'b_f =', model.layers[-1].b_f.get_value()
    
def plot_example(data, predictions, model_labels, feature_indices=[2,], feature_labels=['wind speed'], 
                 model_states=[], state_labels=[], pred_range=[2,42]):
    assert len(predictions) == len(model_labels)
    assert len(feature_indices) == len(feature_labels)
    assert len(model_states) == len(state_labels)
    n_subplots = 1 + len(feature_indices) + len(model_states)  
    gs = gridspec.GridSpec(n_subplots, 1, height_ratios=[2,]*1 + [1]*(len(feature_indices) + len(model_states)))
    i = 0
    
    ax = plt.subplot(gs[i])
    i += 1
    pm25_mean = data[:,-2]
    pm25 = data[:,-1]
    pm25 = pm25 + pm25_mean
    predictions = [pred + pm25_mean[pred_range[0] : pred_range[1]] for pred in predictions]
    plt.plot(pm25[pred_range[0] : pred_range[1]], label='pm25')
    plt.plot(pm25_mean[pred_range[0] : pred_range[1]], '--', label='mean')
#    print 'pm25:',  pm25[pred_range[0] : pred_range[1]]
#    print 'pm25_mean:', pm25_mean[pred_range[0] : pred_range[1]]
    for pred, label in zip(predictions, model_labels):
        plt.plot(pred, label=label)
    plt.legend(loc='upper right')
    
    for state, label in zip(model_states, state_labels):
        ax = plt.subplot(gs[i])
        i += 1
        plt.plot(state, label=label)
        plt.plot(np.zeros_like(state), color='k')
        plt.legend(loc='upper right')
        
    for feature_idx, label in zip(feature_indices, feature_labels):
        ax = plt.subplot(gs[i])
        i += 1
        feature = data[:,feature_idx]
        plt.plot(feature[pred_range[0] : pred_range[1]], label=label)
        plt.legend(loc='upper right')
        
    plt.show()

def filter_data(data):
    pm25 = data[:,:,-1] + data[:,:,-2]
    cond = (pm25[:,:].max(axis=1) > 160) & (pm25[:,:].min(axis=1) < 60)
    return data[cond]
     
if __name__ == '__main__':
#    f = gzip.open('/home/xd/data/pm25data/forXiaodaDataset20151207_t100p100.pkl.gz', 'rb')   
#    data = cPickle.load(f)
#    data[:,:,-2:] -= 80
#    data[:,:,2] = np.sqrt(data[:,:,2]**2 + data[:,:,3]**2)
#    data[:,:,3] = data[:,:,2]
##    data[:,:,1:4] = np.random.randn(data.shape[0], data.shape[1], 3)
#    data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target
#    f.close()
#    train_data, valid_data, test_data = split_data(data)
    
#    train_data, valid_data, test_data = load_data2(segment=True)
#    X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, hist_len=3)  
    
    train_data, valid_data, test_data = load_data2(stations=[u'1003A', u'1004A',u'1005A', u'1006A', u'1007A', u'1011A'], segment=True)
#    X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, hist_len=3, 
#                                                            external_normalize=True)
    
#    for i in range(10):
#        name = 'bj_extnorm_mean'
#        rlstm = build_reduced_lstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, rec_layer_type=ReducedLSTMA, name=name)
#        rlstm.name = name + str(i)
#        rlstm.data = [train_data, valid_data, test_data]
#        print '\ntraining', rlstm.name
#        train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=64)
#    
#    X_train[0][:,:,-1] = 0 # disable pm25_mean feature
#    X_valid[0][:,:,-1] = 0
#    for i in range(10):
#        name = 'bj_extnorm'
#        rlstm = build_reduced_lstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, rec_layer_type=ReducedLSTMA, name=name)
#        rlstm.name = name + str(i)
#        rlstm.data = [train_data, valid_data, test_data]
#        print '\ntraining', rlstm.name
#        train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=64)
#     
    X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, hist_len=3, 
                                                            external_normalize=False)
#    for i in range(10):
#        name = 'bj_mean'
#        rlstm = build_reduced_lstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, rec_layer_type=ReducedLSTMA, name=name)
#        rlstm.name = name + str(i)
#        rlstm.data = [train_data, valid_data, test_data]
#        print '\ntraining', rlstm.name
#        train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=64)
    
    X_train[0][:,:,-1] = 0 # disable pm25_mean feature
    X_valid[0][:,:,-1] = 0
    for i in range(10):
        name = 'test'
        rlstm = build_reduced_lstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, rec_layer_type=ReducedLSTMA, base_name=name)
        rlstm.name = name + str(i)
        rlstm.data = [train_data, valid_data, test_data]
        print '\ntraining', rlstm.name
        train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=64)
              
    train_data, valid_data, test_data = load_data2(segment=True)
    X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, hist_len=3)  
    
    train_data, valid_data, test_data = load_data2(stations=[u'1003A', u'1004A',u'1005A', u'1006A', u'1007A', u'1011A'], segment=True)       
    name = 'bj_extnorm_mean'    
    rlstm = model_from_yaml(open(name + '.yaml').read())
    for i in range(10):
        rlstm.load_weights(name + str(i) + '_weights.hdf5')
        rlstm.name = name + str(i)
        rlstm.data = [train_data, valid_data, test_data]
        test_model(rlstm, dataset='train', show_details=False)
        test_model(rlstm, dataset='valid', show_details=False)    
    name = 'bj_extnorm'    
    rlstm = model_from_yaml(open(name + '.yaml').read())
    for i in range(10):
        rlstm.load_weights(name + str(i) + '_weights.hdf5')
        rlstm.name = name + str(i)
        rlstm.data = [train_data, valid_data, test_data]
        test_model(rlstm, dataset='train', show_details=False)
        test_model(rlstm, dataset='valid', show_details=False)
        
    X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, hist_len=3, 
                                                            external_normalize=False)
    name = 'bj_mean'    
    rlstm = model_from_yaml(open(name + '.yaml').read())
    for i in range(10):
        rlstm.load_weights(name + str(i) + '_weights.hdf5')
        rlstm.name = name + str(i)
        rlstm.data = [train_data, valid_data, test_data]
        test_model(rlstm, dataset='train', show_details=False)
        test_model(rlstm, dataset='valid', show_details=False)    
    name = 'bj'    
    rlstm = model_from_yaml(open(name + '.yaml').read())
    for i in range(10):
        rlstm.load_weights(name + str(i) + '_weights.hdf5')
        rlstm.name = name + str(i)
        rlstm.data = [train_data, valid_data, test_data]
        test_model(rlstm, dataset='train', show_details=False)
        test_model(rlstm, dataset='valid', show_details=False)
        
    #for rlstm in rlstms:
    #    test_model(rlstm)
    #    print rlstm.layers[-1].U_c.get_value(), rlstm.layers[-1].U_f.get_value(), rlstm.layers[-1].b_f.get_value()
    #pred_mlp = predict_all_batch(data[data.shape[0]*3./4:], mlp_predict_batch)  
    #pred = predict_all_batch(data[data.shape[0]*3./4:], rlstm_predict_batch)
    
#y = y_valid[((y_valid[:,:,0].min(axis=1) < 60) & (y_valid[:,:,0].max(axis=1) > 100)), :, 0]
#i = np.random.randint(y.shape[0]); yp = rlstm.predict_on_batch([X_valid[0][i:i+1], X_valid[1][i:i+1], X_valid[2][i:i+1]])[0,:,0]; plt.plot(yp, label='yp'); plt.plot(y[i], label='y'); plt.legend(); plt.show()

#    rlstm = model_from_yaml(open('rlstm_2h.yaml').read())
#    rlstm.load_weights('rlstm_2h0_weights.hdf5')
#    rlstm.name = 'rlstm_2h'
#    rlstm.data = [train_data, valid_data, test_data]
#    test_model(rlstm, dataset='valid', show_details=False)
#    rlstm_predict_batch.model = rlstm
#    data = filter_data(valid_data)
#    pred, fgts, incs, ds, dxs, dhs = predict_all_batch(data, rlstm_predict_batch)
#    i = np.random.randint(data.shape[0]); plot_example(data[i], [pred[i]], ['rlstm'], model_states=[fgts[i], dxs[i], dhs[i]], state_labels=['a', 'dx', 'dh'], pred_range=[2,42])