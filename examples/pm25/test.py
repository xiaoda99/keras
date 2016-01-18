import numpy as np
import pylab as plt
import matplotlib.gridspec as gridspec
import cPickle
import gzip 
from profilehooks import profile
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

def rlstm_predict_batch(gfs, date_time, lonlat, pm25_mean, pm25, pred_range, downsample=1):
#    pm25 = pm25 - pm25_mean
    X, y = transform_sequences(gfs, date_time, lonlat, pm25_mean, pm25, pred_range)
#    print 'In rlstm_predict_batch. X[0] ='
#    print (X[0][0,:,[3,6,9,12,15,16,17,18,19,20]])
    n_steps = pred_range[1] - pred_range[0]
    if rlstm_predict_batch.model is None:
        print 'loading rlstm...'
        rlstm_predict_batch.model = load_rlstm()
        print 'done.'
#    print 'predicting...'
    if hasattr(rlstm_predict_batch.model, 'X_mean'):
        X[0] = normalize_batch(X[0], rlstm_predict_batch.model)
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

def predict_all(data, predict_fn, pred_range):
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

def predict_all_batch(data, predict_fn, pred_range=pred_range, batch_size=1024):
    predictions = []
    fgts = []
    incs = []
    ds = []
    dxs = []
    dhs = []
    for i in range(0, data.shape[0], batch_size):
        start = i
        stop = min(data.shape[0], i + batch_size)
#        pm25 = data[start:stop, :, -1:]
#        gfs = data[start:stop, :, :6]
#        date_time = data[start:stop, :, 6:-2]
#        pm25_mean = data[start:stop, :, -2:-1]
    
        pred_pm25, forgets, increments, delta, delta_x, delta_h = \
            predict_fn(*(parse_data(data[start:stop]) + (pred_range,)))
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

def test_model(model, dataset='test', show_details=True):
#    print dataset
    i = {'train':0, 'valid':1, 'test':1}[dataset]
    data = model.data[i]
    targets = data[:, pred_range[0]:pred_range[1], -1]
    targets_mean = data[:, pred_range[0]:pred_range[1], -2]
    
    if targets.min() >= 0:
        targets_mean = None
        
    rlstm_predict_batch.model = model
    pred, fgts, incx, fgtx, dxs, dhs = predict_all_batch(data, rlstm_predict_batch)
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
    print '%20s%10.4f%10.2f%6.2f%6.2f%10.2f%6.2f%6.2f%10.2f%6.2f%6.2f' % (model.name, mse, 
                                                                  pod1, far1, csi1, 
                                                                  pod4, far4, csi4,
                                                                  pod8, far8, csi8)
    if show_details:
        print 'forget mean min:', fgts.mean(), fgts.min()
#        print 'incx.max(), incx.min(), incx.mean()', incx.max(), incx.min(), incx.mean()
#        print 'fgtx.max(), fgtx.min(), fgtx.mean()', fgtx.max(), fgtx.min(), fgtx.mean()
#        print 'delta_x =', np.abs(dxs).mean(), 'delta_h =', np.abs(dhs).mean()
        print '+dx_pct, abs(+dx).mean, abs(-dx).mean:', (dxs > 0).mean(), np.abs(dxs[dxs>0]).mean(), np.abs(dxs[dxs<0]).mean()
        layer = model.get_rlstm_layer()
        W_c =layer.W_c.get_value()
        W_f =layer.W_f.get_value()
        print 'U_c =',layer.U_c.get_value(), 'U_f =',layer.U_f.get_value(), 'b_c =',layer.b_c.get_value(), 'b_f =',layer.b_f.get_value()
#        print 'W_c max, min, mean, abs_mean:', W_c.max(), W_c.min(), W_c.mean(), np.abs(W_c).mean()
#        print 'W_f max, min, mean, abs_mean:', W_f.max(), W_f.min(), W_f.mean(), np.abs(W_f).mean()
    return [mse, pod1, far1, csi1, pod4, far4, csi4, pod8, far8, csi8, 
            fgts.mean(), fgts.min(), (dxs > 0).mean(), np.abs(dxs[dxs>0]).mean(), np.abs(dxs[dxs<0]).mean()]
          
    
def plot_example(data, predictions, model_labels, feature_indices=[0,], feature_labels=['wind speed',], 
                 model_states=[], state_labels=[], pred_range=pred_range):
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
    plt.legend(loc='upper left')
    
    for state, label in zip(model_states, state_labels):
        ax = plt.subplot(gs[i])
        i += 1
        plt.plot(state, label=label)
        plt.plot(np.zeros_like(state), color='k')
        plt.legend(loc='upper left')
        
    for feature_idx, label in zip(feature_indices, feature_labels):
        ax = plt.subplot(gs[i])
        i += 1
        if feature_idx == 0:
            u = data[:,2]
            v = data[:,3]
            feature = np.sqrt(u**2 + v**2)
        else: 
            feature = data[:,feature_idx]
        plt.plot(feature[pred_range[0] : pred_range[1]], label=label)
        plt.legend(loc='upper left')
        
    plt.show()

def filter_data(data):
    pm25 = data[:,:,-1] + data[:,:,-2]
    cond = (pm25[:,:].max(axis=1) > 160) & (pm25[:,:].min(axis=1) < 60)
    return data[cond]
     
def load_rlstm(base_name, i=None):
    rlstm = model_from_yaml(open(base_name + '.yaml').read())
    rlstm.base_name = base_name
    rlstm.name = base_name
    if i is not None:
        rlstm.name = rlstm.name + str(i)
    rlstm.load_weights(rlstm.name + '_weights.hdf5')
    rlstm.load_normalization_info(rlstm.name + '_norm_info.pkl')
    return rlstm

starttime = '20150916'
endtime = '20160116'

def test_cities(cities, repeat=1):
    beijing_only = True
    
    for city in cities:
        for i in range(repeat):
            train_data, valid_data = load_data3(stations=city2stations[city], 
                                                starttime=starttime, endtime=endtime,
                                                filter=(not beijing_only), normalize_target=False)
            X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, pred_range=pred_range, hist_len=3)
            print 'X_train[0].shape =', X_train[0].shape
            name = city
            rlstm = build_rlstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, 
                                       rec_layer_init='zero', fix_b_f=True, base_name=name,
                                       add_input_noise=beijing_only, add_target_noise=False)
            rlstm.name = name + str(i)
            rlstm.data = [train_data, valid_data]
            
            rlstm.X_mask = np.ones((X_train[0].shape[-1],), dtype='int')
            rlstm.X_mask[-1:] = not beijing_only  # pm25 mean
            
            print '\ntraining', rlstm.name
            X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0], rlstm)
            rlstm.save_normalization_info(name + '_norm_info.pkl')
            batch_size = (1 + (not beijing_only)) * 64
#            patience = (1 + int(beijing_only)) * 10
            patience = 10
            train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=batch_size, patience=patience, nb_epoch=300)
      
    for city in cities:
        for i in range(repeat):
            name = city
            rlstm = model_from_yaml(open(name + '.yaml').read())
            rlstm.name = name + str(i)
            rlstm.load_normalization_info(name + '_norm_info.pkl')
            rlstm.load_weights(rlstm.name + '_weights.hdf5')
            
            train_data, valid_data = load_data3(stations=city2stations[city], 
                                                starttime=starttime, endtime=endtime)
            rlstm.data = [train_data, valid_data]
            print '\n' + rlstm.name
            test_model(rlstm, dataset='train', show_details=False)
            test_model(rlstm, dataset='valid', show_details=True)
     
def test_areas(areas, repeat=1):
    beijing_only = False
    
    for area in areas:
        for i in range(repeat):
            train_data, valid_data = load_data3(lon_range=area2lonlat[area][0], lat_range=area2lonlat[area][1],
                                                starttime=starttime, endtime=endtime,
                                                filter=(not beijing_only))
            X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, pred_range=pred_range, hist_len=3)
            print 'X_train[0].shape =', X_train[0].shape
            name = area
            rlstm = build_rlstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, 
                                       rec_layer_init='zero', fix_b_f=False, base_name=name,
                                       add_input_noise=beijing_only, add_target_noise=False)
            rlstm.name = name + str(i)
            rlstm.data = [train_data, valid_data]
            
            rlstm.X_mask = np.ones((X_train[0].shape[-1],), dtype='int')
            rlstm.X_mask[-1:] = not beijing_only  # pm25 mean
            
            print '\ntraining', rlstm.name
            X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0], rlstm)
            rlstm.save_normalization_info(name + '_norm_info.pkl')
            batch_size = (1 + (not beijing_only)) * 64
#            patience = (1 + int(beijing_only)) * 10
            patience = 10
            train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=batch_size, patience=patience, nb_epoch=300)
      
    for area in areas:
        for i in range(repeat):
            name = area
            rlstm = model_from_yaml(open(name + '.yaml').read())
            rlstm.name = name + str(i)
            rlstm.load_normalization_info(name + '_norm_info.pkl')
            rlstm.load_weights(rlstm.name + '_weights.hdf5')
            
            train_data, valid_data = load_data3(lon_range=area2lonlat[area][0], lat_range=area2lonlat[area][1],
                                                starttime=starttime, endtime=endtime
                                                )
            rlstm.data = [train_data, valid_data]
            print '\n' + rlstm.name
            test_model(rlstm, dataset='train', show_details=False)
            test_model(rlstm, dataset='valid', show_details=True)
            
def test_area_on_cities(area, i, cities):  
    name = area      
    rlstm = model_from_yaml(open(name + '.yaml').read())
    rlstm.name = name + str(i)
    rlstm.load_normalization_info(name + '_norm_info.pkl')
    rlstm.load_weights(rlstm.name + '_weights.hdf5')
    print '\n In' + area
    for city in cities:
        train_data, valid_data = load_data3(stations=city2stations[city], 
                                            starttime=starttime, endtime=endtime,
                                            )
        rlstm.data = [train_data, valid_data]
        print '\n' + city
        test_model(rlstm, dataset='train', show_details=False)
        test_model(rlstm, dataset='valid', show_details=True)
            
if __name__ == '__main__':
#    cities = city2stations.keys()
#    cities = ['haerbin']
#    test_cities(cities, repeat=1)
    
#    areas = area2lonlat.keys()
#    test_areas(areas, repeat=3)
    
    test_area_on_cities('dongbei', 0, dongbei_cities)
    test_area_on_cities('huabei', 0, huabei_cities)
    test_area_on_cities('xibei', 0, xibei_cities)
    test_area_on_cities('huadong', 0, huadong_cities)
    test_area_on_cities('huaxi', 0, huaxi_cities)
    test_area_on_cities('huanan', 0, huanan_cities)

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