import numpy as np
import pylab as plt
import matplotlib.gridspec as gridspec
import cPickle
import gzip 
from profilehooks import profile
from keras.layers.recurrent_xd import RLSTM, ReducedLSTM, ReducedLSTM2, ReducedLSTM3, ReducedLSTMA, ReducedLSTMB
from keras.optimizers import RMSprop
from keras.utils.train_utils import *
from data import load_data, load_data2, load_data3, segment_data
from errors import *
from dataset import *

try: 
    from collections import OrderedDict
except:
    from ordereddict import OrderedDict
    
pred_range = [2, 42]

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
    print 'forget mean min:', fgts.mean(), fgts.min()
    print 'delta_x =', np.abs(dxs).mean(), 'delta_h =', np.abs(dhs).mean()
    layer = model.get_rlstm_layer()
    print 'U_c =',layer.U_c.get_value(), 'U_f =',layer.U_f.get_value(), 'b_c =',layer.b_c.get_value(), 'b_f =',layer.b_f.get_value()
    print 'incx.max(), incx.min(), incx.mean()', incx.max(), incx.min(), incx.mean()
    print 'fgtx.max(), fgtx.min(), fgtx.mean()', fgtx.max(), fgtx.min(), fgtx.mean()
    
    if show_details:
        print 'delta mean, abs_mean, abs_mean+, abs_mean-:', dxs.mean(), np.abs(dxs).mean(), np.abs(dxs[dxs>0]).mean(), np.abs(dxs[dxs<0]).mean()
        
        W_c =layer.W_c.get_value()
        W_f =layer.W_f.get_value()
        print 'W_c max, min, mean, abs_mean:', W_c.max(), W_c.min(), W_c.mean(), np.abs(W_c).mean()
        print 'W_f max, min, mean, abs_mean:', W_f.max(), W_f.min(), W_f.mean(), np.abs(W_f).mean()  
    
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
     
def load_rlstm(base_name, i=0):
    rlstm = model_from_yaml(open(base_name + '.yaml').read())
    rlstm.base_name = base_name
    rlstm.name = base_name + str(i)
    rlstm.load_weights(base_name + str(i) + '_weights.hdf5')
    rlstm.load_normalization_info()
    return rlstm

dongbei_lon_range=[0., 1000.]  #
dongbei_lat_range=[40.4, 1000.] # beijing to ~
#huabei_lon_range=[112.4, 118.3]  #taiyuan-changsha to tangshan
huabei_lon_range=[112.4, 1000.]  #taiyuan-changsha to sea
huabei_lat_range=[34.15, 40.4] # xi'an-xuzhou to beijing

xibei_lon_range=[103.6, 112.4]  #lanzhou-chengdu to taiyuan-changsha
xibei_lat_range=[34.15, 40.4] # xi'an-xuzhou to beijing

huadong_lon_range=[112.4, 1000.] # taiyuan-changsha to sea
#huadong_lat_range=[28.1, 34.15] # changsha to xi'an-xuzhou
huadong_lat_range=[0, 34.15] # sea to xi'an-xuzhou

huaxi_lon_range=[103.6, 112.4] # lanzhou-chengdu to taiyuan-changsha
huaxi_lat_range=[28.1, 34.15] # changsha to xi'an-xuzhou

area2lonlat = OrderedDict([
#                     ('dongbei', (dongbei_lon_range, dongbei_lat_range)),
                     ('huabei', (huabei_lon_range, huabei_lat_range)),
#                     ('xibei', (xibei_lon_range, xibei_lat_range)),
#                     ('huadong', (huadong_lon_range, huadong_lat_range)),
#                     ('huaxi', (huaxi_lon_range, huaxi_lat_range)),
                     ])

beijing_stations = [str(i)+'A' for i in range(1001, 1013)]  #12
#langfang_stations = [str(i)+'A' for i in range(1067, 1071)]  #4
tianjin_stations = [str(i)+'A' for i in range(1013, 1028)]  #15
tangshan_stations = [str(i)+'A' for i in range(1036, 1042)]  #6

baoding_stations = [str(i)+'A' for i in range(1051, 1057)]  #6
shijiazhuang_stations = [str(i)+'A' for i in range(1028, 1036)]  #8
#hengshui_stations = [str(i)+'A' for i in range(1074, 1077)]  #3

xingtai_stations = [str(i)+'A' for i in range(1077, 1081)]  #4
handan_stations = [str(i)+'A' for i in range(1047, 1051)]  #4

jinan_stations = [str(i)+'A' for i in range(1299, 1307)]  #8

xian_stations = [str(i)+'A' for i in range(1462, 1475)] #13

chongqing_stations = [str(i)+'A' for i in range(1414, 1431)] #17
chengdu_stations = [str(i)+'A' for i in range(1431, 1439)] #18

nanjing_stations = [str(i)+'A' for i in range(1151, 1160)] #9
shanghai_stations = [str(i)+'A' for i in range(1141, 1151)] #10
hangzhou_stations = [str(i)+'A' for i in range(1223, 1234)] #11
hefei_stations = [str(i)+'A' for i in range(1270, 1280)] #10
wuhan_stations = [str(i)+'A' for i in range(1325, 1335)] #10
nanchang_stations = [str(i)+'A' for i in range(1290, 1299)] #9
changsha_stations = [str(i)+'A' for i in range(1335, 1345)] #10

city2stations = OrderedDict([
                   ('beijing', beijing_stations),
#                   ('tianjin', tianjin_stations),
#                   ('tangshan', tangshan_stations),
#                   ('baoding', baoding_stations),
#                   ('shijiazhuang', shijiazhuang_stations),
#                   ('xingtai+handan', xingtai_stations + handan_stations),
#                   ('jinan', jinan_stations),
#                   ('xian', xian_stations),
                   
#                   ('nanjing', nanjing_stations),
#                   ('shanghai', shanghai_stations),
#                   ('hangzhou', hangzhou_stations),
#                   ('hefei', hefei_stations),
#                   ('wuhan', wuhan_stations),
#                   ('nanchang', nanchang_stations),
#                   ('changsha', changsha_stations),
                   
#                   ('chongqing', chongqing_stations),
#                   ('chengdu', chengdu_stations),
                   ])

if __name__ == '__main__':
    beijing_only = False
#    if beijing_only:
##        train_data, valid_data, test_data = load_data2(stations=[u'1003A', u'1004A',u'1005A', u'1006A', u'1007A', u'1011A'], segment=True)
#        train_data, valid_data = load_data3(stations=beijing_stations, 
##                                            train_stop=740, valid_start=740) 
#                                            train_stop=630, valid_start=680, valid_stop=840,
#                                            filter=False)
##                                            train_start=200, train_stop=953, valid_start=680, valid_stop=953)
#    else:
##        train_data, valid_data, test_data = load_data2(segment=True) 
#        train_data, valid_data = load_data3(
#                                            lon_range=huabei_lon_range, lat_range=huabei_lat_range,
##                                            lon_range=[110., 117.5], lat_range=[34., 40.4],  
#                                            train_stop=630, valid_start=680, valid_stop=840,
#                                            filter=False)
    
    for area in area2lonlat:
#    for city in city2stations:
        train_data, valid_data = load_data3(
                                            lon_range=area2lonlat[area][0], lat_range=area2lonlat[area][1], 
#                                            stations=city2stations[city],
                                            train_stop=630, valid_start=680, valid_stop=840,
#                                            train_stop=953, valid_start=680, valid_stop=953, 
                                            filter=(not beijing_only))
        X_train, y_train, X_valid, y_valid = build_lstm_dataset(train_data, valid_data, pred_range=pred_range, hist_len=3)
        print 'X_train[0].shape =', X_train[0].shape
        name = area + '20150113'
        rlstm = build_rlstm(X_train[0].shape[-1], h0_dim=20, h1_dim=20, 
                                   rec_layer_init='zero', base_name=name,
                                   add_input_noise=beijing_only, add_target_noise=False)
#        rlstm = build_rlstm2(X_train[0].shape[-1], h0_dim=20, h1_dim=20, base_name=name,
#                                   add_input_noise=beijing_only, add_target_noise=beijing_only)
        rlstm.name = name
        rlstm.data = [train_data, valid_data]
        rlstm.X_mask = np.ones((X_train[0].shape[-1],), dtype='int')
#        rlstm.X_mask[:6] = 0  # wind direction
#        rlstm.X_mask[-4:-3] = 0  # day of week
#        rlstm.X_mask[-3:-1] = 0  # lonlat
        rlstm.X_mask[-1:] = not beijing_only  # pm25 mean
        print '\ntraining', rlstm.name
        X_train[0], X_valid[0] = normalize(X_train[0], X_valid[0], rlstm)
        rlstm.save_normalization_info(name + '_norm_info.pkl')
        batch_size = (1 + (not beijing_only)) * 64
        patience = (1 + int(beijing_only)) * 10
        train(X_train, y_train, X_valid, y_valid, rlstm, batch_size=batch_size, patience=patience, nb_epoch=300)
      
#    name = 'shanghai'         
#    rlstm = model_from_yaml(open(name + '.yaml').read())
##    rlstm.name = name
##    rlstm.load_normalization_info(name + '_norm_info.pkl')
##    rlstm.load_weights(name + '_weights.hdf5')
##    for area in area2lonlat:
#    for city in city2stations:
#        name = city
#        rlstm.name = name
#        rlstm.load_normalization_info(name + '_norm_info.pkl')
#        rlstm.load_weights(name + '_weights.hdf5')
#        
#        train_data, valid_data = load_data3(
##                                            lon_range=area2lonlat[area][0], lat_range=area2lonlat[area][1],
#                                            stations=city2stations[city],
#                                            train_stop=630, valid_start=680, valid_stop=840)
#        rlstm.data = [train_data, valid_data]
#        print '\n' + city
#        test_model(rlstm, dataset='train', show_details=False)
#        test_model(rlstm, dataset='valid', show_details=False)

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