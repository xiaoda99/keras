import os
import ctypes
import numpy as np
import math
import gzip
import cPickle
import pylab as plt
import matplotlib.gridspec as gridspec
from profilehooks import profile

from preprocessor import TickDataItem, Preprocessor, TestPreprocessor

from ifshort_mlp import *

#def remove_extreme(s, cutoff, end):
#    s = np.copy(s)
#    sorted = np.sort(s)
#    cutoff_len = sorted.size * cutoff
##    if sorted[0] * sorted[-1] < 0:
#    if end == -1 or end == 0:
#        low = sorted[cutoff_len]    
#        s[s < low] = low
#    if end == 1 or end == 0:
#        high = sorted[-cutoff_len]
#        s[s > high] = high
#    return s

#from pylearn2.datasets.if_monthly import remove_extreme
def remove_extreme(s, cutoff):
    sorted = np.sort(s)
    cutoff_len = sorted.size * cutoff
    if sorted[0] * sorted[-1] < 0:
        low = sorted[cutoff_len]    
        s[s < low] = low
    high = sorted[-cutoff_len]
    s[s > high] = high
    return s

def normalize(X):
    X = X.astype('float32')
    for i in range(X.shape[1]):
        remove_extreme(X[:,i], cutoff)
    X_train = X
    X_mean = X_train.mean(axis=0)
    X_train = X_train - X_mean
    X_stdev = np.sqrt(X_train.var(axis=0))
    X -= X_mean
    X /= X_stdev
    return X

def postbuild(I, y, hist_len=3):
#    global I
    I = normalize(I)    
    from pylearn2.datasets.if_monthly import form_history
    I = form_history(I, hist_len)
    
    X_train = I[:I.shape[0]*2/3]
    y_train = y[:y.shape[0]*2/3]
    X_test = I[I.shape[0]*2/3:]
    y_test = y[y.shape[0]*2/3:]
    return X_train, y_train, X_test, y_test
            
tick_data = []       
def callback(context, data, len, score, latest_score):
    tick_data.append(TickDataItem(data))
    game.Play(context, 1)
    
game = ctypes.CDLL("game.so")
CMPFUNC = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int) 
_callback = CMPFUNC(callback)
context = game.CreateContext('cfg_test.xml')
game.RegisterCallback(context, _callback)
print 'Running game...'
while not game.IsFinish(context):
    game.RunOne(context)

preprocessor = TestPreprocessor()
cutoff = .001
    
Y = []
for i in range(0, len(tick_data) - 20, 1):
    now = tick_data[i].price
    targets = np.array([td.price for td in tick_data[i : i+20]]) - now
    Y.append(targets)
Y = np.array(Y)

X = []
for i in range(0, len(tick_data) - 20, 1):
    x = preprocessor.step(tick_data[i])        
    X.append(x)

Y = Y.astype('int32')  # this line is very important, I don't know why.
for i in range(Y.shape[1]):
    remove_extreme(Y[:,i], cutoff)
gain_range = [0, 10]
y = Y[:, gain_range[0]:gain_range[1]].max(axis=1, keepdims=True) + \
        Y[:, gain_range[0]:gain_range[1]].min(axis=1, keepdims=True)

price_delta1 = np.array([x.price_delta1 for x in X])
ask_price_delta1 = np.array([x.ask_price_delta1 for x in X])
bid_price_delta1 = np.array([x.bid_price_delta1 for x in X])
price_diff1 = np.array([x.price_diff1 for x in X])
price_ema5 = np.array([x.price_ema[5] for x in X])
price_ema10 = np.array([x.price_ema[10] for x in X])
price_ema20 = np.array([x.price_ema[20] for x in X])
price_delta_ema5 = np.array([x.price_delta_ema[5] for x in X])
price_delta_ema10 = np.array([x.price_delta_ema[10] for x in X])
price_delta_ema20 = np.array([x.price_delta_ema[20] for x in X])
price_delta_ema40 = np.array([x.price_delta_ema[40] for x in X])
price_delta_ema80 = np.array([x.price_delta_ema[80] for x in X])
true_price_delta_ema5 = np.array([x.true_price_delta_ema[5] for x in X])
true_price_delta_ema10 = np.array([x.true_price_delta_ema[10] for x in X])
true_price_delta_ema20 = np.array([x.true_price_delta_ema[20] for x in X])
true_price_delta_ema40 = np.array([x.true_price_delta_ema[40] for x in X])
true_price_delta_ema80 = np.array([x.true_price_delta_ema[80] for x in X])
k5 = np.array([x.k[5] for x in X])
k10 = np.array([x.k[10] for x in X])
k20 = np.array([x.k[20] for x in X])
price_rsi5 = np.array([x.price_rsi[5] for x in X])
price_rsi10 = np.array([x.price_rsi[10] for x in X])
price_rsi20 = np.array([x.price_rsi[20] for x in X])

order_vol_diff1 = np.array([x.order_vol_diff1 for x in X])
order_vol_diff_delta1 = np.array([x.order_vol_diff_delta1 for x in X])
order_vol_ratio1 = np.array([x.order_vol_ratio1 for x in X])
order_vol_ratio_delta1 = np.array([x.order_vol_ratio_delta1 for x in X])
order_info = np.array([x.order_info for x in X])

vol = np.array([x.deal_vol for x in X])
vol_delta1 = np.array([x.vol_delta1 for x in X])
vol_ema5 = np.array([x.vol_ema[5] for x in X])
vol_ema10 = np.array([x.vol_ema[10] for x in X])
vol_delta_ema5 = np.array([x.vol_delta_ema[5] for x in X])
vol_delta_ema10 = np.array([x.vol_delta_ema[10] for x in X])
true_vol_delta_ema5 = np.array([x.true_vol_delta_ema[5] for x in X])
true_vol_delta_ema10 = np.array([x.true_vol_delta_ema[10] for x in X])

P = np.array([ask_price_delta1,
              bid_price_delta1,
#              price_ema5 - price_ema20,  #useless
              price_delta_ema5,  
              price_delta_ema10,  
              price_delta_ema20,  
              price_delta_ema40,  # useful
#              price_delta_ema80,  #27.78
#              price_delta1 - price_delta_ema5,  #useless
#              price_delta_ema5 - price_delta_ema10,  #useless
              k5,
              k10,
              k20,
              price_rsi5,
              price_rsi10,
              price_rsi20,
#              price_rsi10gep5, # less useful than price_rsi10!
#              price_rsi10lemp5, # less useful than price_rsi10!
              ]).T
O1 = np.array([
#               order_vol_diff1,
               order_vol_diff_delta1,
#               order_vol_ratio1,
#               order_vol_ratio_delta1,
               ]).T
O2 = order_info
V = np.array([vol_delta1, 
              vol_delta_ema5,
              vol_delta_ema10,
              ]).T
I = np.hstack([P,])


#@profile
def build_dataset_X():
    print 'Building dataset...'
    X = []
    for i in range(0, len(tick_data) - 20, 1):
        x = preprocessor.step(tick_data[i])
        X.append(x)
    X = np.array(X)
    print 'X.shape =', X.shape
      
    with open('/home/xd/data/trading/IF1503_cut20_43x3_X_raw.npy', 'wb') as f:
        np.save(f,X)
        
    for i in range(X.shape[1]):
        remove_extreme(X[:,i], cutoff)
        
    X_train = X
    X_mean = X_train.mean(axis=0)
    X_train = X_train - X_mean
    X_stdev = np.sqrt(X_train.var(axis=0))
    X = X - X_mean
    X /= X_stdev

    with open('/home/xd/data/trading/IF1503_cut20_43x3_X.npy', 'wb') as f:
        np.save(f,X)
    with open('/home/xd/data/trading/IF1503_cut20_43x3_X_mean.npy', 'wb') as f:
        np.save(f,X_mean)
    with open('/home/xd/data/trading/IF1503_cut20_43x3_X_stdev.npy', 'wb') as f:
        np.save(f,X_stdev)

def build_dataset_Y_old():
    print 'Building dataset...'
    global Y;
    for i in range(0, len(tick_data) - 20, 1):
#        x = ind_extr.step(tick_data[i])
        now = tick_data[i].price
        targets = np.array([td.price for td in tick_data[i : i+20]]) - now
            
        Y.append(targets)
    Y = np.array(Y)
    print Y.shape
    with open('/home/xd/data/trading/IF1503_cut20_Y20.npy', 'wb') as f:
        np.save(f,Y)

def build_dataset_Y():
    print 'Building dataset...'
    global Y;
    for i in range(0, len(tick_data) - 20, 1):
        now = np.array([tick_data[i].ask_price, tick_data[i].bid_price])
        future_ask_prices = [td.ask_price for td in tick_data[i : i+20]]
        future_bid_prices = [td.bid_price for td in tick_data[i : i+20]]
        targets = np.array([future_ask_prices, future_bid_prices]).T - now
            
        Y.append(targets)
    Y = np.array(Y)
    print Y.shape
    with open('/home/xd/data/trading/IF1503_cut20_Y20x2.npy', 'wb') as f:
        np.save(f,Y)
            
def build_dataset():
    print 'Building dataset...'
    global X, Y, I;
    for i in range(0, len(tick_data) - 20, 1):
        x = ind_extr.step(tick_data[i])
        now = np.array([tick_data[i].ask_price, tick_data[i].bid_price])
        future_ask_prices = [td.ask_price for td in tick_data[i : i+20]]
        future_bid_prices = [td.bid_price for td in tick_data[i : i+20]]
        targets = np.array([future_ask_prices, future_bid_prices]).T - now
            
        X.append(x)
        Y.append(targets)
    X = np.array(X)
    Y = np.array(Y)
    
#build_dataset_X()
#build_dataset_Y()
#
#print 'Saving dataset...'
#f = open('/home/xd/data/trading/IF1503_T10_3steps+10MA9.npy', 'wb')
#np.save(f,X)
#np.save(f,Y)
#f.close()

def plot_mean_function(x, y, x_range):
    x_mean = []
    y_mean = []
    cond_mean = []
    for i in np.arange(x_range[0], x_range[1], x_range[2]):
        x_min = i
        x_max = i + x_range[2]
        cond = (x >= x_min) & (x < x_max)
        x_mean.append((x_min + x_max) * 1. / 2.)
        y_mean.append(y[cond].mean())
        cond_mean.append(cond.mean())
    plt.plot(x_mean, y_mean)
    plt.plot(x_mean, np.array(cond_mean)*5)
    plt.show()  
    
def plot_period(start, stop, T):
    if (stop - start) % T != 0:
        print 'stop, start, T =', stop, start, T
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 1])
    ax0 = plt.subplot(gs[0])
    price = [x.price for x in X[start:stop]]
    price_ema1 = [x.price_ema[T] for x in X[start:stop]]
    price_ema2 = [x.price_ema[T*2] for x in X[start:stop]]
    price_ema4 = [x.price_ema[T*4] for x in X[start:stop]]
    ax0.plot(price_ema1)
    ax0.plot(price_ema2)
    ax0.plot(price_ema4)
    ax0.plot(price, color='k')
#    boll4_bw = [x.boll4_bw for x in X[start:stop]]
#    ax0.plot(np.array(price_ema4) + np.array(boll4_bw)*2, color='y')
#    ax0.plot(np.array(price_ema4) - np.array(boll4_bw)*2, color='y')
    ax0.set_xticks(np.arange(0, stop - start, T))
    plt.grid()
    
    ax = plt.subplot(gs[1], sharex=ax0)
    price_delta_ema1 = [x.price_delta_ema[T] for x in X[start:stop]]
    price_delta_ema2 = [x.price_delta_ema[T*2] for x in X[start:stop]]
    price_delta_ema4 = [x.price_delta_ema[T*4] for x in X[start:stop]]
    ax.plot(price_delta_ema1)
    ax.plot(price_delta_ema2)
    ax.plot(price_delta_ema4)
    ax.plot(np.zeros(stop - start), color='k')
    ax.set_xticks(np.arange(0, stop - start, T))
    plt.grid()
    
#    ax = plt.subplot(gs[2], sharex=ax0)
#    ax.plot([x.price_acc_ema1_2nd for x in X[start:stop]])
#    ax.plot([x.price_acc_ema1_4th for x in X[start:stop]])
#    ax.plot(np.zeros(stop - start), color='k')
#    ax.set_xticks(np.arange(0, stop - start, T))
#    plt.grid()
#    
    ax = plt.subplot(gs[3], sharex=ax0)
    ax.plot([x.k[T] for x in X[start:stop]])
#    ax.plot([x.k2 for x in X[start:stop]])
    ax.set_xticks(np.arange(0, stop - start, T))
    plt.grid()
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.show()
    
def plot_condition(condition):
    indices = np.where(condition)[0]
    while True:
        i = indices[np.random.randint(indices.size)]
        if i - T*4 >= 0 and i + T*2 <= len(X):
            plot_period(i - T*4, i + T*2, T) 