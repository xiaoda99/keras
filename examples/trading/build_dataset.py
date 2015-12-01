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

from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from ifshort_rnn import build_rnn, build_mlp, train

#from pylearn2.datasets.if_monthly import form_history
def form_history(X, hist_len, hist_step=1):
    X_hist = np.zeros((X.shape[0], hist_len, X.shape[1]), dtype='float32')
    for i in range(hist_len):
        shift = i * hist_step
        X_hist[:,i,:] = np.roll(X, shift, axis=0)
        X_hist[:shift,i,:] = 0.
    return X_hist.reshape((X_hist.shape[0], X_hist.shape[1] * X_hist.shape[2]))

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

def sequentialize(X, seq_len, skip=1):
    Xs = []
    for i in range(0, X.shape[0] - seq_len, skip):
        Xs.append(X[i : i + seq_len])
    return np.array(Xs)

def postbuild(I, y, hist_len=3):
#    global I
    I = normalize(I)    
    
    I = form_history(I, hist_len)
    
    X_train = I[:I.shape[0]*2/3]
    y_train = y[:y.shape[0]*2/3]
    X_test = I[I.shape[0]*2/3:]
    y_test = y[y.shape[0]*2/3:]
    return X_train, y_train, X_test, y_test

def split_dataset(X, y):
    X_train = X[:X.shape[0]*2/3]
    y_train = y[:y.shape[0]*2/3]
    X_test = X[X.shape[0]*2/3:]
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
    
X = []
for i in range(0, len(tick_data) - 20, 1):
    x = preprocessor.step(tick_data[i])        
    X.append(x)
    
Y = []
for i in range(0, len(tick_data) - 20, 1):
    now = tick_data[i].price
    targets = np.array([td.price for td in tick_data[i : i+20]]) - now
    Y.append(targets)
Y = np.array(Y)

Y = []
for i in range(0, len(tick_data) - 20, 1):
    now = tick_data[i].price
    if i + 61 >= len(tick_data):
        targets = np.zeros(61)
    else:
        targets = np.array([td.price for td in tick_data[i : i+61]]) - now
    gain = targets / now
    Y.append(gain)
Y = np.array(Y)

#Y = Y.astype('int32')  # this line is very important, I don't know why.
for i in range(Y.shape[1]):
    remove_extreme(Y[:,i], cutoff)
#gain_range = [0, 10]
#y = Y[:, gain_range[0]:gain_range[1]].max(axis=1, keepdims=True) + \
#        Y[:, gain_range[0]:gain_range[1]].min(axis=1, keepdims=True)

gain_indices = [5, 10]
y = Y[:, gain_indices]

price_delta1 = np.array([x.price_delta1 for x in X])
ask_price_delta1 = np.array([x.ask_price_delta1 for x in X])
bid_price_delta1 = np.array([x.bid_price_delta1 for x in X])
price_diff1 = np.array([x.price_diff1 for x in X])
price = np.array([x.price for x in X])
ask_price = np.array([x.ask_price for x in X])
bid_price = np.array([x.bid_price for x in X])
price_ema5 = np.array([x.price_ema[5] for x in X])
price_ema10 = np.array([x.price_ema[10] for x in X])
price_ema20 = np.array([x.price_ema[20] for x in X])
price_ema40 = np.array([x.price_ema[40] for x in X])
price_delta_ema5 = np.array([x.price_delta_ema[5] for x in X])
price_delta_ema10 = np.array([x.price_delta_ema[10] for x in X])
price_delta_ema20 = np.array([x.price_delta_ema[20] for x in X])
price_delta_ema40 = np.array([x.price_delta_ema[40] for x in X])
price_delta_ema80 = np.array([x.price_delta_ema[80] for x in X])
price_momentum5 = np.array([x.price_momentum[5] for x in X])
price_momentum10 = np.array([x.price_momentum[10] for x in X])
price_momentum20 = np.array([x.price_momentum[20] for x in X])
price_momentum40 = np.array([x.price_momentum[40] for x in X])
price_momentum80 = np.array([x.price_momentum[80] for x in X])
price_momentum_ema5 = np.array([x.price_momentum_ema[5] for x in X])
price_momentum_ema10 = np.array([x.price_momentum_ema[10] for x in X])
price_momentum_ema20 = np.array([x.price_momentum_ema[20] for x in X])
price_momentum_ema40 = np.array([x.price_momentum_ema[40] for x in X])
price_momentum_ema80 = np.array([x.price_momentum_ema[80] for x in X])
price_momentum_sma5 = np.array([x.price_momentum_sma[5] for x in X])
price_momentum_sma10 = np.array([x.price_momentum_sma[10] for x in X])
price_momentum_sma20 = np.array([x.price_momentum_sma[20] for x in X])
price_momentum_sma40 = np.array([x.price_momentum_sma[40] for x in X])
price_momentum_sma80 = np.array([x.price_momentum_sma[80] for x in X])
price_momentum_sma25 = np.array([x.price_momentum_sma2[5] for x in X])
price_momentum_sma210 = np.array([x.price_momentum_sma2[10] for x in X])
price_momentum_sma220 = np.array([x.price_momentum_sma2[20] for x in X])
price_momentum_sma240 = np.array([x.price_momentum_sma2[40] for x in X])
price_momentum_sma280 = np.array([x.price_momentum_sma2[80] for x in X])
true_price_delta_ema5 = np.array([x.true_price_delta_ema[5] for x in X])
true_price_delta_ema10 = np.array([x.true_price_delta_ema[10] for x in X])
true_price_delta_ema20 = np.array([x.true_price_delta_ema[20] for x in X])
true_price_delta_ema40 = np.array([x.true_price_delta_ema[40] for x in X])
true_price_delta_ema80 = np.array([x.true_price_delta_ema[80] for x in X])
#rsv5 = np.array([x.rsv[5] for x in X])
rsv10 = np.array([x.rsv[10] for x in X])
rsv20 = np.array([x.rsv[20] for x in X])
rsv40 = np.array([x.rsv[40] for x in X])
#k5 = np.array([x.k[5] for x in X])
k10 = np.array([x.k[10] for x in X])
k20 = np.array([x.k[20] for x in X])
k40 = np.array([x.k[40] for x in X])
ke10 = np.array([x.ke[10] for x in X])
ke20 = np.array([x.ke[20] for x in X])
ke40 = np.array([x.ke[40] for x in X])
price_rsi5 = np.array([x.price_rsi[5] for x in X])
price_rsi10 = np.array([x.price_rsi[10] for x in X])
price_rsi20 = np.array([x.price_rsi[20] for x in X])
price_rsi40 = np.array([x.price_rsi[40] for x in X])
atr10 = np.array([x.atr[10] for x in X])
atr20 = np.array([x.atr[20] for x in X])

order_vol_diff1 = np.array([x.order_vol_diff1 for x in X])
order_vol_diff_delta1 = np.array([x.order_vol_diff_delta1 for x in X])
order_vol_ratio1 = np.array([x.order_vol_ratio1 for x in X])
order_vol_ratio_delta1 = np.array([x.order_vol_ratio_delta1 for x in X])
order_info = np.array([x.order_info for x in X])

vol = np.array([x.deal_vol for x in X])
vol_delta1 = np.array([x.vol_delta1 for x in X])
vol_ema5 = np.array([x.vol_ema[5] for x in X])
vol_ema10 = np.array([x.vol_ema[10] for x in X])
vol_ema20 = np.array([x.vol_ema[20] for x in X])
vol_sma5 = np.array([x.vol_sma[5] for x in X])
vol_sma10 = np.array([x.vol_sma[10] for x in X])
vol_sma20 = np.array([x.vol_sma[20] for x in X])
vol_delta_ema5 = np.array([x.vol_delta_ema[5] for x in X])
vol_delta_ema10 = np.array([x.vol_delta_ema[10] for x in X])
vol_delta_ema20 = np.array([x.vol_delta_ema[20] for x in X])
#true_vol_delta_ema5 = np.array([x.true_vol_delta_ema[5] for x in X])
#true_vol_delta_ema10 = np.array([x.true_vol_delta_ema[10] for x in X])
vol_momentum5 = np.array([x.vol_momentum[5] for x in X])
vol_momentum10 = np.array([x.vol_momentum[10] for x in X])
vol_momentum20 = np.array([x.vol_momentum[20] for x in X])
vol_momentum_sma5 = np.array([x.vol_momentum_sma[5] for x in X])
vol_momentum_sma10 = np.array([x.vol_momentum_sma[10] for x in X])
vol_momentum_sma20 = np.array([x.vol_momentum_sma[20] for x in X])

O1 = np.array([
#               order_vol_diff1,
               order_vol_diff_delta1,
#               order_vol_ratio1,
#               order_vol_ratio_delta1,
               ]).T
O2 = order_info
P = np.array([
              ask_price_delta1,
              bid_price_delta1,
#              (ask_price_delta1 + bid_price_delta1) * 1. / 2,  
#              true_price_delta_ema10,  
#              true_price_delta_ema20,  
#              true_price_delta_ema40,  
#              true_price_delta_ema80,  
#              price_delta_ema80,  # hardly useful
              price_momentum5,  
              price_momentum10,  
              price_momentum20,  
              price_momentum40,  # 5-40 20.155, with neuron 60-60
#              price_momentum_sma25,  
#              price_momentum_sma210,  
#              price_momentum_sma220,  # 5-20 combined 20.166
#              price_momentum_sma240,  # 5-40 20.175, combined 20.146, with neuron 60-60 
#              price_momentum_ema80,
#              price_delta_ema80,  #27.78
#              price_delta1 - price_delta_ema5,  #useless
#              price_delta_ema5 - price_delta_ema10,  #useless
              rsv10,
              rsv20,
#              rsv40,
              k10,
              k20,
#              k40,
#              price_rsi5,
              price_rsi10,
              price_rsi20,
#              price_rsi10gep5, # less useful than price_rsi10!
#              price_rsi10lemp5, # less useful than price_rsi10!
#              atr10,
#              atr20
              ]).T
              
V = np.array([
#              vol_delta1, 
#              vol_delta_ema5,
#              vol_delta_ema10,
#              vol_delta_ema20,
              vol,
              vol_sma5,
              vol_sma10, # 20.137!, 20.182 if no vol info is used at all
#              vol_momentum5,
#              vol_momentum10,
#              vol_momentum_sma5,
#              vol_momentum_sma10,  # good when used WITHOUT raw momentums
              ]).T
              
P = np.array([
              ask_price_delta1,
              bid_price_delta1,
#              (ask_price_delta1 + bid_price_delta1) * 1. / 2,  
#              true_price_delta_ema10,  
#              true_price_delta_ema20,  
#              true_price_delta_ema40,  
#              true_price_delta_ema80,  
#              price_delta_ema80,  # hardly useful
              price_momentum_ema10,  
              price_momentum_ema20,  
              price_momentum_ema40,  # 5-40 20.155, with neuron 60-60
              price_momentum_ema80,
#              price_momentum_sma25,  
#              price_momentum_sma210,  
#              price_momentum_sma220,  # 5-20 combined 20.166
#              price_momentum_sma240,  # 5-40 20.175, combined 20.146, with neuron 60-60 
#              price_momentum_ema80,
#              price_delta_ema80,  #27.78
#              price_delta1 - price_delta_ema5,  #useless
#              price_delta_ema5 - price_delta_ema10,  #useless
#              rsv10,
#              rsv20,
#              rsv40,
              k10,
              k20,
              k40,
#              price_rsi5,
              price_rsi10,
              price_rsi20,
              price_rsi40,
#              price_rsi10gep5, # less useful than price_rsi10!
#              price_rsi10lemp5, # less useful than price_rsi10!
#              atr10,
#              atr20
              ]).T
V = np.array([
#              vol_delta1, 
#              vol_delta_ema5,
#              vol_delta_ema10,
#              vol_delta_ema20,
              vol,
              vol_sma5,
              vol_sma10,
              vol_sma20, # 20.137!, 20.182 if no vol info is used at all
#              vol_momentum5,
#              vol_momentum10,
#              vol_momentum_sma5,
#              vol_momentum_sma10,  # good when used WITHOUT raw momentums
              ]).T
              
I = np.hstack([P, V])
I = normalize(I)

Xh = form_history(I, 3)
Xh_train, y_train, Xh_test, y_test = split_dataset(Xh, y)

mlp = build_mlp(Xh.shape[-1], y.shape[-1], 60, 60)
train(Xh_train, y_train, Xh_test, y_test, mlp)

#Xs = sequentialize(I, 20)
#ys = sequentialize(y, 20)
#Xs_train, ys_train, Xs_test, ys_test = split_dataset(Xs, ys)
#
#print 'Building lstm16_rs...'
#lstm16_rs = build_rnn(Xs.shape[-1], ys.shape[-1], h0_dim=16, layer_type=LSTM, return_sequences=True)
#print '\nTraining lstm16_rs, last 10...'
#train(Xs_train[:,-10:,:], ys_train[:,-10:,:], Xs_test[:,-10:,:], ys_test[:,-10:,:], lstm16_rs)
#print '\nTraining lstm40_rs, last 20...'
#train(Xs_train[:,-20:,:], ys_train[:,-20:,:], Xs_test[:,-20:,:], ys_test[:,-20:,:], lstm16_rs)
#
#print 'Building lstm16_16_rs...'
#lstm16_16_rs = build_rnn(Xs.shape[-1], ys.shape[-1], h0_dim=16, h1_dim=16, layer_type=LSTM, return_sequences=True)
#print '\nTraining lstm16_16_rs, last 10...'
#train(Xs_train[:,-10:,:], ys_train[:,-10:,:], Xs_test[:,-10:,:], ys_test[:,-10:,:], lstm16_16_rs)
#print '\nTraining lstm40_rs, last 20...'
#train(Xs_train[:,-20:,:], ys_train[:,-20:,:], Xs_test[:,-20:,:], ys_test[:,-20:,:], lstm16_16_rs)



#print 'Building mlp80_40...'
#mlp80_40 = build_mlp(Xh.shape[-1], y.shape[-1], 80, 40)
#print '\nTraining mlp80_40...'
#train(Xh_train, y_train, Xh_test, y_test, mlp80_40)

#print 'Building lstm40...'
#lstm40 = build_rnn(Xs.shape[-1], ys.shape[-1], h0_dim=30, layer_type=LSTM, return_sequences=False)
#print '\nTraining lstm40, last 5...'
#train(Xs_train[:,-5:,:], ys_train[:,-1,:], Xs_test[:,-5:,:], ys_test[:,-1,:], lstm40)
#print '\nTraining lstm40, last 10...'
#train(Xs_train[:,-10:,:], ys_train[:,-1,:], Xs_test[:,-10:,:], ys_test[:,-1,:], lstm40)
#
#print 'Building lstm40_rs...'
#lstm40_rs = build_rnn(Xs.shape[-1], ys.shape[-1], h0_dim=30, layer_type=LSTM, return_sequences=True)
#print '\nTraining lstm40_rs, last 10...'
#train(Xs_train[:,-10:,:], ys_train[:,-10:,:], Xs_test[:,-10:,:], ys_test[:,-10:,:], lstm40_rs)
#print '\nTraining lstm40_rs, last 20...'
#train(Xs_train[:,-20:,:], ys_train[:,-20:,:], Xs_test[:,-20:,:], ys_test[:,-20:,:], lstm40_rs)
#
#print 'Building srnn40...'
#srnn40 = build_rnn(Xs.shape[-1], ys.shape[-1], h0_dim=30, layer_type=SimpleRNN, return_sequences=False)
#print '\nTraining srnn40, last 5...'
#train(Xs_train[:,-5:,:], ys_train[:,-1,:], Xs_test[:,-5:,:], ys_test[:,-1,:], srnn40)
#print '\nTraining srnn40, last 10...'
#train(Xs_train[:,-10:,:], ys_train[:,-1,:], Xs_test[:,-10:,:], ys_test[:,-1,:], srnn40)
#
#print 'Building srnn40_rs...'
#srnn40_rs = build_rnn(Xs.shape[-1], ys.shape[-1], h0_dim=30, layer_type=SimpleRNN, return_sequences=True)
#print '\nTraining srnn40_rs, last 10...'
#train(Xs_train[:,-10:,:], ys_train[:,-10:,:], Xs_test[:,-10:,:], ys_test[:,-10:,:], srnn40_rs)
#print '\nTraining srnn40_rs, last 20...'
#train(Xs_train[:,-20:,:], ys_train[:,-20:,:], Xs_test[:,-20:,:], ys_test[:,-20:,:], srnn40_rs)


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