import os
import ctypes
import numpy as np
import math
import gzip
import cPickle
import pylab as plt
import matplotlib.gridspec as gridspec
#from collections import OrderedDict
#from ordereddict import OrderedDict  #XD
OrderedDict = dict

from profilehooks import profile

MAX_OUTPUTS_LEN = 1000

class SMA(object):
    def __init__(self, n):
        self.__dict__.update(locals())
        del self.self
        self.history = []
        self.hist_sum = 0.
        self.hist_len = 0
           
    def step(self, x):
        if self.n == 1:
            return x
        
        if self.hist_len == self.n:
            first = self.history.pop(0)
            self.hist_sum -= first
            self.hist_len -= 1
            
        self.history.append(x)
        self.hist_sum += x
        self.hist_len += 1
        
        ma = self.hist_sum * 1. / self.hist_len
        return ma

class MA(object):
    def __init__(self, preprocessor, input_key, t, last_n=1, skip=None):
        self.__dict__.update(locals())
        del self.self
        if self.skip is None:
            self.skip = self.preprocessor.output_step
        assert self.skip % self.preprocessor.output_step == 0
        ma_n = max(1, self.t / self.preprocessor.output_step)
        self.ma = SMA(ma_n)
        self.output_key = self.input_key + '_ma' + str(t)
        self.outputs = []
            
    def _output(self, history):
        ma = self.ma.step(history[-1])
        self.outputs.append(ma)
        return
        
    def output(self, history):
        self._output(history)   
        if len(self.outputs) >= MAX_OUTPUTS_LEN:
            del self.outputs[: MAX_OUTPUTS_LEN/2] 
        rval = OrderedDict()
        skip = self.skip / self.preprocessor.output_step
        for n in range(self.last_n):
            key = self.output_key + '-' + str(n)
            if len(self.outputs) - 1 - skip * n >= 0:
                rval[key] = self.outputs[len(self.outputs) - 1 - skip * n]
            else:
                rval[key] = 0.
        return rval 
    
class Delta(object):
    def __init__(self, preprocessor, input_key, last_n=3):
        self.__dict__.update(locals())
        del self.self
#        self.input_key = 'price'
        self.output_key = self.input_key + '_delta'
               
    def output(self, history):  
        rval = OrderedDict()
        for n in range(self.last_n):
            key = self.output_key + '-' + str(n)
            if len(history) - 2 - n < 0:
                delta = 0.
            else:
                delta = history[-1-n] - history[-2-n]
            rval[key] = delta
        return rval
    
class BarPattern(object):
    def __init__(self, preprocessor, bar_width, last_n=3, skip=None):
        self.__dict__.update(locals())
        del self.self
        if self.skip is None:
            self.skip = self.bar_width
        assert self.skip % self.preprocessor.output_step == 0
        
        self.input_key = 'price'
        self.output_key = 'bar' + str(self.bar_width)
        self.outputs = []
               
    def _output(self, history):  
        if history.size < self.bar_width + 1:
            self.outputs.append((0., 0., 0.))
            return
        bar = history[-self.bar_width :]
        delta = history[-1] - history[-1-self.bar_width]
        if history.size < self.bar_width * 2:
            self.outputs.append((delta, 0., 0.))
            return
        
        prev_bar = history[-self.bar_width * 2 : -self.bar_width]
        high = bar.max()
        low = bar.min()
        prev_high = prev_bar.max()
        prev_low = prev_bar.min()
        breakup = max(0, high -prev_high)
        breakdown = max(0, prev_low - low)
        self.outputs.append((delta, breakup, breakdown))
        
    def output(self, history):
        self._output(history)
        if len(self.outputs) >= MAX_OUTPUTS_LEN:
            del self.outputs[: MAX_OUTPUTS_LEN/2] 
            
        rval = OrderedDict()
        skip = self.skip / self.preprocessor.output_step
        for n in range(self.last_n):
            key = self.output_key + '-' + str(n)
            if len(self.outputs) - 1 - skip * n >= 0:
                rval[key + '_delta'] = self.outputs[len(self.outputs) - 1 - skip * n][0]
                rval[key + '_up'] = self.outputs[len(self.outputs) - 1 - skip * n][1]
                rval[key + '_down'] = self.outputs[len(self.outputs) - 1 - skip * n][2]
            else:
                rval[key + '_delta'] = 0.
                rval[key + '_up'] = 0.
                rval[key + '_down'] = 0.
        return rval 
    
class BarPattern2(object):
    def __init__(self, bar_width, last_n=3, step=None):
        self.__dict__.update(locals())
        del self.self
        if self.step is None:
            self.step = self.bar_wdith
        self.input_key = 'price'
        self.output_key = 'bar' + str(self.bar_width)
            
    def _output(self, history, n=0):
        key = self.output_key
        if n != 0:
            key += ('-' + str(n))
        key += '_'
        if history is None or history.size < self.bar_width + 1:
            return OrderedDict([(key + 'delta', 0.), (key + 'up', 0.), (key + 'down', 0.)])
            
        bar = history[-self.bar_width :]
#        delta = bar[-1] - bar[0]
        delta = history[-1] - history[-1-self.bar_width]
        if history.size < self.bar_width * 2:
            return OrderedDict([(key + 'delta', delta), (key + 'up', 0.), (key + 'down', 0.)])
        
        prev_bar = history[-self.bar_width * 2 : -self.bar_width]
        high = bar.max()
        low = bar.min()
        prev_high = prev_bar.max()
        prev_low = prev_bar.min()
        breakup = max(0, high -prev_high)
        breakdown = max(0, prev_low - low)
#        if high > prev_high and low >= prev_low:
#            direction = 1
#        elif high <= prev_high and low < prev_low:
#            direction = -1
#        else:
#            direction = 0
        return OrderedDict([(key + 'delta', delta), (key + 'up', breakup), (key + 'down', breakdown)])
        
    def output(self, history):
#        history = history[self.input_key]
        rval = OrderedDict()
        for i in range(self.last_n): 
            if len(history) - i * self.step < 1:
                hist_i = None
            else:
                hist_i = history[:len(history) - i * self.step]
            rval.update(self._output(hist_i, i))
        return rval
            
class Breakout(object):
    def __init__(self, bar_width, break_n):
        self.__dict__.update(locals())
        del self.self
        self.input_key = 'price'
        self.output_key = 'break' + str(self.bar_width) + 'x' + str(self.break_n)
           
    def output(self, history):
        key = self.output_key
        if history.size < self.bar_width * self.break_n:
            return OrderedDict([(key, 0)])
            
        bar = history[-self.bar_width :]
        prev_bar = history[-self.bar_width * self.break_n : -self.bar_width]
        high = bar.max()
        low = bar.min()
        prev_high = prev_bar.max()
        prev_low = prev_bar.min()
        breakup = max(0, high -prev_high)
        breakdown = max(0, prev_low - low)
#        if high > prev_high and low >= prev_low:
#            direction = 1
#        elif high <= prev_high and low < prev_low:
#            direction = -1
#        else:
#            direction = 0
        rval = OrderedDict()
        rval[key + '_up'] = breakup
        rval[key + '_down'] = breakdown
#        return OrderedDict([(key + 'up', breakup), (key + 'down', breakdown)])

class Gain(object):
    '''Fully stateless version of DeltaEMA'''
    def __init__(self, preprocessor, t, input_key='price', smooth=None, normalize=True, last_n=1, skip=None):
        self.__dict__.update(locals())
        del self.self
        if self.skip is None:
            self.skip = self.preprocessor.output_step
        assert self.skip % self.preprocessor.output_step == 0
        if self.smooth is None:
            self.smooth = self.t / 2
        ma_n = max(1, self.smooth / self.preprocessor.output_step)
        self.ma = SMA(ma_n)
#        self.input_key = 'price'
#        self.output_key = 'delta' + str(t) + '_sma' + str(self.smooth)
        self.output_key = self.input_key + '_gain' + str(t)
        self.outputs = []
            
    def _output(self, history):
        if history.size < self.t + 1:
            self.outputs.append(0.)
            return
        gain = history[-1] - history[-1-self.t]
        if self.normalize:
            gain /= history[-1-self.t]
        gain_ma = self.ma.step(gain)
        self.outputs.append(gain_ma)
        return
        
    def output(self, history):
        self._output(history)   
        if len(self.outputs) >= MAX_OUTPUTS_LEN:
            del self.outputs[: MAX_OUTPUTS_LEN/2] 
        rval = OrderedDict()
        skip = self.skip / self.preprocessor.output_step
        for n in range(self.last_n):
            key = self.output_key + '-' + str(n)
            if len(self.outputs) - 1 - skip * n >= 0:
                rval[key] = self.outputs[len(self.outputs) - 1 - skip * n]
            else:
                rval[key] = 0.
        return rval 
    
class DeltaSMA2(object):
    '''Fully stateless version of DeltaEMA'''
    def __init__(self, t, smooth=1, last_n=3, step=1):
        self.__dict__.update(locals())
        del self.self
#        if self.smooth is None:
#            self.smooth = self.t / 2
        self.input_key = 'price'
#        self.output_key = 'delta' + str(t) + '_sma' + str(self.smooth)
        self.output_key = 'delta' + str(t)
            
    def _output(self, history, n=0):
        key = self.output_key
        if n != 0:
            key += ('-' + str(n))
        
        if history is None or history.size < self.t + self.smooth:
            return OrderedDict([(key, 0.)])    
        
        delta_sma = history[-self.smooth:].mean() - history[-self.t - self.smooth : -self.t].mean()
        return OrderedDict([(key, delta_sma)])
    
    def output(self, history):
#        history = history[self.input_key]
        rval = OrderedDict()
        for i in range(self.last_n): 
            if len(history) - i * self.step < 1:
                hist_i = None
            else:
                hist_i = history[:len(history) - i * self.step]
            rval.update(self._output(hist_i, i))
        return rval
        
class DeltaEMA(object):
    def __init__(self, bar_width, ema_n, last_n=3, skip=1):
        self.__dict__.update(locals())
        del self.self
        self.input_key = 'price'
        self.output_key = 'delta_ema' + str(self.ema_n) + '_skip' + str(self.skip)
        self.ema = EMA(ema_n)
        self.ema2 = EMA(ema_n/2)
        self.outputs = []
               
    def output(self, history):
        if history.size >= self.bar_width + 1:
            delta = history[-1] - history[-1-self.bar_width]
            delta_ema = self.ema.step(delta)
            delta_ema2 = self.ema2.step(delta_ema)
            if len(self.outputs) == self.last_n * self.skip:
                del self.outputs[0]
            self.outputs.append(delta_ema2)
        
        rval = OrderedDict()
        for n in range(self.last_n):
            key = self.output_key
            if n != 0:
                key += ('-' + str(n))
            rval[key] = self.outputs[len(self.outputs) - 1 - n * self.skip] if len(self.outputs) - 1 - n * self.skip >= 0 else 0.
        return rval 
    
class KD(object):
    def __init__(self, preprocessor, t, sma_n=None, last_n=1, skip=None):
        self.__dict__.update(locals())
        del self.self
        if self.skip is None:
            self.skip = self.preprocessor.output_step
        assert self.skip % self.preprocessor.output_step == 0
        if self.sma_n is None:
            self.sma_n = self.t / 2 / self.preprocessor.output_step
        self.input_key = 'price'
        self.output_key = 'kd' + str(self.t)
        self.sma = SMA(self.sma_n)
        self.outputs = []
               
    def _output(self, history):
        if history.size >= self.t:
            max_t = history[-self.t :].max()
            min_t = history[-self.t :].min()
            if max_t - min_t == 0:
                RSV = 0.5
            else:
                RSV = (history[-1] - min_t) * 1. / (max_t - min_t)
            RSV = RSV * 2. - 1.
            K = self.sma.step(RSV)
            self.outputs.append((RSV, K))
            return
        self.outputs.append((0., 0.))
        
    def output(self, history):
        self._output(history)    
        if len(self.outputs) >= MAX_OUTPUTS_LEN:
            del self.outputs[: MAX_OUTPUTS_LEN/2] 
            
        rval = OrderedDict()
        skip = self.skip / self.preprocessor.output_step
        for n in range(self.last_n):
            key = self.output_key + '-' + str(n)
            if len(self.outputs) - 1 - skip * n >= 0:
#                rval[key + '_RSV'] = self.outputs[len(self.outputs) - 1 - skip * n][0]
                rval[key + '_K'] = self.outputs[len(self.outputs) - 1 - skip * n][1]
            else:
#                rval[key + '_RSV'] = 0.
                rval[key + '_K'] = 0.
        return rval 
    
class RSI(object):
    def __init__(self, preprocessor, t, last_n=1, skip=None):
        self.__dict__.update(locals())
        del self.self
        if self.skip is None:
            self.skip = self.preprocessor.output_step
        assert self.skip % self.preprocessor.output_step == 0
        self.input_key = 'price'
        self.output_key = 'rsi' + str(self.t)
        sma_n = self.t / self.preprocessor.output_step
        self.gain_sma = SMA(sma_n)
        self.loss_sma = SMA(sma_n)
        self.outputs = []
               
    def _output(self, history):
        if len(history) -2 < 0:
            delta = 0.
        else:
            delta = history[-1] - history[-2]
            
        if delta >= 0.:
            gain = delta
            loss = 0.
        else:
            gain = 0.
            loss = -delta
        avg_gain = self.gain_sma.step(gain)
        avg_loss = self.loss_sma.step(loss)
        
        if avg_gain + avg_loss == 0.:
            signed_rsi = 0.
        else:
            signed_rsi = (avg_gain - avg_loss) * 1. / (avg_gain + avg_loss)
        self.outputs.append(signed_rsi)
        
    def output(self, history):
        self._output(history)    
        if len(self.outputs) >= MAX_OUTPUTS_LEN:
            del self.outputs[: MAX_OUTPUTS_LEN/2] 
            
        rval = OrderedDict()
        skip = self.skip / self.preprocessor.output_step
        for n in range(self.last_n):
            key = self.output_key + '-' + str(n)
            if len(self.outputs) - 1 - skip * n >= 0:
                rval[key] = self.outputs[len(self.outputs) - 1 - skip * n]
            else:
                rval[key] = 0.
        return rval 
    
#@profile
def as_arrays(X):
    for k in X:
        X[k] = np.array(X[k])
    return X

#@profile
def dict_append(X, x):
    if len(X) == 0:
        for key in x:
            X[key] = []
    for key in x:
        X[key].append(x[key])

def remove_extreme(s, cutoff):
    sorted = np.sort(s)
    cutoff_len = sorted.size * cutoff
    if sorted[0] * sorted[-1] < 0:
        low = sorted[cutoff_len]    
        s[s < low] = low
    high = sorted[-cutoff_len]
    s[s > high] = high
    return s

class Preprocessor(object):
    def __init__(self, output_step):
        self.__dict__.update(locals())
        del self.self
        self.now = -1
        self.max_hist_len = 32402 * 22
        self.history = OrderedDict()
        
    def set_generators(self, generators):
        self.generators = generators
        
    def input(self, tick):
        if len(self.history) == 0:
            for key in tick:
                self.history[key] = np.zeros(self.max_hist_len)
                
        for key in tick:
            self.history[key][self.now] = tick[key]
        self.now += 1
       
#    @profile 
    def output_xd(self):
        rval = OrderedDict()
        for g in self.generators:
            history = self.history[g.input_key][:self.now+1]
            rval.update(g.output(history))
        return rval
     
#    @profile
    def output_yd(self, transaction_cost):
        if not hasattr(self, 'tick_size'):
            self.tick_size = (self.history['ask_price'][:200000] - self.history['bid_price'][:200000]).mean()
            mean_price = self.history['price'][:200000].mean()
            print 'tick_size =', self.tick_size, 'mean_price =', mean_price
        rval = OrderedDict()
        future_mean = self.history['price'][self.now : self.now + 800*2 + 1]
        future_ask = self.history['ask_price'][self.now : self.now + 800*2 + 1]
        future_bid = self.history['bid_price'][self.now : self.now + 800*2 + 1]
#        for i in [5, 15, 30, 60, 180, 800]:
        for i in [30, 60]:
#            rval['ideal_gain' + str(i)] = (future_mean[i*2] - future_mean[0]) * 1. / future_mean[0]
#            if abs(rval['ideal_gain' + str(i)]) > .1:
#                print 'now =', self.now, ':', future_mean[0], '->', future_mean[i*2] 
#            if rval['ideal_gain' + str(i)] > 0:
#                rval['real_gain' + str(i)] = max(0., (future_bid[i*2] - future_ask[0]) * 1. / future_mean[0])
#                rval['real+_gain' + str(i)] = max(0., (future_bid[i*2] - future_ask[0] - self.tick_size) * 1. / future_mean[0])
#            elif rval['ideal_gain' + str(i)] < 0:
#                rval['real_gain' + str(i)] = min(0, (future_ask[i*2] - future_bid[0]) * 1. / future_mean[0])
#                rval['real+_gain' + str(i)] = min(0, (future_ask[i*2] - future_bid[0] + self.tick_size) * 1. / future_mean[0])
#            else:
#                rval['real_gain' + str(i)] = 0.
#                rval['real+_gain' + str(i)] = 0.
            rval['buy_gain' + str(i)] = (future_mean[i*2] - future_mean[0]) * 1. / future_mean[0] - transaction_cost
            rval['sell_gain' + str(i)] = (future_mean[0] - future_mean[i*2]) * 1. / future_mean[0] - transaction_cost
        return rval
    
#    @profile
    def preprocess(self, ticks, output='X', transaction_cost=.001):
        self.history = ticks
        Xd = OrderedDict()
        for i in range(0, ticks['price'].size - 2000, self.output_step):
            self.now = i
            if output == 'X':
                xd = self.output_xd()
            else:
                assert output == 'Y'
                xd = self.output_yd(transaction_cost)
            dict_append(Xd, xd)
        Xd = as_arrays(Xd) 
        return Xd
    
def build(Xd, includes=None, excludes=[]):
    def find_subs(k, subs):
        for sub in subs:
            if sub in k:
                return True
        return False
    
    if includes is None:
        left = Xd.keys()
    else:
        left = [k for k in Xd.keys() if find_subs(k, includes)]
        
    if len(excludes) > 0:
        left = [k for k in left if not find_subs(k, excludes)]
        
    X = np.array([Xd[k] for k in left]).T
    return X
    
def split(X, extremum_cutoff=0., normalize=True, train_pct=2./3):
    X = np.copy(X)
    if extremum_cutoff != 0.:
        for i in range(X.shape[1]):
            remove_extreme(X[:,i], extremum_cutoff)
            
    if normalize:
#        X_train = X
#        X_mean = X_train.mean(axis=0)
#        print 'X_mean =', X_mean
#        X_train -= X_mean
#        X_stdev = np.sqrt(X_train.var(axis=0))
#        X = X - X_mean
        X -= X.mean(axis=0)
        X_stdev = np.sqrt(X.var(axis=0))
        X /= X_stdev
    X = X.astype('float32')
        
    X_train = X[:X.shape[0] * train_pct]
    X_test = X[X.shape[0] * train_pct:]
    return X_train, X_test
        