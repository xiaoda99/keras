import os
import ctypes
import numpy as np
import math
import gzip
import cPickle
import pylab as plt
import matplotlib.gridspec as gridspec
from profilehooks import profile

class Delta(object):
    def __init__(self):
        pass
        
    def step(self, x):
        if not hasattr(self, 'prev_val'):
            self.prev_val = x
            return 0
        delta = x - self.prev_val
        self.prev_val = x
        return delta
    
class EMA(object):
    def __init__(self, n):
        self.__dict__.update(locals())
        del self.self
        self.history = []
           
    def step(self, x):
        if self.n == 1:
            return x
        
        if len(self.history) < self.n:
            self.history.append(x)
            ma = sum(self.history) * 1. / len(self.history)
            self.prev_ma = ma
            return ma 
        
        alpha = 2./(self.n + 1)
        ma = alpha * x + (1 - alpha) * self.prev_ma
        self.prev_ma = ma
        return ma

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
    
class BOLL(object):
    def __init__(self, n):
        assert n > 1
        self.__dict__.update(locals())
        del self.self
        self.history = []
        self.hist_sum = 0.
        self.hist_len = 0
           
    def step(self, x):
        if self.hist_len == self.n:
            first = self.history.pop(0)
            self.hist_sum -= first
            self.hist_len -= 1
            
        self.history.append(x)
        self.hist_sum += x
        self.hist_len += 1
        
        ma = self.hist_sum * 1. / self.hist_len
        hist = np.array(self.history)
        var = np.square(hist - ma).mean()
        stdev = math.sqrt(var)
        if stdev < .02:
#            print 'stdev =', stdev
            z = 0.
            prob_density = 1.
        else:
            z = (x - ma) * 1. / stdev
            prob_density = math.exp(-z**2 / 2.)   # unnormalized
        sign = 1 if z >= 0 else -1
        return z, (1. - prob_density) * sign, ma, stdev
    
class KD(object):
    def __init__(self, n, m1, m2):
        assert n > 1
        self.__dict__.update(locals())
        del self.self
        self.K_SMA = SMA(m1)
        self.D_SMA = SMA(m2)
        self.history = []
        self.hist_len = 0
           
    def step(self, x):
        if self.hist_len == self.n:
            first = self.history.pop(0)
            self.hist_len -= 1
            
        self.history.append(x)
        self.hist_len += 1
        
        if max(self.history) == min(self.history):
            RSV = .5
#            print 'max == min'
        else:
            RSV = (x - min(self.history)) * 1. / (max(self.history) - min(self.history))
        K = self.K_SMA.step(RSV) 
        D = self.D_SMA.step(K)
        K = K*2. - 1.
        D = D*2. - 1.
        return K, D
     
class ATR(object):
    def __init__(self, t, n):
        assert t > 1
        self.__dict__.update(locals())
        del self.self
        self.ATR_SMA = SMA(n)
        self.history = []
        self.hist_len = 0
           
    def step(self, x):
        if self.hist_len == self.t:
            first = self.history.pop(0)
            self.hist_len -= 1
            
        self.history.append(x)
        self.hist_len += 1
        
        TR = max(self.history) - min(self.history)
        ATR = self.ATR_SMA.step(TR)
        return ATR
        
class Momentum(object):
    def __init__(self, n):
        self.__dict__.update(locals())
        del self.self
        self.history = []
        self.hist_len = 0
           
    def step(self, x):
        assert self.n > 1
        
        if self.hist_len == self.n:
            first = self.history.pop(0)
            self.hist_len -= 1
        elif len(self.history) > 0:
            first = self.history[0]
        else:
            first = x
            
        self.history.append(x)
        self.hist_len += 1
        
        return x - first
            
class Pattern(object):
    ''' Combine PriceChannel, Breakthrough and Momentum. '''
    def __init__(self, n):
        self.__dict__.update(locals())
        del self.self
        self.history = []
           
    def step(self, x):
        assert self.n > 1
        
        if len(self.history) == self.n:
            first = self.history.pop(0)
        elif len(self.history) > 0:
            first = self.history[0]
        else:
            first = x
            
        self.history.append(x)
        hist = np.array(self.history)
        
        if len(self.history) < self.n:
            high, low = x, x
            break_up, break_down = 0, 0
        else:
            hist_len = len(self.history)
            high, low = hist[:hist_len/2].max(), hist[:hist_len/2].min()
            break_up, break_down = max(0, x - hist[:hist_len/2].max()), min(0, x - hist[:hist_len/2].min()) 
        momentum = x - first
        return high, low, break_up, break_down, momentum
        
class RSI(object):
    def __init__(self, n):
        assert n > 1
        self.__dict__.update(locals())
        del self.self
        self.gain_EMA = EMA(n)
        self.loss_EMA = EMA(n)
           
    def step(self, gain, loss):     
        avg_gain = self.gain_EMA.step(gain)
        avg_loss = self.loss_EMA.step(loss)
        
        if avg_gain + avg_loss == 0.:
            return 0.
        else:
            return (avg_gain - avg_loss) * 1. / (avg_gain + avg_loss)
        
class OBV(object):
    def __init__(self, n):
        self.__dict__.update(locals())
        del self.self
        self.history_up = []
        self.history_down = []
        self.hist_len = 0
           
    def step(self, up, down):
        assert self.n > 1
        
        if self.hist_len == self.n:
            self.history_up.pop(0)
            self.history_down.pop(0)
            self.hist_len -= 1
            
        self.history_up.append(up)
        self.history_down.append(down)
        self.hist_len += 1
        
        up_sum = sum(self.history_up)
        down_sum = sum(self.history_down)
        return up_sum - down_sum
    
class TickDataItem(object):
    def __init__(self, data):
        d = data  
        self.ask_price = d[2]
        self.bid_price = d[18]
        self.price = (self.ask_price + self.bid_price) * 1. / 2
        self.deal_vol = sum(d[41:41+7])
        
        self.ask_vol = {}
        self.bid_vol = {}
        for i in range(8):
            self.ask_vol[d[2+i]] = d[10+i]
            self.bid_vol[d[18+i]] = d[26+i]
        self.imbalance = d[0] + d[1]

class IndicatorSet(object):
    def __init__(self):
        pass
           
   
class Preprocessor(object):
    def __init__(self, hist_len=3):
        self.hist_len = hist_len
        self.x_hist = []
        
    def step(self, tick_data_item):
        d = tick_data_item
        x = IndicatorSet()
        x.price = d.price
        x.bid_price = d.bid_price
        x.ask_price = d.ask_price
        x.price_diff1 = d.ask_price - d.bid_price
            
        if not hasattr(self, 'price_Delta1'):
            self.price_Delta1 = Delta()
        x.price_delta1 = self.price_Delta1.step(d.price)
        if not hasattr(self, 'bid_price_Delta1'):
            self.bid_price_Delta1 = Delta()
        x.bid_price_delta1 = self.bid_price_Delta1.step(d.bid_price)
        if not hasattr(self, 'ask_price_Delta1'):
            self.ask_price_Delta1 = Delta()
        x.ask_price_delta1 = self.ask_price_Delta1.step(d.ask_price)
        if not hasattr(self, 'price_diff_Delta1'):
            self.price_diff_Delta1 = Delta()
        x.price_diff_delta1 = self.price_diff_Delta1.step(x.price_diff1)
        
        if not hasattr(self, 'bid_price_delta_EMA'):
            self.bid_price_delta_EMA = {}
        if not hasattr(self, 'ask_price_delta_EMA'):
            self.ask_price_delta_EMA = {}
        x.bid_price_delta_ema = {}
        x.ask_price_delta_ema = {}
        for t in [4,]:
            if not t in self.bid_price_delta_EMA:
                self.bid_price_delta_EMA[t] = EMA(t)
            x.bid_price_delta_ema[t] = self.bid_price_delta_EMA[t].step(x.bid_price_delta1)
            if not t in self.ask_price_delta_EMA:
                self.ask_price_delta_EMA[t] = EMA(t)
            x.ask_price_delta_ema[t] = self.ask_price_delta_EMA[t].step(x.ask_price_delta1)
        
        if x.price_delta1 > 0:
            buy_vol = d.deal_vol
            sell_vol = 0
            x.deal_vol_diff1 = buy_vol
        elif x.price_delta1 < 0:
            buy_vol = 0
            sell_vol = d.deal_vol
            x.deal_vol_diff1 = -sell_vol
        else:
            buy_vol = d.deal_vol / 2
            sell_vol = d.deal_vol / 2
            x.deal_vol_diff1 = 0
#        x.deal_vol_diff1 = d.deal_vol
            
        x.bid_vol = d.bid_vol
        x.ask_vol = d.ask_vol
        
        bid_vol1 = x.bid_vol[d.bid_price]
        ask_vol1 = x.ask_vol[d.ask_price]
        bid_vol2 = x.bid_vol[d.bid_price-1]
        ask_vol2 = x.ask_vol[d.ask_price+1]
#       order_vol_ratio1 = math.log(bid_vol0 * 1. / ask_vol0)
        x.order_vol_diff1 = bid_vol1 - ask_vol1
        x.order_vol_diff2 = bid_vol2 - ask_vol2
        if not hasattr(self, 'order_vol_diff_Delta1'):
            self.order_vol_diff_Delta1 = Delta()
        x.order_vol_diff_delta1 = self.order_vol_diff_Delta1.step(x.order_vol_diff1)
        if not hasattr(self, 'order_vol_diff_EMA'):
            self.order_vol_diff_EMA = {}
        x.order_vol_diff_ema = {}
        for t in [5,]:
            if not t in self.order_vol_diff_EMA:
                self.order_vol_diff_EMA[t] = EMA(t)
            x.order_vol_diff_ema[t] = self.order_vol_diff_EMA[t].step(x.order_vol_diff1)
                
        if not hasattr(self, 'price_EMA'):
            self.price_EMA = {}
        if not hasattr(self, 'price_Delta'):
            self.price_Delta = {}
        if not hasattr(self, 'price_delta_EMA'):
            self.price_delta_EMA = {}
        if not hasattr(self, 'price_DDelta'):
            self.price_DDelta = {}
        if not hasattr(self, 'price_acc_EMA'):
            self.price_acc_EMA = {}
        x.price_ema = {}
#        x.price_delta = {}
        x.price_delta_ema = {}
#        x.price_acc = {}
        x.price_acc_ema = {}
        for t in [5, 10, 20]:
            if not t in self.price_EMA:
                self.price_EMA[t] = EMA(t)
            x.price_ema[t] = self.price_EMA[t].step(d.price)
            
            if not t in self.price_Delta:
                self.price_Delta[t] = Delta()
            price_delta = self.price_Delta[t].step(x.price_ema[t])
            
            if not t in self.price_delta_EMA:
                self.price_delta_EMA[t] = EMA(t/2)
            x.price_delta_ema[t] = self.price_delta_EMA[t].step(price_delta)
            
            if not t in self.price_DDelta:
                self.price_DDelta[t] = Delta()
            price_acc = self.price_DDelta[t].step(price_delta)
            
            if not t in self.price_acc_EMA:
                self.price_acc_EMA[t] = EMA(t/2)
            x.price_acc_ema[t] = self.price_acc_EMA[t].step(price_acc)
            
            
#            if not t in self.price_Delta:
#                self.price_Delta[t] = Delta()
#            price_delta = self.price_Delta[t].step(x.price)
#            
#            if not t in self.price_delta_EMA:
#                self.price_delta_EMA[t] = EMA(t)
#            x.price_delta_ema[t] = self.price_delta_EMA[t].step(price_delta)
#            
#            if not t in self.price_DDelta:
#                self.price_DDelta[t] = Delta()
#            price_acc = self.price_DDelta[t].step(price_delta)
#            
#            if not t in self.price_acc_EMA:
#                self.price_acc_EMA[t] = EMA(t)
#            x.price_acc_ema[t] = self.price_acc_EMA[t].step(price_acc)
            
        if not hasattr(self, 'BOLL'):
            self.BOLL = {}
        if not hasattr(self, 'ATR'):
            self.ATR = {}
        x.boll_z = {}; x.boll_exp = {}; x.boll_ma = {}; x.boll_sd = {}
        x.atr = {}
        for t in [80,]:
            if not t in self.BOLL:
                self.BOLL[t] = BOLL(t)
            x.boll_z[t], x.boll_exp[t], x.boll_ma[t], x.boll_sd[t] = self.BOLL[t].step(d.price)
            
            if not t in self.ATR:
                self.ATR[t] = ATR(t/4, t)
            x.atr[t] = self.ATR[t].step(d.price)
        
        if not hasattr(self, 'KD'):
            self.KD = {}
        x.k = {}; x.d = {}
        for t in [10,]:
            if not t in self.KD:
                self.KD[t] = KD(t, t/2, t/2)
            x.k[t], x.d[t] = self.KD[t].step(d.price)
                        
        if not hasattr(self, 'buy_vol_SMA'):
            self.buy_vol_SMA = {}
        if not hasattr(self, 'sell_vol_SMA'):
            self.sell_vol_SMA = {}
        if not hasattr(self, 'buy_vol_Delta'):
            self.buy_vol_Delta = {}
        if not hasattr(self, 'sell_vol_Delta'):
            self.sell_vol_Delta = {}
        if not hasattr(self, 'buy_vol_delta_SMA'):
            self.buy_vol_delta_SMA = {}
        if not hasattr(self, 'sell_vol_delta_SMA'):
            self.sell_vol_delta_SMA = {}
        
        x.deal_vol_diff = {}
        x.buy_vol_delta = {}
        x.sell_vol_delta = {}
        x.buy_vol_delta_sma = {}
        x.sell_vol_delta_sma = {}
        
        for t in [5,]:
            if not t in self.buy_vol_SMA:
                self.buy_vol_SMA[t] = SMA(t)
            buy_vol_sma = self.buy_vol_SMA[t].step(buy_vol)
            if not t in self.sell_vol_SMA:
                self.sell_vol_SMA[t] = SMA(t)
            sell_vol_sma = self.sell_vol_SMA[t].step(sell_vol)
            x.deal_vol_diff[t] = buy_vol_sma - sell_vol_sma
            if not t in self.buy_vol_Delta:
                self.buy_vol_Delta[t] = Delta()
            buy_vol_delta = self.buy_vol_Delta[t].step(buy_vol_sma)
            if not t in self.sell_vol_Delta:
                self.sell_vol_Delta[t] = Delta()
            sell_vol_delta = self.sell_vol_Delta[t].step(sell_vol_sma)
            if not t in self.buy_vol_delta_SMA:
                self.buy_vol_delta_SMA[t] = SMA(t)
            x.buy_vol_delta_sma[t] = self.buy_vol_delta_SMA[t].step(buy_vol_delta)
            if not t in self.sell_vol_delta_SMA:
                self.sell_vol_delta_SMA[t] = SMA(t)
            x.sell_vol_delta_sma[t] = self.sell_vol_delta_SMA[t].step(sell_vol_delta)
              
        if len(self.x_hist) == self.hist_len * 2 + 1:
            del self.x_hist[0]
        self.x_hist.append(x)  
        indicators = []
        # 3-step price & volume features, 21x3
        for i in reversed(range(len(self.x_hist) - self.hist_len, len(self.x_hist))):
            if i < 1:
                indicators += ([0.] * 14)
                continue
            xi = self.x_hist[i]
            indicators += [ 
                           xi.ask_price_delta1,
                           xi.bid_price_delta1, 
                           xi.price_diff1, 
#                           xi.price_diff_delta1,
#                           xi.deal_vol_diff1, 
#                           xi.order_vol_diff1, 
#                           xi.order_vol_diff2, 
                           xi.order_vol_diff_delta1
                        ]
            
            for j in reversed(range(-2, 3)):
                curr_bid_vol = xi.bid_vol[d.bid_price + j] if d.bid_price + j in xi.bid_vol else 0
                prev_bid_vol = self.x_hist[i-1].bid_vol[d.bid_price + j] if d.bid_price + j in self.x_hist[i-1].bid_vol else 0
                indicators.append(curr_bid_vol - prev_bid_vol)
            for j in range(-2, 3):
                curr_ask_vol = xi.ask_vol[d.ask_price + j] if d.ask_price + j in xi.ask_vol else 0
                prev_ask_vol = self.x_hist[i-1].ask_vol[d.ask_price + j] if d.ask_price + j in self.x_hist[i-1].ask_vol else 0
                indicators.append(curr_ask_vol - prev_ask_vol)
        
        indicators += [x.ask_vol[x.ask_price + j] for j in range(5)]
        indicators += [x.bid_vol[x.bid_price - j] for j in range(5)]
        
        # longer MA features, 9x3
        for i in reversed(range(len(self.x_hist) - self.hist_len, len(self.x_hist))):
#        for i in reversed(range(len(self.x_hist) - self.hist_len * 2 + 1, len(self.x_hist) + 1, 2)):
            if i < 0:
                indicators += ([0.] * 9)
                continue
            xi = self.x_hist[i]
            indicators += [
                           xi.price_ema[5] - xi.price_ema[20], 
                           xi.price_delta_ema[10],
                           xi.price_delta_ema[20],
                           xi.price_acc_ema[10],
                           xi.price_acc_ema[20],
                           xi.boll_z[80],
                           xi.k[10],
                           xi.buy_vol_delta_sma[5],
                           xi.sell_vol_delta_sma[5]
                           ]
        ind_arr = np.array(indicators)
#        print 'ind_arr.sum(), np.abs(ind_arr).sum() =', ind_arr.sum(), np.abs(ind_arr).sum()
        return ind_arr
    
class TestPreprocessor(object):
    def __init__(self, hist_len=3):
        self.hist_len = hist_len
        self.x_hist = []
        
        self.price_Momentum = {}
        self.deal_vol_OBV = {}
        self.deal_vol_RSI = {}
        self.price_RSI = {}
        
    def step(self, tick_data_item):
        d = tick_data_item
        x = IndicatorSet()
        x.price = d.price
        x.bid_price = d.bid_price
        x.ask_price = d.ask_price
        x.price_diff1 = d.ask_price - d.bid_price
            
        if not hasattr(self, 'price_Delta1'):
            self.price_Delta1 = Delta()
        x.price_delta1 = self.price_Delta1.step(d.price)
        if not hasattr(self, 'bid_price_Delta1'):
            self.bid_price_Delta1 = Delta()
        x.bid_price_delta1 = self.bid_price_Delta1.step(d.bid_price)
        if not hasattr(self, 'ask_price_Delta1'):
            self.ask_price_Delta1 = Delta()
        x.ask_price_delta1 = self.ask_price_Delta1.step(d.ask_price)
        if not hasattr(self, 'price_diff_Delta1'):
            self.price_diff_Delta1 = Delta()
        x.price_diff_delta1 = self.price_diff_Delta1.step(x.price_diff1)
        
        if x.price_delta1 > 0:
            buy_vol = d.deal_vol
            sell_vol = 0
        elif x.price_delta1 < 0:
            buy_vol = 0
            sell_vol = d.deal_vol
        else:
            buy_vol = d.deal_vol / 2
            sell_vol = d.deal_vol / 2
        x.deal_vol = d.deal_vol
            
        x.bid_vol = d.bid_vol
        x.ask_vol = d.ask_vol
        
        bid_vol1 = x.bid_vol[d.bid_price]
        ask_vol1 = x.ask_vol[d.ask_price]
        x.order_vol_ratio1 = math.log(bid_vol1 * 1. / ask_vol1)
        x.order_vol_diff1 = bid_vol1 - ask_vol1
        if not hasattr(self, 'order_vol_diff_Delta1'):
            self.order_vol_diff_Delta1 = Delta()
        x.order_vol_diff_delta1 = self.order_vol_diff_Delta1.step(x.order_vol_diff1)
        if not hasattr(self, 'order_vol_ratio_Delta1'):
            self.order_vol_ratio_Delta1 = Delta()
        x.order_vol_ratio_delta1 = self.order_vol_ratio_Delta1.step(x.order_vol_ratio1)
                
        if not hasattr(self, 'price_EMA'):
            self.price_EMA = {}
        if not hasattr(self, 'price_Delta'):
            self.price_Delta = {}
        if not hasattr(self, 'price_delta_EMA'):
            self.price_delta_EMA = {}
        if not hasattr(self, 'true_price_delta_EMA'):
            self.true_price_delta_EMA = {}
        x.price_ema = {}
#        x.price_delta = {}
        x.price_delta_ema = {}
        x.true_price_delta_ema = {}
        for t in [5, 10, 20, 40, 80]:
            if not t in self.price_EMA:
                self.price_EMA[t] = EMA(t)
            x.price_ema[t] = self.price_EMA[t].step(d.price)
            
            if not t in self.price_Delta:
                self.price_Delta[t] = Delta()
            price_delta = self.price_Delta[t].step(x.price_ema[t])
            
            if not t in self.price_delta_EMA:
                self.price_delta_EMA[t] = EMA(t/2)
            x.price_delta_ema[t] = self.price_delta_EMA[t].step(price_delta)
            
            if not t in self.true_price_delta_EMA:
                self.true_price_delta_EMA[t] = EMA(t)
            x.true_price_delta_ema[t] = self.true_price_delta_EMA[t].step(x.price_delta1)
            
#        if not hasattr(self, 'BOLL'):
#            self.BOLL = {}
#        if not hasattr(self, 'ATR'):
#            self.ATR = {}
#        x.boll_z = {}; x.boll_exp = {}; x.boll_ma = {}; x.boll_sd = {}
#        x.atr = {}
#        for t in [80,]:
#            if not t in self.BOLL:
#                self.BOLL[t] = BOLL(t)
#            x.boll_z[t], x.boll_exp[t], x.boll_ma[t], x.boll_sd[t] = self.BOLL[t].step(d.price)
#            
#            if not t in self.ATR:
#                self.ATR[t] = ATR(t/4, t)
#            x.atr[t] = self.ATR[t].step(d.price)
        
        if not hasattr(self, 'KD'):
            self.KD = {}
        x.k = {}; x.d = {}
        for t in [5, 10, 20]:
            if not t in self.KD:
                self.KD[t] = KD(t, t/2, t/2)
            x.k[t], x.d[t] = self.KD[t].step(d.price)
                        
        price_up = max(0, x.price_delta1)
        price_down = max(0, -x.price_delta1)
        x.deal_vol_obv = {}
        x.price_rsi = {}
        for t in [5, 10, 20]:
            if not t in self.deal_vol_OBV:
                self.deal_vol_OBV[t] = OBV(t)
            x.deal_vol_obv[t] = self.deal_vol_OBV[t].step(buy_vol, sell_vol)
            
            if not t in self.price_RSI:
                self.price_RSI[t] = RSI(t)
            x.price_rsi[t] = self.price_RSI[t].step(price_up, price_down) 
            
        # vol
        if not hasattr(self, 'vol_Delta1'):
            self.vol_Delta1 = Delta()
        x.vol_delta1 = self.vol_Delta1.step(d.deal_vol)
        if not hasattr(self, 'vol_EMA'):
            self.vol_EMA = {}
        if not hasattr(self, 'vol_Delta'):
            self.vol_Delta = {}
        if not hasattr(self, 'vol_delta_EMA'):
            self.vol_delta_EMA = {}
        if not hasattr(self, 'true_vol_delta_EMA'):
            self.true_vol_delta_EMA = {}
        x.vol_ema = {}
#        x.vol_delta = {}
        x.vol_delta_ema = {}
        x.true_vol_delta_ema = {}
        for t in [5, 10]:
            if not t in self.vol_EMA:
                self.vol_EMA[t] = EMA(t)
            x.vol_ema[t] = self.vol_EMA[t].step(d.deal_vol)
            
            if not t in self.vol_Delta:
                self.vol_Delta[t] = Delta()
            vol_delta = self.vol_Delta[t].step(x.vol_ema[t])
            
            if not t in self.vol_delta_EMA:
                self.vol_delta_EMA[t] = EMA(t/2)
            x.vol_delta_ema[t] = self.vol_delta_EMA[t].step(vol_delta)
            
            if not t in self.true_vol_delta_EMA:
                self.true_vol_delta_EMA[t] = EMA(t)
            x.true_vol_delta_ema[t] = self.true_vol_delta_EMA[t].step(x.vol_delta1)
            
        order_info = []
        if len(self.x_hist) == 0:
            order_info += [0] * 10
        else:
            for j in reversed(range(-2, 3)):
                curr_bid_vol = x.bid_vol[d.bid_price + j] if d.bid_price + j in x.bid_vol else 0
                prev_bid_vol = self.x_hist[-1].bid_vol[d.bid_price + j] if d.bid_price + j in self.x_hist[-1].bid_vol else 0
                order_info.append(curr_bid_vol - prev_bid_vol)
            for j in range(-2, 3):
                curr_ask_vol = x.ask_vol[d.ask_price + j] if d.ask_price + j in x.ask_vol else 0
                prev_ask_vol = self.x_hist[-1].ask_vol[d.ask_price + j] if d.ask_price + j in self.x_hist[-1].ask_vol else 0
                order_info.append(curr_ask_vol - prev_ask_vol)
    
        order_info += [x.ask_vol[x.ask_price + j] for j in range(5)]
        order_info += [x.bid_vol[x.bid_price - j] for j in range(5)]
        x.order_info = order_info
            
        if len(self.x_hist) == self.hist_len + 1:
            del self.x_hist[0]
        self.x_hist.append(x)  
#        indicators = []
#        xs = []
        # 3-step price & volume features, 21x3
#        for i in reversed(range(len(self.x_hist) - self.hist_len, len(self.x_hist))):
#        for i in range(len(self.x_hist) - self.hist_len, len(self.x_hist)):
#            if i < 1:
#                indicators += ([0.] * 43)
#                continue
#            xi = self.x_hist[i]
#            xs.append(xi)
#            
#            indicators += [ 
#                           xi.ask_price_delta1,
#                           xi.bid_price_delta1, 
#                           xi.price_diff1, 
##                           xi.price_diff_delta1,
##                           xi.deal_vol_diff1, 
##                           xi.order_vol_diff1, 
##                           xi.order_vol_diff2, 
#                           xi.order_vol_diff_delta1
#                        ]
#            
#            for j in reversed(range(-2, 3)):
#                curr_bid_vol = xi.bid_vol[d.bid_price + j] if d.bid_price + j in xi.bid_vol else 0
#                prev_bid_vol = self.x_hist[i-1].bid_vol[d.bid_price + j] if d.bid_price + j in self.x_hist[i-1].bid_vol else 0
#                indicators.append(curr_bid_vol - prev_bid_vol)
#            for j in range(-2, 3):
#                curr_ask_vol = xi.ask_vol[d.ask_price + j] if d.ask_price + j in xi.ask_vol else 0
#                prev_ask_vol = self.x_hist[i-1].ask_vol[d.ask_price + j] if d.ask_price + j in self.x_hist[i-1].ask_vol else 0
#                indicators.append(curr_ask_vol - prev_ask_vol)
#        
#            indicators += [xi.ask_vol[xi.ask_price + j] for j in range(5)]
#            indicators += [xi.bid_vol[xi.bid_price - j] for j in range(5)]
#        
#        # longer MA features, 9x3
##        for i in reversed(range(len(self.x_hist) - self.hist_len, len(self.x_hist))):
##            if i < 0:
##                indicators += ([0.] * 9)
##                continue
##            xi = self.x_hist[i]
#            indicators += [
#                           xi.price_ema[5] - xi.price_ema[20],
#                           xi.buy_vol_delta_sma[5],
#                           xi.sell_vol_delta_sma[5], 
#                           xi.price_acc_ema[10],
#                           xi.price_acc_ema[20],
#                           xi.k[10],
#                           xi.boll_z[80],
#                           xi.price_delta_ema[5], 
#                           xi.price_delta_ema[10],
#                           xi.price_delta_ema[20],
#                           # added from preprocessor.py
#                           xi.price_momentum[5],
#                           xi.price_momentum[10],
#                           xi.price_momentum[20],
#                           xi.buy_vol,
#                           xi.sell_vol,
#                           xi.deal_vol_rsi[5],
#                           xi.deal_vol_rsi[10],
#                           xi.price_rsi[5],
#                           xi.price_rsi[10],
#                           ]
#        ind_arr = np.array(indicators)
#        print 'ind_arr.sum(), np.abs(ind_arr).sum() =', ind_arr.sum(), np.abs(ind_arr).sum()
        return x
             