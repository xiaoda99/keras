import time
import numpy as np
import pylab as plt
import matplotlib.gridspec as gridspec

from build_tick_dataset import *

from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

class EMA(object):
    def __init__(self, n):
        self.__dict__.update(locals())
        del self.self
        self.alpha = 2./(self.n + 1.)
        self.sum = 0.
        self.sum_cnt = 0
           
    def step(self, x):
        if self.n == 1:
            return x
    
        if self.sum_cnt < self.n:
            self.sum += x
            self.sum_cnt += 1
            self.ema = self.sum * 1. / self.sum_cnt
#            if type(self.ema) != type(x):
#                print '1', self.sum_cnt, type(self.ema), type(x)
        else:
            self.ema = self.ema * (1. - self.alpha) + x * self.alpha
#            if type(self.ema) != type(x):
#                print '2', self.sum_cnt, type(self.ema), type(x)
        return self.ema
    
class SaliencyEMA():
    def __init__(self, 
                 simulator,
                 n = 60 * 60 * 2,  # 1 hour
                 stdev_coef = 1.5,
                 days = 5,
                 trading_hours_per_day = 3.75,
                 tick_size = 1,
                 smooth_n = 3
                 ):
        self.__dict__.update(locals())
        del self.self
        
        self.price = self.simulator.history['last_price'].astype('int')
        self.vol = self.simulator.history['vol']
        self.m = int(days * trading_hours_per_day * 60 * 60 * 2)
        self.w = np.ones(self.n + 1) / float(self.n + 1)
        self.y = self.price - np.convolve(self.price, self.w, mode='same')
        denorm = np.sqrt(np.convolve(self.y**2, self.w, mode='same'))
        denorm_mean = denorm.mean()
        denorm2 = (denorm > denorm_mean) * denorm + (denorm <= denorm_mean) * denorm_mean
#        self.ny = self.y
#        self.ny = (denorm > 0) * self.y / denorm
        self.ny = (denorm2 > 0) * self.y / denorm2
        self.ny_mean_EMA = EMA(self.m * 1. / self.n)
        self.ny_stdev_EMA = EMA(self.m * 1. / self.n)
        
        self.start_tick = self.n * 4
        self.saliency = np.zeros(self.price.max() + 1)
        self.saliency2 = np.zeros(self.price.max() + 1)
        self.saliency_EMA = EMA(self.m)
        self.saliency2_EMA = EMA(self.m)
        self.mean_saliency_EMA = EMA(self.m * 1. / self.n)
        
        assert self.smooth_n % 2 == 1
        self.smooth_w = np.ones((self.smooth_n - 1) * self.tick_size + 1) / self.smooth_n 
    
    def normalize(self, saliency_ema, level, name=None): 
        mean_saliency = saliency_ema.mean() * self.tick_size
        if self.saliency_EMA.sum_cnt == self.m and saliency_ema[level] > mean_saliency * 5. and \
                saliency_ema[level] > saliency_ema[level - self.tick_size] * 5.and \
                saliency_ema[level] > saliency_ema[level + self.tick_size] * 5.:
            if name is not None:
                print name
            print 'smooth out abnormal saliency at', level, \
                'mean_saliency =', mean_saliency, 'saliency =', saliency_ema[level]
            saliency_ema[level] = mean_saliency
            
    def smooth(self, saliency_ema):
        return np.convolve(saliency_ema, self.smooth_w, mode='same')
    
    def smooth_all(self):
        self.smoothed_saliency_ema = self.smooth(self.saliency_ema)
#        self.smoothed_saliency2_ema = self.smooth(self.saliency2_ema)
     
    def get_max_of_max(self, sr_level):
        low = sr_level
        high = sr_level
        while self.smoothed_saliency_ema[low] >= self.mean_saliency_ema * 4.:
            low -= 1
        while self.smoothed_saliency_ema[high] >= self.mean_saliency_ema * 4.:
            high += 1
        max_saliency = self.smoothed_saliency_ema[low : high].max()
        sr_level = self.smoothed_saliency_ema[low : high].argmax() + low
        return sr_level, max_saliency
    
    def get_nearby_sr_level(self, distance):
        distance *= self.tick_size
        self.smooth_all()
        last_price = self.price[self.simulator.now - 1]
        nearby_saliency = self.smoothed_saliency_ema[last_price - distance : last_price + distance + 1]
        max_saliency = nearby_saliency.max()
        sr_level = nearby_saliency.argmax() + last_price - distance
        if max_saliency >= self.mean_saliency_ema * 4.:
            sr_level, max_saliency = self.get_max_of_max(sr_level)
        return sr_level, max_saliency 
           
    def step(self):
        if self.simulator.now < self.start_tick:
            return None 
        
        self.saliency *= 0.
        self.saliency2 *= 0.
        if (self.simulator.now - self.start_tick) % self.n == 0: 
            mean = np.mean(self.ny[self.simulator.now - self.start_tick : self.simulator.now])
            self.mean = self.ny_mean_EMA.step(mean)
            stdev = np.sqrt(np.var(self.ny[self.simulator.now - self.start_tick : self.simulator.now]))
            self.stdev = self.ny_stdev_EMA.step(stdev)
                
#        mean = self.ny_mean_EMA.ema
#        stdev = self.ny_stdev_EMA.ema
        ny = self.ny[self.simulator.now - self.n]
        ny = ny * (abs(ny - self.mean) > self.stdev * self.stdev_coef)
        
        level = self.price[self.simulator.now - self.n]
        self.saliency[level] = abs(ny)
        self.saliency_ema = self.saliency_EMA.step(self.saliency)
        
        if (self.simulator.now - self.start_tick) % self.n == 0:  
            floor = self.price[self.simulator.now - self.start_tick : self.simulator.now].min()
            ceil = self.price[self.simulator.now - self.start_tick : self.simulator.now].max()
            mean_saliency = self.saliency_ema[floor : ceil + 1].mean() 
            self.mean_saliency_ema = self.mean_saliency_EMA.step(mean_saliency)
        
#        self.saliency2[level] = self.saliency[level] * self.vol[self.simulator.now - self.n]
#        self.saliency2_ema = self.saliency2_EMA.step(self.saliency2)
        
        if self.simulator.now % 360 == 0:
            self.normalize(self.saliency_ema, level)
#            self.normalize(self.saliency2_ema, level, 'saliency2_ema')
        return self.saliency_ema #, self.saliency2_ema
    
class IntensityEMA():
    def __init__(self, 
                 simulator,
                 days = 5,
                 trading_hours_per_day = 3.75,
                 tick_size = 1,
                 smooth_n = 3
                 ):
        self.__dict__.update(locals())
        del self.self
        
        self.price = self.simulator.history['last_price'].astype('int')
        self.vol = self.simulator.history['vol']
        self.m = int(days * trading_hours_per_day * 60 * 60 * 2)
        
        self.intensity = np.zeros(self.price.max() + 1)
        self.intensity_ema = np.zeros(self.price.max() + 1)
        self.intensity_sum = 0
        self.stay = np.zeros(self.price.max() + 1)
        self.stay_ema = np.zeros(self.price.max() + 1)
        self.stay_sum = 0
        self.sum_cnt = 0
        self.intensity_ema2 = np.zeros(self.price.max() + 1)
        self.alpha = 2. / (self.m + 1)
        
        assert self.smooth_n % 2 == 1
        self.smooth_w = np.ones((self.smooth_n - 1) * self.tick_size + 1) / self.smooth_n 
            
    def smooth(self, ema):
        return np.convolve(ema, self.smooth_w, mode='same')
    
    def smooth_all(self):
        self.smoothed_intensity_ema = self.smooth(self.intensity_ema)
        self.smoothed_intensity_ema2 = self.smooth(self.intensity_ema2)
        
    def step(self):
        self.intensity *= 0.
        self.stay *= 0.
        t = self.simulator.now - 1
        level = self.price[t]
        self.intensity[level] = self.vol[t]
        self.stay[level] = 1.
        if self.sum_cnt < self.m:
            self.intensity_sum += self.intensity
            self.stay_sum += self.stay
            self.sum_cnt += 1
            self.intensity_ema = self.intensity_sum * 1. / self.sum_cnt
            self.stay_ema = self.stay_sum * 1. / self.sum_cnt
        else:
            self.intensity_ema = self.intensity_ema * (1 - self.alpha) + self.intensity * self.alpha
            self.stay_ema = self.stay_ema * (1 - self.alpha) + self.stay * self.alpha
        self.intensity_ema2 *= 0.
        self.intensity_ema2[self.stay_ema > 0] = (self.intensity_ema / self.stay_ema)[self.stay_ema > 0]
        
#        if self.simulator.now % 100 == 0:
#            self.normalize(self.saliency_ema, level)
#            self.normalize(self.saliency2_ema, level, 'saliency2_ema')
        return self.intensity_ema, self.intensity_ema2
    
OPENED = 2
PLACED = 1
CLOSED = 0
class Simulator():
    def __init__(self, history, step_size=1, 
                 show_freq = 30 * 60 * 2, 
                 zoomin_show_freq = 3 * 60 * 2,
                 saliency_scale = 6*1e7,
                 intensity_scale = 3*1e5,
                 intensity2_scale = 3*1e2,
                 tick_size=1,
                 stoploss=6,
                 stopprofit=30,
                 sleeptime_on_action = 2
                 ):
        self.__dict__.update(locals())
        del self.self
        self.stoploss *= self.tick_size
        self.stopprofit *= self.tick_size
    
        self.now = 0
        self.indicators = []
        
        self.sr_level = None
        self.stopprofit_level = None
        self.stoploss_level = None
        self.pending_open_level = None
        self.open_level = None
        self.open_price = None
        self.state = CLOSED
        self.gains = []
        
    def add_indicator(self, indicator):
        self.indicators.append(indicator)
            
    def step(self, step_size=1):
        self.now += step_size
        for indicator in self.indicators:
            indicator.step()
    
    def do(self):
        if self.now - self.indicators[0].m < self.indicators[0].n:
            return
        
        if self.state == OPENED:
            if not self.try_stop_loss():
                self.try_stop_profit()
        elif self.state == PLACED:
            self.try_open()
        else:
            assert self.state == CLOSED
            self.try_place()
            
        if self.state != CLOSED and self.now % self.zoomin_show_freq == 0:
            self.zoomin_show()
        
    def try_stop_loss(self):
        last_price = self.history['last_price'][self.now - 1]
        if (last_price - self.sr_level) * (self.stoploss_level - self.sr_level) > 0 and \
                abs(last_price - self.sr_level) >= abs(self.stoploss_level - self.sr_level):
            self.close(last_price, False)
            return True
        return False
    
    def try_stop_profit(self):
        last_price = self.history['last_price'][self.now - 1]
        sr_level, saliency = self.indicators[0].get_nearby_sr_level(self.stoploss/2)
        if abs(last_price - self.sr_level) < self.stopprofit:
            if abs(last_price - self.sr_level) >= self.stoploss and \
                    abs(sr_level - self.sr_level) > self.stoploss and \
                    saliency >= self.indicators[0].mean_saliency_ema * 4. and \
                    saliency > self.indicators[0].smoothed_saliency_ema[self.sr_level]:
                self.stopprofit_level = sr_level
                self.close(last_price, True)
                return True
            else:
                return False
        else:
            if saliency >= self.indicators[0].mean_saliency_ema * 2.:
                self.stopprofit_level = sr_level
                self.close(last_price, True)
                return True
            else:
                return False
          
    def close(self, last_price, is_stop_profit):
        self.state = CLOSED
        self.zoomin_show(self.sleeptime_on_action)
        
        gain = abs(last_price - self.open_price)
        if not is_stop_profit:
            gain *= -1.
        if gain < 0 and abs(gain) > self.stoploss * 1.5 * 2:
            print gain, '->', -self.stoploss * 1.5 * 2
            gain = -self.stoploss * 1.5 * 2
#        else:
#            print gain
        if gain % self.tick_size != 0:
            print gain, self.tick_size
            assert False
        gain /= self.tick_size
        self.gains.append(gain)
        
        self.sr_level = None
        self.open_level = None
        self.stoploss_level = None
        self.stopprofit_level = None
          
    def try_place(self): 
        last_price = self.history['last_price'][self.now - 1]
        sr_level, saliency = self.indicators[0].get_nearby_sr_level(self.stoploss/2)
        if saliency >= self.indicators[0].mean_saliency_ema * 4. and abs(last_price - sr_level) < self.stoploss:
            self.sr_level = sr_level
            self.pending_open_level = [self.sr_level - self.stoploss, self.sr_level + self.stoploss]
            self.state = PLACED
            self.zoomin_show(self.sleeptime_on_action)
            return True
        return False
             
    def try_open(self): 
        last_price = self.history['last_price'][self.now - 1]
        if abs(last_price - self.sr_level) >= self.stoploss:
            self.pending_open_level = None
            if last_price > self.sr_level:
                self.open_level = self.sr_level + self.stoploss
                self.stoploss_level = self.sr_level - self.stoploss / 2. 
            else:
                self.open_level = self.sr_level - self.stoploss
                self.stoploss_level = self.sr_level + self.stoploss / 2. 
            self.open_price = last_price
            self.state = OPENED
            self.zoomin_show(self.sleeptime_on_action)
            return True
        return False
                     
    def showable(self):
        return self.now - self.indicators[0].m >= self.indicators[0].n and self.now % self.show_freq == 0
      
    def zoomin_show(self, sleeptime=0.5):
        return
        self.indicators[0].smooth_all()
        
        plt.ion()
        plt.clf()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
#        gs = gridspec.GridSpec(1, 1)
        
        y = self.indicators[0].smoothed_saliency_ema
        y_mean = self.indicators[0].mean_saliency_ema
        show_len = self.indicators[0].n
        
        scale = self.saliency_scale
        scale = scale * show_len * 1. / self.indicators[0].m
            
        ax0 = plt.subplot(gs[0])
        price = self.history['last_price'][self.now - show_len : self.now]
        plt.plot(price)
        floor = min(self.sr_level - self.stopprofit, price.min())
        ceil = max(self.sr_level + self.stopprofit, price.max())
        plt.plot(np.ones_like(price) * self.sr_level, color='k')
        if self.state == PLACED:
            assert self.pending_open_level is not None
            plt.plot(np.ones_like(price) * (self.pending_open_level[0]), '--', color='k', label='pending open')
            plt.plot(np.ones_like(price) * (self.pending_open_level[1]), '--', color='k', label='pending open')
        else:
            assert self.state == OPENED or self.state == CLOSED
            assert self.open_level is not None
            assert self.stoploss_level is not None
            assert self.open_level != self.sr_level
            assert self.stoploss_level != self.sr_level
            plt.plot(np.ones_like(price) * self.open_level, '--', color='k', label='open')
            plt.plot(np.ones_like(price) * self.stoploss_level, color='r', label='stoploss')
            if self.open_level > self.sr_level > 0:
                stopprofit_level0 = self.sr_level + self.stopprofit
            else:
                stopprofit_level0 = self.sr_level - self.stopprofit
            plt.plot(np.ones_like(price) * stopprofit_level0, 
                     color='g', label='stopprofit0')
            if self.state == CLOSED and self.stopprofit_level is not None:
                plt.plot(np.ones_like(price) * self.stopprofit_level, color='g', label='stopprofit')
            
        y = y[floor : ceil + 1] * scale
        y[y > show_len * 1.2] = show_len * 1.2
        plt.barh(np.arange(floor, ceil + 1) - 0.5, y, 1.0,
                 alpha=0.2, color='r', edgecolor='none')
        if y_mean is not None:
            y_mean = int(y_mean * 2. * scale)
            ax0.set_xticks(np.arange(0, show_len, y_mean))
        plt.grid()
        plt.legend(loc='upper right')
        
        ax = plt.subplot(gs[1], sharex=ax0)
        pos = self.history['pos'][self.now - show_len : self.now]
        plt.plot(pos)
        plt.grid()
        plt.legend(loc='upper right')
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.draw()
        time.sleep(sleeptime)
        
    def _plot_histogram(self, gs, y, scale, y_mean=None, show_len=None, label=None, sharex=None):
        if show_len is None:
            show_len = self.indicators[0].m
        else:
            scale = scale * show_len * 1. / self.indicators[0].m
            
        ax = plt.subplot(gs, sharex=sharex)
        price = self.history['last_price'][self.now - show_len : self.now]
        plt.plot(price)
        floor = price.min()
        ceil= price.max()
#        floor = self.history['last_price'].min()
#        ceil= self.history['last_price'].max()
        y = y[floor : ceil + 1] * scale
        y[y > show_len * 1.2] = show_len * 1.2
        plt.barh(np.arange(floor, ceil + 1), y, 1.0, label=label,
                 alpha=0.2, color='r', edgecolor='none')
        if y_mean is not None:
            y_mean = int(y_mean * 2. * scale)
            ax.set_xticks(np.arange(0, show_len, y_mean))
        plt.grid()
        plt.legend(loc='upper right')
        return ax
    
    def show(self):
        for indicator in self.indicators:
            indicator.smooth_all()
        
        plt.ion()
        plt.clf()
        gs = gridspec.GridSpec(2, 2, height_ratios=[1,1])
#        gs = gridspec.GridSpec(1, 1)
        
        y = self.indicators[0].smoothed_saliency_ema
        y_mean = self.indicators[0].mean_saliency_ema
        ax0 = self._plot_histogram(gs[0], y, self.saliency_scale, y_mean=y_mean, label='saliency')
        
        y = self.indicators[1].smoothed_intensity_ema
        ax = self._plot_histogram(gs[1], y, self.intensity_scale, label='intensity', sharex=ax0)
        
#        y = self.indicators[2].smoothed_saliency_ema
#        ax = self._plot_histogram(gs[2], y, self.saliency_scale, label='stdev=1.5', sharex=ax0)
        
        y = self.indicators[1].smoothed_intensity_ema2
        ax = self._plot_histogram(gs[3], y, self.intensity2_scale, label='mean_intensity', sharex=ax0)
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.draw()
        
    def finished(self):
        return self.now >= self.history['last_price'].shape[0] 
    
    def run(self):
        while not self.finished():
            self.step()
            self.do()
        print self.gains
        self.gains = np.array(self.gains)
        mean_gain = self.gains[self.gains > 0].mean()
        mean_loss = -self.gains[self.gains < 0].mean()
        print 'total_gain =', self.gains.sum(), 'total_trades =', self.gains.size, \
            'gain/trade =', self.gains.mean(), 'winning rate =', (self.gains > 0).mean(), \
            'mean_gain / mean_loss =', mean_gain, '/', mean_loss, '=', mean_gain / mean_loss 
                
if __name__ == '__main__':
#    ticks = load_ticks('zc', 'SR', 2015, [9,], use_cache=True); tick_size = 1; trading_hours_per_day = 6.25
#    ticks = load_ticks('zc', 'MA', 2015, [11,], use_cache=True); tick_size = 1; trading_hours_per_day = 6.25
#    ticks = load_ticks('zc', 'TA', 2015, [11,], use_cache=True); tick_size = 2; trading_hours_per_day = 6.25
    ticks = load_ticks('dc', 'pp', 2015, [11], use_cache=True); tick_size = 1; trading_hours_per_day = 3.75
#    ticks = load_ticks('sc', 'zn', 2015, [11,], use_cache=True); tick_size = 5; trading_hours_per_day = 7.75
    s = Simulator(ticks, tick_size=tick_size)
    for indicator in [
            SaliencyEMA(s, tick_size=tick_size, trading_hours_per_day=trading_hours_per_day),
#            IntensityEMA(s, tick_size=tick_size, trading_hours_per_day=trading_hours_per_day),
            ]:
        s.add_indicator(indicator)
    s.run()

    
