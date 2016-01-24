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
        else:
            self.ema = self.ema * (1. - self.alpha) + x * self.alpha
        return self.ema
    
class SaliencyEMA():
    def __init__(self, 
                 simulator,
                 n = 15 * 60 * 2,  # 1 hour
                 stdev_coef = 1.5,
                 days = 1,
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
        
        self.saliency = np.zeros(self.price.max() + 1)
        self.saliency2 = np.zeros(self.price.max() + 1)
        self.saliency_EMA = EMA(self.m)
        self.saliency2_EMA = EMA(self.m)
        
        assert self.smooth_n % 2 == 1
        self.smooth_w = np.ones((self.smooth_n - 1) * self.tick_size + 1) / self.smooth_n 
    
    def normalize(self, saliency_ema, level, name=None): 
        mean_saliency = saliency_ema.mean() * self.tick_size
        if self.saliency_EMA.sum_cnt == self.m and saliency_ema[level] > mean_saliency * 4. and \
                saliency_ema[level] > saliency_ema[level - self.tick_size] * 4.and \
                saliency_ema[level] > saliency_ema[level + self.tick_size] * 4.:
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
        
    def step(self):
#        start_tick = self.n * 2:
        start_tick = self.n * 4
        if self.simulator.now < start_tick:
            return None 
        
        self.saliency *= 0.
        self.saliency2 *= 0.
        if (self.simulator.now - start_tick) % self.n == 0: 
            mean = np.mean(self.ny[self.simulator.now - start_tick : self.simulator.now])
            mean = self.ny_mean_EMA.step(mean)
            stdev = np.sqrt(np.var(self.ny[self.simulator.now - start_tick : self.simulator.now]))
            stdev = self.ny_stdev_EMA.step(stdev)
        mean = self.ny_mean_EMA.ema
        stdev = self.ny_stdev_EMA.ema
        ny = self.ny[self.simulator.now - self.n]
        ny = ny * (abs(ny - mean) > stdev * self.stdev_coef)
        
        level = self.price[self.simulator.now - self.n]
        self.saliency[level] = abs(ny)
        self.saliency_ema = self.saliency_EMA.step(self.saliency)
#        self.saliency2[level] = self.saliency[level] * self.vol[self.simulator.now - self.n]
#        self.saliency2_ema = self.saliency2_EMA.step(self.saliency2)
        
        if self.simulator.now % 100 == 0:
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
        t = self.simulator.now
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
    
class Simulator():
    def __init__(self, history, step_size=1, 
                 show_freq = 5 * 60 * 2, # 3 minutes
                 saliency_scale = 6*1e7,
                 saliency2_scale = 8*1e5,
                 intensity_scale = 3*1e5,
                 intensity2_scale = 3*1e2,
                 product_scale = 1e6
                 ):
        self.__dict__.update(locals())
        del self.self
    
        self.now = -1
        self.indicators = []
        
    def add_indicator(self, indicator):
        self.indicators.append(indicator)
            
    def step(self, step_size=1):
        self.now += step_size
        return [indicator.step() for indicator in self.indicators]
    
    def showable(self):
        return self.now - self.indicators[0].m >= self.indicators[0].n and self.now % self.show_freq == 0
        
    def _plot_histogram(self, gs, y, scale, label=None, sharex=None):
        ax = plt.subplot(gs, sharex=sharex)
        price = self.history['last_price'][self.now - self.indicators[0].m : self.now]
        plt.plot(price)
        floor = price.min()
        ceil= price.max()
#        floor = self.history['last_price'].min()
#        ceil= self.history['last_price'].max()
        y = y[floor : ceil + 1] * scale
        y[y > self.indicators[0].m * 1.2] = self.indicators[0].m * 1.2
        plt.barh(np.arange(floor, ceil + 1), y, 1.0, label=label,
                 alpha=0.2, color='r', edgecolor='none')
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
        ax0 = self._plot_histogram(gs[0], y, self.saliency_scale, label='saliency')
        
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
            if self.showable():
                self.show()
                
if __name__ == '__main__':
#    ticks = load_ticks('zc', 'SR', 2015, [10,], use_cache=True); tick_size = 1; trading_hours_per_day = 6.25
    ticks = load_ticks('dc', 'pp', 2015, [11,], use_cache=True); tick_size = 1; trading_hours_per_day = 3.75
#    ticks = load_ticks('sc', 'zn', 2015, [11,], use_cache=True); tick_size = 5; trading_hours_per_day = 7.75
    s = Simulator(ticks)
    for indicator in [
            SaliencyEMA(s, tick_size=tick_size, trading_hours_per_day=trading_hours_per_day),
            IntensityEMA(s, tick_size=tick_size, trading_hours_per_day=trading_hours_per_day),
            ]:
        s.add_indicator(indicator)
    s.run()

    
