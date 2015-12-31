# -*- coding: utf-8 -*-
# 这一版是为pm25做RNN预测使用,新增改变回溯时间长度为一整天
# 在上述基础上增加三个输入维度，分别是每个step时的小时时刻，星期日期，月份
# RNN预测的方式是输入开始-21，-18 ...，-3小时，0小时的gfs和pm25；3小时的gfs，
# 输出3小时的pm25；第二帧输入-18，-15 ...，0小时的gfs和pm25，3小时的gfs和pm25，
# 6小时的gfs，输出6小时的pm25；做n_predict次，一共预测t_predict小时
# 实际上每一个example结构是这样的
'''
[pm25-21h, pm25-18h,..., pm25-3h,pm250h]+[pm25+3h,pm25+6h,...,pm25+120h]
'''
# 并注意到全球gfs数据是格林尼治时间，偏移8小时到北京时间之后之后对准
# gfs分辨率为0.25°，经度从0开始向东经为正，维度从北纬90°开始向南为正。第一维是维度，第二维是经度（721*1440）。

from __future__ import division
#from config import HELPTXT
import os
import cPickle
import gzip
import numpy as np
import datetime
import argparse
import pandas as pd
from pyproj import Proj

NaN = np.NaN

p = Proj('+proj=merc')

gfsdir = '/ldata/pm25data/gfs/'
pm25dir = '/mnt/storm/nowcasting/pm25/'
today = datetime.datetime.today()
pm25meandir = '/ldata/pm25data/pm25mean/mean'+today.strftime('%Y%m%d')+'/'
# savedir = '/ldata/pm25data/pm25dataset/'
savedir = './data_save/'
end_time_norm = (today - datetime.timedelta(days=1)).strftime('%Y%m%d')
end_time = (today - datetime.timedelta(days=2)).strftime('%Y%m%d')
# cause time zone different you should substract 2
start_time_norm = '20150401'
# start_time_norm = '20151225'

t_predict = 120
# n_predict=8

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def preprocess_Pm(Pm, max_missing_rate=.1, max_pm=1000.):
    stations = Pm.keys()
    print len(stations), 'stations before preprocessing.'
    for i, station in enumerate(Pm.keys()):
        print 'Preprocessing', station, i
        pm = np.array(Pm[station]).astype('float')
        pm[pm == 0] = NaN
        if np.isnan(pm).mean() > max_missing_rate:
            # del Pm[station] 
            stations.pop(stations.index(station))
            print 'pop %s' % station
            continue
        nans, fn = nan_helper(pm)
        # if sum(nans) == len(pm):
        #     pm[np.isnan(pm)] = 0
        #     Pm[station] = pm
        #     continue
        pm[nans]= np.interp(fn(nans), fn(~nans), pm[~nans])
        pm[pm > max_pm] = max_pm
        Pm[station] = pm
    print len(stations), 'stations after preprocessing.'
    return Pm, sorted(stations)

f = gzip.open('data_source/' + start_time_norm + '00_' + 
              end_time_norm + '00' + 'pm25.pkl.gz')
Pm = cPickle.load(f)
f.close()
f = gzip.open('pm25lonlat.pkl.gz')
Pmlonlat = cPickle.load(f)
f.close()

# Pm_starttime = datetime.datetime(2015, 4, 1, 0)
Pm_starttime = datetime.datetime.strptime(start_time_norm, '%Y%m%d')
# Pm_stoptime = datetime.datetime(2015, 12, 8, 0)
Pm_stoptime = datetime.datetime.strptime(end_time_norm, '%Y%m%d')
Pm, stations = preprocess_Pm(Pm)

def lonlat2mercator(lon=[116.3883], lat=[39.3289]):
    # p = Proj('+proj=merc')
    radius = [17, 72, 54, 135]
    r = 10000

    cord_x = []
    cord_y = []
    y, x = np.round(np.array(p(radius[1], radius[0]))/r)
    y1, x1 = np.round(np.array(p(radius[3], radius[2]))/r)
    for i in range(len(lon)):
        longitude, latitude = p(lon[i], lat[i])
        latlng = np.array([latitude, longitude])

        if lat[i] < radius[0] or lat[i] > radius[2] \
                or lon[i] < radius[1] or lon[i] > radius[3]:
            raise Exception, 'Out of range  ' + 'lon:' + str(radius[1]) + '-' \
                    + str(radius[3])+' lat:'+str(radius[0])+'-'+str(radius[2])
        latlng = np.abs(np.round(latlng/r)-np.array([x1, y]))
        cord_x.append(latlng[0])
        cord_y.append(latlng[1])
    return cord_x, cord_y


def interp(data, lat_x, lon_y):

    x1 = int(lat_x)
    # 整数部分
    dcm_lat_x = lat_x-x1
    # 小数部分
    y1 = int(lon_y)
    dcm_lon_y = lon_y-y1
    x2 = x1+1
    y2 = y1+1

    R1 = (1 - dcm_lat_x) * data[x1, y1] + dcm_lat_x * data[x2, y1]
    R2 = (1 - dcm_lat_x) * data[x1, y2] + dcm_lat_x * data[x2, y2]
    result = (1 - dcm_lon_y) * R1 + dcm_lon_y * R2
    return result


def savefile(m, path):
    save_file = gzip.open(path, 'wb')  # this will overwrite current contents
    cPickle.dump(m, save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()


class RNNPm25Dataset(object):
    '''
    给出预测点的位置和数据集的时间范围，利用gfs和pm25原始
    数据文件，构建这一时间和地点内的pm25 dataset
    '''
    def __init__(
            self,
            lon=np.hstack((np.array([116.3883,117.20,121.48,106.54,118.78,113.66]),110+7.5*np.random.rand(94))),
            lat=np.hstack((np.array([39.9289,39.13,31.22,29.59,32.04,34.76]),34+6*np.random.rand(94))),
            start='2015080808',
            stop='2015120723',
            # steps=int(t_predict/3+8)
            ):
        '''Initialize the parameters
        :lon:longitude of prediction points, scalar or vector like
        :lat:latitude of prediction points , scalar or vector like
        :start: Beijing time the dataset forecast point starts, e.g.
                '2015022800' better be 3 hours counted(08,11,14),
                since the gfs data are divided into 3 hours.
        :stop: Beijing time the dataset forecast point stops, e.g.
               '2015022818' better be 3 hours counted(08,11,14),
               since the gfs data are divided into 3 hours.
        :steps: timestep lens of predictions
        '''
        #self.cord_x, self.cord_y = lonlat2mercator(lon, lat)
        # 中国地图mercator中坐标

        #XD
        self.lon = np.array([Pmlonlat[station][0] for station in stations])
        self.lat = np.array([Pmlonlat[station][1] for station in stations])

        self.lat_x = (90.0-self.lat.astype(float))/0.25
        # 全球gfs图上坐标
        self.lon_y = self.lon.astype(float)/0.25

        # self.steps = steps
        # steps不止是预测的维度，是第二维全部跳转个数，包括预测之前的时间帧
        self.n_element = 6 + 3 + 2 + 1 + 1
        # 6gfs+3时间特征+1均值+1pm25

        self.starttime = datetime.datetime(
                int(start[0:4]), int(start[4:6]),
                int(start[6:8]), int(start[8:10]))
        self.stoptime = datetime.datetime(
                int(stop[0:4]), int(stop[4:6]),
                int(stop[6:8]), int(stop[8:10]))
        self.n_location = len(self.lon)
        # 从地图上取了n_location个试验点
        timediff = self.stoptime - self.starttime
        assert timediff.seconds % 10800 == 0
        self.time_point = int(timediff.days * 8 + timediff.seconds / 10800 + 1)
        # 总样本数, n_exp按照每隔3小时一组training example来计算

        self.input_data = self.generateinput()
        # input_data包含三个维度全部gfs和pm25数据

    def generateinput(self):
        '''generrate input matrix'''
        # inputs=np.zeros((self.n_exp, self.steps, self.n_element))
        inputs = np.zeros((self.n_location, self.time_point, self.n_element))
        '''gfs,(steps*6)dimentions for every slice'''
        for h in range(0, self.time_point * 3, 3):
            current_time = self.starttime + datetime.timedelta(hours=h) - \
                    datetime.timedelta(hours=8)
            if current_time.hour % 6 == 0:
                file_time_variant = 6
            elif current_time.hour % 3 == 0:
                file_time_variant = 3
            else:
                raise Exception, 'gfs filename error at time:' + \
                        current_time.strftime('%Y%m%d%H')
            # 存在两个两种数据源文件, 6, 12, 18, 00时刻放在xxx_006文件中
            # 而3, 9, 15, 21则放在xxx_003文件中, file_time_variant保存
            # 3或者6
            name = current_time - datetime.timedelta(hours=file_time_variant)
            file_name = name.strftime('%Y%m%d') + '_' + name.strftime('%H') + \
                '_00' + str(file_time_variant) + '.pkl.gz'
            if os.path.exists(gfsdir + file_name) and \
                    os.path.getsize(gfsdir + file_name) > 0:
                        f = gzip.open(gfsdir + file_name)
                        print('current file:' + gfsdir + file_name)
                        cnt = 0
                        for entry in \
                                ['tmp', 'rh', 'ugrd', 'vgrd', 'prate', 'tcdc']:
                            # 填1个step上6个数据
                            temp = cPickle.load(f)
                            for k in range(self.n_location):
                                inputs[k, h / 3, cnt] = interp(
                                        temp.reshape((180 * 4 + 1, 360 * 4)),
                                        self.lat_x[k], self.lon_y[k]
                                        )
                                # 也就是说, numpy是只可以使用index来访问的,
                                # 因为都需要将意义索引值转换为实际的索引值.
                                # cnt是用来找对应dim3 元素的格位置，
                                # （h+3)/3是对应dim2 step位置
                            cnt = cnt + 1
                        f.close()
            else:
                # 该时刻数据找不到，用三小时之前的替换
                cnt = 0
                print('no such file:' + gfsdir + file_name)
                for entry in ['tmp', 'rh', 'ugrd', 'vgrd', 'prate', 'tcdc']:
                    # 填6个数据
                    for k in range(self.n_location):
                        inputs[k, h / 3, cnt] = inputs[k, h / 3 - 1, cnt]
                        # cnt是用来找对应dim3 元素的格位置，（h+3)/3是对应dim2 step位置
                    cnt = cnt+1

        '''pm25,(steps*1)dimentions for every slice'''
        # 同时生成每个location第一个slice的数据
        for h in range(0, self.time_point * 3, 3):
            timediff = self.starttime + datetime.timedelta(hours=h) - Pm_starttime
            offset = int(timediff.days * 24 + timediff.seconds / 3600)
                  
            for k in range(self.n_location):
                inputs[k, h / 3, -2] = Pm[stations[k]][offset-30*24+24:offset+24:24].mean()
                # inputs[k, h / 3, -2] = Pm[stations[k]][offset]
                inputs[k, h / 3, -1] = Pm[stations[k]][offset]

        '''time features, steps*3 dimentions for every slice'''
        for h in range(0, self.time_point * 3, 3):
            #current = self.starttime + datetime.timedelta(hours=3 * h)
            current = self.starttime + datetime.timedelta(hours=h)
            for k in range(self.n_location):
                inputs[k, h / 3, 6] = current.hour/24
                inputs[k, h / 3, 7] = current.weekday()/7
                inputs[k, h / 3, 8] = (current.month + today.day / 30) / 12
                inputs[k, h / 3, 9] = self.lon[k]
                inputs[k, h / 3, 10] = self.lat[k]


        return inputs


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(add_help=True)
    # parser.add_argument('--starttime', type=str,
    #                     help=HELPTXT['starttime'])
    # parser.add_argument('--endtime', type=str,
    #                     help=HELPTXT['endtime'])
    # parser.add_argument('--locations', type=str,
    #                     help=HELPTXT['locations'])
    # args = parser.parse_args()
    # # 处理命令行参数, 获取开始和结束时间

    # stations = pd.read_csv('pm25ss.csv')
    # stations['location'] = tuple(zip(stations['经度'].tolist(),
    #                                  stations['纬度'].tolist()))
    # stations = stations.groupby('地区名')['location'].apply(list)

    # if args.locations:
    #     locations = args.locations.split(' ')

    # else:
    #     locations = stations.index.tolist()

    # lans = reduce(lambda x, y: np.hstack((x, y)),
    #               [np.array([i[0] for i in stations.ix[location]])
    #                for location in locations]
    #               )

    # lons = reduce(lambda x, y: np.hstack((x, y)),
    #               [np.array([i[1] for i in stations.ix[location]])
    #                for location in locations]
    #               )

    # start = args.starttime + '08'
    # end = args.endtime + '08'

    # you could specify it again, but i just hardcode this begining
    # you could specify it use 2015-08-19 like that
    start = start_time_norm + '08'
    end = end_time + '08'
    obj = RNNPm25Dataset(start=start, stop=end)
    savefile(obj.input_data,
             savedir+'Dataset'+'-'+start+'-'+end+'.pkl.gz')
    # 保存导出的矩阵到指定的文件夹
    savefile({s:i for i, s in (enumerate(stations))}, 
             savedir+'stations_all_index.pkl.gz')
    print (savedir+'Dataset'+'-'+start+'-'+end+'.pkl.gz')
