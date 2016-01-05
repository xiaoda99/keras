import cPickle
import gzip
import pickle
import datetime
import pandas as pd
import numpy as np
import copy

savedir = '/home/xd/data/pm25data/'
Pm_starttime = datetime.datetime(2015, 4, 1, 8)
start_time_norm = '20150401'
#end_time = (datetime.datetime.today() - datetime.timedelta(days=2)).strftime('%Y%m%d')
end_time = '20151229'
start = start_time_norm + '08'
end = end_time + '08'
# Pm_stoptime = datetime.datetime(2015, 12, 8, 0)


class generate_data():

    def __init__(self,
                 pm_stations = [],
                 lon_range=None,
                 lat_range=None,
                 starttime='20150801',
                 endtime='20151207'
                 ):
        matrix = cPickle.load(gzip.open(
            savedir + 'Dataset'+'-'+start+'-'+end+'.pkl.gz'
            ))
        stations = pickle.load(gzip.open(savedir + 'stations_all_index.pkl.gz'))
        #data = {key: matrix[stations[key]] for key in stations}
        data = {}
        for key in stations:
            data[key] = matrix[stations[key]]
        d = pd.DataFrame(data.items())
        d.columns = ['station_name', 'pm25_data']
        d.index = d.station_name
        stations_lonlat = cPickle.load(gzip.open(
            savedir + 'pm25lonlat.pkl.gz'
            ))
        d['lon'] = d['station_name'].apply(lambda x: stations_lonlat[x][0])
        d['lat'] = d['station_name'].apply(lambda x: stations_lonlat[x][1])

        assert len(starttime) == 8 and len(endtime) == 8
        self.starttime = datetime.datetime(
                int(starttime[0:4]), int(starttime[4:6]), int(starttime[6:8]), 8)
        self.endtime = datetime.datetime(
                int(endtime[0:4]), int(endtime[4:6]), int(endtime[6:8]), 23)

        if pm_stations:
            assert lon_range == None and lat_range == None
        else:
            assert lon_range != None and lat_range != None

        self.lon_range = lon_range
        self.lat_range = lat_range
        self.pm_stations = pm_stations
        d_index = self.get_special_area_index(d)

        self.matrix = d

        result = self.matrix.ix[d_index]
        # result = self.preprocess_Pm(result)
        result = self.pm25_mean_feature(result)

        self.result = np.array(result.pm25_data.tolist())
        
        # if self.pm_stations:
        #     locate_str = ''.join(self.pm_stations[:2])
        # else:
        #     locate_str = str(self.lat_range[0])+'-'+str(self.lat_range[1])+','+ \
        #                  str(self.lon_range[0])+'-'+str(self.lon_range[1])
        # self.savefile(self.result,
        #               savedir+'Dataset'+'-'+starttime+'-' +
        #               endtime+locate_str+ '.pkl.gz')

        # self.savefile({key: i for i, key in enumerate(result.index)},
        #               savedir+'station_index'+locate_str+'.pkl.gz')

    def get_special_area_index(self, data):
        if self.pm_stations:
            d_index = pd.core.index.Index(self.pm_stations)
            d_index = data.index & d_index
            return d_index.sort_values()
        else:
            da = data[data['lon'] > self.lon_range[0]].index
            db = data[data['lon'] < self.lon_range[1]].index
            dc = data[data['lat'] > self.lat_range[0]].index
            dd = data[data['lat'] < self.lat_range[1]].index
            d_index = da & db & dc & dd
            d_index = d_index.sort_values()
            return d_index

    def nan_helper(self, y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    def preprocess_Pm(self, data, max_missing_rate=.1, max_pm=1000.):
        print len(data.index), 'stations before preprocessing.'
        for i, station in enumerate(data.index):
            print 'Preprocessing', station, i
            # pm = np.array(data.ix[station]['pm25_data'][:, -2]).astype('float')
            # pm = np.array(data.ix[station]['pm25_data'][:, -1]).astype('float')
            pm = data.ix[station]['pm25_data'][:, -1]
            pm[pm == 0] = np.NaN
            if np.isnan(pm).mean() > max_missing_rate:
                data.drop(station, axis=0, inplace=True)
                continue
            nans, fn = self.nan_helper(pm)
            pm[nans] = np.interp(fn(nans), fn(~nans), pm[~nans])
            pm[pm > max_pm] = max_pm
            data.ix[station]['pm25_data'][:, -2] = pm
            data.ix[station]['pm25_data'][:, -1] = pm
        print len(data.index), 'stations after preprocessing.'
        return data

    def pm25_mean_feature(self, data):
        cur_timediff = self.endtime - self.starttime
        assert cur_timediff.seconds % 10800 == 0
        time_point = int(
                cur_timediff.days * 8 + cur_timediff.seconds / 10800 + 1)
        # for i, station in enumerate(data.index):
        #     time_diff = self.starttime - Pm_starttime
        #     offset = int(time_diff.days*8 + time_diff.seconds/10800)
        #     data.ix[station]['pm25_data'] = \
        #             data.ix[station]['pm25_data'][offset:offset+time_point,:]
        #     data.ix[station]['pm25_data'] = \
        #             data.ix[station]['pm25_data'][offset:offset+time_point,:]
        time_diff = self.starttime - Pm_starttime
        offset = int(time_diff.days*8 + time_diff.seconds/10800)
        data['pm25_data'] = data['pm25_data'].apply(lambda x:x[offset:offset+time_point,:])
        return data

    def savefile(self, m, path):
        save_file = gzip.open(path, 'wb')
        # this will overwrite current contents
        cPickle.dump(m, save_file, -1)
        # the -1 is for HIGHEST_PROTOCOL
        save_file.close()

if __name__ == '__main__':
    data = generate_data()
