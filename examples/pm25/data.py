import numpy as np
import cPickle
import gzip
import datetime
from config import heating_start, heating_end, \
        today, startdaytime, startday_string, \
        yesterday_string


from stations_choose import generate_data

def segment_data(data, segment_len=44):
    segments = []
    for i in range(data.shape[1] - segment_len):
        segment = data[:,i:i+segment_len,:]
        
        # filter out segments with too long missing gfs data
        a = segment[0,:,0] # temperature curve of the first station
        fluctuate_indices = np.where(np.diff(a) != 0)[0] + 1
        fluctuate_indices = np.concatenate([np.array([0,]), fluctuate_indices, np.array([a.shape[0]])])
        if np.diff(fluctuate_indices).max() >= 13: 
#            print 'filter out segment', i, i + segment_len
            continue
        
        segments.append(segment)
    return np.vstack(segments)

def filter_data(data, pm_threshold=80):
    pm = data[:,:,-1] + data[:,:,-2]
    if pm.max() < 20:
        pm_threshold /= 100.
    print 'filter', (pm.max(axis=1) > pm_threshold).mean()
    return data[pm.max(axis=1) > pm_threshold]

def preprocess(data):
    wind_x = data[:,:,2]
    wind_y = data[:,:,3]
    rho = np.sqrt(wind_x**2 + wind_y**2)
    phi = np.arctan2(wind_y, wind_x)
    data[:,:,2] = rho
    data[:,:,3] = phi
    data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target
    return data

def normalize_pm25(pm25, threshold=300, decay_coef=.25):
#def normalize_pm25(pm25, threshold=250, decay_coef=.2):
    return pm25 * (pm25 <= threshold) + (threshold + (pm25 - threshold) * decay_coef) * (pm25 > threshold)
  
def load_data3(stations=None, lon_range=None, lat_range=None, starttime=None, endtime=None, latest=False,
               train_start=0, train_stop=None, valid_start=None, valid_stop=None, 
               segment=True, filter=False, normalize_target=False):
    if latest:
        if endtime is None:
            endtime = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')
        if starttime is None:
            starttime = (datetime.datetime.today() - datetime.timedelta(days=121)).strftime('%Y%m%d')
    else:
        assert starttime is not None
        assert endtime is not None
    print 'time range:', starttime, '-', endtime
    data = generate_data(pm_stations=stations, lon_range=lon_range, lat_range=lat_range, 
                         starttime=starttime, endtime=endtime, latest=latest).result
#    print starttime, endtime, data.shape
    if normalize_target:
        data[:,:,-1] = normalize_pm25(data[:,:,-1])
#    data[:,:,-2:] /= 100.
    data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target
    
    if train_stop is None:
        train_stop = int(round(data.shape[1] * 3. / 4))
#        train_stop = 750
    if valid_start is None:
        valid_start = train_stop
    if valid_stop is None:
        valid_stop = data.shape[1]
    train_data = data[:,train_start:train_stop,:]
    valid_data = data[:,valid_start:valid_stop,:]
    train_data2 = data[:,-(train_stop - train_start):,:]
    
    if segment:
        train_data = segment_data(train_data)
        valid_data = segment_data(valid_data)
        train_data2 = segment_data(train_data2)
    if filter:
        train_data = filter_data(train_data)
        valid_data = filter_data(valid_data)
        train_data2 = filter_data(train_data2)
    return train_data, valid_data, train_data2
    
def load_data2(stations=None, segment=True):
#    data = np.load('/home/xd/data/pm25data/raw.npy').astype('float32')
    data = cPickle.load(gzip.open('/home/xd/data/pm25data/forXiaodaDataset-20150401-20151207_huabei+lonlat.pkl.gz'))
#    data = cPickle.load(gzip.open('/home/xd/data/pm25data/forXiaodaDataset-20150401-20151207_huabei.pkl.gz'))
#    data = preprocess(data)
    data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target

    if stations is None:
        train_data = data[:,1310:1890,:]
    #    valid = segment_data(data[:,1750:1890,:])
        valid_data = data[:,1890:,:]
        test_data = data[:,1890:,:]
    else:
        station_indices = cPickle.load(open('/home/xd/data/pm25data/stations_index_huabei.pkl'))
        selected_indices = [station_indices[station] for station in stations if station in station_indices] 
        if len(selected_indices) == 0:
            return None, None, None
        train_data = data[selected_indices,1310:1890,:]
    #    valid = segment_data(data[:,1750:1890,:])
        valid_data = data[selected_indices,1890:,:]
        test_data = data[selected_indices,1890:,:]
    if segment:
        train_data = segment_data(train_data); valid_data = segment_data(valid_data); test_data = segment_data(test_data)
    return train_data, valid_data, test_data

def load_data():
#    data = np.load('/home/xd/data/pm25data/raw.npy').astype('float32')
    data = cPickle.load(gzip.open('/home/xd/data/pm25data/forXiaodaDataset200-2015080108-2015121108.pkl.gz'))
    wind_x = data[:,:,2]
    wind_y = data[:,:,3]
    rho = np.sqrt(wind_x**2 + wind_y**2)
    phi = np.arctan2(wind_y, wind_x)
    data[:,:,2] = rho
    data[:,:,3] = phi
#    data[:,:,2:4] = np.random.randn(data.shape[0], data.shape[1], 2)
#    data[:,:,1] = np.random.randn(data.shape[0], data.shape[1])
    data[:,:,-1] -= 80
    for i in range(300, data.shape[1]):
        data[:, i, -2] = data[:, i-30*8+8:i+8:8, -1].mean(axis=1)
    data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target

    train = segment_data(data[:,340:920,:])
#    valid = segment_data(data[:,780:920,:])
    valid = segment_data(data[:,920:,:])
    test = segment_data(data[:,920:,:])
    return train, valid, test

#station2idx = cPickle.load('stations_index_huabei.pkl')
#huabei = cPickle.load(gzip.open('forXiaodaDataset-20150401-20151207_huabei.pkl.gz'))
#station2lonlat = cPickle.load(gzip.open('pm25lonlat.pkl.gz'))
#a = np.zeros((huabei.shape[0], huabei.shape[1], huabei.shape[2]+2), dtype='float32')
#for s in station2idx:
#    idx = station2idx[s]
#    a[idx,:,:9] = huabei[idx,:,:9]
#    a[idx,:,-2:] = huabei[idx,:,-2:]
#    lonlat = station2lonlat[s]
#    a[idx,:,9] = lonlat[0]
#    a[idx,:,10] = lonlat[1]
#f = gzip.open('forXiaodaDataset-20150401-20151207_huabei+lonlat.pkl.gz', 'wb')
#cPickle.dump(a, f)
#f.close()

def load_data4(
        stations=None,
        lon_range=None,
        lat_range=None,
        heating=False,
        segment=True,
        filter=False,
        normalize_target=False):

    data = generate_data(
        pm_stations=stations,
        lon_range=lon_range,
        lat_range=lat_range,
        starttime=startday_string,
        endtime=yesterday_string,
        )
    data_starttime = data.starttime
    data = data.result
    # print starttime, endtime, data.shape
    if normalize_target:
        data[:, :, -1] = normalize_pm25(data[:, :, -1])
    # data[:,:,-2:] /= 100.
    data[:, :, -1] -= data[:, :, -2]
    # subtract pm25 mean from pm25 target

    if heating:
        if today > heating_start - datetime.timedelta(days=46) and \
                today < heating_start:
                    train_endtime = heating_start.replace(
                            year=heating_start.year - 1) - datetime.timedelta(days=1)
                    train_starttime = train_endtime - datetime.timedelta(days=90)
                    valid_starttime = today - datetime.timedelta(days=31)
                    valid_endtime = valid_starttime + datetime.timedelta(days=30)
        # before 12 month but must in same heating state period
        elif today > heating_end - datetime.timedelta(days=46) and \
                today < heating_end:
                    train_endtime = heating_end.replace(
                            year=heating_end.year - 1) - datetime.timedelta(days=1)
                    train_starttime = train_endtime - datetime.timedelta(days=90)
                    valid_starttime = today - datetime.timedelta(days=31)
                    valid_endtime = valid_starttime + datetime.timedelta(days=30)
        elif today < heating_start + datetime.timedelta(days=46) and \
                today > heating_start:
                    train_starttime = heating_start.replace(
                        year=heating_start.year - 1) + datetime.timedelta(days=1)
                    train_endtime = train_starttime + datetime.timedelta(days=90)
                    if today > heating_start + datetime.timedelta(days=31):
                        valid_starttime = today - datetime.timedelta(days=31)
                        valid_endtime = valid_starttime + datetime.timedelta(days=30)
                    else:
                        valid_starttime = train_endtime
                        valid_endtime = valid_starttime + datetime.timedelta(days=30)
        elif today < heating_end + datetime.timedelta(days=46) and \
                today > heating_end:
                    train_starttime = heating_end.replace(
                        year=heating_end.year - 1) + datetime.timedelta(days=1)
                    train_endtime = train_starttime + datetime.timedelta(days=90)
                    if today > heating_end + datetime.timedelta(days=31):
                        valid_starttime = today - datetime.timedelta(days=31)
                        valid_endtime = valid_starttime + datetime.timedelta(days=30)
                    else:
                        valid_starttime = train_endtime
                        valid_endtime = valid_starttime + datetime.timedelta(days=30)
        else:
            train_starttime = today - datetime.timedelta(days=406)
            train_endtime = train_starttime + datetime.timedelta(days=90)
            valid_starttime = today - datetime.timedelta(days=31)
            valid_endtime = valid_starttime + datetime.timedelta(days=30)
    else:
        train_starttime = today - datetime.timedelta(days=406)
        train_endtime = train_starttime + datetime.timedelta(days=90)
        valid_starttime = today - datetime.timedelta(days=31)
        valid_endtime = valid_starttime + datetime.timedelta(days=30)

    if train_starttime < startdaytime:
        train_starttime = startdaytime
        train_endtime = train_starttime + datetime.timedelta(days=90)
        valid_starttime = train_endtime
        valid_endtime = valid_starttime + datetime.timedelta(days=30)

    timediff = train_starttime - data_starttime
    train_start = timediff.days * 8 + timediff.seconds / 7200
    timediff = train_endtime - data_starttime
    train_stop = timediff.days * 8 + timediff.seconds / 7200
    timediff = valid_starttime - data_starttime
    valid_start = timediff.days * 8 + timediff.seconds / 7200
    timediff = valid_endtime - data_starttime
    valid_stop = timediff.days * 8 + timediff.seconds / 7200

    train_data = data[:, train_start:train_stop, :]
    valid_data = data[:, valid_start:valid_stop, :]

    if segment:
        train_data = segment_data(train_data)
        valid_data = segment_data(valid_data)
    if filter:
        train_data = filter_data(train_data)
        valid_data = filter_data(valid_data)
    return train_data, valid_data
