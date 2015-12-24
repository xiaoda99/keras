import numpy as np
import cPickle
import gzip

def segment_data(data, segment_len=48):
    segments = []
    for i in range(data.shape[1] - segment_len):
        segments.append(data[:,i:i+segment_len,:])
    return np.vstack(segments)

def filter(data, pm_threshold=80):
    pm = data[:,:,-1]
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
