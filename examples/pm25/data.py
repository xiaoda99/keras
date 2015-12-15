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
    data[:,:,2] = np.sqrt(data[:,:,2]**2 + data[:,:,3]**2)
    data[:,:,3] = data[:,:,2]
#    data[:,:,2:4] = np.random.randn(data.shape[0], data.shape[1], 2)
#    data[:,:,1] = np.random.randn(data.shape[0], data.shape[1])
    data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target
    return data

def load_data2():
#    data = np.load('/home/xd/data/pm25data/raw.npy').astype('float32')
    data = cPickle.load(gzip.open('/home/xd/data/pm25data/forXiaodaDataset-20150401-20151207_huabei.pkl.gz'))
    data[:,:,2] = np.sqrt(data[:,:,2]**2 + data[:,:,3]**2)
    data[:,:,3] = data[:,:,2]
#    data[:,:,2:4] = np.random.randn(data.shape[0], data.shape[1], 2)
#    data[:,:,1] = np.random.randn(data.shape[0], data.shape[1])
    data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target

#    train = segment_data(data[:,250:650,:])
#    valid = segment_data(data[:,650:790,:])
#    test = segment_data(data[:,790:,:])
    train = segment_data(data[:,1310:1750,:])
    valid = segment_data(data[:,1750:1890,:])
    test = segment_data(data[:,1890:,:])
    return train, valid, test

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
#    test = segment_data(data[:,920:,:])
    valid = segment_data(data[:,920:,:])
    test = segment_data(data[:,920:,:])
    return train, valid, test