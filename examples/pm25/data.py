import numpy as np

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

def load_data():
    data = np.load('/home/xd/data/pm25data/raw.npy').astype('float32')
    data[:,:,2] = np.sqrt(data[:,:,2]**2 + data[:,:,3]**2)
    data[:,:,3] = data[:,:,2]
#    data[:,:,2:4] = np.random.randn(data.shape[0], data.shape[1], 2)
#    data[:,:,1] = np.random.randn(data.shape[0], data.shape[1])
#    data[:,:,-1] -= data[:,:,-2] # subtract pm25 mean from pm25 target

    train = segment_data(data[:,250:650,:])
    valid = segment_data(data[:,650:790,:])
    test = segment_data(data[:,790:,:])
    return data, train, valid, test

