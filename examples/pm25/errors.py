import numpy as np

def mean_square_error(predictions, targets):
    return np.square(predictions - targets).mean(axis=0)

def absolute_percent_error(predictions, targets, targets_mean):
    return (np.abs(predictions - targets) / np.abs(targets_mean)).mean(axis=0)
        
def absolute_error(predictions, targets):
    return np.abs(predictions - targets).mean(axis=0)

threshold = 80
    
def misclass_error(predictions, targets):
    return ((predictions >= threshold) != (targets >= threshold)).mean(axis=0)

def downsample(sequences, pool_size):
    assert sequences.ndim == 2
    assert sequences.shape[1] % pool_size == 0
    return sequences.reshape((sequences.shape[0], sequences.shape[1] / pool_size, pool_size)).max(axis=2) 

def detection_error(predictions, targets, targets_mean=None, pool_size=1):
    if targets_mean is not None:
        predictions = predictions + targets_mean
        targets = targets + targets_mean
    if pool_size != 1:
        predictions = downsample(predictions, pool_size)
        targets = downsample(targets, pool_size)
    alarm = (predictions >= threshold).mean(axis=0)
    occur = (targets >= threshold).mean(axis=0)
    hit = ((predictions >= threshold) & (targets >= threshold)).mean(axis=0)
    pod = hit / occur
    far = 1. - hit / alarm
    csi = hit / (occur + alarm - hit)
    return pod, far, csi
