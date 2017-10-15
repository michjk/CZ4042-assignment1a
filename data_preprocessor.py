import numpy as np
import os

# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-np.min(X, axis=0))

# load dataset
def load_data(data_path, custom_min_max = False, X_min = None, X_max = None):
    input_txt = np.loadtxt(data_path, delimiter=' ')
    X, _Y = input_txt[:,:36], input_txt[:,-1].astype(int)
    
    if not custom_min_max:
        X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    
    X = scale(X, X_min, X_max)

    _Y[_Y == 7] = 6
    Y = np.zeros((_Y.shape[0], 6))
    Y[np.arange(_Y.shape[0]), _Y-1] = 1

    return X, Y, X_min, X_max