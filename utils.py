import pickle
import numpy as np


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data,f)
        
def get_closest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def change_empty(log):
    log.replace('', '0', inplace=True)
    return log