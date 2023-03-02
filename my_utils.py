import numpy as np

def rad(deg):
    return deg*np.pi/180
def deg(rad):
    return rad*180/np.pi

def normalize(arr, old_bounds, new_bounds):
    """Normalize a 1D array, from old_bounds [min, max] to new desired bounds [new_max, new_min]"""
    return new_bounds[0] + (arr - old_bounds[0]) * (new_bounds[1] - new_bounds[0]) / (old_bounds[1] - old_bounds[0])

def normalize_mode_shape(arr):
    return arr / np.max(np.abs(arr))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def delta_array_func(array):
    """Gives array with consecutive differences between its element values. Half distance for first and last elements"""
    n_array = len(array)
    delta_array = np.zeros(n_array)
    delta_array[0] = (array[1]-array[0])/2
    delta_array[-1] = (array[-1]-array[-2])/2
    delta_array[1:-1] = np.array([array[f+1]-array[f] for f in range(1, n_array-1)])
    return delta_array