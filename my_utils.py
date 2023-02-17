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

