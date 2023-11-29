import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

root_dir = os.path.dirname(os.path.abspath(__file__))  # Root directory that should be used to avoid path problems. See https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure


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


def get_list_of_colors_matching_list_of_objects(list_of_objects, plt_colormap=plt.cm.rainbow):
    """
    Used for the color argument in e.g. plt.scatter()
    list_of_objects: e.g. list of strings, list of floats, etc.
    """
    pd_series = pd.Series(list_of_objects)
    unique_values = pd_series.unique()
    unique_idxs = np.arange(len(unique_values))
    unique_idxs_normalized = unique_idxs / unique_idxs.max()
    my_color_map = dict(zip(unique_values, plt_colormap(unique_idxs_normalized)))
    return pd_series.map(my_color_map).tolist()


def all_equal(iterator):
    """
    True if all elements in a list are identical. False otherwise
    https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


