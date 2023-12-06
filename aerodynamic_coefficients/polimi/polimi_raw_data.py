"""
In late 2023, Politecnico di Milano (Polimi) delivered us new wind tunnel test results in skew winds.
This script allows pre- and post-processing the raw data files provided.

bercos@vegvesen.no
November 2023
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat  # Faster alternative than using mat4py.loadmat
from scipy.optimize import curve_fit
from my_utils import root_dir, get_list_of_colors_matching_list_of_objects, all_equal, flatten_nested_list
import pandas as pd
import logging
matplotlib.use('Qt5Agg')  # to prevent bug in PyCharm


raw_data_path = os.path.join(root_dir, r"aerodynamic_coefficients\polimi\raw_data")
debug = False

# Common variables
scale = 1 / 35  # model scale

# Pontoon dimensions (for normalization)
H_pont = np.round(3.5 * scale, 10)
B_pont = 14.875 * scale
L_pont = 53 * scale

# Raw data types: my labels of the different raw data types. Cues: respective substrings found in Polimi's filenames
raw_data_types_and_cues = {'A01-A06': ['AnnexA1', 'AnnexA2', 'AnnexA3', 'AnnexA4', 'AnnexA5', 'AnnexA6'],
                           'A07-A08': ['AnnexA7', 'AnnexA8'],
                           'A09': ['AnnexA9'],
                           'A10': ['Annex10'],
                           'A11-A12': ['Annex11', 'Annex12'],
                           'A13': ['Annex13'],
                           'A14': ['Annex14'],
                           'A15-A16': ['Annex15', 'Annex16']}

raw_data_types = list(raw_data_types_and_cues.keys())
raw_data_str_cues = list(raw_data_types_and_cues.values())


def unique_file_names_and_paths(file_paths):
    """
    Used for debugging purposes, such that only one raw data file is stored per file type, to save time
    """
    file_names = [os.path.basename(fp) for fp in file_paths]
    logging.warning('Debug mode is ON! Only a few unique raw data files will be processed to speed up')
    case_tags = [fn[6:].split('Ang')[0] for fn in file_names]  # Removing the case id and the angle
    idxs_unique = np.unique(case_tags, return_index=True)[1]
    file_names, file_paths = zip(*[[file_names[i], file_paths[i]] for i in idxs_unique])
    return file_names, file_paths


def overview_all_raw_file_keys():
    """
    Each raw data annex has several .mat files, each being a dict with several keys (e.g.: 'H', 'accZ', 'fsamp', etc.)
    This function provides a df with an overview of all keys in all the raw data files.
    """
    annex_paths = os.listdir(raw_data_path)
    annex_nums = [int(''.join([num for num in path[-3:] if num.isnumeric()])) for path in annex_paths]  # annex numbers
    annex_nums, annex_paths = zip(*sorted(zip(annex_nums, annex_paths)))
    annex_paths = [os.path.join(raw_data_path, s) for s in annex_paths]

    col_annex = []
    col_file = []
    col_keys = []
    for annex_path in annex_paths:
        file_paths = [os.path.join(raw_data_path, annex_path, file_name) for file_name in os.listdir(annex_path)]
        if debug:  # Only consider a few unique raw data files (with unique name tags), to speed up
            _, file_paths = unique_file_names_and_paths(file_paths)
        for file_path in file_paths:
            raw_file = loadmat(file_path, squeeze_me=True)
            set_of_keys = list(raw_file.keys())
            assert len(set(set_of_keys)) == len(set_of_keys), "One key is repeated in the same file!?"
            set_of_keys = set(set_of_keys)
            col_annex.append(os.path.basename(annex_path))
            col_file.append(os.path.basename(file_path))
            col_keys.append(set_of_keys)
    df = pd.DataFrame({'annex': col_annex, 'file': col_file, 'key_set': col_keys})
    # Factorizing, i.e., finding unique key_sets and numbering them
    df['key_set_str'] = [' '.join(df['key_set'][i]) for i in range(len(df.index))]  # so that it is hashable
    df['key_set_id'] = pd.factorize(df['key_set_str'])[0]
    del df['key_set_str']
    set_keys = set(flatten_nested_list([list(k) for k in df['key_set']]))
    return df, set_keys


def get_raw_data_dict(raw_data_path):
    """
    raw_data_path: the absolute path to the folder with all the raw .mat files provided by Polimi
    """

    def get_raw_data_dict_from_file_paths(file_paths, raw_data_type):
        assert raw_data_type in raw_data_types
        print(f'Collecting {raw_data_type} data. May take several minutes...')
        file_names = [os.path.basename(fp) for fp in file_paths]
        data = {}  # prepare for a nested dict

        if debug:  # Only consider a few unique raw data files (with unique name tags), to speed up
            file_names, file_paths = unique_file_names_and_paths(file_paths)

        for file_name, file_path in zip(file_names, file_paths):
            k1 = file_name.split('.mat')[0]  # using the filename for key 1
            data[k1] = loadmat(file_path, squeeze_me=True)  # squeeze unit dimensions
            data_keys = data[k1].keys()

            data[k1]['id'] = int(file_name[2:6])  # case number
            if 'inclinometer' in data_keys:  # time history acquisition time
                data[k1]['rx'] = -1.0 * np.array(data[k1]['inclinometer'])  # rx has opposite sign to inclinometer!
                del data[k1]['inclinometer']  # ditching old key
            if 't' in data_keys:  # time history acquisition time
                data[k1]['t'] = np.array(data[k1]['t'])
            if 'qCeiling' in data_keys:  # time hist of dynamic pressure measured by pitot tube placed on the ceiling
                data[k1]['q_ceil'] = data[k1].pop('qCeiling')  # rename key, ditching old one
                data[k1]['q_ceil'] = np.array(data[k1]['q_ceil']).squeeze()  # squeezing unnecessary dimension
                u_ceil = np.sqrt(data[k1]['q_ceil'] / (1 / 2 * data[k1]['rho']))
                data[k1].update({'u_ceil': u_ceil, 'U_ceil': np.mean(u_ceil)})
            if 'qUpwind' in data_keys:  # ... pitot placed upwind (2m ahead the turn table) at h=0.5m.
                data[k1]['q_upwind'] = data[k1].pop('qUpwind')  # rename key, ditching old one
                data[k1]['q_upwind'] = np.array(data[k1]['q_upwind']).squeeze()  # squeezing unnecessary dimension
            if 'turntable' in data_keys:  # angle imposed to the turning table during the test
                data[k1]['polimi_yaw'] = data[k1].pop('turntable')  # rename key, ditching old one
                # Check if the raw data angle 'turntable' corresponds to the angle in the file name after 'Ang':
                if raw_data_type in ['A10', 'A11-A12', 'A14']:  # file types with different Ang syntax in filename
                    polimi_yaw_2 = float(file_name.split('-Ang')[1].split('-Z')[0])  # find angle between substrings
                else:
                    polimi_yaw_2 = float(file_name.split('_Ang')[1].split('.mat')[0])  # find angle between substrings
                if not np.isclose(data[k1]['polimi_yaw'], polimi_yaw_2, atol=1):  # best to use 1 deg tolerance
                    logging.warning(f"Ang in filename != from 'turntable'. File: {file_path}")
            if 'u' in data_keys:
                assert 'v' in data_keys
                assert 'w' in data_keys
                u = data[k1]['u'].T  # new shape: (n_cobras, n_samples) or (n_samples)
                v = data[k1]['v'].T  # new shape: (n_cobras, n_samples) or (n_samples)
                w = data[k1]['w'].T  # new shape: (n_cobras, n_samples) or (n_samples)
                U = np.mean(u, axis=-1)
                data[k1].update({'U': U, 'u': u, 'v': v, 'w': w})  # updating dict
            if 'THForces' in data_keys:
                data[k1]['F'] = np.array(data[k1]['THForces']).T  # new shape: (dof, n_samples)
                del data[k1]['THForces']  # ditching old key
                assert 'EU' in data_keys
                data[k1]['units'] = data[k1].pop('EU')  # rename key, ditching old one
                assert 'names' in data_keys
                data[k1]['dof_tag'] = data[k1].pop('names')  # rename key, ditching old one
                data[k1]['F_mean'] = np.mean(data[k1]['F'], axis=1)  # new shape: (dof, n_samples)
            if 'H' not in data[k1].keys():
                logging.warning(f"Polimi forgot to add an 'H' key to the data in Annex A9. File: {file_path}")
                data[k1]['H'] = 0.45714285714285713  # taken from the other raw deck data files
            if 'accZ' in data_keys:
                data[k1]['acc_z'] = np.array(data[k1]['accZ']).squeeze()  # squeezing unnecessary dimension
                del data[k1]['accZ']
            if 'conf' in data_keys:
                data[k1]['config'] = data[k1].pop('conf')  # rename key, ditching old one
            if 'fsamp' in data_keys:  # (Hz): sampling frequency.
                data[k1]['fs'] = data[k1].pop('fsamp')  # rename key, ditching old one
            if 'upwind_uvw' in data_keys:
                data[k1]['uvw_upwind'] = data[k1].pop('upwind_uvw')  # rename key, ditching old one
                data[k1]['uvw_upwind'] = np.array(data[k1]['uvw_upwind']).T  # new shape: (3, n_samples)
                data[k1]['U_upwind'] = np.mean(data[k1]['uvw_upwind'][0])
            if 'qRef' in data_keys:
                data[k1]['q_ref'] = data[k1].pop('qRef')  # rename key, ditching old one
            if 'qTildeRef' in data_keys:
                data[k1]['q_tilde_ref'] = data[k1].pop('qTildeRef')  # rename key, ditching old one
            if 'Vupwind_mod' in data_keys:
                assert 'u' in data_keys  # such that U was calculated and set up as new key
                assert np.allclose(data[k1]['Vupwind_mod'], data[k1]['U_upwind'], rtol=1e-02)
                del data[k1]['Vupwind_mod']
            if 'PMot' in data_keys:  # wind tunnel power percentage
                data[k1]['tunnel_power'] = data[k1].pop('PMot')  # rename key, ditching old one

            data[k1] = dict(sorted(data[k1].items()))  # Sorting the dict keys
        assert all_equal([data[k].keys() for k in data.keys()]), \
            "Some files, within the same raw_data_type have different keys! Check the raw data"
        data = dict(sorted(data.items()))  # Sorting the dict keys
        return data

    file_paths = {}  # for all raw_data_types
    data = {}  # for all raw_data_types
    for t, lst_cues in zip(raw_data_types, raw_data_str_cues):
        # Gett all file paths of current raw_data_type
        file_paths[t] = [os.path.join(root, name) for root, dirs, files in os.walk(raw_data_path)
                         for name in files if any(cue in root for cue in lst_cues)]
        data[t] = get_raw_data_dict_from_file_paths(file_paths[t], raw_data_type=t)
    data = dict(sorted(data.items()))  # Sorting the dict keys
    return data


def get_dfs_from_raw_data(raw_data_dict, drop_time_series=True, drop_matlab_info=True, report_all_original_keys=True):
    """
    Takes as input a triply nested dictionary, with the outer keys as raw_data_types.
    drop_time_series: drops the large time series, for efficiency and to get a dataframe with non-iterable cells
    """
    assert list(raw_data_dict.keys()) == raw_data_types, 'new raw_data_type is not implemented!'

    def get_df_from_raw_data(raw_data):
        """
        Dataframe processing that is common to all raw_data_type
        raw_data: doubly nested dictionary
        """
        df = pd.DataFrame.from_dict(raw_data).transpose()
        if drop_time_series:
            cols_w_time_series = ['acc_z', 'q_ceil', 'q_upwind', 'u', 'v', 'w', 't', 't_uvw', 'u_ceil', 'F',
                                  'uvw_upwind']
            df = df.drop(cols_w_time_series, axis=1, errors='ignore')  # ignore already non-existing keys
        if drop_matlab_info:
            cols_w_matlab_info = ['__globals__', '__header__', '__version__']
            df = df.drop(cols_w_matlab_info, axis=1, errors='ignore')
        return df.reset_index(names='case_tag')

    dict_of_dfs = {k: get_df_from_raw_data(raw_data_dict[k]) for k in raw_data_types}

    # Some cols have a list on each cell. These lists will be exploded to multiple rows
    for k in raw_data_types:
        if k in ['A10', 'A11-A12']:
            dict_of_dfs['A10']['U'] = dict_of_dfs['A10']['U'].apply(np.squeeze)  # 'U' is composed of lists of size 1
        if k not in ['A10', 'A11-A12']:
            cols_to_explode = ['cobras', 'U', 'units', 'dof_tag', 'F_mean']  # add columns as needed here
            cols_to_explode_1df = [c for c in dict_of_dfs[k].columns.values if c in cols_to_explode]
            dict_of_dfs[k] = dict_of_dfs[k].explode(cols_to_explode_1df)

    if report_all_original_keys:
        print(f'Collecting all original keys of the raw data. May take several minutes...')
        dict_of_dfs['df_all_original_keys'], dict_of_dfs['set_original_keys'] = overview_all_raw_file_keys()

    return dict_of_dfs


# Getting processed dataframes
raw_data_dict = get_raw_data_dict(raw_data_path)
dict_of_dfs = get_dfs_from_raw_data(raw_data_dict)







raise NotImplementedError

# Further post-processing
coh_df['U/U_ceil'] = coh_df['U'] / coh_df['U_ceil']


# The following plots assess if we should have yaw-dependent coefficients or not. Conclusion: not
def plot_for_cobras():
    """
    Plot normalized U with the cobra ID in the x_axis
    """
    x_axis = 'cobras'
    label_axis = 'yaw'
    coh_df['colors_to_plot'] = get_list_of_colors_matching_list_of_objects(coh_df[label_axis])
    plt.figure(dpi=400)
    for label, color in dict(zip(coh_df[label_axis], coh_df['colors_to_plot'])).items():
        sub_df = coh_df[coh_df[label_axis] == label]  # subset dataframe
        plt.scatter(sub_df[x_axis], sub_df['U/U_ceil'], c=sub_df['colors_to_plot'], label=label, alpha=0.8)
    plt.legend(title='yaw [deg]', bbox_to_anchor=(1.04, 0.5), loc="lower left")
    plt.ylabel(r'$U_{centre}\//\/U_{ceiling}$')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'aerodynamic_coefficients',
                             'polimi', 'preliminary', 'polimi_U_by_Uceil_for_cobras.jpg'))
    plt.close()


def plot_for_yaw():
    """
    Plot normalized U with the yaw angle in the x_axis
    """
    x_axis = 'yaw'
    label_axis = 'cobras'
    coh_df['colors_to_plot'] = get_list_of_colors_matching_list_of_objects(coh_df[label_axis])
    plt.figure(figsize=(8, 4), dpi=400)
    plt.title('(obtained from the coherence measurement tests)')
    for label, color in dict(zip(coh_df[label_axis], coh_df['colors_to_plot'])).items():
        sub_df = coh_df[coh_df[label_axis] == label]  # subset dataframe
        plt.scatter(sub_df[x_axis], sub_df['U/U_ceil'], c=sub_df['colors_to_plot'], label=label, alpha=0.8)
    plt.scatter([0, 30, 60, 90],
                [0.886, 0.839, 0.833, 0.851],
                marker='x', color='black', label='(from wind profile tests)')
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left")
    plt.ylabel(r'$U_{centre}\//\/U_{ceiling}$')
    plt.xlabel('yaw angle [deg]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'aerodynamic_coefficients',
                             'polimi', 'preliminary', 'polimi_U_by_Uceil_for_yaw.jpg'))
    plt.close()


# plot_for_cobras()
# plot_for_yaw()

raise NotImplementedError

# Calculating the coefficients from force measurements. For this, the wind profile needs to be established, because
# the wind forces are to be normalized by the integrated / averaged wind speed along the pontoon / column height
scale = 1 / 35  # wind-tunnel model scale
dof = 'Fy'  #
yaw = 180


# This function is to be re-done when we get the final raw data files
def get_raw_aero_coef(dof, yaw, where='pont', back_engineered_U=False):
    """
    Describe here....
    at: 'pont', 'deck', ...
    """

    #   The wind profile values were taken from a draft raw data Excel file provided below:
    #   https://vegvesen.sharepoint.com/:x:/s/arb-bjffeedwindtunneltests/EVehciF9YZJEoTX42t6B2qkBSay4agGUy_8MMMQHn9lntA?email=bernardo.morais.da.costa%40vegvesen.no&e=BwZEry
    # Wind profile, required for force normalization
    U_profile_raw = np.array([[1E-5, 1E-5],
                              [0.03, 8.3398],
                              [0.05, 8.4902],
                              [0.10, 8.6605],
                              [0.15, 8.9762],
                              [0.20, 9.2538],
                              [0.25, 9.4597],
                              [0.30, 9.854],
                              [0.35, 9.918],
                              [0.40, 10.0355],
                              [0.46, 10.2358],
                              [0.50, 10.4923],
                              [0.60, 10.5613],
                              [0.70, 10.7889]])
    x_raw = U_profile_raw[:, 0]
    y_raw = U_profile_raw[:, 1]

    def func_to_fit(z, a, b):
        """
        this logarithmic function is used to fit the measurements of the wind profile (with parameters 'a' and 'b')
        """
        return a * np.log(z / b)

    def get_fitted_wind_profile(plot=False):
        """
        x_fit: Used for the height above ground (z) [m]
        return: the fitted wind speed [m/s]
        """
        x_fit = np.linspace(x_raw.min(), x_raw.max(), num=1000000)
        y_interp = np.interp(x=x_fit, xp=x_raw, fp=y_raw)
        popt, pcov, *_ = curve_fit(f=func_to_fit, xdata=x_raw, ydata=y_raw, bounds=np.array([[0, 0], [np.inf, np.inf]]))
        y_fit = func_to_fit(x_fit, *popt)
        if plot:
            # Plotting the fitted logarithm
            plt.scatter(x_raw, y_raw, label='raw data')
            plt.plot(x_fit, y_interp, label='interpolation')
            plt.plot(x_fit, y_fit, label='curve fit')
            plt.legend()
            plt.show()
        return x_fit, y_fit

    x_fit, y_fit = get_fitted_wind_profile()

    # # Integrating raw (coarser) data
    # idx = np.where(x_raw <= H)[0][-1]
    # y_raw_ref = np.trapz(y=y_raw[:idx+1], x=x_raw[:idx+1]) / H

    # # Integrating interpolated (finer) data
    # idx = np.where(x_fit <= H)[0][-1]
    # y_interp_ref = np.trapz(y=y_interp[:idx+1], x=x_fit[:idx+1]) / H

    if where == 'pont':
        H = H_pont
        B = B_pont
        L = L_pont
        sub_df = pont_df[(pont_df['dof_tag'] == dof + '-PTot') & (pont_df['yaw'] == yaw)]  # subset of df

    else:
        assert where == 'deck', "Error: Haven't implemented coefficients of other elements"
        H = np.round(3.5 * scale, 10)
        sub_df = deck_df[(deck_df['dof_tag'] == dof + '-Tot') & (deck_df['yaw'] == yaw)]  # subset of df

    if not back_engineered_U:
        # Integrating fitted (finer) data
        idx = np.where(x_fit <= H)[0][-1]
        U_ref = np.trapz(y=y_fit[:idx + 1], x=x_fit[:idx + 1]) / H
    else:  # todo: to be deleted, used just for testing
        U_ref = 7.596  # the speed that gives the q-tilde-ref in the Excel file of Polimi
    qref_tilde = 1 / 2 * rho * U_ref ** 2
    F = sub_df['F_mean']  # force or moment

    rho = sub_df['rho']

    # BERNARDO RE-CODE ALL THIS. SEE ANNEX-DRAFT.PDF AND THE FORMULAS OF EACH COEFFICIENT FOR THE PONTOON AND DECK
    # if dof in ['Fx', 'Fy', 'Fz']:
    #
    #     C = F / (qref_tilde * L * H)
    # else:
    #     assert dof in ['Mx', 'My', 'Mz']

    return NotImplementedError
