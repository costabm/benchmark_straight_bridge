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
from mat4py import loadmat
from scipy.optimize import curve_fit
from my_utils import root_dir, get_list_of_colors_matching_list_of_objects, fill_with_None_where_repeated
import pandas as pd
matplotlib.use('Qt5Agg')  # to prevent bug in PyCharm

folder_path_to_raw_data = os.path.join(root_dir, r"aerodynamic_coefficients\polimi\preliminary")

# Common variables
scale = 1/35  # model scale

# Pontoon dimensions (for normalization)
H_pont = np.round(3.5 * scale, 10)
B_pont = 14.875 * scale
L_pont = 53 * scale


def get_raw_data_dict(folder_path):
    """
    folder_path: the absolute path to the folder with all the raw .mat files provided by Polimi
    """
    lst_dirs = os.listdir(folder_path_to_raw_data)
    coh_file_paths = [path for path in lst_dirs if ('SIW_FLOW' in path)]
    pont_file_paths = [path for path in lst_dirs if ('SIW_ATI' in path)]
    deck_file_paths = [path for path in lst_dirs if ('SIW_DECK' in path)]

    def get_raw_data_dict_from_file_paths(file_paths, raw_data_type):
        """
        raw_data_type: 'coh', 'pont' or 'deck' (for coherence, pontoon, or deck data, respectively)
        """
        data = {}  # prepare for a nested dict
        for file_rel_path in file_paths:
            file_abs_path = os.path.join(folder_path, file_rel_path)
            raw_file = loadmat(file_abs_path)

            # Treat and collect data common to various raw_data_type
            raw_file['t'] = np.array(raw_file['t'])
            raw_file['qCeiling'] = np.array(raw_file['qCeiling']).squeeze()  # squeezing unnecessary dimension
            k1 = file_rel_path
            data[k1] = {}  # prepare for a nested dict
            for k2 in raw_file:  # collecting for all keys in the raw file
                data[k1][k2] = raw_file[k2]
            u_ceil = np.sqrt(data[k1]['qCeiling'] / (1/2*data[k1]['rho']))
            polimi_yaw = float(file_rel_path.split('_Ang')[1].split('.mat')[0])  # find string between substrings
            data[k1].update({'u_ceil': u_ceil, 'U_ceil': np.mean(u_ceil), 'polimi_yaw': polimi_yaw})
            data[k1]['q_ceil'] = data[k1].pop('qCeiling')  # rename key, ditching old one

            # Collect data unique to each raw_data_type
            if raw_data_type == 'coh':
                u = np.array(data[k1]['u']).T  # new shape: (n_cobras, n_samples)
                v = np.array(data[k1]['v']).T  # new shape: (n_cobras, n_samples)
                w = np.array(data[k1]['w']).T  # new shape: (n_cobras, n_samples)
                U = np.mean(u, axis=1)
                data[k1].update({'U': U, 'u': u, 'v': v, 'w': w})  # updating dict
            if raw_data_type in ['pont', 'deck']:
                data[k1]['F'] = np.array(data[k1]['THForces']).T  # new shape: (dof, n_samples)
                del data[k1]['THForces']  # ditching old key
                data[k1]['units'] = data[k1].pop('EU')  # rename key, ditching old one
                data[k1]['dof_tag'] = data[k1].pop('names')  # rename key, ditching old one
                data[k1]['F_mean'] = np.mean(data[k1]['F'], axis=1)  # new shape: (dof, n_samples)
            if raw_data_type == 'deck':
                data[k1]['upwind_uvw'] = np.array(data[k1]['upwind_uvw']).T  # new shape: (3, n_samples)
                data[k1]['upwind_U'] = np.mean(data[k1]['upwind_uvw'][0])
                data[k1]['q_ref'] = data[k1].pop('qRef')  # rename key, ditching old one
            if raw_data_type == 'pont':
                data[k1]['q_tilde_ref'] = data[k1].pop('qTildeRef')  # rename key, ditching old one
        return data

    coh_data = get_raw_data_dict_from_file_paths(coh_file_paths, raw_data_type='coh')
    pont_data = get_raw_data_dict_from_file_paths(pont_file_paths, raw_data_type='pont')
    deck_data = get_raw_data_dict_from_file_paths(deck_file_paths, raw_data_type='deck')

    return {'coh_data': coh_data,
            'pont_data': pont_data,
            'deck_data': deck_data}


def get_dfs_from_raw_data(raw_data_dict, drop_time_series=True):
    """
    Takes as input a triply nested dictionary, with the outer keys as 'coh_data', 'pont_data' and 'deck_data'.
    drop_time_series: drops the large time series, for efficiency and to get a dataframe with non-iterable cells
    """
    assert list(raw_data_dict.keys()) == ['coh_data', 'pont_data', 'deck_data'], 'new raw_data_type is not implemented!'

    def get_df_from_raw_data(raw_data):
        """
        Dataframe processing that is common to all raw_data_type
        raw_data: doubly nested dictionary
        """
        df = pd.DataFrame.from_dict(raw_data).transpose()
        if drop_time_series:
            df = df.drop(['q_ceil', 'u', 'v', 'w', 't', 'u_ceil',
                            'F', 'upwind_uvw'], axis=1, errors='ignore')  # ignore already non-existing keys
        return df.reset_index(names='case_tag')

    coh_df = get_df_from_raw_data(raw_data_dict['coh_data'])
    pont_df = get_df_from_raw_data(raw_data_dict['pont_data'])
    deck_df = get_df_from_raw_data(raw_data_dict['deck_data'])

    # These cols have a list on each cell. Exploding lists to multiple rows
    coh_df = coh_df.explode(['cobras', 'U'])
    pont_df = pont_df.explode(['units', 'dof_tag', 'F_mean'])
    deck_df = deck_df.explode(['units', 'dof_tag', 'F_mean'])

    return coh_df, pont_df, deck_df


# Getting processed dataframes
raw_data_dict = get_raw_data_dict(folder_path_to_raw_data)
coh_df, pont_df, deck_df = get_dfs_from_raw_data(raw_data_dict)

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
    plt.ylabel('$U_{centre}\//\/U_{ceiling}$')
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
    plt.scatter([0,30,60,90],
                [0.886, 0.839, 0.833, 0.851],
                marker='x', color='black', label='(from wind profile tests)')
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left")
    plt.ylabel('$U_{centre}\//\/U_{ceiling}$')
    plt.xlabel('yaw angle [deg]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'aerodynamic_coefficients',
                             'polimi', 'preliminary', 'polimi_U_by_Uceil_for_yaw.jpg'))
    plt.close()


# plot_for_cobras()
# plot_for_yaw()


# Calculating the coefficients from force measurements. For this, the wind profile needs to be established, because
# the wind forces are to be normalized by the integrated / averaged wind speed along the pontoon / column height
scale = 1/35  # wind-tunnel model scale
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
        U_ref = np.trapz(y=y_fit[:idx+1], x=x_fit[:idx+1]) / H
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

