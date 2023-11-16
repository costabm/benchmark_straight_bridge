import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mat4py import loadmat
from my_utils import root_dir, get_list_of_colors_matching_list_of_objects, fill_with_None_where_repeated
import pandas as pd
matplotlib.use('Qt5Agg')

folder_path_to_raw_data = os.path.join(root_dir, r"aerodynamic_coefficients\polimi_raw_data\preliminary")


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
            u_ceiling = np.sqrt(data[k1]['qCeiling'] / (1/2*data[k1]['rho']))
            polimi_yaw = float(file_rel_path.split('_Ang')[1].split('.mat')[0])  # find string between substrings
            data[k1].update({'u_ceiling': u_ceiling, 'U_ceiling': np.mean(u_ceiling), 'polimi_yaw': polimi_yaw})

            # Collect data unique to each raw_data_type
            if raw_data_type == 'coh':
                u = np.array(data[k1]['u']).T  # new shape: (n_cobras, n_samples)
                v = np.array(data[k1]['v']).T  # new shape: (n_cobras, n_samples)
                w = np.array(data[k1]['w']).T  # new shape: (n_cobras, n_samples)
                U = np.mean(u, axis=1)
                data[k1].update({'U': U, 'u': u, 'v': v, 'w': w})  # updating dict
            if raw_data_type in ['pont', 'deck']:
                data[k1]['THForces'] = np.array(data[k1]['THForces']).T  # new shape: (dof, n_samples)
                data[k1]['THForces_mean'] = np.mean(data[k1]['THForces'], axis=1)  # new shape: (dof, n_samples)
            if raw_data_type == 'deck':
                data[k1]['upwind_uvw'] = np.array(data[k1]['upwind_uvw']).T  # new shape: (3, n_samples)
                data[k1]['upwind_U'] = np.mean(data[k1]['upwind_uvw'][0])
        return data

    coh_data = get_raw_data_dict_from_file_paths(coh_file_paths, raw_data_type='coh')
    pont_data = get_raw_data_dict_from_file_paths(pont_file_paths, raw_data_type='pont')
    deck_data = get_raw_data_dict_from_file_paths(deck_file_paths, raw_data_type='deck')

    return {'coh_data': coh_data,
            'pont_data': pont_data,
            'deck_data': deck_data}  # todo: complete the remaining data


def get_dfs_from_raw_data(raw_data_dict, drop_time_series=True):
    """
    Takes as input a triply nested dictionary, with the outer keys given in the first assertion below
    drop_time_series: drops the large time series, for efficiency and to get a dataframe with non-iterable cells
    """
    assert list(raw_data_dict.keys()) == ['coh_data', 'pont_data', 'deck_data'], 'new raw_data_type is not implemented!'

    def get_df_from_raw_data(raw_data):
        """
        The dataframe processing that is common to all raw_data_type is performed in this function
        raw_data: doubly nested dictionary
        """
        df = pd.DataFrame.from_dict(raw_data).transpose()
        if drop_time_series:
            df = df.drop(['qCeiling', 'u', 'v', 'w', 't', 'u_ceiling',
                            'THForces', 'upwind_uvw'], axis=1, errors='ignore')  # ignore already non-existing keys
        return df.reset_index(names='case_tag')

    coh_df = get_df_from_raw_data(raw_data_dict['coh_data'])
    pont_df = get_df_from_raw_data(raw_data_dict['pont_data'])
    deck_df = get_df_from_raw_data(raw_data_dict['deck_data'])

    # These cols have a list on each cell. Exploding lists to multiple rows
    coh_df = coh_df.explode(['cobras', 'U'])
    pont_df = pont_df.explode(['EU', 'names', 'THForces_mean'])
    deck_df = deck_df.explode(['EU', 'names', 'THForces_mean'])

    return coh_df, pont_df, deck_df


# Getting processed dataframes
raw_data_dict = get_raw_data_dict(folder_path_to_raw_data)
coh_df, pont_df, deck_df = get_dfs_from_raw_data(raw_data_dict)

# Further post-processing
coh_df['U/U_ceiling'] = coh_df['U'] /coh_df['U_ceiling']


# Plotting
def plot_for_cobras():
    x_axis = 'cobras'
    label_axis = 'yaw'
    coh_df['colors_to_plot'] = get_list_of_colors_matching_list_of_objects(coh_df[label_axis])
    plt.figure(dpi=400)
    for label, color in dict(zip(coh_df[label_axis], coh_df['colors_to_plot'])).items():
        sub_df = coh_df[coh_df[label_axis] == label]  # subset dataframe
        plt.scatter(sub_df[x_axis], sub_df['U/U_ceiling'], c=sub_df['colors_to_plot'], label=label, alpha=0.8)
    plt.legend(title='yaw [deg]', bbox_to_anchor=(1.04, 0.5), loc="lower left")
    plt.ylabel('$U_{centre}\//\/U_{ceiling}$')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'aerodynamic_coefficients',
                             'polimi_raw_data', 'preliminary', 'polimi_U_by_Uceil_for_cobras.jpg'))
    plt.close()


def plot_for_yaw():
    x_axis = 'yaw'
    label_axis = 'cobras'
    coh_df['colors_to_plot'] = get_list_of_colors_matching_list_of_objects(coh_df[label_axis])
    plt.figure(figsize=(8, 4), dpi=400)
    plt.title('(obtained from the coherence measurement tests)')
    for label, color in dict(zip(coh_df[label_axis], coh_df['colors_to_plot'])).items():
        sub_df = coh_df[coh_df[label_axis] == label]  # subset dataframe
        plt.scatter(sub_df[x_axis], sub_df['U/U_ceiling'], c=sub_df['colors_to_plot'], label=label, alpha=0.8)
    plt.scatter([0,30,60,90],
                [0.886, 0.839, 0.833, 0.851],
                marker='x', color='black', label='(from wind profile tests)')
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left")
    plt.ylabel('$U_{centre}\//\/U_{ceiling}$')
    plt.xlabel('yaw angle [deg]')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'aerodynamic_coefficients',
                             'polimi_raw_data', 'preliminary', 'polimi_U_by_Uceil_for_yaw.jpg'))
    plt.close()


# plot_for_cobras()
# plot_for_yaw()


# Calculating the coefficients from force measurements
scale = 1/35
dof = 'Fx'
yaw = 180
H = 3.5 * scale
B = 14.875 * scale
L = 53 * scale
U_profile_column_beta0 = np.array([[0.0, 0.0],
                                   [0.03, 8.3398],
                                   [0.05, 8.4902],
                                   [0.1, 8.6605],
                                   [0.15, 8.9762],
                                   [0.2, 9.2538],
                                   [0.25, 9.4597],
                                   [0.3, 9.854],
                                   [0.35, 9.918],
                                   [0.4, 10.0355],
                                   [0.46, 10.2358],
                                   [0.5, 10.4923],
                                   [0.6, 10.5613],
                                   [0.7, 10.7889]])

np.interp(x=[0.1], xp=)

U_idx_at_H = np.where(U_profile_column_beta0[:,0]<=0.1)[0][-1]
np.trapz(y=U_profile_column_beta0[:U_idx_at_H+1,1], x=U_profile_column_beta0[:U_idx_at_H+1,0]) / H

sub_df = pont_df[(pont_df['names']==dof+'-PTot') & (pont_df['yaw']==yaw)]  # subset of df

rho = sub_df['rho']

qref_tilde = 1/2 * rho * 7.596**2

Fx = sub_df['THForces_mean']


Cx = Fx / (qref_tilde * L * H)





# # todo: TELL POLIMI ABOUT:
print(" len(t) is 120000, but len(u)=len(v)=len(w) is 119808. File example: J-1658_SIW_FLOW-MD_P7700_Ang-180.mat  ")
print("Polimi hasn't delivered the wind profile at column and pontoon locations yet?")
print("No upwind_uvw information in the pontoon data? (Understandable that it is not in the coherence data)")
print("It would be very useful to have the data from the upwind pitot tube for all measurements")
print("If the Polimi yaw_angle is kept in the file names, then we must know if the data provided is already"
      "transformed to our yaw definition (by multiplying by -1 where needed) or not")
