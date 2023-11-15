import os
import numpy as np
import matplotlib.pyplot as plt
from mat4py import loadmat
from my_utils import root_dir
import pandas as pd


folder_path_to_raw_data = os.path.join(root_dir, r"aerodynamic_coefficients\polimi_raw_data\testing")


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


raw_data_dict = get_raw_data_dict(folder_path_to_raw_data)
coh_df, pont_df, deck_df = get_dfs_from_raw_data(raw_data_dict)



NotImplementedError  # COntinue here Bernardo





plt.title(r'$U\//\/U_{ceiling}$')
for k1 in coh_data:
    data = coh_data[k1]
    plt.scatter(data['cobras'], data['U'] / data['U_ceiling'], label=data['yaw'])
plt.legend(title='Yaw [deg]:', bbox_to_anchor=(1.04, 0), loc="lower left")
plt.ylabel('U [m/s]')
plt.grid()
plt.tight_layout()
plt.show(block=True)

for k1 in coh_data:
    data = coh_data[k1]
    n_cobras = len(data['cobras'])
    for idx, cobra in enumerate(data['cobras']):
        plt.scatter(data['yaw'], data['U'][idx] / data['U_ceiling'], label=data['cobras'])
plt.legend(title='Cobra:', bbox_to_anchor=(1.04, 0), loc="lower left")
plt.ylabel('U [m/s]')
plt.grid()
plt.tight_layout()
plt.show(block=True)

plt.close()

#
# # %% TESTING
# filename1 = folder_name + "J-0026_SIW_ATI_P7000_Ang+000.mat"
# filename2 = folder_name + "J-1658_SIW_FLOW-MD_P7700_Ang-180.mat"
# filename3 = folder_name + "J-0211_SIW_DECK-AERO_P7700_Ang-150.mat"
#
# # data1 = loadmat(filename1)
# # data2 = loadmat(filename2)
# # data3 = loadmat(filename3)
#
# filename = filename2
# # def get_raw_data_dict(filename):
# raw_data = loadmat(filename)
# u = np.array(raw_data["u"]).T  # shape: (n_cobras, n_samples)
# v = np.array(raw_data["v"]).T  # shape: (n_cobras, n_samples)
# w = np.array(raw_data["w"]).T  # shape: (n_cobras, n_samples)
# # pass
#
# np.corrcoef(u[0], u[3])
#
#
# # todo: TELL POLIMI ABOUT:
print(" len(t) is 120000, but len(u)=len(v)=len(w) is 119808. File example: J-1658_SIW_FLOW-MD_P7700_Ang-180.mat  ")
print("Polimi hasn't delivered the wind profile at column and pontoon locations yet?")
print("No upwind_uvw information in the pontoon data? (Understandable that it is not in the coherence data)")
print("It would be very useful to have the data from the upwind pitot tube for all measurements")
print("If the Polimi yaw_angle is kept in the file names, then we must know if the data provided is already"
      "transformed to our yaw definition (by multiplying by -1 where needed) or not")





