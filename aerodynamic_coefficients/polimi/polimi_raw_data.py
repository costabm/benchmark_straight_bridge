"""
In late 2023, Politecnico di Milano (Polimi) delivered SVV new wind tunnel test results in skew winds.
This script allows pre- and post-processing the raw data files provided.

First, a raw_data_dict is generated, made from all the raw data files found in the folder raw_data_path.
Several data keys are renamed and some of the data is treated.

Next, the dict_of_dfs is obtained. The data is organized in Annexes. Several annexes share the same data keys
(U, U_ceil, etc.) so they are grouped together. Each group then forms a dataframe of this dict_of_dfs.

Next, df_all is obtained, simply compiling all the data in dict_of_dfs into one single dataframe, for convenience.
The result can be stored into a .csv file

A few checks are performed in the end.

bercos@vegvesen.no
November 2023
"""

import os
import numpy as np
import matplotlib
import scipy as sp
from my_utils import root_dir, all_equal, flatten_nested_list, deg, rad, from_df_all_get_unique_value_given_key_and_id
from transformations import beta_within_minus_Pi_and_Pi_func, beta_from_beta_rx0_and_rx, theta_from_beta_rx0_and_rx
import pandas as pd
import logging
matplotlib.use('Qt5Agg')  # to prevent bug in PyCharm


# Folder that includes all the raw data as provided by Polimi:
raw_data_path = os.path.join(root_dir, r"aerodynamic_coefficients\polimi\raw_data")
# File (Excel) with the coefficients results as provided by Polimi :
xls_data_path = os.path.join(root_dir, r"aerodynamic_coefficients\polimi\ResultsCoefficients-Rev3.xlsx")

debug = False  # set to True to make debugging of this script faster (only a few unique data files are processed)

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
            raw_file = sp.io.loadmat(file_path, squeeze_me=True)
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
            data[k1] = sp.io.loadmat(file_path, squeeze_me=True)  # squeeze unit dimensions
            data_keys = data[k1].keys()

            data[k1]['id'] = int(file_name[2:6])  # case number
            if 'inclinometer' in data_keys:  # time history acquisition time
                data[k1]['rx'] = -1.0 * data[k1]['inclinometer']  # opposite sign to inclinometer
                del data[k1]['inclinometer']  # ditching old key
            if 't' in data_keys:  # time history acquisition time
                pass
            if 'qCeiling' in data_keys:  # time hist of dynamic pressure measured by pitot tube placed on the ceiling
                data[k1]['q_ceil'] = data[k1].pop('qCeiling')  # rename key, ditching old one
                u_ceil = np.sqrt(data[k1]['q_ceil'] / (1 / 2 * data[k1]['rho']))
                data[k1].update({'u_ceil': u_ceil, 'U_ceil': np.mean(u_ceil)})
            if 'qUpwind' in data_keys:  # ... pitot placed upwind (2m ahead the turn table) at h=0.5m.
                data[k1]['q_upwind'] = data[k1].pop('qUpwind')  # rename key, ditching old one
            if 'turntable' in data_keys:  # angle imposed to the turning table during the test
                data[k1]['polimi_gamma'] = float(data[k1].pop('turntable'))  # sometimes it is int, so forcing to float
                # Check if the raw data angle 'polimi_gamma' corresponds to the angle in the file name after 'Ang':
                if raw_data_type in ['A10', 'A11-A12', 'A14']:  # file types with different Ang syntax in filename
                    polimi_gamma_2 = float(file_name.split('-Ang')[1].split('-Z')[0])  # find angle between substrings
                else:
                    polimi_gamma_2 = float(file_name.split('_Ang')[1].split('.mat')[0])  # find angle between substrings
                if not np.isclose(data[k1]['polimi_gamma'], polimi_gamma_2, atol=1):  # best to use 1 deg tolerance
                    logging.warning(f"Ang in filename != from 'polimi_gamma'. File: {file_path}")
                data[k1]['beta_rx0'] = data[k1]['polimi_gamma'] + 180.0  # AnnexA17 eq.: beta_rx0 = gamma + 180
                data[k1]['beta_rx0'] = deg(beta_within_minus_Pi_and_Pi_func(rad(data[k1]['beta_rx0'])))  # to [-pi,pi]
                if 'rx' in data_keys:
                    data[k1]['beta_svv'] = deg(beta_from_beta_rx0_and_rx(rad(data[k1]['beta_rx0']),
                                                                         rad(data[k1]['rx'])))
                    data[k1]['theta_svv'] = deg(theta_from_beta_rx0_and_rx(rad(data[k1]['beta_rx0']),
                                                                           rad(data[k1]['rx'])))
            if 'u' in data_keys:
                assert 'v' in data_keys
                assert 'w' in data_keys
                u = data[k1]['u'].T  # new shape: (n_cobras, n_samples) or (n_samples)
                v = data[k1]['v'].T  # new shape: (n_cobras, n_samples) or (n_samples)
                w = data[k1]['w'].T  # new shape: (n_cobras, n_samples) or (n_samples)
                U = np.mean(u, axis=-1)
                data[k1].update({'U': U, 'u': u, 'v': v, 'w': w})  # updating dict
            if 'THForces' in data_keys:
                data[k1]['F'] = data[k1]['THForces'].T  # new shape: (dof, n_samples)
                del data[k1]['THForces']  # ditching old key
                assert 'EU' in data_keys
                data[k1]['units'] = data[k1].pop('EU')  # rename key, ditching old one
                assert 'names' in data_keys
                data[k1]['dof_tag'] = data[k1].pop('names')  # rename key, ditching old one
                data[k1]['F_mean'] = np.mean(data[k1]['F'], axis=1)  # new shape: (dof, n_samples)
            if 'H' in data[k1].keys():
                data[k1]['H'] = float(data[k1]['H'])
            if 'accZ' in data_keys:
                data[k1]['acc_z'] = data[k1]['accZ']
                del data[k1]['accZ']
            if 'conf' in data_keys:
                data[k1]['config'] = data[k1].pop('conf')  # rename key, ditching old one
            if 'fsamp' in data_keys:  # (Hz): sampling frequency.
                data[k1]['fs'] = float(data[k1].pop('fsamp'))  # rename key, ditching old one
            if 'h' in data_keys:  # height of the measurements, either from ground or from deck level (Annex10).
                data[k1]['z'] = float(data[k1].pop('h'))  # rename key, ditching old one
            if 'upwind_uvw' in data_keys:
                data[k1]['uvw_upwind'] = data[k1].pop('upwind_uvw')  # rename key, ditching old one
                data[k1]['uvw_upwind'] = data[k1]['uvw_upwind'].T  # new shape: (3, n_samples)
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


def get_dfs_from_raw_data(raw_data_dict, drop_time_series=True, drop_matlab_info=True, report_all_original_keys=False):
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

        keys_and_types = {'B': float, 'F': object, 'F_mean': object, 'H': float, 'L': float, 'U_ceil': float,
                          'U_upwind': float, '__globals__': object, '__header__': object, '__version__': object,
                          'acc_z': object, 'code': object, 'dof_tag': object, 'fs': float, 'id': int,
                          'polimi_gamma': float, 'q_ceil': object, 'q_ref': float, 'q_upwind': object, 'rho': float,
                          'rx': float, 't': object, 't_uvw': object, 'temp': float, 'theta': float, 'u_ceil': object,
                          'units': object, 'uvw_upwind': object, 'yaw': float}
        for key, value in keys_and_types.items():
            if key in df:
                df[key] = df[key].astype(value)

        return df.reset_index(names='case_tag')

    dict_of_dfs = {k: get_df_from_raw_data(raw_data_dict[k]) for k in raw_data_types}

    # Some cols have a list on each cell. These lists will be exploded to multiple rows
    for k in raw_data_types:
        if k not in ['A10', 'A11-A12']:
            cols_to_explode = ['cobras', 'U', 'units', 'dof_tag', 'F_mean']  # add columns as needed here
            cols_to_explode_1df = [c for c in dict_of_dfs[k].columns.values if c in cols_to_explode]
            dict_of_dfs[k] = dict_of_dfs[k].explode(cols_to_explode_1df)

    if report_all_original_keys:
        print(f'Collecting all original keys of the raw data. May take several minutes...')
        dict_of_dfs['df_all_original_keys'], dict_of_dfs['set_original_keys'] = overview_all_raw_file_keys()

    return dict_of_dfs


def get_df_all(dict_of_dfs, save_csv=False):
    """
    Concatenates all the dataframes in the dictionary dict_of_dfs into one big dataframe.
    'annex': the same as raw_data_type
    'explode_id': is just an index resulting from the .explode method in get_dfs_from_raw_data
    """
    df = pd.concat([dict_of_dfs[k] for k in raw_data_types],
                   keys=raw_data_types).reset_index().rename(
                   columns={'level_0': 'annex', 'level_1': 'explode_id'})
    if save_csv:
        df.to_csv(os.path.join(root_dir, r"aerodynamic_coefficients\polimi\df_of_all_polimi_tests.csv"))
    return df


def run_further_checks(df_all):
    """
    Running a few checks on the processed data (input format: dictionary of dataframes)
    """
    # TEST 1: Checking the consistency between the polimi_gamma (turntable) angles, and the reported "yaw" angles
    cols1 = ['polimi_gamma', 'yaw']  # columns to be checked
    test1 = df_all.copy()
    test1 = test1.drop_duplicates(subset=cols1).copy()  # drop duplicate combinations of these 2 angles
    test1['yaw_eq'] = np.array(test1['polimi_gamma'], dtype=float) + 180.0  # AnnexA17 equation: beta = gamma + 180
    test1['yaw_eq_-pi_pi'] = deg(beta_within_minus_Pi_and_Pi_func(rad(test1['yaw_eq'])))  # convert to interval
    test1['error_in_yaw_eq'] = test1['yaw'] - test1['yaw_eq_-pi_pi']  # difference (error)
    fail_condition1 = test1['error_in_yaw_eq'] > 0.6
    if fail_condition1.any():
        logging.warning('The following files have an inconsistent yaw angle definition. The polimi_gamma (turntable) '
                        'angle, when added by 180 deg according to the yaw equation (Polimi Report Annex A17),'
                        'produces a different angle than the reported "yaw" angle.')
        logging.warning(test1[fail_condition1])

    # TEST 2: Calculating beta and theta angles from rx (inclinometer) measurements and polimi_gamma (turntable) angles.
    cols2 = ['polimi_gamma', 'yaw', 'rx', 'beta_rx0', 'beta_svv', 'theta_svv']  # columns to be checked
    test2 = df_all.copy()
    test2 = test2.drop_duplicates(subset=cols2).copy()  # drop duplicate combinations of these 2 angles
    test2['error_in_theta'] = test2['theta_svv'] - test2['theta']
    fail_condition2 = test2['error_in_theta'] > 0.1
    if fail_condition2.any():
        logging.warning('The following files have significant theta angle deviations from my theta estimation.')
        logging.warning(test2[fail_condition2])


def add_sheet_with_svv_adapted_aero_coefs(xls_data_path, df_all):
    """
    The 'K12-G-L' aero coefficients need to be adapted to account for the traffic signs (only tested for a few angles).
    This function opens the Excel file provided by Polimi and adds a new sheet with the new adapted aero coefs.
    If df_all is provided instead of None, the true beta and theta values are calculated and included as "..._svv",
    and the beta_rx0 and rx used to calculate them are also included.
    For the remaining 'K12-G-L-T1', 'K12-G-L-T3', 'K12-G-L-CS' coefs, only formatting changes are performed.
    """

    for in_sheet in ['K12-G-L', 'K12-G-L-T1', 'K12-G-L-T3', 'K12-G-L-CS']:
        out_sheet = in_sheet + '-SVV'
        xl = pd.ExcelFile(xls_data_path)
        assert in_sheet in xl.sheet_names, "Sheet name not found"
        xls_df = xl.parse(sheet_name=in_sheet)  # parses to a dataframe
        xl.close()
        xls_df_svv = xls_df.copy()  # the new sheet '...-SVV' starts as a copy of the in_sheet

        def filter_out_other_quadrants(df):
            return df[(df['Yaw'] >= -0.5) & (df['Yaw'] <= 90.5)]  # with 0.5 deg tolerance

        def reformat_xls(xls_df_svv, df_all):
            """Filter out quadrants; add new keys for SVV angles; change old keys; force values to 0"""

            xls_df_svv = xls_df_svv.copy()  # this is safer and avoids SettingWithWarning!
            xls_df_svv = filter_out_other_quadrants(xls_df_svv)

            # Get pre-calculated angle info from df_all, given the unique 'run' id.
            beta_rx0, rx, beta_svv, theta_svv = [], [], [], []
            for i in xls_df_svv['run']:  # 'run' (Polimi notation) and 'id' (my notation) are the same thing
                beta_rx0.append(from_df_all_get_unique_value_given_key_and_id(df_all, key='beta_rx0', run=i))
                rx.append(from_df_all_get_unique_value_given_key_and_id(df_all, key='rx', run=i))
                beta_svv.append(from_df_all_get_unique_value_given_key_and_id(df_all, key='beta_svv', run=i))
                theta_svv.append(from_df_all_get_unique_value_given_key_and_id(df_all, key='theta_svv', run=i))

            xls_df_svv['beta_rx0'] = beta_rx0
            xls_df_svv['rx'] = rx
            xls_df_svv['beta_svv'] = beta_svv
            xls_df_svv['theta_svv'] = theta_svv
            xls_df_svv['Cx_Ls'] = xls_df_svv.pop('CxTot')
            xls_df_svv['Cy_Ls'] = xls_df_svv.pop('CyTot')
            xls_df_svv['Cz_Ls'] = xls_df_svv.pop('CzTot')
            xls_df_svv['Cxx_Ls'] = xls_df_svv.pop('CMxTot')
            xls_df_svv['Cyy_Ls'] = xls_df_svv.pop('CMyTot')
            xls_df_svv['Czz_Ls'] = xls_df_svv.pop('CMzTot')
            xls_df_svv.rename(columns={'Yaw': 'beta_polimi', 'Theta': 'theta_polimi'}, inplace=True)

            # Formatting changes, e.g. changing key names:
            cols_to_del = ['CxL', 'CyL', 'CzL', 'CMxL', 'CMyL', 'CMzL', 'Cxi', 'Cyi', 'Czi', 'CMxi', 'CMyi', 'CMzi']
            for c in cols_to_del:
                del xls_df_svv[c]

            # Forcing coefficient that should be 0 to 0:
            xls_df_svv.loc[(xls_df_svv['beta_polimi'] == 0) | (xls_df_svv['beta_polimi'] == 180), 'Cx_Ls'] = 0  # Constr.1
            xls_df_svv.loc[(xls_df_svv['beta_polimi'] == 0) | (xls_df_svv['beta_polimi'] == 180), 'Cyy_Ls'] = 0  # Constr.1
            xls_df_svv.loc[(xls_df_svv['beta_polimi'] == 0) | (xls_df_svv['beta_polimi'] == 180), 'Czz_Ls'] = 0  # Constr.1
            xls_df_svv.loc[(xls_df_svv['beta_polimi'] == 90) | (xls_df_svv['beta_polimi'] == -90), 'Cy_Ls'] = 0  # Constr.2
            xls_df_svv.loc[(xls_df_svv['beta_polimi'] == 90) | (xls_df_svv['beta_polimi'] == -90), 'Cxx_Ls'] = 0  # Constr.2
            xls_df_svv.loc[(xls_df_svv['beta_polimi'] == 90) | (xls_df_svv['beta_polimi'] == -90), 'Czz_Ls'] = 0  # Constr.2
            xls_df_svv.loc[(xls_df_svv['beta_polimi'] == 90) | (xls_df_svv['beta_polimi'] == -90)
                           & (xls_df_svv['theta_polimi'] == 0), 'Cz_Ls'] = 0  # Constr. No.3

            return xls_df_svv

        xls_df_svv = reformat_xls(xls_df_svv, df_all)

        # Write new sheet.
        with pd.ExcelWriter(xls_data_path, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
            xls_df_svv.to_excel(writer, sheet_name=out_sheet, index=False)
        print(f'A new sheet {out_sheet}, with the SVV-adapted coefficients, as been created in {xls_data_path}')

        if in_sheet == 'K12-G-L':  # Then create EXTRA sheet with the traffic sign effects (TS).
            xls_df_svv = xls_df.copy()  # RESTART
            out_sheet = in_sheet + '-TS-SVV'

            def scale_coefs_between_2_sheets(xls_df_svv, xls_data_path, from_sheet='K12-G-L', to_sheet='K12-G-L-TS',
                                             dof='CxTot', factor_on_scale_factor=0.44):
                """
                Scales the coefficients of one sheet, e.g. Cx from 'K12-G-L' to those in 'K12-G-L-TS', linearly
                interpolating at beta-missing values (e.g. scales up coefs without traffic signs, to those with traffic
                signs despite only a few angles having been tested with traffic signs). Only the values at theta=0 are being
                used for the scaling.

                xls_df_svv: dataframe (equivalent to one sheet) where the new results will be saved
                xls_data_path: path of the original Excel file with all sheets
                from_sheet: sheet name from where to take original values
                to_sheet: sheet name with the values to which the original values will be scaled
                dof: str with the degree-of-freedom at stake
                factor_on_scale_factor: Another factor to be applied to the scale factor! can be used to account for e.g.
                    the real VS modelled traffic sign area (see my note "SVV supplementary analyses of Polimi’s wind tunnel
                    tests – Bjørnafjord 2023"). Use 0.44 for Cx and 0.42 for Cy.

                Mental exercise:
                (TS = effect of including traffic signs in the model test)
                Cx_no_TS = 2   (at given beta, average of several thetas)
                Cx_w_TS = 2.5  (at given beta, average of several thetas)
                factor_on_scale_factor = 0.44 (the TS effect is overestimated, and only 44% of it is realistic)
                Cs_new_alt =   2 * (1 + (2.5/2 - 1) * 0.44)  # where final_factor = (1 + (2.5/2 - 1) * 0.44)
                Cs_new_alt_1 = 2 + 2 * (2.5/2 - 1) * 0.44  # alternative 1
                Cs_new_alt_2 = 2 + (2.5 - 2) * 0.44  # alternative 2

                """
                xl = pd.ExcelFile(xls_data_path)
                assert (from_sheet in xl.sheet_names) and (to_sheet in xl.sheet_names), "Sheet name not found"
                xls_df_from = filter_out_other_quadrants(xl.parse(sheet_name=from_sheet))  # parses to dataframe
                xls_df_to = filter_out_other_quadrants(xl.parse(sheet_name=to_sheet))  # parses to dataframe
                xl.close()

                beta_from_list = np.unique(xls_df_from['Yaw'])
                beta_to_list = np.unique(xls_df_to['Yaw'])

                # "Alternative 2": Lin. interp. on both _from and _to. Sine Rule of the isolated TS-effect supports this.
                for b in beta_from_list:
                    if b in beta_to_list:
                        C_from = np.array(xls_df_from[(xls_df_from['Yaw'] == b) & (xls_df_from['Theta'] == 0)][dof])
                        C_to = np.array(xls_df_to[(xls_df_to['Yaw'] == b) & (xls_df_to['Theta'] == 0)][dof])
                    else:  # We need to linearly interpolate the scaling of xls_df_from with the nearest values in xls_df_to
                        betas_larger = beta_to_list[np.where(beta_to_list > b)]
                        betas_smaller = beta_to_list[np.where(beta_to_list < b)]
                        beta_below = betas_smaller[np.argmin(abs(betas_smaller - b))]
                        beta_above = betas_larger[np.argmin(abs(betas_larger - b))]
                        C_to_below = np.array(xls_df_to[(xls_df_to['Yaw'] == beta_below) & (xls_df_to['Theta'] == 0)][dof])
                        C_to_above = np.array(xls_df_to[(xls_df_to['Yaw'] == beta_above) & (xls_df_to['Theta'] == 0)][dof])
                        C_to = C_to_below + (b-beta_below) * (C_to_above - C_to_below) / (beta_above - beta_below)
                        C_from_below = np.array(
                            xls_df_from[(xls_df_from['Yaw'] == beta_below) & (xls_df_from['Theta'] == 0)][dof])
                        C_from_above = np.array(
                            xls_df_from[(xls_df_from['Yaw'] == beta_above) & (xls_df_from['Theta'] == 0)][dof])
                        C_from = C_from_below + (b-beta_below) * (C_from_above - C_from_below) / (beta_above - beta_below)
                    scale_factor = C_to / C_from
                    final_factor = 1 + (scale_factor - 1) * factor_on_scale_factor
                    xls_df_svv.loc[xls_df_svv['Yaw'] == b, dof] *= final_factor

                return xls_df_svv

            # The following corrections are described in the document "SVV supplementary analyses of Polimi's ..."
            xls_df_svv = scale_coefs_between_2_sheets(xls_df_svv, xls_data_path, from_sheet='K12-G-L',
                                                      to_sheet='K12-G-L-TS', dof='CxTot', factor_on_scale_factor=0.44)
            xls_df_svv['CyTot'] = 1.014 * xls_df_svv['CyTot']  # Slightly increasing the Cy coefficient
            assert all_equal(xls_df_svv['Code'])
            xls_df_svv['Code'] += '-TS'  # adding -TS to the code because the output will have TS effects

            xls_df_svv = reformat_xls(xls_df_svv, df_all)

            # Write new sheet.
            with pd.ExcelWriter(xls_data_path, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
                xls_df_svv.to_excel(writer, sheet_name=out_sheet, index=False)
            print(f'A new sheet {out_sheet}, with the SVV-adapted coefficients, as been created in {xls_data_path}')


if __name__ == '__main__':  # If this is the file being run (instead of imported) then run the following functions
    # Getting all the raw data into one dictionary:
    raw_data_dict = get_raw_data_dict(raw_data_path)
    # Organizing the raw data into dataframes with similar data formats:
    dict_of_dfs = get_dfs_from_raw_data(raw_data_dict)
    # Compiling the data into one single dataframe:
    df_all = get_df_all(dict_of_dfs, save_csv=True)
    # Running a few checks:
    run_further_checks(df_all)
    # Creating a new sheet with the SVV-adapted aerodynamic coefficients:
    add_sheet_with_svv_adapted_aero_coefs(xls_data_path, df_all)
