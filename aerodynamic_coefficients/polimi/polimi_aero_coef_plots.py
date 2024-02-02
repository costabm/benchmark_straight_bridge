import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from my_utils import (root_dir, get_list_of_colors_matching_list_of_objects,
                      from_df_all_get_unique_value_given_key_and_id)
import os
matplotlib.use('Qt5Agg')  # to prevent bug in PyCharm


folder_path = os.path.join(root_dir, r'aerodynamic_coefficients\polimi')
file_name = r'ResultsCoefficients-Rev3.xlsx'


def plot_yaw_dependency_of_U_by_U_ceil():
    df_name = r"df_of_all_polimi_tests.csv"
    # Plotting U/U_ceil for beta, organized by the different tests (pontoon, column, coherence...)
    df_all = pd.read_csv(os.path.join(folder_path, df_name))
    df = df_all.copy()
    df = df[(df['U'] > 0) & (df['z'] == 0.46)].dropna(axis=1, how='all')
    df['U_by_U_ceil'] = df['U'] / df['U_ceil']
    x_axis = 'beta_rx0'
    y_axis = 'U_by_U_ceil'
    label_axis = 'code'
    color_list = get_list_of_colors_matching_list_of_objects(df['code'])
    plt.figure(dpi=400)
    plt.title('U dependency on the yaw angle (z = 0.46 m)')
    for label, color in dict(zip(df[label_axis], color_list)).items():  # unique values
        sub_df = df[df[label_axis] == label]  # subset dataframe
        plt.scatter(sub_df[x_axis], sub_df[y_axis], color=color, label=label, alpha=0.8)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="lower left")
    plt.xlabel(r'$\beta$ [deg]')
    plt.ylabel(r'$U_{centre}\//\/U_{ceiling}$')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, 'aerodynamic_coefficients',
                             'polimi', 'plots', 'U_dependency_on_the_yaw_angle.jpg'))
    plt.close()


def plot_phase_7_vs_polimi_coefs(sheet_name):
    polimi_file_name = file_name
    phase7_file_name = '../tables/aero_coefs_in_Ls_from_SOH_CFD_scaled_to_Julsund.xlsx'  # Beginning of Phase 7, prelim. coefficients

    # Load results
    path_polimi_data = os.path.join(folder_path, polimi_file_name)
    df_polimi = pd.read_excel(io=path_polimi_data, sheet_name=sheet_name).dropna().sort_values(['Yaw', 'Theta'])

    # Renaming coefficients
    df_polimi.rename(columns={'Code': 'code', 'qRef': 'q_ref', 'Yaw': 'yaw', 'Theta': 'theta',
                              'CMxL': 'CrxL', 'CMxi': 'Crxi',
                              'CMyL': 'CryL', 'CMyi': 'Cryi',
                              'CMzL': 'CrzL', 'CMzi': 'Crzi',
                              'CxTot': 'Cx', 'CyTot': 'Cy', 'CzTot': 'Cz',
                              'CMxTot': 'Crx', 'CMyTot': 'Cry', 'CMzTot': 'Crz'}, inplace=True)

    # Collecting the coefficients used in the start of Phase 7
    phase7_file_path = os.path.join(folder_path, phase7_file_name)
    C_phase7_table = {dof: pd.read_excel(phase7_file_path, header=None, sheet_name=dof).to_numpy()
                      for dof in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']}

    lst_betas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # list of betas to plot
    color_list = plt.cm.turbo(np.linspace(0, 0.95, len(lst_betas))).tolist()

    for str_coef in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']:
        plt.figure(figsize=(5, 4), dpi=400)
        for i, beta in enumerate(lst_betas):
            # Polimi
            thetas_polimi = df_polimi[df_polimi['yaw'] == beta]['theta']
            C_polimi = df_polimi[df_polimi['yaw'] == beta][str_coef]
            C_polimi_L = df_polimi[df_polimi["yaw"] == beta][str_coef + 'L']  # left sensor "L"
            C_polimi_i = df_polimi[df_polimi["yaw"] == beta][str_coef + 'i']  # right sensor "i"
            # Phase 7A coefficients
            C_table_betas = np.linspace(-180, 180, 361).tolist()
            C_table_thetas = np.linspace(12, -12, 25).tolist()
            thetas_phase7 = np.linspace(12, -12, 25).tolist()
            beta_idx = C_table_betas.index(beta)
            # theta_idxs = [C_table_thetas.index(t) for t in list(thetas_polimi)]  # Print only same number of thetas
            theta_idxs = [C_table_thetas.index(t) for t in list(thetas_phase7)]  # Print all available thetas
            C_phase7 = C_phase7_table[str_coef][theta_idxs, beta_idx]
            # New Polimi coefficients
            plt.plot(thetas_polimi, C_polimi, color=color_list[i], lw=2.0, alpha=0.8, ls='-')
            plt.scatter(thetas_polimi, C_polimi_L, color=color_list[i], alpha=0.8, marker='x', s=15)
            plt.scatter(thetas_polimi, C_polimi_i, color=color_list[i], alpha=0.8, marker='+', s=15)
            # Previous Phase 7A coefficients (SOH+CFD & Julsund adjusted):
            plt.plot(thetas_phase7, C_phase7, color=color_list[i], lw=1.0, alpha=0.8, ls='--')
        plt.ylabel(str_coef)
        plt.xlabel(r'$\theta$ [deg]')
        plt.xlim([-10.2, 10.2])
        plt.xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        plt.grid()
        plt.tight_layout()
        plt.savefig(folder_path + r"\plots\polimi_" + sheet_name + '_' + str_coef + ".jpg")
        plt.close()
    # Plot legend
    plt.figure(dpi=400)
    h1 = Line2D([0], [0], color='black', lw=1.0, alpha=0.8, ls='--', label='Phase 7')
    h2 = Line2D([0], [0], color='black', alpha=0.8, marker='x', markersize=5, ls='', label='Polimi (L-cell)')
    h3 = Line2D([0], [0], color='black', alpha=0.8, marker='+', markersize=5, ls='', label='Polimi (R-cell)')
    h4 = Line2D([0], [0], color='black', lw=2.0, alpha=0.8, ls='-', label='Polimi (Total)')
    h5 = Line2D([0], [0], color='white', lw=2.0, alpha=0., ls='-', label=r"$\bf{Yaw\/\/angles:}$")
    h_yaws = [Line2D([0], [0], color=c, lw=2.0, alpha=0.8, ls='-',
                     label=r'$\beta=$' + str(int(round(b, 0))) + r'$\degree$') for b, c in zip(lst_betas, color_list)]
    plt.legend(handles=[h1, h2, h3, h4, h5, *h_yaws], ncols=5)
    plt.axis('off')
    plt.savefig(folder_path + r"\plots\polimi_" + sheet_name + "_legend.jpg", bbox_inches='tight')
    plt.close()


def plot_2_variants_of_polimi_coefs(sheet_1, sheet_2):
    # Load results
    data_path = os.path.join(folder_path, file_name)

    if sheet_1 == r'K12-AG-BAR' and sheet_2 == 'K12-AG-BAR':
        df_0 = pd.read_excel(io=data_path, sheet_name=sheet_1).dropna().sort_values(['Yaw', 'Theta'])
        df_1 = df_0[df_0['Code'] == 'K12-AG-BAR Aero']
        df_2 = df_0[df_0['Code'] == 'K12-AG-BAR Geo']
    else:
        df_1 = pd.read_excel(io=data_path, sheet_name=sheet_1).dropna().sort_values(['Yaw', 'Theta'])
        df_2 = pd.read_excel(io=data_path, sheet_name=sheet_2).dropna().sort_values(['Yaw', 'Theta'])

    def rename(df):
        df = df.rename(columns={'Code': 'code', 'qRef': 'q_ref', 'Yaw': 'yaw', 'Theta': 'theta',
                                'CMxL': 'CrxL', 'CMxi': 'Crxi',
                                'CMyL': 'CryL', 'CMyi': 'Cryi',
                                'CMzL': 'CrzL', 'CMzi': 'Crzi',
                                'CxTot': 'Cx', 'CyTot': 'Cy', 'CzTot': 'Cz',
                                'CMxTot': 'Crx', 'CMyTot': 'Cry', 'CMzTot': 'Crz'})
        return df
    df_1 = rename(df_1)
    df_2 = rename(df_2)

    lst_betas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # list of betas to plot
    color_list = plt.cm.turbo(np.linspace(0, 0.95, len(lst_betas))).tolist()

    for str_coef in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']:
        plt.figure(figsize=(5, 4), dpi=400)
        for i, beta in enumerate(lst_betas):
            # Polimi
            thetas_1 = df_1[df_1['yaw'] == beta]['theta']
            thetas_2 = df_2[df_2['yaw'] == beta]['theta']
            C_1 = df_1[df_1['yaw'] == beta][str_coef]
            C_2 = df_2[df_2['yaw'] == beta][str_coef]
            if len(C_1) > 1:
                plt.plot(thetas_1, C_1, color=color_list[i], lw=2.0, alpha=0.8, ls='-')
            else:  # if only one datapoint is available, e.g. at yaw=90, use marker instead of line
                plt.scatter(thetas_1, C_1, color=color_list[i], alpha=0.8, marker='o', s=20)
            if len(C_2) > 1:
                plt.plot(thetas_2, C_2, color=color_list[i], lw=1.0, alpha=0.8, ls='--')
            else:  # if only one datapoint is available, e.g. at yaw=90, use marker instead of line
                plt.scatter(thetas_2, C_2, color=color_list[i], alpha=0.8, marker='x', s=20)
        plt.ylabel(str_coef)
        plt.xlabel(r'$\theta$ [deg]')
        plt.xlim([-10.2, 10.2])
        plt.xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        plt.grid()
        plt.tight_layout()
        plt.savefig(folder_path + r"\plots\polimi_" + sheet_1 + '_vs_' + sheet_2 + '_' + str_coef + ".jpg")
        plt.close()
    # Plot legend
    plt.figure(dpi=400)
    h1 = Line2D([0], [0], color='black', lw=2.0, alpha=0.8, ls='-', marker='o', markersize=5, label=sheet_1)
    h2 = Line2D([0], [0], color='black', lw=1.0, alpha=0.8, ls='--', marker='x', markersize=5, label=sheet_2)
    h3 = Line2D([0], [0], color='white', lw=0., alpha=0., ls='-', label=r"$\bf{Yaw\/\/angles:}$")
    h_yaws = [Line2D([0], [0], color=c, lw=2.0, alpha=0.8, ls='-',
                     label=r'$\beta=$' + str(int(round(b, 0))) + r'$\degree$') for b, c in zip(lst_betas, color_list)]
    plt.legend(handles=[h1, h2, h3, *h_yaws], ncols=5)
    plt.axis('off')
    plt.savefig(folder_path + r"\plots\polimi_" + sheet_1 + '_vs_' + sheet_2 + "_legend.jpg", bbox_inches='tight')
    plt.close()


def plot_symmetry_check_polimi_coefs(model_name='K12-G-L-T1'):
    data_path_Mil = os.path.join(folder_path, file_name)
    data_path_SVV = os.path.join(root_dir, r'aerodynamic_coefficients\tables',
                                 'aero_coefs_Ls_2D_fit_cons_polimi-' + model_name + '-SVV.xlsx')

    # Load results:
    xls_Mil = pd.read_excel(io=data_path_Mil, sheet_name=model_name).dropna().sort_values(['Yaw', 'Theta'])
    betas_SVV = pd.read_excel(io=data_path_SVV, sheet_name='betas_deg', header=None).to_numpy()
    thetas_SVV = pd.read_excel(io=data_path_SVV, sheet_name='thetas_deg', header=None).to_numpy()[:,0]
    # Unique betas outside first quadrant:
    lst_betas = xls_Mil[(xls_Mil['Yaw'] > 90.5) | (xls_Mil['Yaw'] < -0.5)]['Yaw'].unique()

    # Add beta_SVV and theta_SVV to the xls_Mil
    df_all = pd.read_csv(os.path.join(root_dir, r"aerodynamic_coefficients\polimi\df_of_all_polimi_tests.csv"))
    beta_rx0, rx, beta_svv, theta_svv = [], [], [], []
    for i in xls_Mil['run']:  # 'run' (Polimi notation) and 'id' (my notation) are the same thing
        beta_rx0.append(from_df_all_get_unique_value_given_key_and_id(df_all, key='beta_rx0', run=i))
        rx.append(from_df_all_get_unique_value_given_key_and_id(df_all, key='rx', run=i))
        beta_svv.append(from_df_all_get_unique_value_given_key_and_id(df_all, key='beta_svv', run=i))
        theta_svv.append(from_df_all_get_unique_value_given_key_and_id(df_all, key='theta_svv', run=i))
    xls_Mil['beta_rx0'] = beta_rx0
    xls_Mil['rx'] = rx
    xls_Mil['beta_svv'] = beta_svv
    xls_Mil['theta_svv'] = theta_svv

    lst_betas_for_color = [90, 100, 110, 120, 130, 140, 150, 160, 170, 180][::-1]  # list of betas to plot
    color_list = plt.cm.turbo(np.linspace(0, 0.95, len(lst_betas_for_color))).tolist()
    color_list = [color_list[np.where(lst_betas_for_color==b)[0][0]] for b in lst_betas]

    for str_coef in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']:
        C_SVV = pd.read_excel(io=data_path_SVV, sheet_name=str_coef, header=None).to_numpy()
        dof_Mil = str_coef + 'Tot'
        if 'Cr' in str_coef:
            dof_Mil = dof_Mil.replace("Cr", "CM")
        plt.figure(figsize=(5, 4), dpi=400)

        for i, beta in enumerate(lst_betas):
            beta_idx = np.where(betas_SVV == beta)[1][0]
            C_SVV_at_beta = C_SVV[:,beta_idx]

            sub_xls_Mil = xls_Mil[xls_Mil['Yaw'] == beta]
            C_Mil = sub_xls_Mil[dof_Mil]
            thetas_Mil = sub_xls_Mil['theta_svv']

            plt.plot(thetas_SVV, C_SVV_at_beta, color=color_list[i], lw=2.0, alpha=0.8, ls='-')

            if len(C_Mil) > 1:
                plt.plot(thetas_Mil, C_Mil, color=color_list[i], lw=1.0, alpha=0.8, ls='--')
            else:  # if only one datapoint is available, e.g. at yaw=90, use marker instead of line
                plt.scatter(thetas_Mil, C_Mil, color=color_list[i], alpha=0.8, marker='x', s=20)
        plt.ylabel(str_coef)
        plt.xlabel(r'$\theta$ [deg]')
        plt.xlim([-10.2, 10.2])
        plt.xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
        plt.grid()
        plt.tight_layout()
        plt.savefig(folder_path + r"\plots\polimi_check_symmetry_" + model_name + "_" + str_coef + ".jpg")
        plt.close()
    # Plot legend
    plt.figure(dpi=400)
    h1 = Line2D([0], [0], color='black', lw=2.0, alpha=0.8, ls='-', label=model_name + '-SVV.xlsx')
    h2 = Line2D([0], [0], color='black', lw=1.0, alpha=0.8, ls='--', label=model_name)
    h3 = Line2D([0], [0], color='white', lw=0., alpha=0., ls='-', label=r"$\bf{Yaw\/\/angles:}$")
    h_yaws = [Line2D([0], [0], color=c, lw=2.0, alpha=0.8, ls='-',
                     label=r'$\beta=$' + str(int(round(b, 0))) + r'$\degree$') for b, c in zip(lst_betas, color_list)]
    plt.legend(handles=[h1, h2, h3, *h_yaws], ncols=3)
    plt.axis('off')
    plt.savefig(folder_path + r"\plots\polimi_check_symmetry_" + model_name + "_legend.jpg", bbox_inches='tight')
    plt.close()


# plot_yaw_dependency_of_U_by_U_ceil()
# plot_phase_7_vs_polimi_coefs(sheet_name=r'K12-G-L')
#
# plot_2_variants_of_polimi_coefs(sheet_1=r'K12-G-L-TS', sheet_2=r'K12-G-L')
# plot_2_variants_of_polimi_coefs(sheet_1=r'K12-G-L-T1', sheet_2=r'K12-G-L')
# plot_2_variants_of_polimi_coefs(sheet_1=r'K12-G-L-T3', sheet_2=r'K12-G-L-T1')
# plot_2_variants_of_polimi_coefs(sheet_1=r'K12-AG-BAR', sheet_2=r'K12-AG-BAR')

plot_symmetry_check_polimi_coefs(model_name='K12-G-L')  # there is no available data on 2nd quadrant for K12-G-L-TS
plot_symmetry_check_polimi_coefs(model_name='K12-G-L-T1')
plot_symmetry_check_polimi_coefs(model_name='K12-G-L-T3')
plot_symmetry_check_polimi_coefs(model_name='K12-G-L-CS')
