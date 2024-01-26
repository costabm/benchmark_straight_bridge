import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from my_utils import root_dir, get_list_of_colors_matching_list_of_objects
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

# plot_yaw_dependency_of_U_by_U_ceil()


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
    path_data = os.path.join(folder_path, file_name)

    if sheet_1 == r'K12-AG-BAR' and sheet_2 == 'K12-AG-BAR':
        df_0 = pd.read_excel(io=path_data, sheet_name=sheet_1).dropna().sort_values(['Yaw', 'Theta'])
        df_1 = df_0[df_0['Code'] == 'K12-AG-BAR Aero']
        df_2 = df_0[df_0['Code'] == 'K12-AG-BAR Geo']
    else:
        df_1 = pd.read_excel(io=path_data, sheet_name=sheet_1).dropna().sort_values(['Yaw', 'Theta'])
        df_2 = pd.read_excel(io=path_data, sheet_name=sheet_2).dropna().sort_values(['Yaw', 'Theta'])

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


plot_phase_7_vs_polimi_coefs(sheet_name=r'K12-G-L')
plot_2_variants_of_polimi_coefs(sheet_1=r'K12-G-L-TS', sheet_2=r'K12-G-L')
plot_2_variants_of_polimi_coefs(sheet_1=r'K12-G-L-T1', sheet_2=r'K12-G-L')
plot_2_variants_of_polimi_coefs(sheet_1=r'K12-G-L-T3', sheet_2=r'K12-G-L-T1')
plot_2_variants_of_polimi_coefs(sheet_1=r'K12-AG-BAR', sheet_2=r'K12-AG-BAR')


