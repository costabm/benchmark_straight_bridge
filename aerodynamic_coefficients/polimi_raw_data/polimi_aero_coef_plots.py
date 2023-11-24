import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from my_utils import root_dir
import os
matplotlib.use('Qt5Agg')  # to prevent bug in PyCharm

folder_name = os.path.join(root_dir, r'aerodynamic_coefficients\polimi_raw_data')
file_name = r'ResultsCoefficients-Rev1-BC.xlsx'   # Rev1-BC: sheet "...BAR" splits into "...BAR-AERO" & "...BAR-GEO"


def plot_phase_7_vs_polimi_coefs(sheet_name):
    polimi_file_name = file_name
    phase7_file_name = 'aero_coefs_in_Ls_from_SOH_CFD_scaled_to_Julsund.xlsx'  # Beginning of Phase 7, prelim. coefficients

    # Load results
    path_polimi_data = os.path.join(folder_name, polimi_file_name)
    df_polimi = pd.read_excel(io=path_polimi_data, sheet_name=sheet_name).dropna().sort_values(['Yaw', 'Theta'])

    # Renaming coefficients
    df_polimi.rename(columns={'Code': 'code', 'qRef': 'q_ref', 'Yaw': 'yaw', 'Theta': 'theta',
                              'CMxL': 'CrxL', 'CMxi': 'Crxi',
                              'CMyL': 'CryL', 'CMyi': 'Cryi',
                              'CMzL': 'CrzL', 'CMzi': 'Crzi',
                              'CxTot': 'Cx', 'CyTot': 'Cy', 'CzTot': 'Cz',
                              'CMxTot': 'Crx', 'CMyTot': 'Cry', 'CMzTot': 'Crz'}, inplace=True)

    # Collecting the coefficients used in the start of Phase 7
    phase7_file_path = os.path.join(folder_name, phase7_file_name)
    C_phase7_table = {dof: pd.read_excel(phase7_file_path, header=None, sheet_name=dof).to_numpy()
                      for dof in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']}

    lst_betas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # list of betas to plot
    color_list = plt.cm.plasma(np.linspace(0, 0.95, len(lst_betas) - 1)).tolist()  # except at 90 deg (see next line)
    color_list.append([0.5, 0.5, 0.5, 1])  # add a black color for the markers at beta 90 deg

    for str_coef in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']:
        plt.figure(dpi=400)
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
        plt.xlabel(r'Theta [$\theta$]')
        plt.grid()
        plt.tight_layout()
        plt.savefig(folder_name + r"\plots\polimi_" + sheet_name + '_' + str_coef + ".jpg")
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
    plt.savefig(folder_name + r"\plots\polimi_" + sheet_name + "_legend.jpg", bbox_inches='tight')
    plt.close()


def plot_2_variants_of_polimi_coefs(sheet_1, sheet_2):
    # Load results
    path_data = os.path.join(folder_name, file_name)
    df_1 = pd.read_excel(io=path_data, sheet_name=sheet_1).dropna().sort_values(['Yaw', 'Theta'])
    df_2 = pd.read_excel(io=path_data, sheet_name=sheet_2).dropna().sort_values(['Yaw', 'Theta'])

    def rename(df):
        return df.rename(columns={'Code': 'code', 'qRef': 'q_ref', 'Yaw': 'yaw', 'Theta': 'theta',
                                  'CMxL': 'CrxL', 'CMxi': 'Crxi',
                                  'CMyL': 'CryL', 'CMyi': 'Cryi',
                                  'CMzL': 'CrzL', 'CMzi': 'Crzi',
                                  'CxTot': 'Cx', 'CyTot': 'Cy', 'CzTot': 'Cz',
                                  'CMxTot': 'Crx', 'CMyTot': 'Cry', 'CMzTot': 'Crz'}, inplace=True)
    rename(df_1)
    rename(df_2)

    lst_betas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # list of betas to plot
    color_list = plt.cm.plasma(np.linspace(0, 0.95, len(lst_betas) - 1)).tolist()  # except at 90 deg (see next line)
    color_list.append([0.5, 0.5, 0.5, 1])  # add a black color for the markers at beta 90 deg

    for str_coef in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']:
        plt.figure(dpi=400)
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
        plt.xlabel(r'Theta [$\theta$]')
        plt.grid()
        plt.tight_layout()
        plt.savefig(folder_name + r"\plots\polimi_" + sheet_1 + '_vs_' + sheet_2 + '_' + str_coef + ".jpg")
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
    plt.savefig(folder_name + r"\plots\polimi_" + sheet_1 + '_vs_' + sheet_2 + "_legend.jpg", bbox_inches='tight')
    plt.close()


# plot_phase_7_vs_polimi_coefs(sheet_name = r'K12-G-L')
# plot_2_variants_of_polimi_coefs(sheet_1=r'K12-G-L-TS', sheet_2=r'K12-G-L')
# plot_2_variants_of_polimi_coefs(sheet_1=r'K12-G-L-T1', sheet_2=r'K12-G-L')
# plot_2_variants_of_polimi_coefs(sheet_1=r'K12-G-L-T3', sheet_2=r'K12-G-L-T1')
plot_2_variants_of_polimi_coefs(sheet_1=r'K12-AG-BAR-AERO', sheet_2=r'K12-AG-BAR-GEO')

# TODO: BERNARDO, THE K12-AG-BAR PLOTS ARE ONLY AT YAW=0, SO THEY ARE QUITE DIFFERENT. SO THE BEST IS TO MAKE A NEW
# FUNCTION FOR THIS PLOT, AND TO DELETE MY "Rev1-BC" WHICH THEN BECOMES UNECESSARY!! DELETE IT IN TWO PLACES: HERE IN
# THIS PYCHARM PROJECT, AND IN THE SHARED FOLDER 14_FAG_AERODYNAMIKK
