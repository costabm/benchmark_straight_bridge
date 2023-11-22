import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

folder_name = os.path.join(os.getcwd(), r'aerodynamic_coefficients\polimi_raw_data')
polimi_file_name = r'ResultsCoefficients-Rev1.xlsx'
polimi_sheet_name = r'K12-G-L'
phase7_file_name = 'aero_coefs_in_Ls_from_SOH_CFD_scaled_to_Julsund.xlsx'  # Beginning of Phase 7, prelim. coefficients

# Load results
path_polimi_data = os.path.join(folder_name, polimi_file_name)
df_polimi = pd.read_excel(io=path_polimi_data, sheet_name=polimi_sheet_name).dropna().sort_values(['Yaw', 'Theta'])

# Renaming coefficients
df_polimi.rename(columns={'Code': 'code', 'qRef': 'q_ref', 'Yaw': 'yaw', 'Theta': 'theta',
                          'CMxL': 'CrxL', 'CMxi': 'Crxi',
                          'CMyL': 'CryL', 'CMyi': 'Cryi',
                          'CMzL': 'CrzL', 'CMzi': 'Crzi',
                          'CxTot':   'Cx', 'CyTot':   'Cy', 'CzTot':   'Cz',
                          'CMxTot': 'Crx', 'CMyTot': 'Cry', 'CMzTot': 'Crz'}, inplace=True)

# Collecting the coefficients used in the start of Phase 7
phase7_file_path = os.path.join(folder_name, phase7_file_name)
C_phase7_table = {dof: pd.read_excel(phase7_file_path, header=None, sheet_name=dof).to_numpy()
                  for dof in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']}

lst_betas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # list of betas to plot

for str_coef in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']:
    plt.figure(dpi=400)
    for i, beta in enumerate(lst_betas):
        color_list = plt.cm.plasma(np.linspace(0, 0.95, len(lst_betas) - 1)).tolist()
        color_list.append([0.5, 0.5, 0.5, 1])  # add a black color for the markers at beta 90 deg

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
        theta_idxs = [C_table_thetas.index(t) for t in list(thetas_polimi)]  # Print only same number of thetas
        theta_idxs = [C_table_thetas.index(t) for t in list(thetas_phase7)]  # Print all available thetas

        C_phase7 = C_phase7_table[str_coef][theta_idxs, beta_idx]

        # New Polimi coefficients
        plt.plot(thetas_polimi, C_polimi, color=color_list[i], linewidth=2.0, alpha=0.8, linestyle='-')
        plt.scatter(thetas_polimi, C_polimi_L, color=color_list[i], alpha=0.8, marker='x', s=15)
        plt.scatter(thetas_polimi, C_polimi_i, color=color_list[i], alpha=0.8, marker='+', s=15)
        # Previous Phase 7A coefficients (SOH+CFD & Julsund adjusted)
        plt.plot(thetas_phase7, C_phase7, color=color_list[i], linewidth=1.0, alpha=0.8, linestyle='--')

    plt.ylabel(str_coef)
    plt.xlabel(r'Theta [$\theta$]')
    plt.grid()
    plt.savefig(folder_name + r"\plots\polimi_" + polimi_sheet_name + '_' + str_coef + ".jpg")
    plt.close()
