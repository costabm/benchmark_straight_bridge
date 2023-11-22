import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


which_tests = 'K12_G_L_T1'  # K12_G_L_no_traffic_sign ; K12_G_L_with_traffic_sign ; K12_G_L_T1

# Load results
path_polimi_data = os.path.join(os.getcwd(), r'aerodynamic_coefficients\polimi_raw_data/coefficients.xlsx')
df_polimi = pd.read_excel(io=path_polimi_data, sheet_name='coeff ' + which_tests).dropna().sort_values(['yaw', 'theta'])

# Renaming coefficients
df_polimi.rename(columns={'CMxL':'CrxL', 'CMxi':'Crxi',
                          'CMyL':'CryL', 'CMyi':'Cryi',
                          'CMzL':'CrzL', 'CMzi':'Crzi'}, inplace=True)

# Multiplying Cx by (-1)
df_polimi['CxL'] = df_polimi['CxL'] * -1
df_polimi['Cxi'] = df_polimi['Cxi'] * -1


path_phase7_data = os.path.join(os.getcwd(), r'aerodynamic_coefficients\polimi_raw_data\aero_coefs_in_Ls_from_SOH_CFD_scaled_to_Julsund.xlsx')  # First part of the Phase 7, with the preliminary coefficients
C_phase7_table = {
    'Cx' : pd.read_excel(path_phase7_data, header=None, sheet_name='Cx').to_numpy(),
    'Cy' : pd.read_excel(path_phase7_data, header=None, sheet_name='Cy').to_numpy(),
    'Cz' : pd.read_excel(path_phase7_data, header=None, sheet_name='Cz').to_numpy(),
    'Crx': pd.read_excel(path_phase7_data, header=None, sheet_name='Crx').to_numpy(),
    'Cry': pd.read_excel(path_phase7_data, header=None, sheet_name='Cry').to_numpy(),
    'Crz': pd.read_excel(path_phase7_data, header=None, sheet_name='Crz').to_numpy()}

# Create new results from the average of both sensors
df_polimi['Cx'] = (df_polimi['CxL'] + df_polimi['Cxi']) / 2
df_polimi['Cy'] = (df_polimi['CyL'] + df_polimi['Cyi']) / 2
df_polimi['Cz'] = (df_polimi['CzL'] + df_polimi['Czi']) / 2
df_polimi['Crx'] = (df_polimi['CrxL'] + df_polimi['Crxi']) / 2
df_polimi['Cry'] = (df_polimi['CryL'] + df_polimi['Cryi']) / 2
df_polimi['Crz'] = (df_polimi['CrzL'] + df_polimi['Crzi']) / 2

lst_betas = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

for str_coef in ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']:
    plt.figure(dpi=400)
    for i, beta in enumerate(lst_betas):
        color_list = plt.cm.plasma(np.linspace(0, 0.95, len(lst_betas)-1)).tolist()
        color_list.append([0.5, 0.5, 0.5, 1])  # add a black color for the markers at beta 90 deg

        # Polimi
        thetas_polimi = df_polimi[df_polimi['yaw']==beta]['theta']
        C_polimi = df_polimi[df_polimi['yaw']==beta][str_coef]
        C_polimi_L = df_polimi[df_polimi["yaw"] == beta][str_coef+'L']  # left sensor "L"
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
    plt.savefig(r"aerodynamic_coefficients\polimi_raw_data\polimi_" + which_tests + '_' + str_coef + ".jpg")
    plt.close()




