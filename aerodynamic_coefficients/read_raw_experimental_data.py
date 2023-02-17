"""
Raw data field names: 'Xforce', 'Yforce', 'Moment', 'TiltAngle', 'SkewAngle', 'Velocity', 'Density', 'Drag', 'Lift', 'Zforce', 'Xmoment', 'Ymoment'
Note that 'Moment' is the Z moment

The 'Lnw', 'Ls' and 'Lw' used here are right-hand coordinate systems.
Ls and Lw as in L.D.Zhu
Lnw's x-axis = along-normal-wind (cos-rule-drag). y-axis = along bridge girder, respecting the M rotation in SOH report, picture C.7. z-axis = perpendicular but in the same 2D plane (cos-rule-lift)

author: Jungao and Bernardo
"""

import os
import numpy as np
import scipy.io
import pandas as pd
from transformations import T_LsLw_func, T_LnwLs_func, T_LsSOH_func, theta_yz_bar_func, C_Ci_Ls_to_C_Ci_Lnw
from aero_coefficients import from_SOH_to_Zhu_angles, rad, deg

fix_data = True  # should the raw data be adapted?: C_SOH_z == C_Ls_x < 0; Aproximate beta 0.1 to 0 to avoid numerical errors (we want some coefficients to be zero at beta 0); C_SOH_z, C_SOH_xx and C_SOH_yy at (beta=0) = 0.
flow_used = 'smooth'  #  'smooth', 'turbulent', or the 'average' of both. 'turbulent' was used in my PhD

if flow_used == 'average':
    flow_list = ['smooth', 'turbulent']
elif flow_used == 'smooth':
    flow_list = ['smooth']
elif flow_used == 'turbulent':
    flow_list = ['turbulent']

C_Ci_SOH_ZHUnorm = []
for flow in flow_list:
    # Importing raw data.
    # path_raw_data = os.path.join(os.getcwd(), r'aerodynamic_coefficients', 'wind_tunnel_test_raw_data_treatment', 'raw_data_smooth_flow')  # smooth flow makes more sense
    path_raw_data = os.path.join(os.getcwd(), r'aerodynamic_coefficients', 'wind_tunnel_test_raw_data_treatment', 'raw_data_' +str(flow) +'_flow')  # smooth flow makes less sense but leads to better fits
    all_raw_files_names = [i for i in os.listdir(path_raw_data) if 'traffic' not in i]  # Only saving results without traffic.
    n_raw_files = len(all_raw_files_names)
    raw_all = [scipy.io.loadmat(os.path.join(path_raw_data, all_raw_files_names[i]))['StaticResults'][0][0] for i in range(n_raw_files)]
    # Organizing data. Aerodynamic forces:
    raw_x = np.array([raw_all[i]['Xforce'] for i in range(n_raw_files)])[:,:,0]  # N.
    raw_y = np.array([raw_all[i]['Yforce'] for i in range(n_raw_files)])[:,:,0]  # N.
    raw_z = - np.array([raw_all[i]['Zforce'] for i in range(n_raw_files)])[:,:,0]  # N. Needs to be negative to conform with xyz right-hand-rule-coordinate-system.
    raw_xx = np.array([raw_all[i]['Xmoment'] for i in range(n_raw_files)])[:,:,0]  # Nm.
    raw_yy = np.array([raw_all[i]['Ymoment'] for i in range(n_raw_files)])[:,:,0]  # Nm.
    raw_zz = - np.array([raw_all[i]['Moment'] for i in range(n_raw_files)])[:,:,0]  # Nm. Needs to be negative to conform with xyz right-hand-rule-coordinate-system.
    raw_V = np.array([raw_all[i]['Velocity'] for i in range(n_raw_files)])[0,:,0]  # m/s. U+u. Should have been different for each test, but it was only measured in one test, and copied.
    raw_rho = np.array([raw_all[i]['Density'] for i in range(n_raw_files)])[0,0,0]  # kg/m3. Should have been different for each test, but it was only measured in one test, and copied.
    raw_alpha_angles = - np.array([raw_all[i]['TiltAngle'] for i in range(n_raw_files)])[:, 0, 0]  # deg. Needs to be negative to conform with xyz right-hand-rule-coordinate-system. This alpha represents
    # a rotation around Ls_x, and not a rotation in the opposite direction according figure C.7. of SOH report.
    raw_SOH_skew_angles = np.array([raw_all[i]['SkewAngle'] for i in range(n_raw_files)])[:, 0, 0]  # deg.
    # Fixing data, part 1:
    if fix_data:
        # Converting skew angles that are less then 0.2 deg, to 0 deg.
        raw_SOH_skew_angles = np.array(list(map(lambda x: 0. if x < 0.2 else x, raw_SOH_skew_angles)))
    # Other input parameters (Model name is K71. Model scale is 1/80)
    raw_betas, raw_thetas = [deg(i) for i in from_SOH_to_Zhu_angles(rad(raw_SOH_skew_angles), rad(raw_alpha_angles))]  # deg.
    h_model = 0.043  # m. Model height.
    b_model = 0.386  # m. Model width.
    L_model = 2.4  # m. Model length.
    P_model = 62.4 / 80  # m. Model perimeter (full scale perimeter divided by scale factor 80). Only used in Jungao normalization.
    f = 500  # Hz. Sample rate. Data was collected for 60 sec for each case, with 500hz, giving 30000 datapoints.
    T = 60  # sec. Duration of each test.
    T_array = np.arange(0, T, 1/f)  # sec. Time points.
    U = np.mean(raw_V)  # m/s. Mean wind speed, for all tests.
    # Transformation matrices
    raw_theta_yz = deg(theta_yz_bar_func(rad(raw_betas), rad(raw_thetas)))
    T_LnwLs = T_LnwLs_func(rad(raw_betas), rad(raw_theta_yz), dim='6x6')
    T_LsSOH = T_LsSOH_func(dim='6x6')
    T_LwLs = np.array([T_LsLw_func(rad(raw_betas), rad(raw_thetas), dim='6x6')[i].T for i in range(n_raw_files)])
    # Post processing. Respecting SOH (Svend Ole Hansen's) normalization:
    # Coefficients in local SOH xyz according to Figure C.7 in SOH report (not figure C.8!!).
    C_SOH_x =         np.mean(raw_x, axis=1)  / (0.5 * raw_rho * U**2 * L_model * h_model)  # [-]. Note that the h_model is used here to normalize. Equivalent to "C_SOH_x" in SOH tables.
    C_SOH_x_ZHUnorm = np.mean(raw_x, axis=1)  / (0.5 * raw_rho * U**2 * L_model * b_model)  # [-]. Only normalization difference is here. Zhu normalizes everything by B (or B**2)
    C_SOH_y =         np.mean(raw_y, axis=1)  / (0.5 * raw_rho * U**2 * L_model * b_model)  # [-]
    C_SOH_z =         np.mean(raw_z, axis=1)  / (0.5 * raw_rho * U**2 * L_model * b_model)  # [-].
    C_SOH_xx =        np.mean(raw_xx, axis=1) / (0.5 * raw_rho * U**2 * L_model * b_model ** 2)  # [-]
    C_SOH_yy =        np.mean(raw_yy, axis=1) / (0.5 * raw_rho * U**2 * L_model * b_model ** 2)  # [-]
    C_SOH_zz =        np.mean(raw_zz, axis=1) / (0.5 * raw_rho * U**2 * L_model * b_model ** 2)  # [-]. Equivalent to "CM" in SOH tables.
    # Fixing data, part 2:
    if fix_data:
        # Converting C_SOH_z, C_SOH_xx and C_SOH_yy to 0, when beta = 0.
        for n in range(n_raw_files):
            if raw_SOH_skew_angles[n] < 0.2:
                C_SOH_z[n] = 0.
                C_SOH_xx[n] = 0.
                C_SOH_yy[n] = 0.
            # Forcing C_SOH_z <= 0 for all cases (SOH_z and Ls_x are equal and point in opposite direction to the wind, for 0<beta<90):
            C_SOH_z = -np.abs(C_SOH_z)
    # Storing values. Coefficients in SOH sensor orientation "X+, Y+, Z+", according to Figure C.8 in SOH report. ZHUnorm = Le-Dong Zhu's normalization (dividing C_SOH_x by B).
    C_Ci_SOH_ZHUnorm.append(np.array([C_SOH_x_ZHUnorm, C_SOH_y, C_SOH_z, C_SOH_xx, C_SOH_yy, C_SOH_zz]))

# Average.
C_Ci_SOH_ZHUnorm =  sum(C_Ci_SOH_ZHUnorm) / len(C_Ci_SOH_ZHUnorm)  # averaging if 2 arrays in list. No changes otherwise

# ZHUnorm C_Ci_SOH cannot be directly transformed, because C_SOH_x and C_SOH_y have different normalizations.
C_Ci_Ls_ZHUnorm = T_LsSOH @ C_Ci_SOH_ZHUnorm

# Coefficients in "Local-normal-wind" right-hand system "Drag=x, -Axial=y, Lift=z, _=xx, Moment=yy, _=zz", using "Drag, Lift, M" from Figure C.7 in SOH report.
C_Ci_Lnw_U_HBB = C_Ci_Ls_to_C_Ci_Lnw(rad(raw_betas), rad(raw_thetas), C_Ci_Ls_ZHUnorm, h_model, b_model, C_Ci_Lnw_normalized_by='U', drag_normalized_by='H')  # HBBB^2B^2B^2 as opposed to BBBB^2B^2B^2
C_Ci_Lnw_Uyz_HBB = C_Ci_Ls_to_C_Ci_Lnw(rad(raw_betas), rad(raw_thetas), C_Ci_Ls_ZHUnorm, h_model, b_model, C_Ci_Lnw_normalized_by='Uyz', drag_normalized_by='H')

# Coefficients in L.D.Zhu "Lw" <=> "(q_bar, p_bar, h_bar)" (local wind coordinate system):
C_Ci_Lw = np.einsum('icd,di->ci', T_LwLs, C_Ci_Ls_ZHUnorm)  # c and d for DOF (component). i for each angle combination data.

# Exporting to a dataframe and to Excel
df = pd.DataFrame()
df['test_case_name'] = all_raw_files_names
df['SOH_beta_uncorrected[deg]'] = raw_SOH_skew_angles
df['alpha[deg]'] = raw_alpha_angles
df['beta[deg]'] = raw_betas
df['theta[deg]'] = raw_thetas
df['Cd_U_H_SOHreport'] = C_Ci_Lnw_U_HBB[0]
df['Ca_U_SOHreport'] = C_Ci_Lnw_U_HBB[1]
df['Cl_U_SOHreport'] = C_Ci_Lnw_U_HBB[2]
df['Cra_U_SOHreport'] = C_Ci_Lnw_U_HBB[4]
df['Cd_Uyz_H'] = C_Ci_Lnw_Uyz_HBB[0]
df['Ca_Uyz'] = C_Ci_Lnw_Uyz_HBB[1]
df['Cl_Uyz'] = C_Ci_Lnw_Uyz_HBB[2]
df['Crd_Uyz'] = C_Ci_Lnw_Uyz_HBB[3]
df['Cra_Uyz'] = C_Ci_Lnw_Uyz_HBB[4]
df['Crl_Uyz'] = C_Ci_Lnw_Uyz_HBB[5]
df['Cx_Ls'] = C_Ci_Ls_ZHUnorm[0]
df['Cy_Ls'] = C_Ci_Ls_ZHUnorm[1]
df['Cz_Ls'] = C_Ci_Ls_ZHUnorm[2]
df['Cxx_Ls'] = C_Ci_Ls_ZHUnorm[3]
df['Cyy_Ls'] = C_Ci_Ls_ZHUnorm[4]
df['Czz_Ls'] = C_Ci_Ls_ZHUnorm[5]
df['Cq_bar_Lw'] = C_Ci_Lw[0]
df['Cp_bar_Lw'] = C_Ci_Lw[1]
df['Ch_bar_Lw'] = C_Ci_Lw[2]
df['Cqq_bar_Lw'] = C_Ci_Lw[3]
df['Cpp_bar_Lw'] = C_Ci_Lw[4]
df['Chh_bar_Lw'] = C_Ci_Lw[5]
# df = df.sort_values(['SOH_beta_uncorrected[deg]', 'theta[deg]'])  # todo: This was the previous version! Roll back if unexpected results occur
df = df.sort_values(['SOH_beta_uncorrected[deg]', 'theta[deg]'], ascending=[True,False])
df.to_csv(r"aerodynamic_coefficients\\aero_coef_experimental_data_"+flow_used+".csv", index=False)

