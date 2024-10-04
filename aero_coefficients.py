"""
This script collects experimental data_in on drag, lift, moment and axial coefficients and interpolates and extrapolates it

The available data_in from the SOH tests is limited (30 points, 6 betas(uncorrected) * 5 alphas).
Different methods to interpolate and extrapolate were studied in the "aero_coefficients_study_different_extrapolations".

The main conclusion is:
It is reasonable to use a 2-variable (2D) 2nd order polynomial to fit all the data_in points.
Another source of data_in (either data_in from a similar bridge girder, or artificial data_in from the cosine rule) could be
partially used for the parts of the domain far from the existing data_in points, using a weight function.

Other conclusions are:
The data_in in SOH coordinates (betas_uncorrected, alphas) is in a regular grid.
The data_in in Zhu coordinates (betas, thetas) is in an irregular grid.
It is also reasonable to use a 1-variable (1D) 2nd order polynomial to fit strips of the data_in in one direction (alphas),
and then repeating for the other direction (betas), and finally connecting all the points into one surface. However,
some unexpected fit-curvatures appear, given that this method is more sensitive to bad outliers in the
(even more limited) data_in of each strip. On the other hand, it can be more easily tuned locally as desired.
The 1D fitting needs to be done on a regular grid. This can be overcome by 1) using SOH coordinates for all inter- and
extrapolations and changing to Zhu coordinates in the end. Or 2) interpolating the irregular grid data_in into a
pseudo-regular grid (the convex-hull domain will not be a rectangle) and then doing 1D operations on this interpolated
data_in in a similar way.

created: 12/2019
author: Bernardo Costa
email: bernamdc@gmail.com
"""

import os
import numpy as np
import pandas as pd
from aerodynamic_coefficients.polynomial_fit import cons_poly_fit
from transformations import T_LnwLs_func, theta_yz_bar_func, T_LsGw_func
from my_utils import root_dir, deg, rad
from scipy import interpolate
import copy


lst_methods = ['cos_rule', 'hybrid', 'table', '2D_fit_free', '2D_fit_cons', '2D_fit_cons_scale_to_Jul',
               '2D_fit_cons_w_CFD_scale_to_Jul', '2D_fit_free_polimi',
               '2D_fit_cons_polimi-K12-G-L-TS-SVV', '2D_fit_cons_polimi-K12-G-L-T1-SVV',
               '2D_fit_cons_polimi-K12-G-L-T3-SVV', '2D_fit_cons_polimi-K12-G-L-CS-SVV',
               '2D_fit_cons_polimi-K12-G-L-SVV',
               'aero_coefs_Ls_2D_fit_cons_polimi-K12-G-L-TS-SVV.xlsx',
               'cos_rule_aero_coefs_Ls_2D_fit_cons_polimi-K12-G-L-TS-SVV.xlsx',
               'aero_coefs_in_Ls_from_SOH_CFD_scaled_to_Julsund.xlsx',
               'aero_coefs_Gw_2D_fit_cons_polimi-K12-G-L-TS-SVV.xlsx']

# Factor for when using aero_coef_method == '2D_fit_cons_w_CFD_adjusted':
Cx_factor = 2.0  # To make CFD results conservative, better match SOH and reflect friction and other bridge equipment
Cy_factor = 1.0  # MAKE SURE IF THIS HAS ALREADY BEEN DONE IN THE CSV FILE "aero_coef_experimental_data.csv"   # 4.0 / 3.5  # H has increased from 3.5 to 4.0 in Phase 7 of the BJF project, but since Cy is normalized by B, this is overlooked...

# Converting to L.D. Zhu "beta" and "theta" definition. Points are no longer in a regular grid. Angles are converted to a [-180,180] deg interval
def from_SOH_to_Zhu_angles(betas_uncorrected, alphas):
    # DEPRECATED FUNCTION. SEE INSTEAD: transformations.beta_theta_from_beta_rx0_and_rx
    assert np.all(np.logical_and(rad(-90) <= betas_uncorrected, betas_uncorrected <= rad(90)))
    betas = np.arctan(np.tan(betas_uncorrected) / np.cos(alphas))  # [Rad]. Correcting the beta values (when alpha != 0). See the "Angles of the skewed wind" document in the _basis folder for explanation.
    thetas = -np.arcsin(np.cos(betas_uncorrected) * np.sin(alphas))  # [Rad]. thetas as in L.D.Zhu definition. See the "Angles of the skewed wind" document in the _basis folder for explanation.
    return betas, thetas


def df_aero_coef_measurement_data(method):
    """
    Creates a dataframe compiling the necessary aerodynamic coefficients from wind tunnel tests (SOH or e.g. Julsund) and CFD results.
    These can be later used as input for polynomial fits of this data
    """

    df_SOH = pd.read_csv(os.path.join(root_dir, 'aerodynamic_coefficients', 'aero_coef_experimental_data.csv'))  # raw original values
    df = df_SOH[['test_case_name', 'beta[deg]', 'theta[deg]', 'Cx_Ls', 'Cy_Ls', 'Cz_Ls', 'Cxx_Ls', 'Cyy_Ls', 'Czz_Ls']]  # keeping only relevant columns

    include_CFD = False
    increase_CFD_Cx = False
    multiply_2018_Cy_by_4_by_3p5 = False
    upscale_Cy_Cz_Crx_to_Julsund = False
    include_Julsund = False
    polimi_only = False
    discard_data_outside_1st_quadrant = False

    if method == '2D_fit_cons':
        pass

    if method == '2D_fit_cons_w_CFD':
        include_CFD = True

    if method == '2D_fit_cons_w_CFD_adjusted':
        include_CFD = True
        increase_CFD_Cx = True
        multiply_2018_Cy_by_4_by_3p5 = True

    if method == '2D_fit_cons_scale_to_Jul':
        upscale_Cy_Cz_Crx_to_Julsund = True

    if method == '2D_fit_cons_w_CFD_scale_to_Jul':
        include_CFD = True
        increase_CFD_Cx = True
        upscale_Cy_Cz_Crx_to_Julsund = True

    if '_polimi' in method:
        polimi_only = True
        discard_data_outside_1st_quadrant = True

    # if '_polimi' in method and '_w_cos_rule_data' in method:
    #     add_cos_rule_data = True

    if method == 'cos_rule':
        include_CFD = True
        increase_CFD_Cx = True
        upscale_Cy_Cz_Crx_to_Julsund = True

    if include_CFD:
        # Import the results from NablaFlow 3D CFD
        df_CFD = pd.read_csv(os.path.join(root_dir, 'aerodynamic_coefficients', 'aero_coef_CFD_data.csv'))  # raw original values
        if increase_CFD_Cx:
            df_CFD['Cx_Ls'] = 2 * df_CFD['Cx_Ls']
        df = pd.concat([df, df_CFD])

    if multiply_2018_Cy_by_4_by_3p5:  # to be applied on SOH (and eventually on CFD) results, relative to the 2018 cross-section. This is to account that Phase 7 CS is 4 m high, and that Cy is B-normalized instead of H-normalized.
        df['Cy_Ls'] = 4.0 / 3.5 * df['Cy_Ls']

    if upscale_Cy_Cz_Crx_to_Julsund:
        # Import the latest wind tunnel tests (Milano) from the Julsundbru project
        df_Jul = pd.read_csv(os.path.join(root_dir, 'aerodynamic_coefficients', 'aero_coef_Julsundet_data.csv'))  # raw original values

        Cy_beta_0_func  = interpolate.interp1d(df[df['beta[deg]']==0]['theta[deg]'], df[df['beta[deg]']==0][ 'Cy_Ls'], fill_value='extrapolate')
        Cz_beta_0_func  = interpolate.interp1d(df[df['beta[deg]']==0]['theta[deg]'], df[df['beta[deg]']==0][ 'Cz_Ls'], fill_value='extrapolate')
        Cxx_beta_0_func = interpolate.interp1d(df[df['beta[deg]']==0]['theta[deg]'], df[df['beta[deg]']==0]['Cxx_Ls'], fill_value='extrapolate')
        Cy_Jul_func = interpolate.interp1d(df_Jul['theta[deg]'], df_Jul['Cy_Ls'], fill_value='extrapolate')
        Cz_Jul_func = interpolate.interp1d(df_Jul['theta[deg]'], df_Jul['Cz_Ls'], fill_value='extrapolate')
        Cxx_Jul_func = interpolate.interp1d(df_Jul['theta[deg]'], df_Jul['Cxx_Ls'], fill_value='extrapolate')
        df = df.assign(Cy_Ls =   df['Cy_Ls'] + ( Cy_Jul_func(df['theta[deg]']) -  Cy_beta_0_func(df['theta[deg]'])) * np.cos(np.deg2rad(df['beta[deg]'])),
                       Cz_Ls =   df['Cz_Ls'] + ( Cz_Jul_func(df['theta[deg]']) -  Cz_beta_0_func(df['theta[deg]'])) * np.cos(np.deg2rad(df['beta[deg]'])),
                       Cxx_Ls = df['Cxx_Ls'] + (Cxx_Jul_func(df['theta[deg]']) - Cxx_beta_0_func(df['theta[deg]'])) * np.cos(np.deg2rad(df['beta[deg]'])))

        # df['Cy_Ls']  =  df['Cy_Ls'].copy() * ( Cy_Jul_func(df['theta[deg]']) /  Cy_beta_0_func(df['theta[deg]']))  # Jungao's suggestion

    if include_Julsund:
        df_Jul = pd.read_csv(os.path.join(root_dir, 'aerodynamic_coefficients', 'aero_coef_Julsundet_data.csv'))  # raw original values
        df = pd.concat([df, df_Jul])

    if polimi_only:
        assert any([s in method for s in ['K12-G-L-TS-SVV', 'K12-G-L-T1-SVV', 'K12-G-L-T3-SVV', 'K12-G-L-CS-SVV',
                                          'K12-G-L-SVV']]), \
            '"method" not covered'
        sheet_name = method.split('_polimi-')[1]  # use the last part of the method-string as the sheet name
        # Load results
        path_polimi_data = os.path.join(root_dir, r'aerodynamic_coefficients\polimi\ResultsCoefficients-Rev3.xlsx')
        df = pd.read_excel(io=path_polimi_data, sheet_name=sheet_name).dropna()
        try:
            df['Cx_Ls'] =  df.pop('CxTot')  # change name to respect the original aero_coefficients.py notations
            df['Cy_Ls'] =  df.pop('CyTot')  # change name to respect the original aero_coefficients.py notations
            df['Cz_Ls'] =  df.pop('CzTot')  # change name to respect the original aero_coefficients.py notations
            df['Cxx_Ls'] = df.pop('CMxTot')  # change name to respect the original aero_coefficients.py notations
            df['Cyy_Ls'] = df.pop('CMyTot')  # change name to respect the original aero_coefficients.py notations
            df['Czz_Ls'] = df.pop('CMzTot')  # change name to respect the original aero_coefficients.py notations
        except:  # The names are already the appropriate ones
            assert all([key in df.columns for key in ['Cx_Ls', 'Cy_Ls', 'Cz_Ls', 'Cxx_Ls', 'Cyy_Ls', 'Czz_Ls']])
        try:
            df['beta[deg]'] = df.pop('beta_svv')  # respect the original aero_coefficients.py notations
            df['theta[deg]'] = df.pop('theta_svv')  # respect the original aero_coefficients.py notations
        except:  # The names are already the appropriate ones
            assert all([key in df.columns for key in ['beta[deg]', 'theta[deg]']])
        df = df.sort_values(['beta[deg]', 'theta[deg]'])

    # if add_cos_rule_data:
    #     df_at_beta_theta_0 = df[(df['beta[deg]']==0) & (np.isclose(df['theta[deg]'], 0, atol=0.3))]
    #     Cy_b_0_t_0 = df_at_beta_theta_0['Cy_Ls']
    #     Cz_b_0_t_0 = df_at_beta_theta_0['Cz_Ls']
    #     NotImplementedError


    if discard_data_outside_1st_quadrant:  # just to be sure I'm not changing the previous PhD methods
        df = df[(df['beta[deg]']>=0) & (df['beta[deg]']<=90)]  # removing the new results from Polimi outside the 0-90 quadrant

    return df


def get_C_signs_and_change_betas_extrap(betas_extrap):
    # Converting all [-180,180] angles into equivalent [0,90] angles. The sign information outside [0,90] is lost and stored manually for each coefficient. Assumes symmetric cross-section.
    size = len(betas_extrap)
    Cx_sign, Cy_sign, Cz_sign, Cxx_sign, Cyy_sign, Czz_sign = np.zeros((6, size))
    # Signs for axes in Ls.
    for b in range(size):
        if rad(0) <= betas_extrap[b] <= rad(90):  # all other intervals will be transformations to this one
            Cx_sign[b] = 1
            Cy_sign[b] = 1
            Cz_sign[b] = 1
            Cxx_sign[b] = 1
            Cyy_sign[b] = 1
            Czz_sign[b] = 1
        elif rad(90) < betas_extrap[b] <= rad(180):
            betas_extrap[b] = rad(180) - betas_extrap[b]  # if beta = 110, then becomes 180-110=70
            # the following signs will conserve the fact that beta was in another quadrant.
            Cx_sign[b] = 1
            Cy_sign[b] = -1
            Cz_sign[b] = 1
            Cxx_sign[b] = -1
            Cyy_sign[b] = 1
            Czz_sign[b] = -1
        elif -rad(90) <= betas_extrap[b] < 0:
            betas_extrap[b] = -betas_extrap[b]  # if beta = -60, then becomes 60
            # the following signs will conserve the fact that beta was in another quadrant.
            Cx_sign[b] = -1
            Cy_sign[b] = 1
            Cz_sign[b] = 1
            Cxx_sign[b] = 1
            Cyy_sign[b] = -1
            Czz_sign[b] = -1
        elif -rad(180) <= betas_extrap[b] < -rad(90):
            betas_extrap[b] = rad(180) + betas_extrap[b]  # if beta = -160, then becomes 180+(-160)=20
            # the following signs will conserve the fact that beta was in another quadrant.
            Cx_sign[b] = -1
            Cy_sign[b] = -1
            Cz_sign[b] = 1
            Cxx_sign[b] = -1
            Cyy_sign[b] = -1
            Czz_sign[b] = 1
    return betas_extrap, Cx_sign, Cy_sign, Cz_sign, Cxx_sign, Cyy_sign, Czz_sign


def get_C_signs_and_change_betas_NEW(betas):
    # todo: Bernardo check if we can use this function instead of the one above. It's vectorized, so should be faster
    # Ensure betas is a NumPy array
    betas = np.asarray(betas)

    # Initialize sign arrays
    signs = np.ones((6, betas.size))  # Shape (6, size) for Cx, Cy, Cz, Crx, Cry, Crz

    # Define masks for different beta quadrants
    mask_90_180 = (betas > rad(90)) & (betas <= rad(180))
    mask_neg90_0 = (betas >= -rad(90)) & (betas < 0)
    mask_neg180_neg90 = (betas >= -rad(180)) & (betas < -rad(90))

    # Transform betas and update signs
    betas[mask_90_180] = rad(180) - betas[mask_90_180]
    signs[:, mask_90_180] = [[1], [-1], [1], [-1], [1], [-1]]

    betas[mask_neg90_0] = -betas[mask_neg90_0]
    signs[:, mask_neg90_0] = [[-1], [1], [1], [1], [-1], [-1]]

    betas[mask_neg180_neg90] = rad(180) + betas[mask_neg180_neg90]
    signs[:, mask_neg180_neg90] = [[-1], [-1], [1], [-1], [-1], [1]]

    return (betas,) + tuple(signs)



def aero_coef_table_method(betas_extrap, thetas_extrap, method, coor_system):
    assert coor_system in ['Ls', 'Gw']
    table_path = os.path.join(root_dir, 'aerodynamic_coefficients', 'tables', method)
    betas_table = np.deg2rad(pd.read_excel(table_path, header=None, sheet_name='betas_deg').to_numpy())
    thetas_table = np.deg2rad(pd.read_excel(table_path, header=None, sheet_name='thetas_deg').to_numpy())
    if coor_system == 'Ls':
        sheet_names = ['Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']
    elif coor_system == 'Gw':
        sheet_names = ['CXu', 'CYv', 'CZw', 'CrXu', 'CrYv', 'CrZw']
    C_Ci_Ls_table = np.array([
        pd.read_excel(table_path, header=None, sheet_name=sheet_names[0]).to_numpy(),
        pd.read_excel(table_path, header=None, sheet_name=sheet_names[1]).to_numpy(),
        pd.read_excel(table_path, header=None, sheet_name=sheet_names[2]).to_numpy(),
        pd.read_excel(table_path, header=None, sheet_name=sheet_names[3]).to_numpy(),
        pd.read_excel(table_path, header=None, sheet_name=sheet_names[4]).to_numpy(),
        pd.read_excel(table_path, header=None, sheet_name=sheet_names[5]).to_numpy()])
    # NOTE THAT THE TABLE SHOULD BE C_Ci_Ls_table[:,::-1,:] SINCE THE THETAS WERE, IN THE ORIGINAL TABLE, IN DESCENDING ORDER, BUT ARE FORCED BY RectBivariateSpline to be ascending
    C_C0_func = interpolate.RectBivariateSpline(np.unique(betas_table), np.unique(thetas_table),
                                                np.moveaxis(C_Ci_Ls_table[:, ::-1, :], 1, 2)[0], kx=1, ky=1)
    C_C1_func = interpolate.RectBivariateSpline(np.unique(betas_table), np.unique(thetas_table),
                                                np.moveaxis(C_Ci_Ls_table[:, ::-1, :], 1, 2)[1], kx=1, ky=1)
    C_C2_func = interpolate.RectBivariateSpline(np.unique(betas_table), np.unique(thetas_table),
                                                np.moveaxis(C_Ci_Ls_table[:, ::-1, :], 1, 2)[2], kx=1, ky=1)
    C_C3_func = interpolate.RectBivariateSpline(np.unique(betas_table), np.unique(thetas_table),
                                                np.moveaxis(C_Ci_Ls_table[:, ::-1, :], 1, 2)[3], kx=1, ky=1)
    C_C4_func = interpolate.RectBivariateSpline(np.unique(betas_table), np.unique(thetas_table),
                                                np.moveaxis(C_Ci_Ls_table[:, ::-1, :], 1, 2)[4], kx=1, ky=1)
    C_C5_func = interpolate.RectBivariateSpline(np.unique(betas_table), np.unique(thetas_table),
                                                np.moveaxis(C_Ci_Ls_table[:, ::-1, :], 1, 2)[5], kx=1, ky=1)
    C_Ci_Ls_table_interp = np.array(
        [C_C0_func.ev(betas_extrap, thetas_extrap),  # .ev means "evaluate" the interpolation, at given points
         C_C1_func.ev(betas_extrap, thetas_extrap),
         C_C2_func.ev(betas_extrap, thetas_extrap),
         C_C3_func.ev(betas_extrap, thetas_extrap),
         C_C4_func.ev(betas_extrap, thetas_extrap),
         C_C5_func.ev(betas_extrap, thetas_extrap)])

    return C_Ci_Ls_table_interp


def aero_coef(betas_extrap, thetas_extrap, method, coor_system,
              degree_list={'2D_fit_free':[2,2,1,1,3,4], '2D_fit_cons':[3,4,4,4,4,4],
                           '2D_fit_cons_scale_to_Jul':[3,4,4,4,4,4], '2D_fit_cons_w_CFD_scale_to_Jul':[3,4,4,5,4,4],
                           '2D_fit_cons_polimi-K12-G-L-TS-SVV':[9,9,9,9,9,9], '2D_fit_free_polimi':[4,4,4,4,4,4],
                           '2D_fit_cons_polimi-K12-G-L-T1-SVV':[9,9,9,9,9,9],
                           '2D_fit_cons_polimi-K12-G-L-T3-SVV':[9,9,9,9,9,9],
                           '2D_fit_cons_polimi-K12-G-L-CS-SVV':[9,9,9,9,9,9],
                           '2D_fit_cons_polimi-K12-G-L-SVV':[9,9,9,9,9,9]}):  # constr_fit_adjusted_degree_list=[3,5,5,5,4,4]  BEST FIT FOR '2D_fit_cons_polimi' is [7,9,7,7,-,-]
    """
    betas: 1D-array
    thetas: 1D-array (same size as betas)

    plot = True or False. Plots and saves pictures of the generated surface for Cd

    method = '2D_fit_free', '2D_fit_cons', 'cos_rule', 'hybrid', 'table'.

    coor_system = 'Lnw', 'Ls', 'Lnw&Ls'

    Returns the interpolated / extrapolated values of the aerodynamic coefficients in Local-normal-wind "Lnw" at each given
    coordinate (betas[i], thetas[i]), using the results from the SOH wind tunnel tests. The results are
    in "Local normal wind coordinates" (the same as the Lwbar system from L.D.Zhu when rotated by beta back to non-skew position) or in Ls.
    Regardless, in the process, Local Structural coordinates are used for correctness in symmetry transformations and in constraints.
    """
    betas_extrap = copy.deepcopy(betas_extrap)
    thetas_extrap = copy.deepcopy(thetas_extrap)
    # Data error checks
    if any(abs(thetas_extrap) > rad(90)):
        raise ValueError('At least one theta is outside [-90,90] deg (in radians)')
    if any(abs(betas_extrap) > rad(180)):
        raise ValueError('At least one beta is outside [-180,180] deg (in radians)')
    if len(betas_extrap.flatten()) != len(thetas_extrap.flatten()):
        raise ValueError('Both arrays need to have same size')
    if len(np.shape(betas_extrap)) != 1 or len(np.shape(thetas_extrap)) != 1:
        raise TypeError('Input should be 1D array')

    size = len(betas_extrap)

    # IF TABLE:
    if method[-5:] == '.xlsx':
        if not method[:8] == 'cos_rule':  # NO cosine rule
            return aero_coef_table_method(betas_extrap, thetas_extrap, method, coor_system)
        else:  # COSINE RULE
            assert method[:9] == 'cos_rule_'
            # Get the table in Ls coordinates (must exist first!). Only then, if necessary, transform to Gw coordinates.
            _, Cx_sign, Cy_sign, Cz_sign, Cxx_sign, Cyy_sign, Czz_sign = get_C_signs_and_change_betas_extrap(betas_extrap)  # Get coefficient signs.
            zeros = np.zeros(size)
            table_name = method[9:]
            table_name_Ls = table_name.replace('_Gw_', '_Ls_')
            C_Ci_Ls_table_interp_beta0 = aero_coef_table_method(zeros, thetas_extrap, method=table_name_Ls, coor_system="Ls")
            C_Ci_Ls_table_interp_beta0 = np.array([zeros,
                                                   C_Ci_Ls_table_interp_beta0[1]*Cy_sign,
                                                   C_Ci_Ls_table_interp_beta0[2]*Cz_sign,
                                                   C_Ci_Ls_table_interp_beta0[3]*Cxx_sign,
                                                   zeros,
                                                   zeros])
            C_Ci_Ls_cos_rule = C_Ci_Ls_table_interp_beta0 * np.cos(betas_extrap) ** 2
            if coor_system == 'Ls':
                return C_Ci_Ls_cos_rule
            elif coor_system == 'Gw':
                T_GwLs = np.transpose(T_LsGw_func(betas_extrap, thetas_extrap, dim='6x6'), axes=(0, 2, 1))
                C_Ci_Gw_cos_rule = np.einsum('nij,jn->in', T_GwLs, C_Ci_Ls_cos_rule, optimize=True)
                return C_Ci_Gw_cos_rule


    # If NOT TABLE
    # Importing input data
    df = df_aero_coef_measurement_data(method)

    betas_SOH = rad(df['beta[deg]'].to_numpy())
    thetas_SOH = rad(df['theta[deg]'].to_numpy())
    Cx_Ls = df['Cx_Ls'].to_numpy()
    Cy_Ls = df['Cy_Ls'].to_numpy()
    Cz_Ls = df['Cz_Ls'].to_numpy()
    Cxx_Ls = df['Cxx_Ls'].to_numpy()
    Cyy_Ls = df['Cyy_Ls'].to_numpy()
    Czz_Ls = df['Czz_Ls'].to_numpy()

    # Get coefficient signs and then change all betas back to the 0-90 quadrant.
    betas_extrap, Cx_sign, Cy_sign, Cz_sign, Cxx_sign, Cyy_sign, Czz_sign = get_C_signs_and_change_betas_extrap(betas_extrap)

    # Input data and desired output coordinates (betas and thetas)
    data_in_Cx_Ls = np.array([betas_SOH, thetas_SOH, Cx_Ls])
    data_in_Cy_Ls = np.array([betas_SOH, thetas_SOH, Cy_Ls])
    data_in_Cz_Ls = np.array([betas_SOH, thetas_SOH, Cz_Ls])
    data_in_Cxx_Ls = np.array([betas_SOH, thetas_SOH, Cxx_Ls])
    data_in_Cyy_Ls = np.array([betas_SOH, thetas_SOH, Cyy_Ls])
    data_in_Czz_Ls = np.array([betas_SOH, thetas_SOH, Czz_Ls])

    data_coor_out = np.array([betas_extrap.flatten(), thetas_extrap.flatten()])
    data_bounds = np.array([[0, np.pi / 2], [-np.pi / 2, np.pi / 2]])  # [[beta bounds], [theta bounds]]
    data_bounds_Cy = np.array([[0, np.pi / 2], [-rad(30), rad(30)]])  # [[beta bounds], [theta bounds]]

    # Transforming the coefficients to Local normal wind "Lnw" coordinates, whose axes are defined as:
    # x-axis <=> along-normal-wind (i.e. a "cos-rule-drag"), aligned with the (U+u)*cos(beta) that lies in a 2D plane normal to the bridge girder.
    # y-axis <=> along bridge girder but respecting a M rotation in SOH report where wind is from left and leading edge goes up.
    # z-axis <=> cross product of x and y (i.e. a "cos-rule-lift"), in the same 2D normal plane as x-axis

    if 'Lnw' in coor_system:
        print('WARNING: Avoid using coor_system=Lnw. This has not been carefully looked at')
        theta_yz = theta_yz_bar_func(betas_extrap, thetas_extrap)
        T_LnwLs = T_LnwLs_func(betas=betas_extrap, theta_yz=theta_yz, dim='6x6')

    # 2D polynomial fitting. Note: wrong signs if outside [0,90]
    if '2D_fit_free' in method or method == 'hybrid':
        Cx_Ls_2D_fit_free = cons_poly_fit(data_in_Cx_Ls, data_coor_out, data_bounds, degree=degree_list[method][0], ineq_constraint=False,
                                          other_constraint=False, degree_type='max')[1] * Cx_sign
        Cy_Ls_2D_fit_free = cons_poly_fit(data_in_Cy_Ls, data_coor_out, data_bounds, degree=degree_list[method][1], ineq_constraint=False,
                                          other_constraint=False, degree_type='max')[1] * Cy_sign
        Cz_Ls_2D_fit_free = cons_poly_fit(data_in_Cz_Ls, data_coor_out, data_bounds, degree=degree_list[method][2], ineq_constraint=False,
                                          other_constraint=False, degree_type='max')[1] * Cz_sign
        Cxx_Ls_2D_fit_free = cons_poly_fit(data_in_Cxx_Ls, data_coor_out, data_bounds, degree=degree_list[method][3], ineq_constraint=False,
                                           other_constraint=False, degree_type='max')[1] * Cxx_sign
        Cyy_Ls_2D_fit_free = cons_poly_fit(data_in_Cyy_Ls, data_coor_out, data_bounds, degree=degree_list[method][4], ineq_constraint=False,
                                           other_constraint=False, degree_type='max')[1] * Cyy_sign
        Czz_Ls_2D_fit_free = cons_poly_fit(data_in_Czz_Ls, data_coor_out, data_bounds, degree=degree_list[method][5], ineq_constraint=False,
                                           other_constraint=False, degree_type='max')[1] * Czz_sign
        C_Ci_Ls_2D_fit_free = np.array(
            [Cx_Ls_2D_fit_free, Cy_Ls_2D_fit_free, Cz_Ls_2D_fit_free, Cxx_Ls_2D_fit_free, Cyy_Ls_2D_fit_free,
             Czz_Ls_2D_fit_free])
        if 'Lnw' in coor_system:
            C_Ci_Lnw_2D_fit_free = np.einsum('icd,di->ci', T_LnwLs, C_Ci_Ls_2D_fit_free, optimize=True)

    # Cosine rule: Coefficients(beta,theta) = Coefficients(0,theta)*cos(beta)**2. See LDZhu PhD Thesis, Chapter 6.7. Note: wrong signs if outside [0,90]
    if method in ['cos_rule', 'hybrid', '2D']:
        if method == '2D':
            theta_yz = theta_yz_bar_func(betas_extrap, thetas_extrap)
            # First step: find the C(0,theta) values for all thetas (even if repeated), using the 2D fit on all SOH data_in.
            data_coor_out_beta_0_theta_all = np.array([np.zeros(size), theta_yz.flatten()])
            Cx_Ls_2D_fit_beta_0_theta_all = np.zeros(size)
            # Cx_Ls_2D_fit_beta_0_theta_all = cons_poly_fit( data_in_Cx_Ls , data_coor_out_beta_0_theta_all, data_bounds, degree=2, ineq_constraint=False, other_constraint=False, degree_type='total')[1] * Cx_sign
            Cy_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cy_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=degree_list['2D_fit_free'][1], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cy_sign
            Cz_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cz_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=degree_list['2D_fit_free'][2], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cz_sign
            Cxx_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cxx_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=degree_list['2D_fit_free'][3], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cxx_sign
            Cyy_Ls_2D_fit_beta_0_theta_all = np.zeros(size)
            Czz_Ls_2D_fit_beta_0_theta_all = np.zeros(size)

            factor = np.sin(thetas_extrap)**2 + np.cos(betas_extrap)**2 * np.cos(thetas_extrap)**2

        elif method == 'cos_rule':
            # First step: find the C(0,theta) values for all thetas (even if repeated), using the 2D fit on all SOH data_in.
            data_coor_out_beta_0_theta_all = np.array([np.zeros(size), thetas_extrap.flatten()])
            Cx_Ls_2D_fit_beta_0_theta_all = np.zeros(size)
            # Cx_Ls_2D_fit_beta_0_theta_all = cons_poly_fit( data_in_Cx_Ls , data_coor_out_beta_0_theta_all, data_bounds, degree=2, ineq_constraint=False, other_constraint=False, degree_type='total')[1] * Cx_sign
            Cy_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cy_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=degree_list['2D_fit_free'][1], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cy_sign
            Cz_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cz_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=degree_list['2D_fit_free'][2], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cz_sign
            Cxx_Ls_2D_fit_beta_0_theta_all = \
            cons_poly_fit(data_in_Cxx_Ls[:,:5], data_coor_out_beta_0_theta_all, data_bounds, degree=degree_list['2D_fit_free'][3], ineq_constraint=False,
                          other_constraint=False, degree_type='total')[1] * Cxx_sign
            Cyy_Ls_2D_fit_beta_0_theta_all = np.zeros(size)
            Czz_Ls_2D_fit_beta_0_theta_all = np.zeros(size)

            factor = np.cos(betas_extrap) ** 2

        Cx_Ls_cos = Cx_Ls_2D_fit_beta_0_theta_all * factor
        Cy_Ls_cos = Cy_Ls_2D_fit_beta_0_theta_all * factor
        Cz_Ls_cos = Cz_Ls_2D_fit_beta_0_theta_all * factor
        Cxx_Ls_cos = Cxx_Ls_2D_fit_beta_0_theta_all * factor
        Cyy_Ls_cos = Cyy_Ls_2D_fit_beta_0_theta_all * factor
        Czz_Ls_cos = Czz_Ls_2D_fit_beta_0_theta_all * factor
        C_Ci_Ls_cos = np.array([Cx_Ls_cos, Cy_Ls_cos, Cz_Ls_cos, Cxx_Ls_cos, Cyy_Ls_cos, Czz_Ls_cos])

        if 'Lnw' in coor_system:
            C_Ci_Lnw_cos = np.einsum('icd,di->ci', T_LnwLs, C_Ci_Ls_cos, optimize=True)
        # Note: The cos^2 rule, from L.D.Zhu eq. (6-10), is to be performed on structural xyz coordinates.

    # Hybrid Model: If inside SOH domain = 2D fit, if outside: Cosine. Smooth function between them. Different for Ca.
    if method == 'hybrid':
        # NEW FORMULATION (with cos**2)
        # Cd_hyb = Cd_2D_fit_free * np.cos(betas_extrap)**2 + Cd_cos * np.sin(betas_extrap)**2
        # Cl_hyb = Cl_2D_fit_free * np.cos(betas_extrap)**2 + Cl_cos * np.sin(betas_extrap)**2
        # Cm_hyb = Cm_2D_fit_free * np.cos(betas_extrap)**2 + Cm_cos * np.sin(betas_extrap)**2
        # Ca_hyb = -0.011  * np.sin(betas_extrap)**2  # Alternative: Ca_hyb = -0.011 * np.sin( (betas_extrap/rad(90))**(1/1.5)*rad(90) )**2
        pass

    if '2D_fit_cons' in method:
        # Ls coordinates.
        # The constraints are reasoned for a 0-90 deg interval, but applicable to a -180 to 180 deg interval when the symmetry signs (defined above) are also used.
        # Cx
        ineq_constraint_Cx = False  # False or 'positivity' or 'negativity'
        other_constraint_Cx = ['F_is_0_at_x0_start', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end']
        # Cy
        ineq_constraint_Cy = False  # False or 'positivity' or 'negativity'
        other_constraint_Cy = ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_start', 'dF/dx0_is_0_at_x0_end_at_x1_middle']  # , 'F_is_0p13_at_x0_start_at_x1_middle']
        # Cz
        ineq_constraint_Cz = False  # False or 'positivity' or 'negativity'. we could have: dF/dx1_is_positive_at_x0_end', but difficult to implement with little gain.
        other_constraint_Cz = ['F_is_0_at_x0_end_at_x1_middle', 'dF/dx0_is_0_at_x0_start', 'dF/dx0_is_0_at_x0_end', 'F_is_CFD_at_x0_end_at_x1_-10', 'F_is_CFD_at_x0_end_at_x1_10']  #, 'F_is_-2_at_x1_start', 'F_is_2_at_x1_end']  # , 'dF/dx1_is_16p4_at_x0_start_at_x1_middle', 'F_is_-0p19_at_x0_start_at_x1_middle'], # can eventually remove derivative constraint
        # Cxx
        ineq_constraint_Cxx = False  # False or 'positivity' or 'negativity'
        other_constraint_Cxx = ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_start', 'dF/dx0_is_0_at_x0_end_at_x1_middle']   # 'dF/dx0_is_0_at_x0_start'
        # Cyy
        ineq_constraint_Cyy = False  # False or 'positivity' or 'negativity'
        other_constraint_Cyy = ['F_is_0_at_x0_start', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end']
        # Czz
        ineq_constraint_Czz = False  # False or 'positivity' or 'negativity'
        other_constraint_Czz = ['F_is_0_at_x0_start', 'F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end']
        Cx_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cx_Ls, data_coor_out, data_bounds, degree_list[method][0], ineq_constraint_Cx, other_constraint_Cx,
                      degree_type='max')[1] * Cx_sign
        Cy_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cy_Ls, data_coor_out, data_bounds, degree_list[method][1], ineq_constraint_Cy, other_constraint_Cy,
                      degree_type='max')[1] * Cy_sign  #,, minimize_method='trust-constr', init_guess=[3.71264795e-22, -8.86505000e+00, 4.57056472e+01, -7.39911989e+01, 3.71506016e+01, -6.12248467e-22, -8.75830974e+00,  5.74817737e+01, -1.10425715e+02, 6.17022514e+01, -1.09522498e-21, -2.46382690e+01, 7.14658962e+01, -4.41460857e+01, -2.68154157e+00,  0.00000000e+00, 4.21168758e+01, -1.42475723e+02,  1.30059436e+02, -2.97005883e+01, 0.00000000e+00,  1.44752923e-01, -3.21775938e+01,  9.85035640e+01, -6.64707231e+01])[1] * Cy_sign  # minimize_method='trust-constr'
        Cz_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cz_Ls, data_coor_out, data_bounds, degree_list[method][2], ineq_constraint_Cz, other_constraint_Cz,
                      degree_type='max')[1] * Cz_sign
        Cxx_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cxx_Ls, data_coor_out, data_bounds, degree_list[method][3], ineq_constraint_Cxx, other_constraint_Cxx,
                      degree_type='max')[1] * Cxx_sign
        Cyy_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Cyy_Ls, data_coor_out, data_bounds, degree_list[method][4], ineq_constraint_Cyy, other_constraint_Cyy,
                      degree_type='max')[1] * Cyy_sign
        Czz_Ls_2D_fit_cons = \
        cons_poly_fit(data_in_Czz_Ls, data_coor_out, data_bounds, degree_list[method][5], ineq_constraint_Czz, other_constraint_Czz,
                      degree_type='max')[1] * Czz_sign
        C_Ci_Ls_2D_fit_cons = np.array([Cx_Ls_2D_fit_cons, Cy_Ls_2D_fit_cons, Cz_Ls_2D_fit_cons, Cxx_Ls_2D_fit_cons, Cyy_Ls_2D_fit_cons,Czz_Ls_2D_fit_cons])

    # if method == '2D_fit_cons_2':
    #     # Ls coordinates.
    #     # The constraints are reasoned for a 0-90 deg interval, but applicable to a -180 to 180 deg interval when the symmetry signs (defined above) are also used.
    #     # Cx
    #     ineq_constraint_Cx = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Cx = ['F_is_0_at_x0_start', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']
    #     # Cy
    #     ineq_constraint_Cy = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Cy = ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']  # ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end','dF/dx0_is_0_at_x0_start'] # ] # 'dF/dx0_is_0_at_x0_start']  #  is a cheap way to replace the malfunctioning "positivity" constraint.
    #     # Cz
    #     ineq_constraint_Cz = False  # False or 'positivity' or 'negativity'. we could have: dF/dx1_is_positive_at_x0_end', but difficult to implement with little gain.
    #     other_constraint_Cz = ['F_is_-2_at_x1_start', 'F_is_2_at_x1_end', 'F_is_0_at_x0_end_at_x1_middle', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']  # 'dF/dx0_is_0_at_x0_start']  # can eventually remove derivative constraint
    #     # Cxx
    #     ineq_constraint_Cxx = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Cxx = ['F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']  # 'dF/dx0_is_0_at_x0_start'
    #     # Cyy
    #     ineq_constraint_Cyy = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Cyy = ['F_is_0_at_x0_start', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']
    #     # Czz
    #     ineq_constraint_Czz = False  # False or 'positivity' or 'negativity'
    #     other_constraint_Czz = ['F_is_0_at_x0_start', 'F_is_0_at_x0_end', 'F_is_0_at_x1_start', 'F_is_0_at_x1_end', 'dF/dx0_is_0_at_x0_end', 'dF/dx0_is_0_at_x0_start']
    #
    #     Cx_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cx_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[0], ineq_constraint_Cx, other_constraint_Cx,
    #                   degree_type='max')[1] * Cx_sign
    #     Cy_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cy_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[1], ineq_constraint_Cy, other_constraint_Cy,
    #                   degree_type='max')[1] * Cy_sign  #,, minimize_method='trust-constr', init_guess=[3.71264795e-22, -8.86505000e+00, 4.57056472e+01, -7.39911989e+01, 3.71506016e+01, -6.12248467e-22, -8.75830974e+00,  5.74817737e+01, -1.10425715e+02, 6.17022514e+01, -1.09522498e-21, -2.46382690e+01, 7.14658962e+01, -4.41460857e+01, -2.68154157e+00,  0.00000000e+00, 4.21168758e+01, -1.42475723e+02,  1.30059436e+02, -2.97005883e+01, 0.00000000e+00,  1.44752923e-01, -3.21775938e+01,  9.85035640e+01, -6.64707231e+01])[1] * Cy_sign  # minimize_method='trust-constr'
    #     Cz_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cz_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[2], ineq_constraint_Cz, other_constraint_Cz,
    #                   degree_type='max', minimize_method='SLSQP')[1] * Cz_sign
    #     Cxx_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cxx_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[3], ineq_constraint_Cxx, other_constraint_Cxx,
    #                   degree_type='max')[1] * Cxx_sign
    #     Cyy_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Cyy_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[4], ineq_constraint_Cyy, other_constraint_Cyy,
    #                   degree_type='max')[1] * Cyy_sign
    #     Czz_Ls_2D_fit_cons = \
    #     cons_poly_fit(data_in_Czz_Ls, data_coor_out, data_bounds, constr_fit_2_degree_list[5], ineq_constraint_Czz, other_constraint_Czz,
    #                   degree_type='max')[1] * Czz_sign
    #     C_Ci_Ls_2D_fit_cons = np.array(
    #         [Cx_Ls_2D_fit_cons, Cy_Ls_2D_fit_cons, Cz_Ls_2D_fit_cons, Cxx_Ls_2D_fit_cons, Cyy_Ls_2D_fit_cons,
    #          Czz_Ls_2D_fit_cons])

    if method == 'benchmark1':
        # C_Ci_Ls_benchmark = np.zeros((6, size))
        # C_Ci_Ls_benchmark[1,:] = 0.07455
        # C_Ci_Ls_benchmark[2, :] = -0.14749
        T_LsGw = T_LsGw_func(betas=betas_extrap, thetas=thetas_extrap, dim='6x6')
        C_Ci_Gw_benchmark = np.zeros((6, size))
        C_Ci_Gw_benchmark[0,:] = 0.0745517584974706
        factor = np.cos(betas_extrap) ** 2
        C_Ci_Ls_benchmark = factor * np.einsum('icd,di->ci', T_LsGw, C_Ci_Gw_benchmark, optimize=True)
        C_Ci_Ls_benchmark = np.array([C_Ci_Ls_benchmark[0] * Cx_sign,
                                      C_Ci_Ls_benchmark[1] * Cy_sign,
                                      C_Ci_Ls_benchmark[2] * Cz_sign,
                                      C_Ci_Ls_benchmark[3] * Cxx_sign,
                                      C_Ci_Ls_benchmark[4] * Cyy_sign,
                                      C_Ci_Ls_benchmark[5] * Czz_sign])
        return C_Ci_Ls_benchmark  # shape is (6, g_node_num)
    
    if method == 'benchmark2':
        T_LsGw = T_LsGw_func(betas=betas_extrap, thetas=thetas_extrap, dim='6x6')
        C_Ci_Gw_benchmark = np.zeros((6, size))
        C_Ci_Gw_benchmark[2,:] = -0.14749
        factor = np.cos(betas_extrap) ** 2
        C_Ci_Ls_benchmark = factor * np.einsum('icd,di->ci', T_LsGw, C_Ci_Gw_benchmark, optimize=True)
        C_Ci_Ls_benchmark = np.array([C_Ci_Ls_benchmark[0] * Cx_sign,
                                      C_Ci_Ls_benchmark[1] * Cy_sign,
                                      C_Ci_Ls_benchmark[2] * Cz_sign,
                                      C_Ci_Ls_benchmark[3] * Cxx_sign,
                                      C_Ci_Ls_benchmark[4] * Cyy_sign,
                                      C_Ci_Ls_benchmark[5] * Czz_sign])
        return C_Ci_Ls_benchmark  # shape is (6, g_node_num)
    
    if method == 'benchmark3':
        C_Ci_Ls_benchmark = np.zeros((6, size))
        C_Ci_Ls_benchmark[1,:] = 0.0745517584974706
        C_Ci_Ls_benchmark = np.array([C_Ci_Ls_benchmark[0] * Cx_sign,
                                      C_Ci_Ls_benchmark[1] * Cy_sign,
                                      C_Ci_Ls_benchmark[2] * Cz_sign,
                                      C_Ci_Ls_benchmark[3] * Cxx_sign,
                                      C_Ci_Ls_benchmark[4] * Cyy_sign,
                                      C_Ci_Ls_benchmark[5] * Czz_sign])
        return C_Ci_Ls_benchmark  # shape is (6, g_node_num)
    

    if 'Lnw' in coor_system:
        C_Ci_Lnw_2D_fit_cons = np.einsum('icd,di->ci', T_LnwLs, C_Ci_Ls_2D_fit_cons, optimize=True)

    if coor_system == 'Ls':
        if '2D_fit_free' in method:
            return C_Ci_Ls_2D_fit_free
        elif method in ['2D_fit_cons', '2D_fit_cons_2', '2D_fit_cons_w_CFD', '2D_fit_cons_w_CFD_adjusted',
                        '2D_fit_cons_scale_to_Jul', '2D_fit_cons_w_CFD_scale_to_Jul',
                        '2D_fit_cons_polimi-K12-G-L-TS-SVV', '2D_fit_cons_polimi-K12-G-L-T1-SVV',
                        '2D_fit_cons_polimi-K12-G-L-T3-SVV', '2D_fit_cons_polimi-K12-G-L-CS-SVV',
                        '2D_fit_cons_polimi-K12-G-L-SVV']:
            return C_Ci_Ls_2D_fit_cons
        elif method in ['cos_rule','2D']:
            return C_Ci_Ls_cos
        elif method == 'hybrid':
            return None
    elif coor_system == 'Lnw':
        if method == '2D_fit_free':
            return C_Ci_Lnw_2D_fit_free
        elif method in ['2D_fit_cons', '2D_fit_cons_2']:
            return C_Ci_Lnw_2D_fit_cons
        elif method in ['cos_rule','2D']:
            return C_Ci_Lnw_cos
        elif method == 'hybrid':
            return None
    elif coor_system == 'Lnw&Ls':
        if method == '2D_fit_free':
            return C_Ci_Lnw_2D_fit_free, C_Ci_Ls_2D_fit_free
        elif method in ['2D_fit_cons', '2D_fit_cons_2']:
            return C_Ci_Lnw_2D_fit_cons, C_Ci_Ls_2D_fit_cons
        elif method in ['cos_rule','2D']:
            return C_Ci_Lnw_cos, C_Ci_Ls_cos
        elif method == 'hybrid':
            return None


def aero_coef_derivatives(betas, thetas, method, coor_system):
    betas = copy.deepcopy(betas)
    thetas = copy.deepcopy(thetas)
    # Attention: The Lnw will produce wrong errors since Lnw adapts to all Ci(theta), Ci(theta_prev) and Ci(theta_next) and then the gradient is wrong, and very different for beta -180 and 0 deg,
    # since Lnw is only physical in the [0,90] beta-interval.
    if coor_system == 'Lnw': print('WARNING: coor_system should be "Ls" otherwise the dtheta derivatives will be WRONG!')

    delta_angle = rad(0.001)  # rad. Small variation in beta and theta to calculate the gradients.

    # Attention: if some beta are super close to the boundaries -180,-90,0,90,180 since aero_coef function mirrors f.ex: Ci_before = -0.1 back to 0.1 and then the derivative is wrong and huge (an example gave 10**5 bigger value)! Solution: decrease delta_angle.
    # Correting the error when a beta is exactly at the boundary, by deliberatelly changing problematic betas to very close values.
    angle_correction = delta_angle * 2.1  # rad.
    for i, beta in enumerate(betas):
        if abs(beta - rad(-180)) < delta_angle:
            betas[i] += angle_correction
        if abs(beta - rad(-90)) < delta_angle:
            betas[i] += angle_correction
        if abs(beta - rad(0)) < delta_angle:
            betas[i] += angle_correction
        if abs(beta - rad(90)) < delta_angle:
            betas[i] -= angle_correction
        if abs(beta - rad(180)) < delta_angle:
            betas[i] -= angle_correction

    # Check if previous correction worked:
    if any(abs(rad(180) - abs(betas)) <= delta_angle) or any(abs(rad(90) - abs(betas)) <= delta_angle) or any(
            abs(rad(0) - abs(betas)) <= delta_angle):
        print("WARNING !!! : at least one aero coef derivative could be wrong.")

    # Values "previous" and "next" meaning negative and positive infinitesimal variation of the respective angles.
    beta_prev = betas - delta_angle
    beta_next = betas + delta_angle
    theta_prev = thetas - delta_angle
    theta_next = thetas + delta_angle

    # The centered value of the coefficients
    Cx, Cy, Cz, Cxx, Cyy, Czz = aero_coef(copy.deepcopy(betas), copy.deepcopy(thetas), method=method, coor_system=coor_system)

    # The immediately before and after values of the coefficients
    Cx_beta_prev, Cy_beta_prev, Cz_beta_prev, Cxx_beta_prev, Cyy_beta_prev, Czz_beta_prev = aero_coef(copy.deepcopy(beta_prev), copy.deepcopy(thetas), method=method, coor_system=coor_system)
    Cx_beta_next, Cy_beta_next, Cz_beta_next, Cxx_beta_next, Cyy_beta_next, Czz_beta_next = aero_coef(copy.deepcopy(beta_next), copy.deepcopy(thetas), method=method, coor_system=coor_system)
    Cx_theta_prev, Cy_theta_prev, Cz_theta_prev, Cxx_theta_prev, Cyy_theta_prev, Czz_theta_prev = aero_coef(copy.deepcopy(betas), copy.deepcopy(theta_prev), method=method, coor_system=coor_system)
    Cx_theta_next, Cy_theta_next, Cz_theta_next, Cxx_theta_next, Cyy_theta_next, Czz_theta_next = aero_coef(copy.deepcopy(betas), copy.deepcopy(theta_next), method=method, coor_system=coor_system)

    # Calculating the derivatives = delta(Coef)/delta(angle)
    Cx_dbeta = np.gradient( np.array([ Cx_beta_prev,  Cx,  Cx_beta_next]), axis=0)[1] / delta_angle  # Confirmed. For cos_rule method, compared with d(cos(x)**2) = -sin(2x)
    Cy_dbeta = np.gradient( np.array([ Cy_beta_prev,  Cy,  Cy_beta_next]), axis=0)[1] / delta_angle
    Cz_dbeta = np.gradient( np.array([ Cz_beta_prev,  Cz,  Cz_beta_next]), axis=0)[1] / delta_angle
    Cxx_dbeta = np.gradient(np.array([Cxx_beta_prev, Cxx, Cxx_beta_next]), axis=0)[1] / delta_angle  # Confirmed. For cos_rule method, compared with d(cos(x)**2) = -sin(2x)
    Cyy_dbeta = np.gradient(np.array([Cyy_beta_prev, Cyy, Cyy_beta_next]), axis=0)[1] / delta_angle
    Czz_dbeta = np.gradient(np.array([Czz_beta_prev, Czz, Czz_beta_next]), axis=0)[1] / delta_angle

    Cx_dtheta = np.gradient( np.array([ Cx_theta_prev,  Cx,  Cx_theta_next]), axis=0)[1] / delta_angle  # Confirmed. For cos_rule method, compared with d(cos(x)**2) = -sin(2x)
    Cy_dtheta = np.gradient( np.array([ Cy_theta_prev,  Cy,  Cy_theta_next]), axis=0)[1] / delta_angle
    Cz_dtheta = np.gradient( np.array([ Cz_theta_prev,  Cz,  Cz_theta_next]), axis=0)[1] / delta_angle
    Cxx_dtheta = np.gradient(np.array([Cxx_theta_prev, Cxx, Cxx_theta_next]), axis=0)[1] / delta_angle  # Confirmed. For cos_rule method, compared with d(cos(x)**2) = -sin(2x)
    Cyy_dtheta = np.gradient(np.array([Cyy_theta_prev, Cyy, Cyy_theta_next]), axis=0)[1] / delta_angle
    Czz_dtheta = np.gradient(np.array([Czz_theta_prev, Czz, Czz_theta_next]), axis=0)[1] / delta_angle

    return np.array([[Cx_dbeta, Cy_dbeta, Cz_dbeta, Cxx_dbeta, Cyy_dbeta, Czz_dbeta],
                     [Cx_dtheta, Cy_dtheta, Cz_dtheta, Cxx_dtheta, Cyy_dtheta, Czz_dtheta]])


def from_SOH_to_Zhu_coef_normalization(Cd, Cl, Cm, Ca):
    """
    All in bridge local reference frame "Ls" (not wind reference frame "Lw")
    Converting the experimental coefficients normalized according to SOH, to Zhu's normalization.
    This function could have only 4 lines, but the following is for understanding (de-normalizing and normalizing again)
    """
    rho = 1.25  # [kg/m3]. air density. Not important since it cancels itself.
    U_model = 10  # [m/s]. model wind speed. Not important since it cancels itself.

    # SOH model in the wind tunnel.
    h_model = 0.043  # [m]. model height
    b_model = 0.386  # [m]. model width
    L_model = 2.4  # [m]. model length
    P_model = 62.4 / 80  # [m]. model cross-section perimeter (real scale perimeter divided by scale factor)

    # From SOH's coefficients, to Aerodynamic forces in the model. See SOH report, eq.(C.1)-(C.3)
    Fd_model = 1 / 2 * rho * U_model ** 2 * L_model * h_model * Cd  # [N]
    Fl_model = 1 / 2 * rho * U_model ** 2 * L_model * b_model * Cl  # [N]
    Fm_model = 1 / 2 * rho * U_model ** 2 * L_model * b_model ** 2 * Cm  # [Nm]
    Fa_model = 1 / 2 * rho * U_model ** 2 * L_model * P_model * Ca  # [N]

    # Normalizing according to L.D.Zhu. Note that these are still in bridge ref. frame, not in the wind ref. frame.
    # See You-Lin Xu book, eq. (10.13).
    Cd_Zhu = Fd_model / L_model / (1 / 2 * rho * U_model ** 2 * b_model)  # [-]
    Cl_Zhu = Fl_model / L_model / (1 / 2 * rho * U_model ** 2 * b_model)  # [-]
    Cm_Zhu = Fm_model / L_model / (1 / 2 * rho * U_model ** 2 * b_model ** 2)  # [-]
    Ca_Zhu = Fa_model / L_model / (1 / 2 * rho * U_model ** 2 * b_model)  # [-]

    return Cd_Zhu, Cl_Zhu, Cm_Zhu, Ca_Zhu
