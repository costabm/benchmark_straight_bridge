import numpy as np
import pandas as pd
from buffeting import C_Ci_func
from my_utils import deg, rad


beta_step = 1  # deg
theta_step = 1  # deg

beta = rad(np.arange(-180, 180+beta_step, beta_step))
theta = rad(np.arange(-12, 12+theta_step, theta_step))

method = '2D_fit_cons_polimi'  # "2D_fit_cons_w_CFD_scale_to_Jul"
coor_system = 'Ls'  # Ls or Gw
n_aero_coef = 4  # Choose 4 to neglect Cry and Crz

if coor_system == 'Ls':
    d_keys = ['betas_deg', 'thetas_deg', 'Cx', 'Cy', 'Cz', 'Crx', 'Cry', 'Crz']
elif coor_system == 'Gw':
    d_keys = ['betas_deg', 'thetas_deg', 'CXu', 'CYv', 'CZw', 'CrXu', 'CrYv', 'CrZw']
else:
    raise NotImplementedError
d = {key:[] for key in d_keys}

for t in list(reversed(theta)):
    arr_equal_thetas = t * np.ones(len(beta))
    Ci_row = C_Ci_func(beta=beta, theta=arr_equal_thetas, aero_coef_method=method, n_aero_coef=n_aero_coef,
                       coor_system=coor_system)

    d['betas_deg'].append(deg(beta))
    d['thetas_deg'].append(deg(arr_equal_thetas))
    for i, label in enumerate(d_keys[2:]):
        assert 'C' in label
        d[label].append(Ci_row[i])

writer = pd.ExcelWriter(r'aerodynamic_coefficients\tables\aero_coefs_'+f'{coor_system}_{method}'+'.xlsx',
                        engine='xlsxwriter')  # You need to: pip install xlsxwriter
for key in d:
    d[key] = np.array(d[key])
    df = pd.DataFrame(d[key])
    df.to_excel(writer, sheet_name=key, index=False, header=False)
writer.close()

