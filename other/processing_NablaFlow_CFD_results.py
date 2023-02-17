import numpy as np
from my_utils import deg, rad
from transformations import T_GsGw_func

T_GsGw = T_GsGw_func(beta_0=rad(-50), theta_0=rad(0), dim='6x6')
L = 12  # meters. length of each FEM slice
rho = 1.225  # kg/m3. Given in AeroCloud report
U = 30  # m/s. Chosen by us
B = 31  # m. Cross-section width


F_Gw = np.array([4022.009961, -1*-2881.000384, -27590.56009,
                 2170427.311, -1*-2803685.952, 611391.1483])  # u=Nabla_D; v=-Nabla_L; w=Nabla_S

F_Gs = T_GsGw @ F_Gw

Cx = F_Gs[0] / L / (1/2 * rho * U**2 * B)
Cy = F_Gs[1] / L / (1/2 * rho * U**2 * B)
Cz = F_Gs[2] / L / (1/2 * rho * U**2 * B)
Crx = F_Gs[3] / L / (1/2 * rho * U**2 * B**2)

C = F_Gs / L / (1/2 * rho * U**2 * np.diag([B,B,B,B**2,B**2,B**2]))

np.linalg.inv(np.diag([B,B,B,B**2,B**2,B**2])) * np.diag([B,B,B,B**2,B**2,B**2])


