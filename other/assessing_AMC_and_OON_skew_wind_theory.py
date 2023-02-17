"""
Using the document "Aerodynamic modelling in Orcaflex external function.pdf" as a basis for comparison
"""

import numpy as np
from sympy import symbols, simplify, atan, atan2, asin, acos, sqrt, sign, pi
from my_utils import deg, rad

U, Ux, Uy, Uz = symbols('U, Ux Uy Uz', real=True)
Uxy = sqrt(Ux**2 + Uy**2)

Ux_PS = Uy
Uy_PS = Uz
Uz_PS = Ux
Uxy_PS = sqrt(Ux_PS**2 + Uy_PS**2)

beta_PS = atan(Uz_PS/Uxy_PS)
alpha_PS = atan(Uy_PS/Ux_PS)  # In the document, atan2(Uy_PS, Ux_PS) is probably wrong.

beta_BC = -acos(Uy/Uxy)*sign(Ux)
theta_BC = asin(Uz/U)

theta_OON = atan2(Uz, Uxy)


for i in range(100):
    Ux_ = np.random.uniform(-10, 10)
    Uy_ = np.random.uniform(-10, 10)
    Uz_ = np.random.uniform(-10, 10)
    U_ = np.sqrt(Ux_**2 + Uy_**2 + Uz_**2)
    beta_PS_ = beta_PS.subs(  [(U, U_), (Ux, Ux_), (Uy, Uy_), (Uz, Uz_)])
    beta_BC_ = beta_BC.subs(  [(U, U_), (Ux, Ux_), (Uy, Uy_), (Uz, Uz_)])
    # alpha_PS_ = alpha_PS.subs([(U, U_), (Ux, Ux_), (Uy, Uy_), (Uz, Uz_)])
    theta_OON_ = theta_OON.subs([(U, U_), (Ux, Ux_), (Uy, Uy_), (Uz, Uz_)])
    theta_BC_ = theta_BC.subs([(U, U_), (Ux, Ux_), (Uy, Uy_), (Uz, Uz_)])

    # print('theta OON:', deg(theta_OON_.evalf()))
    # print('theta BC:', deg(theta_BC_.evalf()))
    print('theta OON - theta BC:', deg(theta_BC_.evalf()) - deg(theta_OON_.evalf()))
    # print('theta BC:', deg(theta_BC_.evalf()))




