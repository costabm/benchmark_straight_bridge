import numpy as np
import sympy as sp
from sympy import integrate, exp, pi, simplify, cos

s1 = sp.Symbol('s1', real=True, positive=True)
s2 = sp.Symbol('s2', real=True, positive=True)
w = sp.Symbol('w', real=True, positive=True)
C = sp.Symbol('C', real=True, positive=True)
L = sp.Symbol('L', real=True, positive=True)
U = sp.Symbol('U', real=True, positive=True)

# Functions from AMC Appendix E Aerodynamics - Enclosure 3 - eq. (8) and (6)
R_fun = exp(-C*w*(s2-s1)/(2*pi*U))
phi1_fun = cos(s1)
phi2_fun = cos(s2)

fun = integrate(R_fun*phi1_fun*phi2_fun, (s2, 0, L), (s2, 0, L))
simplify(fun)




