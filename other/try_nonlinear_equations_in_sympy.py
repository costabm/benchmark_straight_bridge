import numpy as np
from sympy import symbols, solve, evalf, sqrt

a_list = np.array([-2,4,6,-8,12])
b_list = np.array([-12,-4,6,18,14])
c_list = np.array([-5,14,1,0,44])

a = symbols('a', real=True)
b = symbols('b', real=True)
c = symbols('c', real=True)

x = symbols('x', real=True)

eq = a*x**2 + b*x - c
sols_exact = []
sols_approx = []
for i in range(5):
    sols_exact.append(solve(eq.subs(a, a_list[i]).subs(b, b_list[i]).subs(c, c_list[i])))
    sols_approx.append([s.evalf() for s in sols_exact[-1]])


