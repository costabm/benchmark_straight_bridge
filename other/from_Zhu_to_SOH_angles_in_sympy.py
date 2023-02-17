from sympy import symbols, I, Q, log, exp, refine, Eq, Matrix, simplify, cos, sin, tan, acos, asin, atan, sqrt, eye, diag, expand, solve, linsolve, Eq, O, Poly, pi, Dummy, S, apart, sign, diff, Derivative, Abs, pprint, Q, factorial, posify, atan2, sec, expand_trig, collect, factor, invert, BlockMatrix, limit, nsimplify, csc, cot, sec, atan
import numpy as np

#Symbols
b = symbols(r'b', real=True, positive=True)  # beta
t = symbols(r't', real=True)  # theta
a = symbols(r'a', real=True)  # alpha
g = symbols(r'g', real=True, negative=True)  # gama

cosa = symbols(r'cosa', real=True, positive=True)
cosg = symbols(r'cosg', real=True, negative=True)


# b = atan(tan(-g)/cos(a))
# t = -asin(cos(-g)*sin(a))


eq1 = Eq(atan(-(sin(g)/cosg)/cosa), b)
eq2 = Eq(-asin(cosg*sin(a)), t)

sols = solve((eq1,eq2), (g,a), dict=True)

sol1_a = sols[0]['a'].subs(cosg, cos(g)).subs(cosa, cos(a))


#
# solve([3*a+2*b-10, 4*a-2*b-4], symbols=[)
#
# solve((b,t), (g,a))


# NUMERICAL TESTS INSTEAD
import numpy as np

def deg(rad):
    return rad*180/np.pi

for i in range(10):  # number of random tests
    # Random numbers in the desired domain
    b = np.random.uniform(-np.pi, np.pi)  # beta
    t = np.random.uniform(-np.pi/2, np.pi/2)  # theta
    # Jungao's equations in the Model Test Specaification
    g = np.arcsin(-1*np.cos(t)*np.sin(b))  # gamma
    a = np.arcsin(-np.sin(t) / np.sqrt(1-np.sin(b)**2 * np.cos(t)**2))  # alpha
    # Confirmation using Bernardo's equations in Paper 2
    b_confirm = np.arctan(np.tan(-g)/np.cos(a))
    t_confirm = -np.arcsin(np.cos(-g)*np.sin(a))

    error_margin = 1E-3
    if deg(b) - deg(b_confirm) < error_margin and deg(t) - deg(t_confirm) < error_margin:
        print('Good!')
    else:
        print(f'Not good when: \nbeta        :  {deg(b)} \nbeta_confirm:  {deg(b_confirm)} \ntheta        : {deg(t)}\ntheta_confirm: {deg(t_confirm)} \ngamma: {deg(g)} \nalpha: {deg(a)} \n')




