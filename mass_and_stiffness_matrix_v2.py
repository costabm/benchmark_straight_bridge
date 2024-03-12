"""
Mass and stiffness matrices
"""

import numpy as np

def mass_matrix_12b_local_func(L, mx, my, mz, mxx='default', ey=0, ez=0, matrix_type='consistent'):
    """
    L - element length
    mx, my, mz - linear masses (along the x-, y- and z-axis). Usually equal.
    mxx - rotational mass (denoted as m_theta in the book by Strommen)
    Reference: Structural Dynamics - Einar N. Strommen. Appendix C.1
    """
    if matrix_type == 'consistent':
        m11 = np.array([[140*mx, 0, 0, 0, 0, 0],
                        [0, 156*my, 0, -147*my*ez, 0, 22*my*L],
                        [0, 0, 156*mz, 147*mz*ey, -22*mz*L, 0],
                        [0, -147*my*ez, 147*mz*ey, 140*mxx, -21*mz*L*ey, -21*my*L*ez],
                        [0, 0, -22*mz*L, -21*mz*L*ey, 4*mz*L**2, 0],
                        [0, 22*my*L, 0, -21*my*L*ez, 0, 4*my*L**2]]) * L/420  # todo: double check all values
        m12 = np.array([[70*mx, 0, 0, 0, 0, 0],
                        [0, 54*my, 0, -63*my*ez, 0, -13*my*L],
                        [0, 0, 54*mz, 63*mz*ey, 13*mz*L, 0],
                        [0, -63*my*ez, 63*mz*ey, 70*mxx, 14*mz*L*ey, -14*my*L*ez],
                        [0, 0, -13*mz*L, -14*mz*L*ey, -3*mz*L**2, 0],
                        [0, 13*my*L, 0, -14*my*L*ez, 0, -3*my*L**2]]) * L/420
        m21 = m12.T
        m22 = np.array([[140*mx, 0, 0, 0, 0, 0],
                        [0, 156*my, 0, -147*my*ez, 0, -22*my*L],
                        [0, 0, 156*mz, 147*mz*ey, 22*mz*L, 0],
                        [0, -147*my*ez, 147*mz*ey, 140*mxx, 21*mz*L*ey, 21*my*L*ez],
                        [0, 0, 22*mz*L, 21*mz*L*ey, 4*mz*L**2, 0],
                        [0, -22*my*L, 0, 21*my*L*ez, 0, 4*my*L**2]]) * L/420

        m11 = np.array([[140*mx,          0,         0,           0,           0,           0],
                        [     0,     156*my,         0,  -147*my*ez,           0,     22*my*L],
                        [     0,          0,    156*mz,   147*mz*ey,    -22*mz*L,           0],
                        [     0, -147*my*ez, 147*mz*ey,     140*mxx, -21*mz*L*ey, -21*my*L*ez],
                        [     0,          0,  -22*mz*L, -21*mz*L*ey,   4*mz*L**2,           0],
                        [     0,     22*my*L,        0, -21*my*L*ez,           0,   4*my*L**2]]) * L/420

        m12 = np.array([[70*mx,         0,        0,           0,          0,           0],
                        [    0,     54*my,        0,   -63*my*ez,          0,    -13*my*L],
                        [    0,         0,    54*mz,    63*mz*ey,    13*mz*L,           0],
                        [    0, -63*my*ez, 63*mz*ey,      70*mxx, 14*mz*L*ey, -14*my*L*ez],
                        [    0,         0, -13*mz*L, -14*mz*L*ey, -3*mz*L**2,           0],
                        [    0,   13*my*L,        0, -14*my*L*ez,          0,  -3*my*L**2]]) * L/420
        m21 = m12.T
        m22 = np.array([[140*mx,          0,         0,          0,          0,          0],
                        [     0,     156*my,         0, -147*my*ez,          0,   -22*my*L],
                        [     0,          0,    156*mz,  147*mz*ey,    22*mz*L,          0],
                        [     0, -147*my*ez, 147*mz*ey,    140*mxx, 21*mz*L*ey, 21*my*L*ez],
                        [     0,          0,   22*mz*L, 21*mz*L*ey,  4*mz*L**2,          0],
                        [     0,   -22*my*L,         0, 21*my*L*ez,          0,  4*my*L**2]]) * L/420
        # todo: double check all values
    # todo: check if matrix is symmetric with a simple test