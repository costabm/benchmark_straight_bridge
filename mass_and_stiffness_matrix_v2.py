"""
Mass and stiffness matrices
"""

import numpy as np


def mass_matrix_12dof_beam(L, mx, my, mz, mxx, ey=None, ez=None, matrix_type='consistent'):
    """
    Local mass matrix of a 12 degree-of-freedom uniform beam element of a 3D frame.
    Params:
        L - [m] Finite element length.
        mx, my, mz - [kg/m] Linear masses (along the x-, y- and z-axis). Usually equal.
        mxx - [(kg/m)*m2] Rotational mass (denoted as m_theta in the book by Strommen).
        ey, ez - [m] Eccentricities.
    Input type:
        scalar or array (when using array, len is the number of elements)
    References:
        Structural Dynamics - Einar N. Strommen. Appendix C.1.
        The local coordinate system is consistent with "Ls" (see Fig. 4.3)
    """

    if hasattr(L, '__len__'):
        assert len(L) == len(mx) == len(my) == len(mz) == len(mxx)

    if ey is None and ez is None:
        ey = np.zeros_like(L)
        ez = np.zeros_like(L)

    o = np.zeros_like(L)

    if matrix_type == 'consistent':
        m11 = L/420 * np.array([[140*mx,          o,         o,           o,           o,           o],
                                [     o,     156*my,         o,  -147*my*ez,           o,     22*my*L],
                                [     o,          o,    156*mz,   147*mz*ey,    -22*mz*L,           o],
                                [     o, -147*my*ez, 147*mz*ey,     140*mxx, -21*mz*L*ey, -21*my*L*ez],
                                [     o,          o,  -22*mz*L, -21*mz*L*ey,   4*mz*L**2,           o],
                                [     o,     22*my*L,        o, -21*my*L*ez,           o,   4*my*L**2]])

        m12 = L/420 * np.array([[70*mx,         o,        o,           o,          o,           o],
                                [    o,     54*my,        o,   -63*my*ez,          o,    -13*my*L],
                                [    o,         o,    54*mz,    63*mz*ey,    13*mz*L,           o],
                                [    o, -63*my*ez, 63*mz*ey,      70*mxx, 14*mz*L*ey, -14*my*L*ez],
                                [    o,         o, -13*mz*L, -14*mz*L*ey, -3*mz*L**2,           o],
                                [    o,   13*my*L,        o, -14*my*L*ez,          o,  -3*my*L**2]])

        m21 = m12.T

        m22 = L/420 * np.array([[140*mx,          o,         o,          o,          o,          o],
                                [     o,     156*my,         o, -147*my*ez,          o,   -22*my*L],
                                [     o,          o,    156*mz,  147*mz*ey,    22*mz*L,          o],
                                [     o, -147*my*ez, 147*mz*ey,    140*mxx, 21*mz*L*ey, 21*my*L*ez],
                                [     o,          o,   22*mz*L, 21*mz*L*ey,  4*mz*L**2,          o],
                                [     o,   -22*my*L,         o, 21*my*L*ez,          o,  4*my*L**2]])

        return np.vstack((np.hstack((m11, m12)),
                          np.hstack((m21, m22))))

    elif matrix_type == 'lumped':
        tol = 1E-5
        return L/2 * np.diag([mx, my, mz, mxx, o+tol, o+tol, mx, my, mz, mxx, o+tol, o+tol])

    else:
        raise ValueError("Unsupported matrix type provided")


def stiff_matrix_12dof_beam(L, E, G, A, Iyy, Izz, J, matrix_type='euler-bernoulli'):
    """
    Local mass matrix of a 12 degree-of-freedom uniform beam element of a 3D frame.
    Params:
        L - [m] Finite element length.
        E - [Pa] Young's modulus.
        G - [Pa] Shear modulus.
        A - [m2] Area.
        Iyy - [m4] Second moment of area about the y-axis (usually the weak axis).
        Izz - [m4] Second moment of area about the z-axis (usually the strong axis).
        J - [m] Saint-Venant torsion constant.
    Input type:
        scalar or array (when using array, len is the number of elements)
    References:
        Structural Dynamics - Einar N. Strommen. Appendix C.1.
        The local coordinate system is consistent with "Ls" (see Fig. 4.3)
    """

    if hasattr(L, '__len__'):
        assert len(L) == len(E) == len(G) == len(A) == len(Iyy) == len(Izz) == len(J)

    o = np.zeros_like(L)

    if matrix_type == 'euler-bernoulli':
        k11 = np.array([[E*A/L,             o,             o,     o,             o,            o],
                        [    o, 12*E*Izz/L**3,             o,     o,             o, 6*E*Izz/L**2],
                        [    o,             o, 12*E*Iyy/L**3,     o, -6*E*Iyy/L**2,            o],
                        [    o,             o,             o, G*J/L,             o,            o],
                        [    o,             o, -6*E*Iyy/L**2,     o,     4*E*Iyy/L,            o],
                        [    o,  6*E*Izz/L**2,             o,     o,             o,    4*E*Izz/L]])

        k12 = np.array([[-E*A/L,              o,              o,      o,             o,            o],
                        [     o, -12*E*Izz/L**3,              o,      o,             o, 6*E*Izz/L**2],
                        [     o,              o, -12*E*Iyy/L**3,      o, -6*E*Iyy/L**2,            o],
                        [     o,              o,              o, -G*J/L,             o,            o],
                        [     o,              o,   6*E*Iyy/L**2,      o,     2*E*Iyy/L,            o],
                        [     o,  -6*E*Izz/L**2,              o,      o,             o,    2*E*Izz/L]])

        k21 = k12.T

        k22 = np.array([[E*A/L,             o,             o,     o,            o,             o],
                        [    o, 12*E*Izz/L**3,             o,     o,            o, -6*E*Izz/L**2],
                        [    o,             o, 12*E*Iyy/L**3,     o, 6*E*Iyy/L**2,             o],
                        [    o,             o,             o, G*J/L,            o,             o],
                        [    o,             o,  6*E*Iyy/L**2,     o,    4*E*Iyy/L,             o],
                        [    o, -6*E*Izz/L**2,             o,     o,            o,     4*E*Izz/L]])

        return np.vstack((np.hstack((k11, k12)),
                          np.hstack((k21, k22))))

    else:
        raise ValueError("Unsupported matrix type provided")


def geom_stiff_matrix_12dof_beam(L, N, My, Mz, e0):
    """
    Local mass matrix of a 12 degree-of-freedom uniform beam element of a 3D frame.
    Params:
        L - [m] Finite element length.
        N - [N]
        My - [Nm] Shear modulus.
        Mz - [m2] Area.
        e0 - [m4] Radius of gyration. Defined in eq. (1.115) of Strommen's book Structural Dynamics.
        Izz - [m4] Second moment of area about the z-axis (usually the strong axis).
        J - [m] Saint-Venant torsion constant.
    Input type:
        scalar or array (when using array, len is the number of elements)
    References:
        Structural Dynamics - Einar N. Strommen. Appendix C.1.
        The local coordinate system is consistent with "Ls" (see Fig. 4.3)
    """

    if hasattr(L, '__len__'):
        assert len(L) == len(N) == len(My) == len(Mz) == len(e0)

    o = np.zeros_like(L)

    raise NotImplementedError
