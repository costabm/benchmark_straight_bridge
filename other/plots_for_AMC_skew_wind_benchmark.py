from static_loads import static_wind_func, R_loc_func
from buffeting import U_bar_func
from straight_bridge_geometry import g_node_coor, p_node_coor, g_node_coor_func, R, arc_length, zbridge, bridge_shape, g_s_3D_func

U_bar = U_bar_func(g_node_coor)  # Homogeneous wind only depends on g_node_coor_z...
# Displacements
g_node_coor_sw, p_node_coor_sw, D_glob_sw = static_wind_func(g_node_coor, p_node_coor, alpha, U_bar, beta_DB, theta_0,
                                                             aero_coef_method, n_aero_coef, skew_approach)
D_loc_sw = mat_Ls_node_Gs_node_all_func(D_glob_sw, g_node_coor, p_node_coor, alpha)
# Internal forces
R_loc_sw = R_loc_func(D_glob_sw, g_node_coor, p_node_coor, alpha)  # orig. coord. + displacem. used to calc. R.