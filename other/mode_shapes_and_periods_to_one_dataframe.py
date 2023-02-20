import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from my_utils import normalize_mode_shape
from transformations import mat_Ls_node_Gs_node_all_func
from mass_and_stiffness_matrix import mass_matrix_func, stiff_matrix_func, geom_stiff_matrix_func
from modal_analysis import simplified_modal_analysis_func
from straight_bridge_geometry import g_node_coor, p_node_coor

n_modes_plot = 100

########################################################################################################################
# Initialize structure:
########################################################################################################################
bridge_concept = 'K11'
from straight_bridge_geometry import g_node_coor, p_node_coor

g_node_num = len(g_node_coor)
g_elem_num = g_node_num - 1
p_node_num = len(p_node_coor)
all_node_num = g_node_num + p_node_num
all_elem_num = g_elem_num + p_node_num
R_loc = np.zeros((all_elem_num, 12))  # No initial element internal forces
D_loc = np.zeros((all_node_num, 6))  # No initial nodal displacements
girder_N = copy.deepcopy(R_loc[:g_elem_num, 0])  # No girder axial forces
c_N = copy.deepcopy(R_loc[g_elem_num:, 0])  # No columns axial forces
alpha = copy.deepcopy(D_loc[:g_node_num, 3])  # No girder nodes torsional rotations

mass_matrix = mass_matrix_func(g_node_coor, p_node_coor, alpha)  # (N)
stiff_matrix = stiff_matrix_func(g_node_coor, p_node_coor, alpha)  # (N)
geom_stiff_matrix = geom_stiff_matrix_func(g_node_coor, p_node_coor, girder_N, c_N, alpha)

_, _, omegas, shapes = simplified_modal_analysis_func(mass_matrix, stiff_matrix - geom_stiff_matrix)
periods = 2 * np.pi / omegas

periods = 2 * np.pi / omegas
n_g_nodes = len(g_node_coor)
n_p_nodes = len(p_node_coor)

flat_shapes_Gs = shapes[:n_modes_plot].copy()
assert len(flat_shapes_Gs.shape) == 2
assert flat_shapes_Gs.shape[1] == (n_g_nodes + n_p_nodes) * 6
assert (flat_shapes_Gs.shape[1] / 6).is_integer()
shapes_Gs = np.reshape(flat_shapes_Gs, (flat_shapes_Gs.shape[0], int(flat_shapes_Gs.shape[1] / 6), 6))
shapes_Ls = np.array([mat_Ls_node_Gs_node_all_func(shapes_Gs[f], g_node_coor, p_node_coor, alpha) for f in range(
    shapes_Gs.shape[
        0])])  # this matrix requires the shapes matrix to be in format 'ni', with n: number of nodes and i: 6 DOF.
g_shapes_Ls = shapes_Ls[:, :n_g_nodes]
g_shapes_Ls = np.array([normalize_mode_shape(x) for x in g_shapes_Ls])

df_modes = pd.DataFrame()
for i in range(n_modes_plot):
    df_modes = pd.concat([df_modes, pd.DataFrame(g_shapes_Ls[i].T)], axis=0)






