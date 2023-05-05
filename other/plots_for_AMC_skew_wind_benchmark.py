import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from static_loads import static_wind_func, R_loc_func
from buffeting import U_bar_func
from straight_bridge_geometry import g_node_coor, p_node_coor, g_node_coor_func, R, arc_length, zbridge, bridge_shape, g_s_3D_func
import os
import pandas as pd
from buffeting import beta_0_func
from my_utils import deg, rad

plt_TD = True

#%% Standard benchmark, with different betas
results = {}
results_path_list_FD = []
results_path_list_TD = []
results_folder_path = os.path.join(os.getcwd(), r'results')
# results_folder_path = r'C:\\Users\\bercos\\PycharmProjects\\benchmark_straight_bridge\\results'

# Getting Frequency domain file
for item in os.listdir(results_folder_path):
    if item[:12] == "FD_all_nodes":
        results_path_list_FD.append(item)
input_idx_FD = int(input(pd.Series(results_path_list_FD)))
results_path_FD = results_path_list_FD[input_idx_FD]
df_FD = pd.read_csv(os.path.join(results_folder_path, results_path_FD))

# Getting Time domain file
if plt_TD:
    for item in os.listdir(results_folder_path):
        if item[:12] == "TD_all_nodes":
            results_path_list_TD.append(item)
    input_idx_TD = int(input(pd.Series(results_path_list_TD)))
    results_path_TD = results_path_list_TD[input_idx_TD]
    df_TD = pd.read_csv(os.path.join(results_folder_path, results_path_TD))

x_coords = g_node_coor[:,0]
beta_DBs = df_FD['beta_DB']
dof_string = ['x [m]', 'y [m]', 'z [m]', 'rx [deg]', 'ry [deg]', 'rz [deg]']
analysis_string = ['mean', 'std']
color_list = ['blue', 'orange', 'green', 'red', 'purple']
handles, labels = [], []
fig, axs = plt.subplots(3, 2, figsize=(8, 10), dpi=300, constrained_layout=False)
for a, analysis_type in enumerate(['static', 'std']):
    for b, beta_DB in enumerate(beta_DBs):
        for d, dof in enumerate([1,2,3]):
            # Frequency domain results:
            # Organizing results
            df_one_beta = df_FD.loc[np.isclose(df_FD['beta_DB'], beta_DB)]
            mask = df_one_beta.columns[[f'{analysis_type}_dof_{dof}' in c for c in df_one_beta.columns]]  # for analysis and dof
            results = np.squeeze(np.array(df_one_beta[mask]))
            # Plotting
            ax = axs[d, a]
            ax.set_title(f'{analysis_string[a]} of {dof_string[dof]}')
            if dof in [3,4,5]:
                results = np.rad2deg(results)
            ax.plot(x_coords, results, c=color_list[b], alpha=0.75, label=r'$\beta = $' + f'{round(deg(beta_0_func(beta_DB)))}' + r'$\degree$' if a==d==0 else None)
            if plt_TD:
                for SE_l in ['NL']:
                    df_one_beta = df_TD.loc[np.isclose(df_TD['beta_DB'], beta_DB) & (df_TD['SE_linearity'] == SE_l)]
                    mean_mask = df_one_beta.columns[[f'{analysis_type}_dof_{dof}' in c and c[:3]!='std' for c in df_one_beta.columns]]
                    std_mask = df_one_beta.columns[[f'{analysis_type}_dof_{dof}' in c and c[:3]=='std' for c in df_one_beta.columns]]
                    mean_results = np.squeeze(np.array(df_one_beta[mean_mask]))  # mean of all seeds
                    std_results = np.squeeze(np.array(df_one_beta[std_mask]))  # std of all seeds
                    if dof in [3, 4, 5]:
                        mean_results = np.rad2deg(mean_results)
                        std_results = np.rad2deg(std_results)
                    # ax.plot(x_coords, mean_results, c=color_list[b], linestyle='dotted', alpha=0.3, linewidth=3)
                    # ax.plot(x_coords, mean_results + std_results, c=color_list[b], linestyle='dashdot', alpha=0.3, linewidth=3)
                    # ax.plot(x_coords, mean_results - std_results, c=color_list[b], linestyle='dashdot', alpha=0.3, linewidth=3)
                    ax.fill_between(x_coords, mean_results - std_results, mean_results + std_results, facecolor=color_list[b], alpha=0.3)
            ax.grid(visible=True, which='major', alpha=0.6)
            ax.minorticks_on()
            ax.grid(visible=True, which='minor', alpha=0.15)
            handles.append(ax.get_legend_handles_labels()[0])
            labels.append(ax.get_legend_handles_labels()[1])

legend1 = fig.legend(loc=8, ncol=len(beta_DBs), frameon=False)

black_line = matplotlib.lines.Line2D([0], [0], color='black', alpha=0.5, label=r'Frequency-domain')
black_patch = matplotlib.patches.Patch(facecolor='black', alpha=0.3, label=r'Time-domain ($\mu \pm \sigma$)')
legend2 = fig.legend(handles=[black_line, black_patch], loc='center', bbox_to_anchor=[0.5,0.05], ncol=2, frameon=False)

fig.tight_layout()
fig.subplots_adjust(bottom=0.1)  # play with this number to adjust legend placement
U_check = U_bar_func(g_node_coor)[0]
plt.savefig(r'results\benchmark' + f'_U_{int(U_check)}_' + f"SW_{str(df_FD['SWind'][0])[0]}_KG_{str(df_FD['KG'][0])[0]}_SE_{str(df_FD['SE'][0])[0]}"
                                  + f"_nfreq_{df_FD['n_freq'][0]}_fmin_{df_FD['f_min'][0]}_fmax_{df_FD['f_max'][0]}_zeta_{df_FD['damping_ratio'][0]}_ModalDamping.jpg")
plt.close()



# #%% Benchmark of different simpler cases
#
# results = {}
# results_path_list = []
# results_folder_path = os.path.join(os.getcwd(), r'results')
#
# for item in os.listdir(results_folder_path):
#     if item[:12] == "FD_all_nodes":
#         results_path_list.append(item)
#
# input_idx_list = [int(i) for i in eval(input(pd.Series(results_path_list)))]
# color_list = ['blue', 'orange', 'green', 'red', 'purple']
# label_list = ['base case', 'case 1', 'case 2', 'case 3', 'case 4']
# # label_list = ['case 1', 'case 4']
# # label_list = ['case 2', 'case 3']
#
# fig, axs = plt.subplots(3, 2, figsize=(8, 10), dpi=300, constrained_layout=False)
# for input_idx in input_idx_list:
#     results_path = results_path_list[input_idx]
#
#     # Getting the DataFrames of the results
#     df = pd.read_csv(os.path.join(results_folder_path, results_path))
#
#     x_coords = g_node_coor[:,0]
#     beta_DBs = df['beta_DB']
#     dof_string = ['x', 'y', 'z', 'rx', 'ry', 'rz']
#     analysis_string = ['mean', 'std']
#
#     for a, analysis_type in enumerate(['static', 'std']):
#         for b, beta_DB in enumerate(beta_DBs):
#             for dof in range(3):
#                 # Organizing results
#                 df_one_beta = df.loc[df['beta_DB'] == beta_DB]
#                 mask = df_one_beta.columns[[f'{analysis_type}_dof_{dof}' in c for c in df_one_beta.columns]]  # for analysis and dof
#                 results = np.squeeze(np.array(df_one_beta[mask]))
#                 # Plotting
#                 ax = axs[dof, a]
#                 ax.set_title(f'{analysis_string[a]} of {dof_string[dof]}')
#                 ax.plot(x_coords, results, c=color_list[input_idx], label=label_list[input_idx] if a==dof==0 else None)
#
# fig.legend(loc=8, ncol=len(label_list))
# fig.tight_layout()
# fig.subplots_adjust(bottom=0.08)  # play with this number to adjust legend placement
#
# U_check = U_bar_func(g_node_coor)[0]
#
# plt.savefig(r'results\simple_benchmark_2_3' + f'_U_{int(U_check)}_' + f"SW_{str(df['SWind'][0])[0]}_KG_{str(df['KG'][0])[0]}_SE_{str(df['SE'][0])[0]}"
#                                 + f"_nfreq_{df['n_freq'][0]}_fmin_{df['f_min'][0]}_fmax_{df['f_max'][0]}_zeta_{df['damping_ratio'][0]}_ModalDamping.jpg")

