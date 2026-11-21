import numpy as np
import MDAnalysis as mda
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from scipy.spatial import Voronoi
from matplotlib.collections import PolyCollection

from code_libraries.bilayer_operations import Bilayer
from code_libraries.acyl_chain_preprocessing import *
from code_libraries.miscellaneous_functions import *
from lo_ld_classifier import *



save_dir = "../straigh_line_plots"
os.makedirs(save_dir, exist_ok=True)

start_frame , stop_frame = 0 , 50
sys_type = "sym"
sys_name_85 = "dopc_85"
sys_name_80 = "dopc_80"
sys_name_75 = "dopc_75"
sys_name_70 = "dopc_70"
sys_name_65 = "dopc_65"

dopc_85 = clf_lo_ld(sys_name_85 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)
dopc_80 = clf_lo_ld(sys_name_80 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)
dopc_75 = clf_lo_ld(sys_name_75 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)
dopc_70 = clf_lo_ld(sys_name_70 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)
dopc_65 = clf_lo_ld(sys_name_65 , sys_type , "ft_last" , "xgb" , 11 , start_frame , stop_frame)


dopc_85.process_all_input()
dopc_80.process_all_input()
dopc_75.process_all_input()
dopc_70.process_all_input()
dopc_65.process_all_input()

dopc_85.classify(uses_predict_proba=False)
dopc_80.classify(uses_predict_proba=False)
dopc_75.classify(uses_predict_proba=False)
dopc_70.classify(uses_predict_proba=False)
dopc_65.classify(uses_predict_proba=False)

chol_fracs = [0.15, 0.20, 0.25, 0.30, 0.35]
systems = [dopc_85, dopc_80, dopc_75, dopc_70, dopc_65]

lower_means = []
upper_means = []

for sys in systems:
    lower, upper = sys.lo_ld_proportions()   # returns (lower_array, upper_array)
    lower_means.append(np.mean(lower))
    upper_means.append(np.mean(upper))

save_path = os.path.join(save_dir, "lo_fraction_vs_chol_11pt_ft_last.png")

plt.figure(figsize=(12,8))

plt.plot(chol_fracs , 0.5*(np.array(lower_means) + np.array(upper_means)) , marker = "o")

plt.legend(["Combined"])
plt.ylim(0, 1.0)

plt.xlabel("CHOL fraction")
plt.ylabel("Fraction of Lo classified DOPC")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

plt.show()

dopc_85.process_all_input()
dopc_85.classify(uses_predict_proba=False)

dopc_85.plot_predicted_average_all_frames_simple("test_plot" , figure_size=[15,8])
plt.show()
