# Machine-learning classification of Lo / Ld acyl chains in DOPC/cholesterol bilayers

Code for classifying the sn-1 and sn-2 chains of DOPC as liquid-ordered (Lo) or liquid-disordered (Ld) with
logistic regression (LR), XGBoost (XGB) and a neural network (NN), and for the analyses in the paper.

## Layout

```
code_libraries/       shared Python functions (imported by every notebook)
common/               notebooks used by all three models
LR/  XGB/  NN/        one folder per classifier, the same steps in each
heavy_data_NO_GIT/    data folder (not on GitHub; only the empty folder structure is)
```

Every notebook is run from its own folder; paths go to `../heavy_data_NO_GIT/` and `../code_libraries/`.

## Data

The trajectories are not in the repository. Put them in `heavy_data_NO_GIT/gro_xtc/sym/`:

| file | system |
|---|---|
| `dopc_pure.gro/.xtc` | pure DOPC (χ_C = 0, Ld training class) |
| `dopc_94` … `dopc_65` `.gro/.xtc` | DOPC/CHOL 94/6, 90/10, 85/15, 80/20, 75/25, 70/30, 65/35 |
| `dopc_60.gro/.xtc` | DOPC/CHOL 60/40 (χ_C = 0.40, Lo training class) |
| `dopc_75_rep1`, `dopc_75_rep2` `.gro/.xtc` | two independent χ_C = 0.25 replicas (300 DOPC per leaflet) |

All other folders in `heavy_data_NO_GIT/` are filled by the notebooks.

## Requirements

See `requirements.txt` (numpy, pandas, scipy, matplotlib, MDAnalysis, scikit-learn, xgboost, tensorflow/keras, scikeras, jupyter):

```
pip install -r requirements.txt
```

## Run order

1. `common/npy_array_generation` – positions of every system from the trajectories.
2. `common/training_df_generation` – training tables (pure DOPC = Ld, DOPC/CHOL 60/40 = Lo), per leaflet and window.
3. In `LR/`, `XGB/` and `NN/`:
   1. `*_training` – train on the lower leaflet, test on the upper leaflet, for every window.
   2. `*_acc_ece` – accuracy, F1 and calibration error; choice of the window (LR 7 pt, XGB 17 pt, NN 5 pt).
   3. `*_save_classifier_data` – classify all systems, endpoints and replicas with the chosen window.
   4. Analyses: `*_lo_ld_fracs`, `*_chain_mix_+_entropy`, `*_chain_height_lat_d`, and the others below.
4. `common/entropy_all_models`, `common/op_profiles_reference`, `common/S_avg_distribution_pure_vs_60`, `common/hg_rdfs_no_ML`.

## Figures and tables

| paper | notebook |
|---|---|
| Fig. 5 (Lo fraction) | `LR/LR_lo_ld_fracs`, `XGB/XGB_lo_ld_fracs`, `NN/NN_lo_ld_fracs` |
| Fig. 6 (order parameters) | `common/op_profiles_reference` (a, b), `NN/NN_order_parameters` (c–h) |
| Fig. 7 (chain populations) | `*/…_chain_mix_+_entropy` |
| Fig. 8 (three-state entropy) | `common/entropy_all_models` |
| Fig. 9 (CHOL–chain RDF) | `NN/NN_chol_dopc_chain_rdfs` |
| Fig. 10 (headgroup RDF, no classification) | `common/hg_rdfs_no_ML` |
| Fig. 11 (headgroup–CHOL RDF) | `NN/NN_hg_rdfs` |
| Fig. S1 (⟨S_CH⟩ distributions) | `common/S_avg_distribution_pure_vs_60`, then `NN/NN_S_avg_distribution_chol25` |
| Table 1 (height / lateral displacement, Lo vs Ld) | `LR/LR_height_lat_d_lo_ld_table` |
| Table 2 (Lo–Ld percentage differences) | `*/…_chain_height_lat_d` |
| Table 3 (which chain is Lo in mixed lipids) | `*/…_chain_mix_+_entropy` |

## Notes

- Training uses the lower leaflet of pure DOPC and DOPC/CHOL 60/40; for these two systems the analyses use only the
  upper (unseen) leaflet.
- Uncertainties: SD (ddof = 1) over the three χ_C = 0.25 runs (`dopc_75`, `dopc_75_rep1`, `dopc_75_rep2`),
  shown at χ_C = 0.25.
