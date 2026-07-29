

## Dependencies

- pandas
- TensorFlow
- scikeras
- scikit-learn
- XGBoost
- MDAnalysis

```
pip install pandas tensorflow scikeras scikit-learn xgboost MDAnalysis
```

## Repository Structure

- `code_repo/` — all code needed to generate the tables and figures in the manuscript
- `heavy_data_NO_GIT/` — large trajectory and array data, not tracked by git

### Getting the trajectory data

`heavy_data_NO_GIT/gro_xtc/sym` is an empty subfolder where the relevant GROMACS trajectories (`.gro` and `.xtc`) go. Download them from:

https://doi.org/10.18738/T8/2KBDZA

There is also an `asym` subfolder, originally intended for asymmetric systems — these will be addressed in a future study.

> **Warning:** once these files are added, subsequent pushes to GitHub may fail due to storage limits.

## Workflow

Run notebooks in `code_repo/` in the following order:

1. **`npy_array_generation.ipynb`** — generates and stores the numpy arrays required in `../heavy_data_NO_GIT/position_npys`
2. **`training_only_ops.ipynb`** — generates order parameters for pure and 40% DOPC systems only (no ML)
3. **`training_df_generation.ipynb`** — generates pandas dataframes with all features (order parameters, chain heights, lateral displacements) for every running-average (n) value used in training

`apl_blt.ipynb` can also be run independently to generate the Figure 3 plot.

## Training Notebooks

Notebooks are prefixed by model: `NN` (neural network), `XGB` (XGBoost), `LR` (logistic regression).

| Suffix | Output |
|---|---|
| `_training` | Model training on all n |
| `_acc_ece` | Model accuracies and ECE scores |
| `_save_classifier_data` | Saves classifier data and metadata for downstream analyses — **run before proceeding with anything below** |
| `_lo_ld_fracs` | Data points for the Section 4.3 plots. The manuscript uses the statistical model described there; the curves here are simple sigmoid fits for reference — the points themselves are what matters |
| `_chain_height_lat_d` | Classified chain heights and lateral displacements, and their percentage differences (Table 4) |
| `_order_parameters` | **NN only** — lo/ld classified sn1/sn2 order parameters (Figure 6) |
| `_chain_mix_+entropy` | Mixed chain fractions (Table 5, Figure 7) and entropy plots (Figure 8) |
| `_chol_dopc_chain_rdfs` | **NN only** — lo/ld classified acyl chain COM to cholesterol COM RDFs (Figure 9) |
| `_hg_rdfs` | **NN only** — lo/ld classified umbrella coverage (Figure 11) |
