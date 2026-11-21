#!/usr/bin/env python3
"""
Fast LR post-classification summary of acyl-chain height and xy radius.

This script uses already saved LR classification labels only.
It saves only the compact printed-summary table:

    chi_c, predicted_class,
    sn-1 Height (nm), sn-1 xy_rad (nm),
    sn-2 Height (nm), sn-2 xy_rad (nm)

Each cell is mean ± std.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from code_libraries.acyl_chain_preprocessing import chain_height_df
from code_libraries.bilayer_operations import Bilayer
from code_libraries.op_by_chain import convolve_ops, split_ops_by_label
from code_libraries.predict_acyl_chains_NN_by_chain import clf_lo_ld_by_chain


# ---------------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------------

start_frame, stop_frame = 0, 2000
sys_type = "sym"
window = 15
red_type = "ft_last"
model_type = "LR"
validation_frames = 3

project_root = Path(__file__).resolve().parent.parent
data_array_dir = project_root / "heavy_data_NO_GIT/data_arrays_by_chain_logreg"
output_dir = project_root / "heavy_data_NO_GIT/results_by_chain_new"
summary_csv = output_dir / "classified_LR_height_rad_fast_printed_summary.csv"

systems = [
    ("dopc_94", 0.0625),
    ("dopc_90", 0.10),
    ("dopc_85", 0.15),
    ("dopc_80", 0.20),
    ("dopc_75", 0.25),
    ("dopc_70", 0.30),
    ("dopc_65", 0.35),
]

chains = ["sn1", "sn2"]
leaflets = ["lower", "upper"]
feature_columns = ["z_height", "xy_radius"]


# Constants mirror chain_height_df(); validation below checks they still match.
sn1_indices = [41, 42, 43, *range(91, 138)]
sn2_indices = [32, 33, 34, *range(44, 91)]
chain_masses = np.array(
    [
        12., 1., 1., 12., 1., 1., 12., 1., 1., 12., 1., 1., 12.,
        1., 1., 12., 1., 1., 12., 1., 1., 12., 1., 12., 1., 12.,
        1., 1., 12., 1., 1., 12., 1., 1., 12., 1., 1., 12., 1.,
        1., 12., 1., 1., 12., 1., 1., 12., 1., 1., 1.,
    ],
    dtype=float,
)


Bilayer.main_directory = str(project_root)
Bilayer.heavy_data_directory = "heavy_data_NO_GIT/"
Bilayer.gro_xtc_directory = str(project_root / "heavy_data_NO_GIT/gro_xtc")
Bilayer.position_directory = str(project_root / "heavy_data_NO_GIT/position_npys")


def fast_chain_height_array(dopc_leaflet, chain):
    chain_indices = sn1_indices if chain == "sn1" else sn2_indices
    chain_positions = dopc_leaflet[:, :, chain_indices, :]

    chain_com = np.sum(chain_positions * chain_masses[None, None, :, None], axis=2) / chain_masses.sum()
    shifted_chain_positions = chain_positions - chain_com[:, :, None, :]

    cstart = shifted_chain_positions[:, :, 0, :]
    cend = shifted_chain_positions[:, :, -4, :]

    z_height = np.abs(cstart[:, :, 2] - cend[:, :, 2])
    xy_radius = np.linalg.norm(cstart[:, :, :2] - cend[:, :, :2], axis=2)

    return np.stack([z_height, xy_radius], axis=2)


def mean_pm_std(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return "nan ± nan"
    return f"{values.mean():.4f} ± {values.std(ddof=0):.4f}"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

summary_rows = []

for sys_name, cholesterol_concentration in systems:
    print(f"Loading DOPC positions for {sys_name}")

    dopc_pos_raw = clf_lo_ld_by_chain(
        sys_name,
        sys_type,
        "raw",
        model_type,
        "sn1",
        3,
        start_frame,
        stop_frame,
    ).bilayer_all_dopc_data()

    dopc_pos_raw_lower, dopc_pos_raw_upper = dopc_pos_raw
    dopc_by_leaflet = {
        "lower": dopc_pos_raw_lower,
        "upper": dopc_pos_raw_upper,
    }

    chain_class_values = {}

    for chain in chains:
        label_file = data_array_dir / f"{sys_name}_{chain}_{red_type}_{window}pt_prediction_labels.npy"
        print(f"  Loading saved labels: {label_file.name}")

        preds_lower, preds_upper = np.load(label_file, allow_pickle=True)
        preds_by_leaflet = {
            "lower": np.asarray(preds_lower, dtype=bool),
            "upper": np.asarray(preds_upper, dtype=bool),
        }

        height_lateral_by_leaflet = {}
        labels_by_leaflet = {}

        for leaflet in leaflets:
            dopc_leaflet = dopc_by_leaflet[leaflet]
            n_frames, n_lipids = dopc_leaflet.shape[:2]

            n_check = min(validation_frames, n_frames)
            check_df = chain_height_df(dopc_leaflet[:n_check], n_check, chain, iqr_filter=False)
            check_fast = fast_chain_height_array(dopc_leaflet[:n_check], chain).reshape(-1, 2)

            if list(check_df.columns) != feature_columns:
                raise ValueError(f"{sys_name} {chain} {leaflet}: unexpected feature columns")

            np.testing.assert_allclose(
                check_fast,
                check_df[feature_columns].to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
            )

            height_lateral = fast_chain_height_array(dopc_leaflet, chain)
            height_lateral_averaged = convolve_ops(height_lateral, window)

            labels = preds_by_leaflet[leaflet]
            if height_lateral_averaged.shape[:2] != labels.shape:
                raise ValueError(
                    f"{sys_name} {chain} {leaflet}: feature shape "
                    f"{height_lateral_averaged.shape[:2]} does not match label shape {labels.shape}"
                )

            height_lateral_by_leaflet[leaflet] = height_lateral_averaged
            labels_by_leaflet[leaflet] = labels

        height_lateral_both = np.concatenate(
            [height_lateral_by_leaflet["lower"], height_lateral_by_leaflet["upper"]],
            axis=1,
        )
        labels_both = np.concatenate(
            [labels_by_leaflet["lower"], labels_by_leaflet["upper"]],
            axis=1,
        )

        lo_features, ld_features = split_ops_by_label(height_lateral_both, labels_both)

        chain_class_values[chain] = {}
        for predicted_class, split_features in [("Ld", ld_features), ("Lo", lo_features)]:
            selected_frames = [frame for frame in split_features if len(frame) > 0]
            selected = np.vstack(selected_frames) if selected_frames else np.empty((0, 2))
            chain_class_values[chain][predicted_class] = selected

    for predicted_class in ["Ld", "Lo"]:
        sn1_values = chain_class_values["sn1"][predicted_class]
        sn2_values = chain_class_values["sn2"][predicted_class]

        summary_rows.append(
            {
                "χc": cholesterol_concentration,
                "predicted_class": predicted_class,
                "sn-1 Height (nm)": mean_pm_std(sn1_values[:, 0]),
                "sn-1 xy_rad (nm)": mean_pm_std(sn1_values[:, 1]),
                "sn-2 Height (nm)": mean_pm_std(sn2_values[:, 0]),
                "sn-2 xy_rad (nm)": mean_pm_std(sn2_values[:, 1]),
            }
        )


summary_df = pd.DataFrame(summary_rows)

output_dir.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(summary_csv, index=False)

print("\nLR height/xy_rad printed summary table:")
print(summary_df.to_string(index=False))
print(f"\nSaved:\n{summary_csv}")
