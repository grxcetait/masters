#!/usr/bin/env python3

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# === ARGUEMENTS ====
# Folder containing the sumpt, gaussian_weighting and two_steps result folders
base_dir = "outputs/sem1week4"

# Subfolder for each algorithm
sumpt_dir = os.path.join(base_dir, "sumpt")
gaussian_dir = os.path.join(base_dir, "gaussian_weighting")
two_step_dir = os.path.join(base_dir, "two_steps")

# Where to save the tables
tables_dir = os.path.join(base_dir, "tables")

# Matching tolerances to use when recomputing the two-step results [mm]
tolerances = [0.3, 1.0]

# Folder names look like "1.5ptmin_100ptmax_2.5etamax" (inf = no cut)
folder_pattern = re.compile(r"^([\d.]+|inf)ptmin_([\d.]+|inf)ptmax_([\d.]+|inf)etamax$")

# Names for the box weights
weight_names = {0: "track count", 1: "pT", 2: "pT^2"}

# Column names for the cuts
cut_columns = ["pT min (GeV)", "pT max (GeV)", "|eta| max"]

# ===== FUNCTIONS =====


def find_cut_folders(algorithm_dir):

    # Find every subfolder whose name records the cuts, and read the cuts from the name
    folders = []

    if not os.path.isdir(algorithm_dir):
        print(f"Not found, skipping: {algorithm_dir}")
        return folders

    for name in sorted(os.listdir(algorithm_dir)):

        match = folder_pattern.match(name)
        if match is None:
            continue

        cuts = {"pT min (GeV)": float(match.group(1)),
                "pT max (GeV)": float(match.group(2)),
                "|eta| max": float(match.group(3))}

        folders.append((cuts, os.path.join(algorithm_dir, name)))

    return folders


def finish_table(df, sort_columns):

    # Sort the rows, then write infinite cuts as "none" and round the numbers
    df = df.sort_values(sort_columns).reset_index(drop = True)

    for col in cut_columns:
        df[col] = df[col].apply(lambda x: "none" if np.isinf(x) else f"{x:g}")

    return df.round(3)


def box_table():

    # Read every vertex_scan_summary.csv from the box (sum pT) runs
    rows = []

    for cuts, folder in find_cut_folders(sumpt_dir):

        path = os.path.join(folder, "vertex_scan_summary.csv")
        if not os.path.isfile(path):
            continue

        summary = pd.read_csv(path)

        for _, r in summary.iterrows():
            row = dict(cuts)
            row["Weight"] = weight_names.get(int(r["power"]), f"pT^{int(r['power'])}")
            row["res (mm)"] = r["res"]
            for tol in tolerances:
                row[f"Eff tol {tol} (%)"] = 100 * r[f"eff_tol{tol}"]
            row["Median |dz| (mm)"] = r["median_abs_dz"]
            rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # Keep the weights in the order count, pT, pT^2 when sorting
    df["_weight_order"] = df["Weight"].map({v: k for k, v in weight_names.items()})
    df = finish_table(df, cut_columns + ["_weight_order", "res (mm)"])

    return df.drop(columns = "_weight_order")


def gaussian_table():

    # Read every gaussian_scan_summary.csv from the Gaussian weighting runs
    rows = []

    for cuts, folder in find_cut_folders(gaussian_dir):

        path = os.path.join(folder, "gaussian_scan_summary.csv")
        if not os.path.isfile(path):
            continue

        summary = pd.read_csv(path)

        for _, r in summary.iterrows():
            row = dict(cuts)
            row["d0 factor"] = "yes" if bool(r["use_d0_weight"]) else "no"
            row["Normalised"] = "yes" if bool(r["normalise"]) else "no"
            row["Tracks per event"] = r["mean_tracks"]
            for tol in tolerances:
                row[f"Eff tol {tol} (%)"] = 100 * r[f"eff_tol{tol}"]
            row["Median |dz| (mm)"] = r["median_abs_dz"]
            rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    return finish_table(df, cut_columns + ["d0 factor", "Normalised"])


def select_candidate(score, cand_ntrk):

    # Index of the candidate with the highest score in each event
    # (empty candidate slots can never be chosen)
    score = np.where(cand_ntrk > 0, score, -np.inf)
    return np.argmax(score, axis = 1)


def two_step_table():

    # Recompute the two-step results from the saved candidates in each .npz file
    rows = []

    for cuts, folder in find_cut_folders(two_step_dir):

        for name in sorted(os.listdir(folder)):

            match = re.match(r"^vertex_candidates_s([\d.]+)\.npz$", name)
            if match is None:
                continue

            s_max = float(match.group(1))
            data = np.load(os.path.join(folder, name))

            cand_z = data["cand_z"]
            cand_ntrk = data["cand_ntrk"]
            truth_z = data["truth_z"]
            truth_valid = data["truth_valid"].astype(bool)

            # Mean tracks per event, from the per-event CSV saved alongside
            csv_path = os.path.join(folder, f"vertex_candidates_s{match.group(1)}.csv")
            if os.path.isfile(csv_path):
                tracks = pd.read_csv(csv_path, usecols = ["n_tracks_passed"])["n_tracks_passed"].mean()
            else:
                tracks = np.nan

            # Distance from every candidate to the truth (empty slots = infinitely far)
            dist = np.abs(cand_z - truth_z[:, None])
            dist = np.where(np.isnan(dist), np.inf, dist)

            # The four ways of choosing a candidate
            selectors = {
                "Densest": data["cand_height"],
                "Track count": cand_ntrk.astype(np.float64),
                "Sum pT": data["cand_sumpt"],
                "Sum pT^2": data["cand_sumpt2"],
            }

            # Which candidate each selector chooses (same for every tolerance)
            event_rows = np.arange(len(truth_z))
            chosen = {sel: select_candidate(score, cand_ntrk) for sel, score in selectors.items()}

            for tol in tolerances:

                # Reconstruction efficiency: is the truth close to ANY candidate?
                reco = (dist < tol).any(axis = 1)

                row = dict(cuts)
                row["s max"] = s_max
                row["tol (mm)"] = tol
                row["Tracks per event"] = tracks
                row["Reco eff (%)"] = 100 * np.mean(reco[truth_valid])

                # Selection efficiency for each selector, overall and in events where the truth was found
                for sel in selectors:
                    sel_dist = dist[event_rows, chosen[sel]]
                    matched = sel_dist < tol
                    found = truth_valid & reco
                    row[f"{sel} eff (%)"] = 100 * np.mean(matched[truth_valid])
                    row[f"{sel} eff if found (%)"] = 100 * np.mean(matched[found]) if np.any(found) else np.nan

                # Median distance of the chosen candidate from the truth
                for sel in selectors:
                    sel_dist = dist[event_rows, chosen[sel]][truth_valid]
                    row[f"{sel} median |dz| (mm)"] = np.median(sel_dist[np.isfinite(sel_dist)])

                rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    return finish_table(df, ["tol (mm)"] + cut_columns + ["s max"])


def main():

    # Create the output directory if it doesn't exist
    os.makedirs(tables_dir, exist_ok = True)

    tables = {
        "box_sliding_window": box_table(),
        "gaussian_density": gaussian_table(),
        "two_step": two_step_table(),
    }

    # Save each table as its own CSV
    for name, df in tables.items():

        if df is None:
            print(f"No results found for {name}")
            continue

        path = os.path.join(tables_dir, f"{name}.csv")
        df.to_csv(path, index = False)
        print(f"\n===== {name} ({len(df)} rows) =====")
        print(df.to_string(index = False))
        print(f"Saved: {Path(path).resolve()}")

    # Also save all tables in one Excel file, one sheet per algorithm (needs openpyxl)
    try:
        xlsx_path = os.path.join(tables_dir, "all_results.xlsx")
        with pd.ExcelWriter(xlsx_path) as writer:
            for name, df in tables.items():
                if df is not None:
                    df.to_excel(writer, sheet_name = name[:31], index = False)
        print(f"\nSaved: {Path(xlsx_path).resolve()}")
    except ImportError:
        print("\nopenpyxl not installed, so no Excel file (pip install openpyxl). The CSVs are saved.")


if __name__ == "__main__":

    main()