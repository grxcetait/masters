#!/usr/bin/env python3

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# === ARGUEMENTS ===
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
#weight_names = {0: "track count", 1: "pT", 2: "pT^2"}
#weight_names = {0: "track count"}
weight_names = {1: "pT"}

# Column names for the cuts
cut_columns = ["pT min (GeV)", "pT max (GeV)", "|eta| max"]

# Only read folders with these cuts (np.inf = no cut); leave a cut out to keep all its values
fixed_cuts = {"pT max (GeV)": np.inf, "|eta| max": np.inf}

# Output file name -> "box_sum_pt_{output_file_name}.csv"
#output_file_name = "all_cuts"
#output_file_name = "all_weight_pt_min"
#output_file_name = "track_count"
output_file_name = "pT"


# === FUNCTIONS ===

def find_folders(algorithm_dir):

    # Empty list to hold folders
    folders = []

    # Check if folder is found
    if not os.path.isdir(algorithm_dir):
        print(f"Not found, skipping: {algorithm_dir}")
        return folders

    # Find every subfolder whose name records the cuts
    # Read the cuts from the name
    for name in sorted(os.listdir(algorithm_dir)):

        match = folder_pattern.match(name)
        if match is None:
            continue

        cuts = {"pT min (GeV)": float(match.group(1)),
                "pT max (GeV)": float(match.group(2)),
                "|eta| max": float(match.group(3))}

        # Skip folders whose cuts don't match the fixed ones
        if any(cuts[col] != value for col, value in fixed_cuts.items()):
            continue

        folders.append((cuts, os.path.join(algorithm_dir, name)))

    return folders

def finalise_table(df, sort_columns):

    # Sort the rows
    df = df.sort_values(sort_columns).reset_index(drop = True)

    # Write infinite cuts as "none"
    for col in cut_columns:
        df[col] = df[col].apply(lambda x: "none" if np.isinf(x) else f"{x:g}")

    # Drop the cut columns that are fixed, since every row has the same value
    df = df.drop(columns = [col for col in fixed_cuts if col in df.columns])

    # Round the numbers
    return df.round(3)

def box_sum_pt_table():

    # Empty list to hold rows
    rows = []

    # Read every vertex_scan_summary.csv from the box sum pt runs 
    for cuts, folder in find_folders(sumpt_dir):

        path = os.path.join(folder, "vertex_scan_summary.csv")
        if not os.path.isfile(path):
            continue

        # Read dataframe
        summary = pd.read_csv(path)

        # Loop through rows and obtain data
        for _, r in summary.iterrows():
    
            # Only keep the weights listed in weight_names
            if int(r["power"]) not in weight_names:
                continue

            row = dict(cuts)
            row["Weight"] = weight_names.get(int(r["power"]), f"pT^{int(r['power'])}")
            row["res (mm)"] = r["res"]
            for tol in tolerances:
                row[f"Eff tol {tol} (%)"] = 100 * r[f"eff_tol{tol}"]
            row["Median |dz| (mm)"] = r["median_abs_dz"]
            rows.append(row)

    if not rows:
        return None

    # Turn into a data frame 
    df = pd.DataFrame(rows)

    # Keep the weights in the order count, pT, pT^2 
    df["_weight_order"] = df["Weight"].map({v: k for k, v in weight_names.items()})
    df = finalise_table(df, cut_columns + ["_weight_order", "res (mm)"])

    # Return the final df without the weight ordering column
    return df.drop(columns = "_weight_order")

def gaussian_weighting_table():

    # Empty list to hold rows
    rows = []

    # Read every gaussian_scan_summary.csv from the gaussian weighting runs 
    for cuts, folder in find_folders(gaussian_dir):

        path = os.path.join(folder, "gaussian_scan_summary.csv")
        if not os.path.isfile(path):
            continue

        # Read dataframe
        summary = pd.read_csv(path)

        # Loop through rows and obtain data 
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

    # Turn into a data frame
    df = pd.DataFrame(rows)

    # Finishing touches on the table
    return finalise_table(df, cut_columns + ["d0 factor", "Normalised"])


def select_candidate(score, cand_ntrk):

    # Index of the candidate with the highest score in each event
    # (empty candidate slots can never be chosen)
    score = np.where(cand_ntrk > 0, score, -np.inf)
    return np.argmax(score, axis = 1)


def two_steps_table():

    # Recompute the two-step results from the saved candidates in each .npz file
    rows = []

    for cuts, folder in find_folders(two_step_dir):

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
                #"Densest": data["cand_height"],
                #"Track count": cand_ntrk.astype(np.float64),
                "Sum pT": data["cand_sumpt"],
                #"Sum pT^2": data["cand_sumpt2"],
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

    # Turn into dataframe
    df = pd.DataFrame(rows)

    # Finishing touches on the table
    return finalise_table(df, ["tol (mm)"] + cut_columns + ["s max"])

def main():

    # Create the output directory if it doesn't exist
    os.makedirs(tables_dir, exist_ok = True)

    tables = {
        "box_sum_pt": box_sum_pt_table(),
        "gaussian_weighting": gaussian_weighting_table(),
        "two_steps": two_steps_table()
    }

    # Save each table as its own CSV
    for name, df in tables.items():

        if df is None:
            print(f"No results found for {name}")
            continue

        path = os.path.join(tables_dir, f"{name}_{output_file_name}.csv") 
        df.to_csv(path, index = False)
        print(f"Saved: {Path(path).resolve()}")


if __name__ == "__main__":

    main()