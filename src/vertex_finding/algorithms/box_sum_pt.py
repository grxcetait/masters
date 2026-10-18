#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import os

import pandas as pd

import awkward as ak
import numpy as np
import uproot

from numba import njit, prange

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# === ARGUEMENTS ====
filename = "CERNbox/initial_data/root_files/HHbbyy_200PU.root"
tree_name = "Events"
prefix = "HLTTrack_"

# Create a list of the branches we are interested in
track_branches = ["HLTTrack_pt", "HLTTrack_eta", "HLTTrack_d0", "HLTTrack_z0",
                  "HLTTrack_d0Err", "HLTTrack_z0Err", "TruthVertex_z", 
                  "TruthVertex_isValid"]

# Track selection (Run 1 vertex selection: |pt| > 0.5 GeV, |d0| < 4 mm, sig_d0 < 5 mm, sig_z0 < 10 mm)
pt_min_values = [1, 1.5]       # [GeV] 
pt_max_values = [np.inf, 100]    # [GeV] where np.inf = None
eta_max_values = [2.5, np.inf]  # where np.inf = None
d0_max = 4.0        # [mm]
sig_d0_max = 5.0    # [mm]
sig_z0_max = 10.0   # [mm]

# Grid along the beam axis
z_min = -200.0      # [mm]
z_max = 200.0       # [mm]
step = 0.1          # [mm] grid spacing, keep this smaller than res and tol

# Values to scan
res_values = [0.1, 0.25, 0.5, 1.0, 2.0]     # [mm] half-width of the window each track votes in
powers = [0, 1, 2]                           # weight = pt ** power (where 0 = track count)
tolerances = [0.3, 1.0]                      # [mm] matching tolerances to report

# ===== FUNCTIONS =====


@njit(parallel = True, fastmath = True)
def iterate_events_number(z0, pt, offsets, truth_z_arr, z_min, step, n_grid, tol, res, power):

    # Find out number of events and length of grid
    n_events = len(offsets) - 1

    # Create empty arrays to hold data
    # Events with no useable tracks are left as NaN rather than 0
    # So they can't be mistaken for a real vertex at z=0
    top5_z = np.full((n_events, 5), np.nan, dtype=np.float32)
    top5_w = np.zeros((n_events, 5), dtype=np.float32)
    top5_dz = np.full((n_events, 5), np.nan, dtype=np.float32)
    lead_z = np.full(n_events, np.nan, dtype=np.float32)
    lead_dz = np.full(n_events, np.nan, dtype=np.float32)
    matched = np.zeros(n_events, dtype=np.int32)

    # Loop through all events in parallel
    for ev in prange(n_events):

        start = offsets[ev]
        end = offsets[ev + 1]
        n_tracks = end - start

        if n_tracks == 0:
            continue

        # Create an array to hold summed weight at each grid point
        sum_w = np.zeros(n_grid, dtype = np.float64)
    
        # Iterate over the grid points in parallel 
        # And add each track's weight
        # To every grid point within the resolution window
        for j in range(start, end):

            # Weight of the track (power 0 = count, 1 = pt, 2 = pt^2)
            w = pt[j] ** power

            # Find the first and last grid points covered by this track
            low = int(np.ceil((z0[j] - res - z_min) / step))
            high = int(np.floor((z0[j] + res - z_min) / step))
    
            # Keep the indices inside the grid
            if low < 0:
                low = 0
            if high > n_grid - 1:
                high = n_grid - 1

            for i in range(low, high + 1):
                sum_w[i] += w

        # Obtain the true vertex value
        truth_z = truth_z_arr[ev]

        # Find top 5 candidates
        sorted_indices = np.argsort(sum_w)[::-1]
        for rank in range(5):
            idx = sorted_indices[rank]
            top5_z[ev, rank] = z_min + idx * step
            top5_w[ev, rank] = sum_w[idx]
            top5_dz[ev, rank] = top5_z[ev, rank] - truth_z

        # Find the lead position in case there are tracks clumped together in its window
        # This gives a position finer than the grid spacing
        z_peak = top5_z[ev, 0]
        total_z = 0.0
        n_in = 0

        # Within the window of z_peak, add up all the tracks within 
        for j in range(start, end):
            if np.abs(z0[j] - z_peak) <= res:
                total_z += z0[j]
                n_in += 1

        # Obtain the average z
        if n_in > 0:
            lead_z[ev] = total_z / n_in

        # If there are no tracks in the window, just give z_peak
        else:
            lead_z[ev] = z_peak

        # Calculate dz from this new z position
        lead_dz[ev] = lead_z[ev] - truth_z
 
        # If the distance between the lead and true z is within the given tolerance
        # It is considered a match
        if np.abs(lead_dz[ev]) < tol:
            matched[ev] = 1

    return top5_z, top5_w, top5_dz, lead_z, lead_dz, matched



def main():

    # Number of grid points
    n_grid = int(round((z_max - z_min) / step)) + 1

    # Load in all events from the root file
    with uproot.open(f"{filename}:{tree_name}") as tree:
        events = tree.arrays(track_branches)

    # Calculate the number of events
    n_events = len(events)

    # Extract variables 
    pt = events["HLTTrack_pt"]
    eta = events["HLTTrack_eta"]
    d0 = events["HLTTrack_d0"]
    z0 = events["HLTTrack_z0"]

    # Convert variances to normal uncertainties 
    # (because they're squared currently )
    sig_d0 = np.sqrt(np.maximum(events["HLTTrack_d0Err"], 0.0))
    sig_z0 = np.sqrt(np.maximum(events["HLTTrack_z0Err"], 0.0))

    # Truth information
    truth_z = ak.to_numpy(events["TruthVertex_z"]).astype(np.float64)
    truth_valid = ak.to_numpy(events["TruthVertex_isValid"]).astype(np.bool_)

    for pt_min in pt_min_values:
        for pt_max in pt_max_values:
            for eta_max in eta_max_values:

                # Create a Boolean mask with the track selection criteria
                mask = ((pt > pt_min)
                        & (pt < pt_max)
                        & (np.abs(eta) < eta_max)
                        & (np.abs(d0) < d0_max)
                        & (sig_d0 > 0.0)
                        & (sig_d0 < sig_d0_max)
                        & (sig_z0 > 0.0)
                        & (sig_z0 < sig_z0_max))

                # Filter arrays 
                filtered_pt = pt[mask]
                filtered_z0 = z0[mask]

                # Count how many tracks are in an event and convert into numpy array
                counts = ak.num(filtered_pt).to_numpy()

                # Create an array filled with zeros, 1 length more than the number of events
                offsets = np.zeros(len(counts) + 1, dtype=np.int64)

                # Calcualte the cumulative sum of the track counts 
                offsets[1:] = np.cumsum(counts)

                # Concert to numpy arrays and flatten for numba 
                flat_pt = ak.to_numpy(ak.flatten(filtered_pt)).astype(np.float32)
                flat_z0 = ak.to_numpy(ak.flatten(filtered_z0)).astype(np.float32)

                # Empty list to hold data
                scan_rows = []
                best = None

                # Specify output subfloder
                output_subfolder = f"sem1week4/sumpt/{pt_min}ptmin_{pt_max}ptmax_{eta_max}etamax"

                # Create the output directory if it doesn't exist
                out_dir = os.path.join("outputs", output_subfolder)
                os.makedirs(out_dir, exist_ok = True)

                # Go through each resolution value and each power 
                for res in res_values:
                    for power in powers:
            
                        # Find the primary vertices 
                        top5_z, top5_w, top5_dz, lead_z, lead_dz, matched = iterate_events_number(
                            z0 = flat_z0, pt = flat_pt, offsets = offsets, truth_z_arr = truth_z,
                            z_min = z_min, step = step, n_grid = n_grid, tol = tolerances[0],
                            res = res, power = float(power))
            
                        # Only use events where the truth vertex is valid
                        abs_dz = np.abs(lead_dz[truth_valid])
            
                        # Efficiency at each tolerance
                        row = {"res": res, "power": power}
                        for tol in tolerances:
                            row[f"eff_tol{tol}"] = np.mean(abs_dz < tol)
            
                        # Median distance from the truth, as a resolution check
                        row["median_abs_dz"] = np.nanmedian(abs_dz)
                        scan_rows.append(row)

                        # Print the results for this configuration
                        print(f"res = {res:4.2f} mm, pt^{power}: "
                            + ", ".join(f"eff(tol={tol}) = {row[f'eff_tol{tol}'] * 100:5.2f} %" for tol in tolerances)
                            + f", median |dz| = {row['median_abs_dz']:.3f} mm")

                        # Save the per-event results for this configuration
                        results_df = pd.DataFrame(
                            {
                                "event_id": np.arange(n_events),
                                "truth_z": truth_z,
                                "truth_valid": truth_valid,
                                "n_tracks_passed": counts,
                                "lead_z_refined": lead_z,
                                "delta_z": lead_dz,
                                "is_matched": matched,
                                "lead_seed_z": top5_z[:, 0],
                                "lead_sum_w": top5_w[:, 0],
                            }
                        )

                        filepath = os.path.join(out_dir, f"vertex_results_res{row['res']}_pt{row['power']}.csv")
                        results_df.to_csv(filepath, index = False)
                        print(f"Saved: {Path(filepath).resolve()}")
            
                        # Keep the best configuration at the main tolerance
                        if best is None or row[f"eff_tol{tolerances[0]}"] > best[f"eff_tol{tolerances[0]}"]:
                            best = row

                # Save the scan summary
                scan_df = pd.DataFrame(scan_rows)
                scan_path = os.path.join(out_dir, "vertex_scan_summary.csv")
                scan_df.to_csv(scan_path, index = False)
                print(f"Saved: {Path(scan_path).resolve()}")

                # Print the best configuration
                print(f"Best: res = {best['res']} mm, pt^{best['power']}, "
                    f"eff = {best[f'eff_tol{tolerances[0]}'] * 100:.2f} %")


if __name__ == "__main__":
     
    main()

    