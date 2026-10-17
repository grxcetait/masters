#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import os
import math

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

# Each track's Gaussian is only added out to n_sigma standard deviations
# (beyond 4 sigma the Gaussian is < 0.04 % of its peak, so this is just for speed)
n_sigma = 4.0

# Values to scan
tolerances = [0.3, 1.0]                      # [mm] matching tolerances to report

use_d0_weight_values = [False, True] # include the d0 factor from ATL-PHYS-PUB-2019-015 Sec. 5.1
#normalise_values = [True, False]     # True  = each track's Gaussian has area 1 (as in the paper)
                                     # False = each track's Gaussian has peak height 1 (like the box count)
normalise_values = [True]
 
# Constants
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

# ===== FUNCTIONS =====


@njit(parallel = True, fastmath = True)
def iterate_events_gaussian(z0, sig_z0, d0, sig_d0, offsets, truth_z_arr,
                            z_min, step, n_grid, tol, n_sigma, use_d0_weight, normalise):

    # Find out number of events and length of grid
    n_events = len(offsets) - 1

    # Create empty arrays to hold data
    # Events with no useable tracks are left as NaN rather than 0
    # So they can't be mistaken for a real vertex at z=0
    seed_z = np.full(n_events, np.nan, dtype=np.float32)
    lead_z = np.full(n_events, np.nan, dtype=np.float32)
    lead_dz = np.full(n_events, np.nan, dtype=np.float32)
    lead_density = np.zeros(n_events, dtype=np.float32)
    matched = np.zeros(n_events, dtype=np.int32)

    # Loop through all events in parallel
    for ev in prange(n_events):

        start = offsets[ev]
        end = offsets[ev + 1]
        n_tracks = end - start

        if n_tracks == 0:
            continue

        # Create an array to hold the track density at each grid point
        density = np.zeros(n_grid, dtype = np.float32)
    
        # Iterate over the grid points in parallel 
        # To every grid point within the resolution window
        for j in range(start, end):

            # d0 factor: z-independent weight from the track's d0 significance
            # Without normalisation it is just the d0 compatibility, between 0 and 1
            if use_d0_weight:
                w_d0 = math.exp(-0.5 * (d0[j] / sig_d0[j]) ** 2)
                if normalise:
                    w_d0 = w_d0 / (SQRT2PI * sig_d0[j])
            else:
                w_d0 = 1.0
 
            # Normalised: area 1, so the peak height is 1 / (sqrt(2 pi) sig_z0)
            # Not normalised: multiply by sqrt(2 pi) sig_z0 so the peak height is 1
            if normalise:
                scale = 1.0
            else:
                scale = SQRT2PI * sig_z0[j]

            # Find the grid points within n_sigma of the track
            width = n_sigma * sig_z0[j]
            low = int(np.ceil((z0[j] - width - z_min) / step))
            high = int(np.floor((z0[j] + width - z_min) / step))
    
            # Keep the indices inside the grid
            if low < 0:
                low = 0
            if high > n_grid - 1:
                high = n_grid - 1

            # Integrate the Gaussian over each grid bin rather than sampling it at
            # the bin centre, so very narrow tracks (sig_z0 << step) are not missed.
            # The integral of a Gaussian between two edges is a difference of erf values.
            edge = z_min + (low - 0.5) * step
            cdf_prev = 0.5 * math.erf((edge - z0[j]) / (SQRT2 * sig_z0[j]))
 
            for i in range(low, high + 1):
 
                edge = z_min + (i + 0.5) * step
                cdf_next = 0.5 * math.erf((edge - z0[j]) / (SQRT2 * sig_z0[j]))
 
                # Fraction of the track's Gaussian in this bin, divided by the bin width
                # (second bracket of rho(z), averaged over the bin)
                density[i] += w_d0 * scale * (cdf_next - cdf_prev) / step
 
                cdf_prev = cdf_next

        # Seed = grid point with the highest track density
        idx = np.argmax(density)
        seed_z[ev] = z_min + idx * step
        lead_density[ev] = density[idx]
 
        # Refine the seed with the weighted average
        # (correlation term dropped): each track is weighted by its density at the
        # current estimate divided by sig_z0^2. Two passes, as in the paper's peak search.
        z_est = seed_z[ev]
        for it in range(2):
 
            total_wz = 0.0
            total_w = 0.0
 
            for j in range(start, end):
 
                # Only tracks within n_sigma of the current estimate contribute
                pull = (z_est - z0[j]) / sig_z0[j]
                if np.abs(pull) > n_sigma:
                    continue
 
                if use_d0_weight:
                    w_d0 = math.exp(-0.5 * (d0[j] / sig_d0[j]) ** 2) / (SQRT2PI * sig_d0[j])
                else:
                    w_d0 = 1.0
 
                # Track density at the current estimate (always the paper's normalised form here,
                # so precisely measured tracks dominate the position, which is what we want)
                rho = w_d0 * math.exp(-0.5 * pull ** 2) / (SQRT2PI * sig_z0[j])
 
                # Weight for the average
                w = rho / sig_z0[j] ** 2
                total_wz += w * z0[j]
                total_w += w
 
            if total_w > 0:
                z_est = total_wz / total_w
 
        lead_z[ev] = z_est
 
        # Distance between the refined lead and the truth
        lead_dz[ev] = lead_z[ev] - truth_z_arr[ev]
 
        # If the distance between the lead and true z is within the given tolerance
        # It is considered a match
        if np.abs(lead_dz[ev]) < tol:
            matched[ev] = 1
 
    return seed_z, lead_z, lead_dz, lead_density, matched



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
                filtered_d0 = d0[mask]
                filtered_sig_z0 = sig_z0[mask]
                filtered_sig_d0 = sig_d0[mask]

                # Count how many tracks are in an event and convert into numpy array
                counts = ak.num(filtered_pt).to_numpy()

                # Create an array filled with zeros, 1 length more than the number of events
                offsets = np.zeros(len(counts) + 1, dtype=np.int64)

                # Calcualte the cumulative sum of the track counts 
                offsets[1:] = np.cumsum(counts)

                # Concert to numpy arrays and flatten for numba 
                flat_pt = ak.to_numpy(ak.flatten(filtered_pt)).astype(np.float32)
                flat_z0 = ak.to_numpy(ak.flatten(filtered_z0)).astype(np.float32)
                flat_d0 = ak.to_numpy(ak.flatten(filtered_d0)).astype(np.float32)
                flat_sig_z0 = ak.to_numpy(ak.flatten(filtered_sig_z0)).astype(np.float32)
                flat_sig_d0 = ak.to_numpy(ak.flatten(filtered_sig_d0)).astype(np.float32)

                # Empty list to hold data
                scan_rows = []
                best = None

                # Specify output subfloder
                output_subfolder = f"sem1week4/gaussian_weighting/{pt_min}ptmin_{pt_max}ptmax_{eta_max}etamax"

                # Create the output directory if it doesn't exist
                out_dir = os.path.join("outputs", output_subfolder)
                os.makedirs(out_dir, exist_ok = True)

                # Run this for (not)using d0 weight and normalising values
                for use_d0_weight in use_d0_weight_values:
                        for normalise in normalise_values:
            
                            # Find the primary vertices 
                            seed_z, lead_z, lead_dz, lead_density, matched = iterate_events_gaussian(
                                z0 = flat_z0, sig_z0 = flat_sig_z0, d0 = flat_d0, sig_d0 = flat_sig_d0,
                                offsets = offsets, truth_z_arr = truth_z, z_min = z_min, step = step,
                                n_grid = n_grid, tol = tolerances[0], n_sigma = n_sigma,
                                use_d0_weight = use_d0_weight, normalise = normalise)

                            # Only use events where the truth vertex is valid
                            abs_dz = np.abs(lead_dz[truth_valid])

                            # Efficiency at each tolerance
                            row = {"eta_max": eta_max, "use_d0_weight": use_d0_weight,
                                        "normalise": normalise, "mean_tracks": counts.mean()}
                            for tol in tolerances:
                                row[f"eff_tol{tol}"] = np.mean(abs_dz < tol)
                            row["median_abs_dz"] = np.nanmedian(abs_dz)
                            scan_rows.append(row)

                            print(f"|eta| < {eta_max}, d0 weight = {use_d0_weight!s:5}, normalised = {normalise!s:5}, "
                                    f"{counts.mean():6.1f} tracks: "
                                    + ", ".join(f"eff(tol={tol}) = {row[f'eff_tol{tol}'] * 100:5.2f} %" for tol in tolerances)
                                    + f", median |dz| = {row['median_abs_dz']:.3f} mm")

                            # Save the per-event results for this configuration
                            results_df = pd.DataFrame(
                                {
                                    "event_id": np.arange(n_events),
                                    "truth_z": truth_z,
                                    "truth_valid": truth_valid,
                                    "n_tracks_passed": counts,
                                    "seed_z": seed_z,
                                    "lead_z_refined": lead_z,
                                    "delta_z": lead_dz,
                                    "lead_density": lead_density,
                                    "is_matched": matched,
                                }
                            )
    
                            filepath = os.path.join(out_dir, f"vertex_results_d0{use_d0_weight}_norm{normalise}.csv")
                            results_df.to_csv(filepath, index = False)
                            print(f"Saved: {Path(filepath).resolve()}")

                            # Keep the best configuration at the main tolerance
                            if best is None or row[f"eff_tol{tolerances[0]}"] > best[f"eff_tol{tolerances[0]}"]:
                                best = row

                # Save the scan summary
                scan_df = pd.DataFrame(scan_rows)
                scan_path = os.path.join(out_dir, "gaussian_scan_summary.csv")
                scan_df.to_csv(scan_path, index = False)
                print(f"Saved: {Path(scan_path).resolve()}")
            
                # Save the per-event results for the best configuration
                print(f"Best: |eta| < {best['eta_max']}, d0 weight = {best['use_d0_weight']}, "
                    f"normalised = {best['normalise']}, "
                    f"eff = {best[f'eff_tol{tolerances[0]}'] * 100:.2f} %")


if __name__ == "__main__":
     
    main()

    