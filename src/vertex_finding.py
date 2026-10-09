#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import os

import ROOT as root
import pandas as pd

import awkward as ak
import numpy as np
import uproot

import matplotlib.pyplot as plt

from numba import njit, prange
from tqdm import tqdm

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import branches function from plot_ttree_tracks.py
import utils.navigating_ttrees as nt

# === ARGUEMENTS ====
filename = "CERNbox/initial_data/root_files/HHbbyy_200PU.root"
tree_name = "Events"
prefix = "HLTTrack_"

# Create a list of the branches we are interested in
track_branches = ["HLTTrack_pt", "HLTTrack_eta", "HLTTrack_d0", "HLTTrack_z0",
                  "HLTTrack_d0Err", "HLTTrack_z0Err", "TruthVertex_z", 
                  "TruthVertex_isValid"]

# ===== FUNCTIONS =====

@njit(parallel = True, fastmath = True)
def iterate_z_axis_numba(z0, pt, z_min, z_max, steps, tol):

    # Create an empty grid to hold sum_pt values
    z_grid = np.arange(-200, 200.1, 1.0, dtype = np.float32)

    # Obtain lengths
    n_tracks = len(z0)
    n_grid = len(z_grid)

    # Create a grid to hold sum pt values
    sum_pt = np.zeros(n_grid, dtype = np.float32)

    # Iterate over the grid points in parallel 
    for i in prange(n_grid):

        # Obtain the z_grid value
        z_val = z_grid[i]

        # Set the total value as 0 
        total = 0.0

        # Iterate through the tracks
        for j in range(n_tracks):

            # Calculate the distance between z_trial and z0
            dz = np.abs(z_val - z0[j])

            # Check if the value is within a tolerance
            # (Check if the track is consistent with coming from that position)
            if dz <= tol:

                total += pt[j]

        # Add the total to the 
        sum_pt[i] = total

    return sum_pt


@njit(parallel = True, fastmath = True)
def iterate_events_number(z0, pt, offsets, truth_z_arr, z_grid, tol):

    # Find out number of events and length of grid
    n_events = len(offsets) - 1
    n_grid = len(z_grid)

    # Create empty arrays to hold data
    top5_z = np.zeros((n_events, 5), dtype=np.float32)
    top5_pt = np.zeros((n_events, 5), dtype=np.float32)
    top5_dz = np.zeros((n_events, 5), dtype=np.float32)
    delta_z = np.full(n_events, np.nan, dtype=np.float32)
    matched = np.zeros(n_events, dtype=np.int32)

    # Loop through all events in parallel
    for ev in prange(n_events):

        start = offsets[ev]
        end = offsets[ev + 1]
        n_tracks = end - start

        if n_tracks == 0:
            continue

        # Create an array to hold sum pt values
        sum_pt = np.zeros(n_grid, dtype = np.float32)
    
        # Iterate over the grid points in parallel 
        for i in prange(n_grid):
    
            # Obtain the z_grid value
            z_trial = z_grid[i]
    
            # Set the total value as 0 
            total = 0.0
    
            # Iterate through the tracks
            for j in range(start, end):
    
                # Calculate the distance between z_trial and z0
                dz = np.abs(z_trial - z0[j])
    
                # Check if the value is within a tolerance
                # (Check if the track is consistent with coming from that position)
                if dz <= tol:
    
                    total += pt[j]
    
            # Add the total to the 
            sum_pt[i] = total

        # Obtain the true vertex value
        truth_z = truth_z_arr[ev]

        # Find top 5 candidates
        sorted_indices = np.argsort(sum_pt)[::-1]
        for rank in range(5):
            idx = sorted_indices[rank]
            top5_z[ev, rank] = z_grid[idx]
            top5_pt[ev, rank] = sum_pt[idx]
            top5_dz[ev, rank] = top5_z[ev, rank] - truth_z

        # If the distance between the top and true z is within the given tolerance
        # It is considered a match
        if np.abs(top5_dz[ev, 0]) < tol:
            matched[ev] = 1

    return top5_z, top5_pt, top5_dz, matched



def main():

    # Create an empty grid of all z_grid values
    z_grid = np.arange(-200, 200.1, 1.0, dtype = np.float32)

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

    # Create a Boolean mask with the track selection criteria
    mask = ((pt > 1.5)      # GeV
            & (np.abs(eta) < 2.5)
            & (np.abs(d0) < 4.0)
            & (sig_d0 < 5.0)
            & (sig_z0 < 10.0))

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
    truth_z = ak.to_numpy(events["TruthVertex_z"]).astype(np.float32)
    truth_valid = ak.to_numpy(events["TruthVertex_isValid"]).astype(np.bool_)

    # Find the primary vertices 
    top5_z, top5_pt, top5_dz, matched = iterate_events_number(z0 = flat_z0, pt = flat_pt, offsets = offsets,
                                                              truth_z_arr = truth_z, z_grid = z_grid, tol = 1)

    # Add values to the dataframe
    results_df = pd.DataFrame(
        {
            "event_id": np.arange(n_events),
            "truth_z": truth_z,
            "truth_valid": truth_valid,
            "n_tracks_passed": counts,
            "lead_seed_z": top5_z[:, 0],
            "lead_sum_pt2": top5_pt[:, 0],
            "delta_z": top5_dz[:, 0],
            "is_matched": matched,
            "seed_z_rank2": top5_z[:, 1],
            "seed_z_rank3": top5_z[:, 2],
            "seed_z_rank4": top5_z[:, 3],
            "seed_z_rank5": top5_z[:, 4],
            "dz_rank2": top5_dz[:, 1],
            "dz_rank3": top5_dz[:, 2],
            "dz_rank4": top5_dz[:, 3],
            "dz_rank5": top5_dz[:, 4],
            "sum_pt2_rank2": top5_pt[:, 1],
            "sum_pt2_rank3": top5_pt[:, 2],
            "sum_pt2_rank4": top5_pt[:, 3],
            "sum_pt2_rank5": top5_pt[:, 4],
        }
    )

    # Only obtain events where the vertex is valid
    valid = results_df[results_df["truth_valid"]]

    # Check if the event is matched with the true vertex
    matched_ev = valid[valid["is_matched"] == 1]

    # Calculate efficiency
    # Number of matched events / number of total events 
    eff = len(matched_ev) / len(valid) if len(valid) > 0 else 0.0

    # Print results 
    print(f"Total Processed Events: {n_events:,}")
    print(f"Overall Efficiency: {eff * 100:.2f} %")

    # Specify output subfolder
    output_subfolder = "sem1week3"

    # Create the output directory if it doesn't exist
    out_dir = os.path.join("outputs", output_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    # Construct the full file path for saving the CSV file
    filepath = os.path.join(out_dir, f"vertex_reconstruction_results.csv")
    results_df.to_csv(filepath, index = False)
    print(f"Saved: {Path(filepath).resolve()}")


if __name__ == "__main__":
     
    main()

    