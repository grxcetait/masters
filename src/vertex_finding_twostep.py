#!/usr/bin/env python3

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

# Create a list of the branches we are interested in
track_branches = ["HLTTrack_pt", "HLTTrack_eta", "HLTTrack_d0", "HLTTrack_z0",
                  "HLTTrack_d0Err", "HLTTrack_z0Err", "TruthVertex_z", 
                  "TruthVertex_isValid"]

# Track selection
pt_min_values = [1, 1.5]       # [GeV] 
pt_max_values = [np.inf, 100]    # [GeV] where np.inf = None
eta_max_values = [2.5, np.inf]  # where np.inf = None
d0_max = 4.0        # [mm]
sig_d0_max = 5.0    # [mm]
sig_z0_max = 10.0   # [mm]

# Grid along the beam axis
z_min = -200.0      # [mm]
z_max = 200.0       # [mm]
step = 0.1          # [mm]

# Gaussian density settings (ATL-PHYS-PUB-2019-015 Sec. 5.1)
n_sigma = 4.0           # each track's Gaussian is added out to n_sigma
use_d0_weight = True    # include the d0 factor

# Candidate finding settings
n_cand = 30             # number of candidate vertices to find per event
min_tracks = 2          # a candidate needs at least this many tracks (as in the paper)
s_max_values = [3.0, 5.0]   # track-to-vertex compatibility cut on the significance
                            # s = sqrt((d0/sig_d0)^2 + (dz/sig_z0)^2), the AMVF uses s < 5

# Matching tolerances to report
tolerances = [0.3, 1.0]     # [mm]

# Constants
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

# ===== FUNCTIONS =====


@njit(fastmath = True)
def add_track(density, z0, sig_z0, weight, z_min, step, n_grid, n_sigma):

    # Add (or, with a negative weight, remove) one track's normalised Gaussian
    # to the density, integrated over each grid bin

    # Find the grid points within n_sigma of the track
    width = n_sigma * sig_z0
    low = int(math.ceil((z0 - width - z_min) / step))
    high = int(math.floor((z0 + width - z_min) / step))

    # Keep the indices inside the grid
    if low < 0:
        low = 0
    if high > n_grid - 1:
        high = n_grid - 1
    if high < low:
        return

    # Fraction of the Gaussian in each bin = difference of erf at the bin edges
    edge = z_min + (low - 0.5) * step
    cdf_prev = 0.5 * math.erf((edge - z0) / (SQRT2 * sig_z0))

    for i in range(low, high + 1):

        edge = z_min + (i + 0.5) * step
        cdf_next = 0.5 * math.erf((edge - z0) / (SQRT2 * sig_z0))

        density[i] += weight * (cdf_next - cdf_prev) / step

        cdf_prev = cdf_next


@njit(fastmath = True)
def d0_factor(d0, sig_d0, use_d0_weight):

    # z-independent weight from the track's d0 significance
    # (first bracket of the simplified rho(z), ATL-PHYS-PUB-2019-015 Sec. 5.1)
    if use_d0_weight:
        return math.exp(-0.5 * (d0 / sig_d0) ** 2) / (SQRT2PI * sig_d0)

    return 1.0


@njit(parallel = True, fastmath = True)
def find_candidates(z0, sig_z0, d0, sig_d0, pt, offsets, z_min, step, n_grid,
                    n_sigma, use_d0_weight, n_cand, s_max, min_tracks):

    # Find out number of events
    n_events = len(offsets) - 1

    # Create empty arrays to hold the candidates
    # Unfilled candidates are left as NaN / 0
    cand_z = np.full((n_events, n_cand), np.nan, dtype=np.float64)
    cand_height = np.zeros((n_events, n_cand), dtype=np.float64)
    cand_ntrk = np.zeros((n_events, n_cand), dtype=np.int32)
    cand_sumpt = np.zeros((n_events, n_cand), dtype=np.float64)
    cand_sumpt2 = np.zeros((n_events, n_cand), dtype=np.float64)

    # Loop through all events in parallel
    for ev in prange(n_events):

        start = offsets[ev]
        end = offsets[ev + 1]
        n_tracks = end - start

        if n_tracks == 0:
            continue

        # Build the track density once, as before
        density = np.zeros(n_grid, dtype = np.float64)
        w = np.empty(n_tracks, dtype = np.float64)

        for j in range(n_tracks):
            jj = start + j
            w[j] = d0_factor(d0[jj], sig_d0[jj], use_d0_weight)
            add_track(density, z0[jj], sig_z0[jj], w[j], z_min, step, n_grid, n_sigma)

        # Keep track of which tracks have already been given to a vertex
        assigned = np.zeros(n_tracks, dtype = np.bool_)
        n_left = n_tracks

        # Find candidates one at a time
        k = 0
        n_iter = 0
        while k < n_cand and n_left >= min_tracks and n_iter < 5 * n_cand:

            n_iter += 1

            # 1) Seed = highest point of the density of the remaining tracks
            idx = np.argmax(density)
            height = density[idx]
            if height <= 0.0:
                break
            z_est = z_min + idx * step

            # Refine the seed with the weighted average from Sec. 5.3
            # (only tracks not yet assigned to a vertex)
            for it in range(2):

                total_wz = 0.0
                total_w = 0.0

                for j in range(n_tracks):

                    if assigned[j]:
                        continue

                    jj = start + j
                    pull = (z_est - z0[jj]) / sig_z0[jj]
                    if np.abs(pull) > n_sigma:
                        continue

                    rho = w[j] * math.exp(-0.5 * pull * pull) / (SQRT2PI * sig_z0[jj])
                    ww = rho / sig_z0[jj] ** 2
                    total_wz += ww * z0[jj]
                    total_w += ww

                if total_w > 0.0:
                    z_est = total_wz / total_w

            # Assign compatible tracks to this vertex, remove them from the density,
            # and add up their pT and pT^2
            ntrk = 0
            sumpt = 0.0
            sumpt2 = 0.0

            for j in range(n_tracks):

                if assigned[j]:
                    continue

                jj = start + j

                # Track-to-vertex significance 
                dz_sig = (z0[jj] - z_est) / sig_z0[jj]
                if use_d0_weight:
                    d0_sig = d0[jj] / sig_d0[jj]
                else:
                    d0_sig = 0.0
                s = math.sqrt(dz_sig * dz_sig + d0_sig * d0_sig)

                if s < s_max:
                    assigned[j] = True
                    n_left -= 1
                    add_track(density, z0[jj], sig_z0[jj], -w[j], z_min, step, n_grid, n_sigma)
                    ntrk += 1
                    sumpt += pt[jj]
                    sumpt2 += pt[jj] * pt[jj]

            # If no track was compatible, remove the track closest to the seed
            # so the loop always makes progress (as in the paper, the pool always shrinks)
            if ntrk == 0:

                best_j = -1
                best_pull = 1e30

                for j in range(n_tracks):
                    if assigned[j]:
                        continue
                    jj = start + j
                    pull = np.abs((z0[jj] - z_est) / sig_z0[jj])
                    if pull < best_pull:
                        best_pull = pull
                        best_j = j

                if best_j < 0:
                    break

                assigned[best_j] = True
                n_left -= 1
                jj = start + best_j
                add_track(density, z0[jj], sig_z0[jj], -w[best_j], z_min, step, n_grid, n_sigma)
                continue

            # Keep the candidate if it has enough tracks
            # (its tracks stay removed either way)
            if ntrk >= min_tracks:
                cand_z[ev, k] = z_est
                cand_height[ev, k] = height
                cand_ntrk[ev, k] = ntrk
                cand_sumpt[ev, k] = sumpt
                cand_sumpt2[ev, k] = sumpt2
                k += 1

    return cand_z, cand_height, cand_ntrk, cand_sumpt, cand_sumpt2



def select_candidate(score, cand_ntrk):

    # Index of the candidate with the highest score in each event
    # (empty candidate slots can never be chosen)
    score = np.where(cand_ntrk > 0, score, -np.inf)
    return np.argmax(score, axis = 1)



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
    # (because they're squared currently )
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

                # Count how many tracks are in an event and convert into numpy array
                counts = ak.num(pt[mask]).to_numpy()
                print(f"Mean tracks per event after selection: {counts.mean():.1f}")

                # Create an array filled with zeros, 1 length more than the number of events
                offsets = np.zeros(len(counts) + 1, dtype=np.int64)

                # Calcualte the cumulative sum of the track counts 
                offsets[1:] = np.cumsum(counts)

                # Convert to numpy arrays and flatten for numba 
                flat_pt = ak.to_numpy(ak.flatten(pt[mask])).astype(np.float64)
                flat_z0 = ak.to_numpy(ak.flatten(z0[mask])).astype(np.float64)
                flat_sig_z0 = ak.to_numpy(ak.flatten(sig_z0[mask])).astype(np.float64)
                flat_d0 = ak.to_numpy(ak.flatten(d0[mask])).astype(np.float64)
                flat_sig_d0 = ak.to_numpy(ak.flatten(sig_d0[mask])).astype(np.float64)

                # Specify output subfloder
                output_subfolder = f"sem1week4/two_steps/{pt_min}ptmin_{pt_max}ptmax_{eta_max}etamax"

                # Create the output directory if it doesn't exist
                out_dir = os.path.join("outputs", output_subfolder)
                os.makedirs(out_dir, exist_ok = True)

                for s_max in s_max_values:

                    # Find the candidate vertices
                    cand_z, cand_height, cand_ntrk, cand_sumpt, cand_sumpt2 = find_candidates(
                        z0 = flat_z0, sig_z0 = flat_sig_z0, d0 = flat_d0, sig_d0 = flat_sig_d0,
                        pt = flat_pt, offsets = offsets, z_min = z_min, step = step, n_grid = n_grid,
                        n_sigma = n_sigma, use_d0_weight = use_d0_weight, n_cand = n_cand,
                        s_max = s_max, min_tracks = min_tracks)

                    # Distance from every candidate to the truth (empty slots = infinitely far)
                    dist = np.abs(cand_z - truth_z[:, None])
                    dist = np.where(np.isnan(dist), np.inf, dist)

                    n_found = (cand_ntrk > 0).sum(axis = 1)
                    print(f"\ns < {s_max}: mean candidates found per event = {n_found.mean():.1f}")

                    # Choose one candidate in four different ways
                    selectors = {
                        "densest": cand_height,    
                        "n_tracks": cand_ntrk.astype(np.float64),
                        "sum_pt": cand_sumpt,
                        "sum_pt2": cand_sumpt2,
                    }

                    rows = np.arange(n_events)

                    for tol in tolerances:

                        # Reconstruction efficiency: is the truth close to ANY candidate?
                        reco = (dist < tol).any(axis = 1)
                        reco_eff = np.mean(reco[truth_valid])
                        print(f"  tol = {tol} mm: reconstruction efficiency (truth among candidates) = {reco_eff * 100:5.2f} %")

                        # Selection efficiency for each way of choosing
                        for name, score in selectors.items():
                            chosen = select_candidate(score, cand_ntrk)
                            sel = dist[rows, chosen] < tol
                            sel_eff = np.mean(sel[truth_valid])
                            cond_eff = np.mean(sel[truth_valid & reco]) if np.any(truth_valid & reco) else 0.0
                            print(f"    choose by {name:8}: efficiency = {sel_eff * 100:5.2f} %"
                                f"  (chosen correctly in {cond_eff * 100:5.2f} % of events where it was found)")

                    # Per-event results at the main tolerance
                    tol = tolerances[0]
                    chosen_pt2 = select_candidate(cand_sumpt2, cand_ntrk)
                    chosen_dense = select_candidate(cand_height, cand_ntrk)

                    # Which candidate is the truth, and where does it rank by sum pT^2?
                    truth_idx = np.argmin(dist, axis = 1)
                    truth_found = dist[rows, truth_idx] < tol
                    truth_sumpt2 = cand_sumpt2[rows, truth_idx]
                    truth_rank_pt2 = np.where(truth_found,
                                            1 + np.sum(np.where(cand_ntrk > 0, cand_sumpt2, -np.inf)
                                                        > truth_sumpt2[:, None], axis = 1),
                                            -1)

                    results_df = pd.DataFrame(
                        {
                            "event_id": np.arange(n_events),
                            "truth_z": truth_z,
                            "truth_valid": truth_valid,
                            "n_tracks_passed": counts,
                            "n_candidates": n_found,
                            "truth_found": truth_found,
                            "truth_candidate_index": np.where(truth_found, truth_idx, -1),
                            "truth_rank_by_sum_pt2": truth_rank_pt2,
                            "densest_z": cand_z[rows, chosen_dense],
                            "densest_dz": cand_z[rows, chosen_dense] - truth_z,
                            "sum_pt2_z": cand_z[rows, chosen_pt2],
                            "sum_pt2_dz": cand_z[rows, chosen_pt2] - truth_z,
                            "sum_pt2_chosen": cand_sumpt2[rows, chosen_pt2],
                            "sum_pt2_truth": np.where(truth_found, truth_sumpt2, np.nan),
                        }
                    )

                    filepath = os.path.join(out_dir, f"vertex_candidates_s{s_max}.csv")
                    results_df.to_csv(filepath, index = False)
                    print(f"  Saved: {Path(filepath).resolve()}")

                    # Save every candidate too, for plots later (e.g. sum pT^2 of truth vs pile-up candidates)
                    npz_path = os.path.join(out_dir, f"vertex_candidates_s{s_max}.npz")
                    np.savez_compressed(npz_path, cand_z = cand_z, cand_height = cand_height,
                                        cand_ntrk = cand_ntrk, cand_sumpt = cand_sumpt,
                                        cand_sumpt2 = cand_sumpt2, truth_z = truth_z,
                                        truth_valid = truth_valid)
                    print(f"  Saved: {Path(npz_path).resolve()}")


if __name__ == "__main__":
     
    main()