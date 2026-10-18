#!/usr/bin/env python3

import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === ARGUEMENTS ====
filename = "CERNbox/initial_data/root_files/HHbbyy_200PU.root"
tree_name = "Events"

# Create a list of the branches we are interested in
track_branches = ["HLTTrack_pt", "HLTTrack_eta", "HLTTrack_d0", "HLTTrack_z0",
                  "HLTTrack_d0Err", "HLTTrack_z0Err", "TruthVertex_z",
                  "TruthVertex_isValid"]

# Track selection (must match the runs whose results are read below)
pt_min = 1.5        # [GeV]
pt_max = np.inf     # [GeV]
eta_max = np.inf
d0_max = 4.0        # [mm]
sig_d0_max = 5.0    # [mm]
sig_z0_max = 10.0   # [mm]

# Grid along the beam axis
z_min = -200.0      # [mm]
z_max = 200.0       # [mm]
step = 0.1          # [mm]

# Gaussian density settings
n_sigma = 4.0
use_d0_weight = True

# Box (sum pT window) settings
box_res = 0.1       # [mm]
box_power = 1       # 1 = pT weighted

# Two-step settings
s_max = 3.0
n_cand = 30
min_tracks = 2

# Matching tolerance [mm]
tol = 0.3

# Saved per-event results, used to choose the events
base_dir = "outputs/sem1week4"
output_dir = "outputs/sem1week5"
selection_folder = f"{pt_min}ptmin_{pt_max}ptmax_{eta_max}etamax"
gaussian_csv = os.path.join(base_dir, "gaussian_weighting", selection_folder, "vertex_results_d0True_normTrue.csv")
box_csv = os.path.join(base_dir, "sumpt", selection_folder, f"vertex_results_res{box_res}_pt{box_power}.csv")
two_step_npz = os.path.join(base_dir, "two_steps", selection_folder, f"vertex_candidates_s{s_max}.npz")

# Which events to plot
# None = choose one automatically from that category; or give an event number, e.g. 1234
event_ids = {
    "all_correct": None,     # all three methods find the true vertex
    "all_wrong": None,       # all three methods choose a wrong vertex
    "one_correct": None,     # two methods wrong, one right
}

# For "one_correct": which method should be the right one
# (None = any; or "Gaussian", "Sum pT window", "Two-step")
one_correct_method = "Sum pT window"

# Random seed for choosing events (change it to get different examples)
seed = 2

# Half-width of the zoomed view around the truth [mm]
zoom_half_width = 3.0

# Extra space around the truth and chosen vertices in the full view [mm]
view_margin = 5.0

# Where to save the plots
out_dir = os.path.join(output_dir, "event_displays", selection_folder)

# Constants
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
erf = np.vectorize(math.erf, otypes = [float])

# Method names and colours (same colours as the dz plots)
METHODS = ["Two-step", "Gaussian", "Sum pT window"]
COLOURS = {"Two-step": "tab:green", "Gaussian": "tab:red", "Sum pT window": "tab:blue"}

# ===== CHOOSING EVENTS =====


def load_saved_results():

    # Chosen-vertex dz of each method in every event, from the saved results
    gaussian = pd.read_csv(gaussian_csv)
    box = pd.read_csv(box_csv)

    data = np.load(two_step_npz)
    cand_z = data["cand_z"]
    cand_ntrk = data["cand_ntrk"]
    truth_z = data["truth_z"]
    score = np.where(cand_ntrk > 0, data["cand_sumpt"], -np.inf)
    chosen = np.argmax(score, axis = 1)
    two_step_dz = cand_z[np.arange(len(truth_z)), chosen] - truth_z

    return pd.DataFrame({
        "truth_valid": gaussian["truth_valid"].astype(bool),
        "Gaussian": gaussian["delta_z"].to_numpy(),
        "Sum pT window": box["delta_z"].to_numpy(),
        "Two-step": two_step_dz,
    })


def choose_events(saved):

    # Classify every event and pick one event per case
    valid = saved["truth_valid"].to_numpy()
    right = {m: (np.abs(saved[m].to_numpy()) < tol) & valid for m in METHODS}
    n_right = sum(right[m].astype(int) for m in METHODS)

    categories = {
        "all_correct": valid & (n_right == 3),
        "all_wrong": valid & (n_right == 0),
        "one_correct": valid & (n_right == 1),
    }

    if one_correct_method is not None:
        categories["one_correct"] = categories["one_correct"] & right[one_correct_method]

    rng = np.random.default_rng(seed)
    chosen = {}

    for cat, mask in categories.items():
        if event_ids.get(cat) is not None:
            chosen[cat] = int(event_ids[cat])
        elif mask.any():
            chosen[cat] = int(rng.choice(np.flatnonzero(mask)))
        else:
            print(f"No events found for {cat}")

    return chosen

# ===== RERUNNING THE ALGORITHMS FOR ONE EVENT =====


def load_event(ev):

    # Read one event from the ROOT file and apply the track selection
    with uproot.open(f"{filename}:{tree_name}") as tree:
        arrays = tree.arrays(track_branches, entry_start = ev, entry_stop = ev + 1)

    event = arrays[0]
    pt = np.asarray(event["HLTTrack_pt"], dtype = np.float64)
    eta = np.asarray(event["HLTTrack_eta"], dtype = np.float64)
    d0 = np.asarray(event["HLTTrack_d0"], dtype = np.float64)
    z0 = np.asarray(event["HLTTrack_z0"], dtype = np.float64)

    # Convert variances to normal uncertainties
    sig_d0 = np.sqrt(np.maximum(np.asarray(event["HLTTrack_d0Err"], dtype = np.float64), 0.0))
    sig_z0 = np.sqrt(np.maximum(np.asarray(event["HLTTrack_z0Err"], dtype = np.float64), 0.0))

    # Same track selection as the scan scripts
    mask = ((pt > pt_min) & (pt < pt_max) & (np.abs(eta) < eta_max)
            & (np.abs(d0) < d0_max)
            & (sig_d0 > 0.0) & (sig_d0 < sig_d0_max)
            & (sig_z0 > 0.0) & (sig_z0 < sig_z0_max))

    return {"pt": pt[mask], "z0": z0[mask], "d0": d0[mask],
            "sig_z0": sig_z0[mask], "sig_d0": sig_d0[mask],
            "truth_z": float(event["TruthVertex_z"])}


def d0_factor(d0, sig_d0):

    # z-independent weight from the d0 significance (ATL-PHYS-PUB-2019-015 Sec. 5.1)
    if use_d0_weight:
        return np.exp(-0.5 * (d0 / sig_d0) ** 2) / (SQRT2PI * sig_d0)
    return np.ones_like(d0)


def add_track(density, z0, sig_z0, weight, n_grid):

    # Add (negative weight: remove) one track's Gaussian, integrated over each grid bin
    width = n_sigma * sig_z0
    low = max(int(math.ceil((z0 - width - z_min) / step)), 0)
    high = min(int(math.floor((z0 + width - z_min) / step)), n_grid - 1)
    if high < low:
        return

    edges = z_min + (np.arange(low, high + 2) - 0.5) * step
    cdf = 0.5 * erf((edges - z0) / (SQRT2 * sig_z0))
    density[low:high + 1] += weight * np.diff(cdf) / step


def track_density(ev_data, n_grid):

    # Gaussian track density W(z) of all selected tracks
    w = d0_factor(ev_data["d0"], ev_data["sig_d0"])
    density = np.zeros(n_grid)
    for j in range(len(w)):
        add_track(density, ev_data["z0"][j], ev_data["sig_z0"][j], w[j], n_grid)
    return density, w


def refine(z_est, ev_data, w, use):

    # Weighted average from ATL-PHYS-PUB-2019-015 Sec. 5.3, two passes, using tracks in "use"
    z0, sz = ev_data["z0"], ev_data["sig_z0"]
    for _ in range(2):
        pull = (z_est - z0) / sz
        ok = use & (np.abs(pull) <= n_sigma)
        rho = w[ok] * np.exp(-0.5 * pull[ok] ** 2) / (SQRT2PI * sz[ok])
        ww = rho / sz[ok] ** 2
        if ww.sum() > 0:
            z_est = np.sum(ww * z0[ok]) / ww.sum()
    return z_est


def run_gaussian(ev_data, density, w):

    # Gaussian method: highest point of the density, then refined
    seed_z = z_min + np.argmax(density) * step
    return refine(seed_z, ev_data, w, np.ones(len(w), dtype = bool))


def run_box(ev_data, n_grid):

    # Sum pT window: each track adds pT ** power to every grid point within +/- res
    z0, pt = ev_data["z0"], ev_data["pt"]
    sum_w = np.zeros(n_grid)
    for j in range(len(z0)):
        low = max(int(np.ceil((z0[j] - box_res - z_min) / step)), 0)
        high = min(int(np.floor((z0[j] + box_res - z_min) / step)), n_grid - 1)
        sum_w[low:high + 1] += pt[j] ** box_power

    # Highest grid point, refined with the mean z0 of the tracks in its window
    z_peak = z_min + np.argsort(sum_w)[::-1][0] * step
    in_window = np.abs(z0 - z_peak) <= box_res
    lead_z = z0[in_window].mean() if in_window.any() else z_peak

    return lead_z, sum_w


def run_two_step(ev_data, density_all, w, n_grid):

    # Two-step method: find candidates one at a time, then choose by sum pT
    z0, sz, d0, sd0, pt = (ev_data[k] for k in ["z0", "sig_z0", "d0", "sig_d0", "pt"])
    density = density_all.copy()
    assigned = np.zeros(len(z0), dtype = bool)
    candidates = []
    n_iter = 0

    while len(candidates) < n_cand and (~assigned).sum() >= min_tracks and n_iter < 5 * n_cand:

        n_iter += 1

        # Seed = highest point of the density of the remaining tracks, then refined
        idx = np.argmax(density)
        height = density[idx]
        if height <= 0.0:
            break
        z_est = refine(z_min + idx * step, ev_data, w, ~assigned)

        # Assign compatible tracks (significance s < s_max) and remove them from the density
        d0_sig = d0 / sd0 if use_d0_weight else np.zeros_like(d0)
        s = np.sqrt(((z0 - z_est) / sz) ** 2 + d0_sig ** 2)
        new = (~assigned) & (s < s_max)

        if not new.any():
            # Remove the closest remaining track so the loop always makes progress
            pull = np.where(assigned, np.inf, np.abs((z0 - z_est) / sz))
            j = int(np.argmin(pull))
            assigned[j] = True
            add_track(density, z0[j], sz[j], -w[j], n_grid)
            continue

        for j in np.flatnonzero(new):
            add_track(density, z0[j], sz[j], -w[j], n_grid)
        assigned |= new

        if new.sum() >= min_tracks:
            candidates.append({"z": z_est, "height": height, "ntrk": int(new.sum()),
                               "sumpt": float(pt[new].sum())})

    cands = pd.DataFrame(candidates)
    chosen_z = cands.loc[cands["sumpt"].idxmax(), "z"] if len(cands) else np.nan

    return chosen_z, cands

# ===== PLOTTING =====


def make_plot(ev, category, ev_data, results, saved_row):

    n_grid = int(round((z_max - z_min) / step)) + 1
    z_grid = z_min + np.arange(n_grid) * step
    truth = ev_data["truth_z"]

    density = results["density"]
    sum_w = results["sum_w"]
    cands = results["cands"]
    chosen = {m: results[m] for m in METHODS}

    # Full view: covers the truth and every chosen vertex; zoomed view: around the truth
    points = [truth] + [z for z in chosen.values() if np.isfinite(z)]
    lo, hi = min(points) - view_margin, max(points) + view_margin
    views = [(max(lo, z_min), min(hi, z_max)),
             (truth - zoom_half_width, truth + zoom_half_width)]

    fig, axes = plt.subplots(4, 2, figsize = (14, 9), sharex = "col",
                             gridspec_kw = {"width_ratios": [2, 1]})

    for col, (x_lo, x_hi) in enumerate(views):

        # Two-step: track density (grey) with each candidate's sum pT as a stem
        ax = axes[0, col]
        ax.plot(z_grid, density / density.max(), color = "lightgrey", lw = 1, label = "Track density")
        if len(cands):
            heights = cands["sumpt"] / cands["sumpt"].max()
            ax.vlines(cands["z"], 0, heights, color = COLOURS["Two-step"], alpha = 0.5, lw = 1.5,
                      label = "Candidate sum pT")
            best = cands["sumpt"].idxmax()
            ax.vlines(cands.loc[best, "z"], 0, heights[best], color = COLOURS["Two-step"], lw = 3)

        # Gaussian: track density
        ax = axes[1, col]
        ax.plot(z_grid, density / density.max(), color = COLOURS["Gaussian"], lw = 1.2)

        # Sum pT window: summed pT in each window
        ax = axes[2, col]
        ax.step(z_grid, sum_w / sum_w.max(), where = "mid", color = COLOURS["Sum pT window"], lw = 1)

        # Truth
        ax = axes[3, col]
        ax.vlines(truth, 0, 1, color = "k", lw = 2.5)

        for row, name in enumerate(METHODS + ["Truth"]):

            ax = axes[row, col]
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(0, 1.18)
            ax.axvline(truth, color = "k", ls = "--", lw = 0.8, alpha = 0.5)

            if name in chosen and np.isfinite(chosen[name]):
                # Triangle marks the vertex this method chose
                ax.plot(chosen[name], 1.1, marker = "v", color = COLOURS[name], ms = 9)

            if col == 0:
                ax.set_ylabel("Weight")
                if name in chosen:
                    dz = chosen[name] - truth
                    verdict = "correct" if abs(dz) < tol else "wrong"
                    title = f"{name}: chosen z = {chosen[name]:.2f} mm, dz = {dz:+.3f} mm ({verdict})"
                else:
                    title = f"Truth: z = {truth:.2f} mm"
                ax.text(0.01, 0.93, title, transform = ax.transAxes, fontsize = 8, va = "top")

        if col == 0:
            axes[0, 0].legend(fontsize = 7, loc = "upper right")

        axes[3, col].set_xlabel("z [mm]")
        axes[0, col].set_title("Full view" if col == 0 else f"Zoom: truth ± {zoom_half_width} mm",
                               fontsize = 9)

    labels = {"all_correct": "all methods correct", "all_wrong": "all methods wrong",
              "one_correct": "only one method correct"}
    fig.suptitle(f"Event {ev} ({labels.get(category, category)})", fontsize = 11)
    fig.tight_layout()

    path = os.path.join(out_dir, f"event{ev}_{category}.png")
    fig.savefig(path, dpi = 150)
    plt.close(fig)
    print(f"Saved: {Path(path).resolve()}")


def main():

    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok = True)

    n_grid = int(round((z_max - z_min) / step)) + 1

    saved = load_saved_results()
    events = choose_events(saved)

    for category, ev in events.items():

        ev_data = load_event(ev)

        density, w = track_density(ev_data, n_grid)
        gaussian_z = run_gaussian(ev_data, density, w)
        box_z, sum_w = run_box(ev_data, n_grid)
        two_step_z, cands = run_two_step(ev_data, density, w, n_grid)

        results = {"density": density, "sum_w": sum_w, "cands": cands,
                   "Gaussian": gaussian_z, "Sum pT window": box_z, "Two-step": two_step_z}

        make_plot(ev, category, ev_data, results, saved.loc[ev])


if __name__ == "__main__":

    main()