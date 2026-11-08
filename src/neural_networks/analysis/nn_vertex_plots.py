#!/usr/bin/env python3
"""
Event displays of the neural network's choice: a few successes and a few failures.

Top panel:    track density, with the true vertex (dashed line)
Bottom panel: NN probability of each candidate (thick = NN choice, star = true candidate)
"""

import math
import os
from pathlib import Path

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === ARGUMENTS ===
filename = "CERNbox/initial_data/root_files/HHbbyy_200PU.root"
tree_name = "Events"
pred_file = "outputs/sem1week8/nn_test_predictions.npz"
out_dir = "outputs/sem1week8/nn_event_displays"

n_events = 3          # how many successes and how many failures to plot
seed = 4

# Track selection and density settings (must match the two-step run)
pt_min = 1.5
d0_max, sig_d0_max, sig_z0_max = 4.0, 5.0, 10.0
z_min, z_max, step = -200.0, 200.0, 0.1
n_sigma = 4.0

zoom_half_width = 3.0     # [mm]
view_margin = 5.0         # [mm]

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

# ===== FUNCTIONS =====


def track_density(ev):

    # Read one event, apply the track selection and build the Gaussian track density
    branches = ["HLTTrack_pt", "HLTTrack_d0", "HLTTrack_z0", "HLTTrack_d0Err", "HLTTrack_z0Err"]
    with uproot.open(f"{filename}:{tree_name}") as tree:
        event = tree.arrays(branches, entry_start = ev, entry_stop = ev + 1)[0]

    pt = np.asarray(event["HLTTrack_pt"], dtype = float)
    d0 = np.asarray(event["HLTTrack_d0"], dtype = float)
    z0 = np.asarray(event["HLTTrack_z0"], dtype = float)
    sig_d0 = np.sqrt(np.maximum(np.asarray(event["HLTTrack_d0Err"], dtype = float), 0.0))
    sig_z0 = np.sqrt(np.maximum(np.asarray(event["HLTTrack_z0Err"], dtype = float), 0.0))

    mask = ((pt > pt_min) & (np.abs(d0) < d0_max) & (sig_d0 > 0) & (sig_d0 < sig_d0_max)
            & (sig_z0 > 0) & (sig_z0 < sig_z0_max))
    d0, z0, sig_d0, sig_z0 = d0[mask], z0[mask], sig_d0[mask], sig_z0[mask]

    # Each track: a Gaussian in z with a d0 weight
    z_grid = np.arange(z_min, z_max + step / 2, step)
    w = np.exp(-0.5 * (d0 / sig_d0) ** 2) / (SQRT2PI * sig_d0)
    density = np.zeros_like(z_grid)
    for zj, sj, wj in zip(z0, sig_z0, w):
        near = np.abs(z_grid - zj) < n_sigma * sj
        density[near] += wj * np.exp(-0.5 * ((z_grid[near] - zj) / sj) ** 2) / (SQRT2PI * sj)

    return z_grid, density


def make_plot(ev, label, z_grid, density, cand_z, cand_ntrk, logits, target, truth_z):

    real = cand_ntrk > 0
    chosen = int(np.argmax(logits))

    # Probability of each candidate (softmax of the network's scores)
    prob = np.exp(logits - logits.max())
    prob /= prob.sum()

    # Full view covers the truth and the chosen candidate; zoom is around the truth
    zs = [truth_z, cand_z[chosen]] if real[chosen] else [truth_z]
    views = [(min(zs) - view_margin, max(zs) + view_margin),
             (truth_z - zoom_half_width, truth_z + zoom_half_width)]

    fig, axes = plt.subplots(2, 2, figsize = (13, 6), sharex = "col",
                             gridspec_kw = {"width_ratios": [2, 1]})

    for col, (lo, hi) in enumerate(views):

        # Track density
        ax = axes[0, col]
        ax.fill_between(z_grid, density / density.max(), color = "lightgrey")
        ax.set_ylabel("Track density" if col == 0 else "")

        # NN probability per candidate
        ax = axes[1, col]
        ax.vlines(cand_z[real], 0, prob[real], color = "tab:purple", alpha = 0.5, lw = 1.5)
        if real[chosen]:
            ax.vlines(cand_z[chosen], 0, prob[chosen], color = "tab:purple", lw = 3.5, label = "NN choice")
        if target >= 0:
            ax.plot(cand_z[target], prob[target] + 0.05, "*", color = "gold", mec = "k", ms = 12,
                    label = "True candidate")
        ax.set_ylabel("NN probability" if col == 0 else "")
        ax.set_xlabel("z [mm]")

        for ax in axes[:, col]:
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1.15)
            ax.axvline(truth_z, color = "k", ls = "--", lw = 0.8)

        axes[0, col].set_title("Full view" if col == 0 else f"Zoom: truth ± {zoom_half_width} mm", fontsize = 9)

    axes[1, 0].legend(fontsize = 8, loc = "upper right")

    dz = cand_z[chosen] - truth_z if real[chosen] else np.nan
    fig.suptitle(f"Event {ev}: NN {label}   (chosen dz = {dz:+.3f} mm, probability = {prob[chosen]:.2f})")
    fig.tight_layout()

    path = os.path.join(out_dir, f"event{ev}_{label}.png")
    fig.savefig(path, dpi = 300)
    plt.close(fig)
    print(f"Saved: {Path(path).resolve()}")


def main():

    os.makedirs(out_dir, exist_ok = True)
    p = np.load(pred_file)

    # Success = NN chose the true candidate; failure = truth was among the candidates but NN chose another
    chosen = p["logits"].argmax(axis = 1)
    found = p["target"] >= 0
    success = found & (chosen == p["target"])
    failure = found & (chosen != p["target"])

    rng = np.random.default_rng(seed)

    for label, mask in [("success", success), ("failure", failure)]:

        rows = rng.choice(np.flatnonzero(mask), size = min(n_events, mask.sum()), replace = False)

        for i in rows:
            ev = int(p["event_id"][i])
            z_grid, density = track_density(ev)
            make_plot(ev, label, z_grid, density, p["cand_z"][i], p["cand_ntrk"][i],
                      p["logits"][i], int(p["target"][i]), float(p["truth_z"][i]))


if __name__ == "__main__":
    main()