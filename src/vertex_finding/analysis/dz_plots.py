#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === ARGUEMENTS ====
# Folder containing the sumpt, gaussian_weighting and two_steps result folders
base_dir = "outputs/sem1week4"
output_dir = "outputs/sem1week5"

# Track selection to study (no pT cap and no eta cut, so np.inf)
pt_min = 1.5        # [GeV]
pt_max = np.inf     # [GeV]
eta_max = np.inf

# The results folder for this selection, named the same way as in the scan scripts
selection_folder = f"{pt_min}ptmin_{pt_max}ptmax_{eta_max}etamax"

# Which configuration of each method to plot
box_res = 0.1                   # [mm] box window half-width
box_power = 1                   # 1 = pT weighted
gaussian_file = "vertex_results_d0True_normTrue.csv"
two_step_s_max = 3.0            # track-to-vertex significance cut used in the run

# Matching tolerances [mm]: tol sets which events count as matched,
# and both are drawn as lines on the |dz| plot
tol = 0.3
tol_lines = [0.3, 1.0]

# Binning for the matched dz plots [mm]
matched_range = tol             # plot from -matched_range to +matched_range
matched_bin_width = 0.005       # finer bins to resolve the core

# How to normalise the histograms
# "none":     number of events in each bin
# "fraction": each histogram sums to 1 (fraction of events in each bin)
# "peak":     each histogram's tallest bin is scaled to 1
normalisation = "peak"

# Where to save the plots
out_dir = os.path.join(output_dir, "statistics", selection_folder)

# ===== FUNCTIONS =====


def select_candidate(score, cand_ntrk):

    # Index of the candidate with the highest score in each event
    # (empty candidate slots can never be chosen)
    score = np.where(cand_ntrk > 0, score, -np.inf)
    return np.argmax(score, axis = 1)


def load_gaussian():

    # Signed dz of the Gaussian density method (densest peak), and truth validity
    path = os.path.join(base_dir, "gaussian_weighting", selection_folder, gaussian_file)
    if not os.path.isfile(path):
        print(f"Not found, skipping: {path}")
        return None, None

    df = pd.read_csv(path)
    return df["delta_z"].to_numpy(), df["truth_valid"].to_numpy().astype(bool)


def load_box():

    # Signed dz of the box (sum pT window) method
    path = os.path.join(base_dir, "sumpt", selection_folder,
                        f"vertex_results_res{box_res}_pt{box_power}.csv")
    if not os.path.isfile(path):
        print(f"Not found, skipping: {path}")
        return None, None

    df = pd.read_csv(path)
    return df["delta_z"].to_numpy(), df["truth_valid"].to_numpy().astype(bool)


def load_two_step():

    # Signed dz of the two-step method choosing by sum pT, recomputed from the saved candidates
    path = os.path.join(base_dir, "two_steps", selection_folder,
                        f"vertex_candidates_s{two_step_s_max}.npz")
    if not os.path.isfile(path):
        print(f"Not found, skipping: {path}")
        return None, None

    data = np.load(path)
    cand_z = data["cand_z"]
    truth_z = data["truth_z"]
    chosen = select_candidate(data["cand_sumpt"], data["cand_ntrk"])

    dz = cand_z[np.arange(len(truth_z)), chosen] - truth_z
    return dz, data["truth_valid"].astype(bool)


def normalise(counts):

    # Scale a histogram according to the chosen normalisation
    if normalisation == "peak":
        return counts / counts.max() if counts.max() > 0 else counts
    if normalisation == "fraction":
        return counts / counts.sum() if counts.sum() > 0 else counts
    return counts


def plot_hist(ax, values, bins, style):

    # Draw one histogram as a step line
    counts, edges = np.histogram(values, bins = bins)
    ax.stairs(normalise(counts), edges, **style)


def y_label():

    # Axis label matching the normalisation
    return {"none": "Events", "fraction": "Fraction of events",
            "peak": "Normalised events"}[normalisation]


def plot_matched(results):

    # dz for matched events, with each method's efficiency at tol in the legend
    fig, ax = plt.subplots(figsize = (8, 5))
    n_bins = int(round(2 * matched_range / matched_bin_width))
    bins = np.linspace(-matched_range, matched_range, n_bins + 1)

    for label, dz, style in results:
        matched = np.abs(dz) < tol
        eff = 100 * np.mean(matched)
        plot_hist(ax, dz[matched], bins,
                  #{"label": f"{label}: eff (tol = {tol} mm) = {eff:.2f} %", "linewidth": 1.5, **style})
                    {"label": f"{label}: eff = {eff:.2f} %", "linewidth": 1.5, **style})

    ax.set_xlabel(f"dz for matched events (|dz| < {tol} mm) [mm]")
    ax.set_ylabel(f"{y_label()}")
    ax.set_xlim(-matched_range, matched_range)
    ax.set_ylim(bottom = 0)
    name = f"dz_matched_{normalisation}.png"

    ax.legend(fontsize = 8)
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi = 300)
    plt.close(fig)
    print(f"Saved: {Path(path).resolve()}")


def plot_abs_dz(results):

    # |dz| between the chosen vertex and the truth, on a log x axis
    fig, ax = plt.subplots(figsize = (8, 5))
    bins = np.logspace(-4, np.log10(400), 80)

    for label, dz, style in results:
        abs_dz = np.abs(dz)
        plot_hist(ax, abs_dz[np.isfinite(abs_dz)], bins, {"label": label, "linewidth": 1.5, **style})

    for t, ls in zip(tol_lines, [":", "-."]):
        ax.axvline(t, color = "k", linestyle = ls, linewidth = 1, label = f"tol = {t} mm")

    ax.set_xscale("log")
    ax.set_xlabel("|dz| between chosen vertex and truth [mm]")
    ax.set_ylabel(f"{y_label()}")
    ax.set_ylim(bottom = 0)
    ax.legend(fontsize = 8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"abs_dz_distribution_{normalisation}.png")
    fig.savefig(path, dpi = 300)
    plt.close(fig)
    print(f"Saved: {Path(path).resolve()}")


def main():

    # Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok = True)

    # The three methods: (label, loader, line style)
    methods = [
        ("Gaussian weighting", load_gaussian, {"color": "tab:red"}),
        #(f"Sum pT window (res = {box_res} mm)", load_box, {"color": "tab:blue"}),
        (f"Sum pT window", load_box, {"color": "tab:blue"}),
        #(f"Two-step (s < {two_step_s_max:g})", load_two_step, {"color": "tab:green"}),
        (f"Two steps", load_two_step, {"color": "tab:green"}),
    ]

    # Load every method, keeping only events with a valid truth vertex
    results = []
    for label, loader, style in methods:
        dz, valid = loader()
        if dz is None:
            continue
        dz = dz[valid]
        results.append((label, dz, style))

    if not results:
        print("No results found.")
        return

    plot_matched(results)
    plot_abs_dz(results)


if __name__ == "__main__":

    main()