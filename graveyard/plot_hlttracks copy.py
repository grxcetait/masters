#!/usr/bin/env python3
"""
Plot HLT track multiplicity and track parameters from the Events TTree.

Uses ROOT's RDataFrame so the loop over events runs in C++ and is therefore fast. The plotting is done with matplotlib.
Instead of a Python for-loop.
"""

import argparse
import os

import matplotlib.pyplot as plt
import mplhep as hep
import ROOT as root

import plotting_functions as pf

hep.style.use("ATLAS")  # Use ATLAS style for plots
root.gROOT.SetBatch(True)  # Don't pop up canvases

# Statistics that can be shown in the box, in the order they are displayed
STAT_CHOICES = ["entries", "integral", "mean", "std", "fwhm"]
DEFAULT_STATS = ["entries", "mean", "std", "fwhm"]

# Default legend / statistics box font size
DEFAULT_LEGEND_SIZE = 14


# ========================== Inspecting the file ==========================


def find_branches(df, prefix):
    """Return a list of branches in the RDataFrame that start with the given prefix."""

    # Get all the avaliable column names in the RDataFrame
    # These are returned as a special string-like object std::vector<std::string> (C++)
    names = [str(n) for n in df.GetColumnNames()]

    # If a branch begins with the desired prefix,
    # Convert them into a standard python string and return
    return {n[len(prefix):].lower(): n for n in names if n.startswith(prefix)}


# ========================== Plotting ==========================

def plot_histogram(hist, xlabel, ylabel, title, folder, stem, log_y, formats):

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize = (8, 6), layout = "constrained")

    # Plot the histogram with error bars
    hep.histplot(hist, ax = ax, histtype = "errorbar", yerr = True, color = "black", 
                 marker = "o", markersize = 4, elinewidth = 1.2, capsize = 2)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel, loc = "right")
    ax.set_ylabel(ylabel, loc = "top")

    # Set the y-axis to log scale if requested
    if log_y:
        ax.set_yscale("log")

    # Leave room at the top for the statistics box and legend
    y_low, y_high = ax.get_ylim()

    if log_y:
        ax.set_ylim(y_low, y_high * (y_high / y_low) ** 0.4)
    else: 
        ax.set_ylim(0, y_high * 1.3)

    # Add a statistics box with the number of entries, mean, and standard deviation
    stats = (f"Entries = {int(hist.GetEntries()):,}\n"
             f"Mean = {hist.GetMean():.3g}\n"
             f"Std dev = {hist.GetStdDev():.3g}")
    ax.text(0.97, 0.97, stats, transform = ax.transAxes, ha = "right", va = "top", 
            fontsize = 14, bbox = dict(facecolor = "white", alpha = 0.9, edgecolor = "0.6"))

    # Save the figure in the requested formats
    for file_format in formats:
        path = os.path.join(folder, f"{stem}.{file_format}")
        fig.savefig(path, dpi = 300, bbox_inches = "tight")
        print(f"Saved {file_format.upper()} to: {path}")

    plt.close(fig)



# ========================== Main ==========================

def resolve_stats(choices):
    """Turn the --stats arguments into a list of statistic names."""
    if "none" in choices:
        return []
    if "all" in choices:
        return list(STAT_CHOICES)
    return list(choices)

def main():
    parser = argparse.ArgumentParser(description = "Plot HLT track multiplicity and track parameters from a ROOT TTree.")

    parser.add_argument("-f", "--filename", required = True, 
                         help = "Path to the ROOT file.")
    parser.add_argument("-t", "--tree", default = "Events",
                         help = "Name of the TTree in the ROOT file (default: Events).")
    parser.add_argument("-p", "--prefix", default = "HLTTrack_",
                         help = "Branch prefix of the track variables. (default: HLTTrack_).")
    parser.add_argument("-o", "--output", default = "hlt_tracks",
                         help = "Output folder for the plots inside outputs/. (default: hlt_tracks).")
    parser.add_argument("--max-events", type = int, default = None,
                         help = "Maximum number of events to process. (default: all).")
    parser.add_argument("--threads", type = int, default = 4,
                         help = "Number of threads to use for processing. (default: 4).")
    parser.add_argument("--formats", nargs = "+", default = ["png"], choices = ["png", "pdf", "svg"],
                        help = "Output file formats for the plots. (default: png).")

    # Statistics box
    parser.add_argument("--stats", nargs = "+", default = DEFAULT_STATS,
                        choices = STAT_CHOICES + ["all", "none"], metavar = "STAT",
                        help = "Statistics to show in the box. Choose any of: "
                                + ", ".join(STAT_CHOICES)
                                + ", or 'all' / 'none'. Default: "
                                + " ".join(DEFAULT_STATS) + ".")
    parser.add_argument("--stat-errors", action = argparse.BooleanOptionalAction, default = False,
                        help = "Show the uncertainty (±) on the mean and std dev.")
    
    args = parser.parse_args()


    # Enable multithreading if requested and no maximum number of events is set
    if args.threads > 0 and args.max_events is None:
        root.ROOT.EnableImplicitMT(args.threads)
        print(f"Using {args.threads} threads for processing.")

    # Create an RDataFrame from the TTree in the ROOT file
    df = root.RDataFrame(args.tree, args.filename)
    if args.max_events is not None:
        df = df.Range(args.max_events)
        print(f"Processing a maximum of {args.max_events} events.")

    # Find the branches that start with the given prefix and print them
    branches = find_branches(df, args.prefix)
    if not branches:
        print(f"No branches found with prefix '{args.prefix}'.")
        return
    print(f"Found track branches: ", ", ".join(branches.values()))

    # Create the output folder if it doesn't exist
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "outputs", args.output)
    os.makedirs(folder, exist_ok = True)

    # Make a list to hold histogram data
    histogram_list = [] # (histogram result, xlabel, ylabel, title, stem, log_y)

    # Track multiplicity: number of tracks in each event. Any per-track branch works
    # Since they all have the same length within an event
    ref_column = branches.get("pt") or next(iter(branches.values()))
    df = df.Define("n_tracks", f"{ref_column}.size()")
    h_mult = df.Histo1D(("n_tracks", "", MULT["nbins"], MULT["xmin"], MULT["xmax"]), "n_tracks")
    histogram_list.append((h_mult, MULT["xlabel"], "Events", "HLT track multiplicity", "n_tracks", False))

    # Track parameters: Histo1D on a vector branch fills one entry per track
    # Loop through the variables dictionary for each suffix and plotting configuration
    for suffix, cfg in VARIABLES.items():

        # Get the exact branch name corresponding to this variable
        column = branches.get(suffix)

        # If the branch does not exist, print a warning
        if column is None:
            print(f"Warning: no '{args.prefix}{suffix}' branch found, skipping.")
            continue

        # Fill the histogram
        h = df.Histo1D((f"h_{suffix}", "", cfg["nbins"], cfg["xmin"], cfg["xmax"]), column)

        # Append the histogram data to the list
        histogram_list.append((h, cfg["xlabel"], "Tracks / bin", f"HLT track {suffix}", 
                       f"hlttrack_{suffix}", cfg["log_y"]))

    print("Looping over events ...")
    # Iterate thorugh histogram list and print each histogram
    for result, xlabel, ylabel, title, stem, log_y in histogram_list:

        # GetValue() triggers a single loop over the events for all of them
        hist = result.GetValue()

        # Plot histogram
        plot_histogram(hist, xlabel, ylabel, title, folder, stem, log_y, args.formats)


if __name__ == "__main__":
    main()