#!/usr/bin/env python3
"""
Plot HLT track multiplicity and arbitrary track parameters from a ROOT TTree.
Uses RDataFrame for fast C++ event loops and matplotlib/mplhep for plotting.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import ROOT as root

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import custom plotting functions from the utils module (this also sets the ATLAS style and ROOT batch mode)
import utils.plotting_functions as pf

# Aesthetic properties for thin lines and clean caps
PLOT_KWARGS = {
    "histtype": "errorbar",
    "markersize": 4,
    "elinewidth": 1.2,
    "capsize": 2
}

# ========================== Inspecting the file ==========================

def find_branches(df, prefix, requested_suffixes):
    """
    Returns a dictionary of branch names matching the prefix.
    If 'all' is in requested_suffixes, it returns all matching branches.
    Otherwise, it strictly filters to the requested suffixes.
    """

    # If the user didn't ask for any suffixes, return empty
    if not requested_suffixes:
        return {}

    # Get all column names from the RDataFrame and filter by prefix
    names = [str(n) for n in df.GetColumnNames()]
    branches = {}
    plot_all = "all" in requested_suffixes

    # Loop through the column names and check for matches with the prefix
    for n in names:

        # If the column name starts with the prefix, extract the suffix
        if n.startswith(prefix):

            # Extract the suffix by removing the prefix from the column name
            suffix = n[len(prefix):]

            # If the user requested all branches or this specific suffix, add it to the dictionary
            if plot_all or suffix in requested_suffixes:

                # Add the suffix and full branch name to the dictionary
                branches[suffix] = n

    return branches


# ========================== Plotting ==========================

def plot_histogram(hist, default_xlabel, default_ylabel, default_title, stem, args):
    """Draws and saves a single histogram using the globally requested styles."""

    # Determine the labels and title, using defaults if not overridden by the user
    xlabel = args.xlabel if args.xlabel else default_xlabel
    ylabel = args.ylabel if args.ylabel else default_ylabel
    title = args.title if args.title else default_title

    # If normalisation is requested, scale the histogram to unit area and adjust the Y-axis label accordingly
    if args.normalise and hist.Integral() > 0:

        # Make sure errors are stored so they scale with the content
        if hist.GetSumw2N() == 0:
            hist.Sumw2()

        hist.Scale(1.0 / hist.Integral())
        if not args.ylabel:
            ylabel = "Fraction of Entries"

    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    # Format the legend label with statistics if requested
    lbl = pf.format_stats_label(hist, label=args.label, stats_choices=args.stats, stat_errors=args.stat_errors)

    # adjust plotting kwargs based on the --connect argument
    plot_args = PLOT_KWARGS.copy()
    if args.connect:
        plot_args["linestyle"] = "-"   # Connects the points with a solid line

    # Draw the histogram using mplhep's histplot function
    hep.histplot(hist, ax=ax, color="black", marker="o", label=lbl, **plot_args)

    # Apply common styling to the axes, including titles, labels, ranges, and scales
    title_kw = {"fontsize": args.title_size} if args.title_size else {}
    font_kw = {"fontsize": args.label_size} if args.label_size else {}

    # Set the title if not suppressed by the user, with optional font size
    if not args.no_title:
        ax.set_title(title, pad=15, **title_kw)

    # Set the X and Y axis labels with optional font sizes
    ax.set_xlabel(xlabel, loc="right", **font_kw)
    ax.set_ylabel(ylabel, loc="top", **font_kw)

    # Apply user-specified axis ranges and log scales if requested
    if args.xrange:
        ax.set_xlim(args.xrange[0], args.xrange[1])
    if args.yrange:
        ax.set_ylim(args.yrange[0], args.yrange[1])

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")

    # Add a legend
    ax.legend(frameon=True, fontsize=args.legend_size, loc="best")

    # Save the plot to the specified output directory with the appropriate filename
    if args.normalise:
        stem += "_norm"
    if args.logx:
        stem += "_logx"
    if args.logy:
        stem += "_logy"
    pf.save_plot(fig, args.output, stem, args.formats)


# ========================== Main ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="Generalized RDataFrame plotter for TTree branches.")

    # Files and Processing
    parser.add_argument("-f", "--filename", required=True, help="Path to the ROOT file.")
    parser.add_argument("-t", "--tree", default="Events", help="Name of the TTree (default: Events).")
    parser.add_argument("-p", "--prefix", default="HLTTrack_", help="Branch prefix to search for (default: HLTTrack_).")
    parser.add_argument("-o", "--output", default="hlt_tracks", help="Output subfolder inside outputs/.")

    parser.add_argument("--max-events", type=int, default=None, help="Maximum events to process.")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for RDataFrame.")

    # What to plot
    parser.add_argument("-s", "--suffixes", nargs="*", default=[], help="Specific suffixes to plot (e.g., pt eta phi). Pass 'all' to plot everything.")
    parser.add_argument("--multiplicity", action="store_true", help="Include the track multiplicity plot.")

    # Custom Labels, Titles, and Limits
    parser.add_argument("--title", type=str, default=None, help="Override the plot title.")
    parser.add_argument("--xlabel", type=str, default=None, help="Override the X-axis label.")
    parser.add_argument("--ylabel", type=str, default=None, help="Override the Y-axis label.")
    parser.add_argument("-l", "--label", default=None, help="Legend header label.")
    parser.add_argument("--xrange", nargs=2, type=float, metavar=("MIN", "MAX"), help="Force X-axis limits (min max)")
    parser.add_argument("--yrange", nargs=2, type=float, metavar=("MIN", "MAX"), help="Force Y-axis limits (min max)")

    # Formatting & Customization
    parser.add_argument("--title-size", type=float, default=None, help="Font size for plot titles")
    parser.add_argument("--label-size", type=float, default=None, help="Font size for axis labels")
    parser.add_argument("--legend-size", type=float, default=11, help="Font size for legend text")
    parser.add_argument("--no-title", action="store_true", help="Omit all plot titles")
    parser.add_argument("-lx", "--logx", action="store_true", help="Force log-x scale on ALL plots")
    parser.add_argument("-ly", "--logy", action="store_true", help="Force log-y scale on ALL plots")
    parser.add_argument("--normalise", action="store_true", help="Normalise all histograms to unit area")
    parser.add_argument("--connect", action="store_true", help="Draw lines connecting the data points.")

    # Stats and Formats
    parser.add_argument(
        "--stats", nargs="*", default=["entries", "mean", "std"],
        choices=["entries", "integral", "mean", "std", "none"],
        help="Statistics to list inside the legend. Pass 'none' to disable."
    )
    parser.add_argument("--stat-errors", action="store_true", help="Show error bounds on mean/std")
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf"])

    args = parser.parse_args()

    if "none" in args.stats or not args.stats:
        args.stats = []

    return args


def main():
    args = parse_args()

    # Enable multi-threading if requested and no max events limit is set
    if args.threads > 0 and args.max_events is None:
        root.ROOT.EnableImplicitMT(args.threads)
        print(f"Using {args.threads} threads for processing.")

    # Load the ROOT file and create an RDataFrame for the specified TTree
    df = root.RDataFrame(args.tree, args.filename)

    # If a maximum number of events is specified, limit the RDataFrame to that range
    if args.max_events is not None:
        df = df.Range(args.max_events)
        print(f"Processing a maximum of {args.max_events} events.")

    # Queue for storing histograms to be plotted
    histogram_queue = []

    # If multiplicity plotting is requested, define a new column for the number of tracks and create a histogram for it
    if args.multiplicity:

        # Find the first branch that starts with the specified prefix to use for multiplicity calculation
        ref_column = None

        # Loop through the column names to find the first one that starts with the specified prefix
        for n in df.GetColumnNames():

            # If the column name starts with the prefix, store it and break the loop
            if str(n).startswith(args.prefix):
                ref_column = str(n)
                break

        # If a reference column was found, define a new column for the number of tracks and create a histogram for it
        if ref_column:

            # Define a new column "n_tracks" that counts the number of tracks by taking the size of the reference column
            df = df.Define("n_tracks", f"{ref_column}.size()")

            # Create a histogram for the number of tracks
            h_mult = df.Histo1D("n_tracks")

            # Add the multiplicity histogram to the queue with default labels and titles
            histogram_queue.append((
                h_mult,
                "Number of Tracks",
                "Events",
                f"{args.prefix} Multiplicity",
                f"{args.prefix.strip('_').lower()}_multiplicity"
            ))

        # If no reference column was found, print a warning message
        else:
            print(f"Warning: Could not find any branch starting with '{args.prefix}' for multiplicity.")

    # Find branches with the specified prefix and requested suffixes
    branches = find_branches(df, args.prefix, args.suffixes)

    # If any branches were found, create histograms for each and add them to the queue
    if branches:

        print(f"Found {len(branches)} parameter branches to plot.")

        # Loop through the found branches and create histograms for each
        for suffix, column in branches.items():

            # Create a histogram for the current branch
            h = df.Histo1D(column)

            # Add the histogram to the queue with default labels and titles
            histogram_queue.append((
                h,
                suffix,
                "Tracks / Bin",
                f"{args.prefix}{suffix}",
                f"{args.prefix.strip('_').lower()}_{suffix}"
            ))

    # If no histograms were queued for plotting, print a message and exit
    if not histogram_queue:
        print("Nothing to plot! Use '--multiplicity' or specify branches.")
        return

    print("Executing event loop...")

    # Loop through the queued histograms, retrieve their values, and plot them using the specified styles and labels
    for result, default_xlabel, default_ylabel, default_title, stem in histogram_queue:
        hist = result.GetValue()
        plot_histogram(hist, default_xlabel, default_ylabel, default_title, stem, args)


if __name__ == "__main__":
    main()