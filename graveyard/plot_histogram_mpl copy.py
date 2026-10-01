#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import mplhep as hep
import ROOT as root
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import plotting_functions as pf

hep.style.use("ATLAS")  # Use ATLAS style for plots
root.gROOT.SetBatch(True)  # Don't pop up canvases

# Colours / markers for the first and second histogram. The different marker
# shapes keep the two distinguishable in greyscale prints.
COLOURS = ("black", "#c1272d")
MARKERS = ("o", "s")

# Statistics that can be shown in the box, in the order they are displayed
STAT_CHOICES = ["entries", "integral", "mean", "std", "fwhm"]
DEFAULT_STATS = ["entries", "mean", "std", "fwhm"]

# Default legend / statistics box font size
DEFAULT_LEGEND_SIZE = 14


# =============================================================================
# Loading
# =============================================================================

def load_histogram(filename, histogram_path):

    histogram_path = histogram_path.strip("/")

    # Open the ROOT file
    root_file = root.TFile.Open(filename)

    # Check if the file was successfully opened
    if not root_file or root_file.IsZombie():
        print("Error: Could not open ROOT file.")
        return

    # Retrieve the object from the ROOT file
    hist = root_file.Get(histogram_path)

    # Check if the object was successfully retrieved
    if not hist:
        print(f"Error: Could not retrieve {histogram_path} from the ROOT file.")
        root_file.Close()
        return

    # TEfficiency isn't a TH1, so build a TH1 holding the efficiency values
    if hist.InheritsFrom("TEfficiency"):
        eff = hist

        # The total histogram is used to get the binning and axis titles,
        # but the efficiency values are stored in the TEfficiency object
        template = eff.GetTotalHistogram()

        # 2D and 3D efficiencies are not supported, so skip them
        if template.GetDimension() != 1:
            print(f"Skipping {histogram_path}: not a 1D efficiency.")
            root_file.Close()
            return

        # Create a new TH1 to hold the efficiency values and errors
        hist = template.Clone(eff.GetName() + "_th1")

        # Set the histogram to be independent of the file and reset its contents
        hist.SetDirectory(0)
        hist.Reset()
        hist.Sumw2()

        hist.SetTitle(eff.GetTitle().split(";")[0])

        # Set the axis titles
        hist.GetXaxis().SetTitle(template.GetXaxis().GetTitle())
        hist.GetYaxis().SetTitle("Efficiency")

        # Loop over the bins and set the efficiency values and errors
        # Bins with zero total entries are skipped
        for i in range(1, hist.GetNbinsX() + 1):

            # If the bin is empty, leave at 0 with zero error
            if template.GetBinContent(i) == 0:
                continue

            hist.SetBinContent(i, eff.GetEfficiency(i))

            # Symmetric approximation of the asymmetric errors
            hist.SetBinError(i, 0.5 * (eff.GetEfficiencyErrorLow(i)
                                       + eff.GetEfficiencyErrorUp(i)))

    # Check if the histogram is 1D, since only 1D histograms are supported
    elif hist.GetDimension() != 1:
        print(f"Skipping {histogram_path}: not a 1D histogram.")
        root_file.Close()
        return

    # Detach from the file so it survives closing, then close
    hist.SetDirectory(0)
    root_file.Close()
    return hist


def find_histograms(filename, directory = ""):

    # Remove any leading or trailing slashes, since ROOT's Get() does not accept them
    directory = directory.strip("/")

    # Open the ROOT file
    root_file = root.TFile.Open(filename)

    # Check if the file was successfully opened
    if not root_file or root_file.IsZombie():
        print("Error: Could not open ROOT file.")
        return []

    # Retrieve the starting directory from the ROOT file
    start_directory = root_file.Get(directory) if directory else root_file

    # Check if the directory was successfully retrieved
    if not start_directory:
        print(f"Error: Directory '{directory}' not found in file.")
        root_file.Close()
        return []

    histogram_paths = []

    def walk(current_directory, prefix):

        # Skip duplicate cycles of the same object (name;1, name;2)
        seen_names = set()

        # Loop over all keys in the current directory
        for key in current_directory.GetListOfKeys():

            # Get the name of the object and skip if already seen
            name = key.GetName()
            if name in seen_names:
                continue

            # Mark this name as seen to avoid duplicates in the same directory
            seen_names.add(name)

            # Work out what kind of object this key is
            key_class = root.TClass.GetClass(key.GetClassName())
            if not key_class:
                continue

            # Build the full path to the object, including the current directory prefix
            path = f"{prefix}/{name}" if prefix else name

            # Subdirectories are searched recursively
            if key_class.InheritsFrom("TDirectory"):
                walk(current_directory.Get(name), path)

            # Only keep 1D efficiencies
            elif key_class.InheritsFrom("TEfficiency"):

                # Get the TEfficiency object to check its dimension
                efficiency = current_directory.Get(name)

                if efficiency.GetTotalHistogram().GetDimension() == 1:
                    histogram_paths.append(path)

            # Only keep 1D histograms (TH2 and TH3 also inherit from TH1)
            elif (key_class.InheritsFrom("TH1")

                  # Exclude 2D and 3D histograms, which also inherit from TH1
                  and not key_class.InheritsFrom("TH2")
                  and not key_class.InheritsFrom("TH3")):

                histogram_paths.append(path)

    walk(start_directory, directory)
    root_file.Close()
    return histogram_paths


# =============================================================================
# Plot functions
# =============================================================================

def plot_one_file(filename, histogram_path, output, title, xlabel, ylabel, label, opts):

    # === HISTOGRAMS ===
    hist = load_histogram(filename, histogram_path)

    # Check if a histogram was obtained
    if hist is None:
        return

    plot_hist, stats_hist = pf.prepare(hist, opts)
    normalised = plot_hist is not stats_hist

    # === OUTPUT NAME ===
    output_folder = pf.output_folder_for(output)
    histogram_name = os.path.basename(histogram_path.strip("/"))
    stem = histogram_name + pf.output_tags(opts, normalised)

    # === PLOTTING ===
    fig, ax = plt.subplots(figsize = (8, 6), layout = "constrained")
    pf.draw(ax, plot_hist, COLOURS[0], MARKERS[0])

    # Label as given by user or pulled from the ROOT histogram (or fallback if empty)
    lk = pf.font_kw(opts.label_size)
    pf.set_title(ax, title or hist.GetTitle(), opts)
    ax.set_xlabel(xlabel or hist.GetXaxis().GetTitle() or "X Axis", loc = "right", **lk)
    ax.set_ylabel(ylabel or hist.GetYaxis().GetTitle() or "Events / Bin", loc = "top", **lk)
    ax.tick_params(axis = "both", which = "major", labelsize = 11)

    if opts.log_x:
        ax.set_xscale("log")
    if opts.log_y:
        ax.set_yscale("log")

    # Statistics box, headed by the label if one was given
    lines = pf.stat_lines(stats_hist, opts)
    if lines or label:
        pf.add_stats_legend(ax, [(pf.box_text(label, lines), COLOURS[0], None)],
                         opts.legend_size)

    pf.add_headroom([ax], opts.log_y, len(lines) + (1 if label else 0))
    pf.apply_ranges(ax, opts.xrange, opts.yrange1)

    # === SAVING ===
    pf.save_figure(fig, output_folder, stem, opts)


def plot_two_files(filename1, filename2, histogram_path1, histogram_path2,
                   output, title1, title2, x1_label, y1_label, x2_label, y2_label,
                   label1, label2, opts):

    # === HISTOGRAMS ===
    if filename2 is None:
        filename2 = filename1

    hist1 = load_histogram(filename1, histogram_path1)
    hist2 = load_histogram(filename2, histogram_path2)

    if hist1 is None or hist2 is None:
        return

    plot1, stats1 = pf.prepare(hist1, opts)
    plot2, stats2 = pf.prepare(hist2, opts)
    normalised1 = plot1 is not stats1
    normalised2 = plot2 is not stats2

    # === OUTPUT NAME ===
    output_folder = pf.output_folder_for(output)
    name1 = os.path.basename(histogram_path1.strip("/"))
    name2 = os.path.basename(histogram_path2.strip("/"))
    side_by_side = opts.layout == "side-by-side"
    stem = (f"{name1}_vs_{name2}"
            + pf.output_tags(opts, normalised1 or normalised2, side_by_side))

    # === LABELS ===
    x1_label = x1_label or hist1.GetXaxis().GetTitle() or "X1 Axis"
    x2_label = x2_label or hist2.GetXaxis().GetTitle() or "X2 Axis"
    y1_label = y1_label or hist1.GetYaxis().GetTitle() or "Events / Bin"
    y2_label = y2_label or hist2.GetYaxis().GetTitle() or "Events / Bin"

    lines1 = pf.stat_lines(stats1, opts)
    lines2 = pf.stat_lines(stats2, opts)

    lk = pf.font_kw(opts.label_size)

    # === PLOTTING ===
    if not side_by_side:

        # ---------- Both histograms on the same graph ----------
        fig, ax = plt.subplots(figsize = (8, 6), layout = "constrained")
        separate = opts.yaxes == "separate"

        # A twin axis gives the second histogram its own y-axis
        ax2 = ax.twinx() if separate else ax
        axes = [ax, ax2] if separate else [ax]

        pf.draw(ax, plot1, COLOURS[0], MARKERS[0])
        pf.draw(ax2, plot2, COLOURS[1], MARKERS[1])

        # Synchronize X-axis ranges across both histograms
        x_min, x_max = pf.x_range([hist1, hist2], opts.log_x)
        ax.set_xlim(x_min, x_max)
        ax2.set_xlim(x_min, x_max)

        # Title (the legend labels are only used in the legend / statistics box)
        legend_label1 = label1 or name1
        legend_label2 = label2 or name2
        if title1:
            pf.set_title(ax, title1, opts)
        elif title2:
            pf.set_title(ax, title2, opts)
        else:
            pf.set_title(ax, f"{name1} and {name2}", opts)

        # Axis labels: coloured y labels tell the reader which axis belongs to which
        ax.set_xlabel(x1_label, loc = "right", **lk)
        if separate:
            ax.set_ylabel(y1_label, loc = "top", color = COLOURS[0], **lk)
            ax2.set_ylabel(y2_label, loc = "top", color = COLOURS[1], **lk)
            ax.tick_params(axis = "y", labelcolor = COLOURS[0])
            ax2.tick_params(axis = "y", labelcolor = COLOURS[1])
        else:
            ax.set_ylabel(y1_label, loc = "top", color = COLOURS[0], **lk)

        if opts.log_x:
            ax.set_xscale("log")
        if opts.log_y:
            for a in axes:
                a.set_yscale("log")

        # Combined legend + statistics, one block per histogram. It lives on the
        # top-most axis so that the data of the twin axis cannot be drawn over it.
        blocks = [(pf.box_text(legend_label1, lines1), COLOURS[0], MARKERS[0]),
                  (pf.box_text(legend_label2, lines2), COLOURS[1], MARKERS[1])]
        pf.add_stats_legend(ax2, blocks, opts.legend_size)

        pf.add_headroom(axes, opts.log_y, len(lines1) + len(lines2) + 2)

        # User-given ranges. With a shared y-axis only --yrange1 applies.
        pf.apply_ranges(ax, opts.xrange, opts.yrange1)
        if separate:
            pf.apply_ranges(ax2, opts.xrange, opts.yrange2)
        else:
            if opts.yrange2:
                print("Note: --yrange2 is ignored when both histograms share one y-axis.")

    else:

        # ---------- Two panels next to each other ----------
        share_y = opts.yaxes == "shared"
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize = (13, 6), sharey = share_y,
                                         layout = "constrained")

        panels = [(ax_a, plot1, hist1, title1, x1_label, y1_label, label1, lines1, 0),
                  (ax_b, plot2, hist2, title2, x2_label, y2_label, label2, lines2, 1)]

        for ax, plot_hist, hist, title, x_label, y_label, label, lines, i in panels:
            pf.draw(ax, plot_hist, COLOURS[i], MARKERS[i])
            pf.set_title(ax, title or hist.GetTitle(), opts)
            ax.set_xlabel(x_label, loc = "right", **lk)
            ax.tick_params(axis = "both", which = "major", labelsize = 11)

            # With a shared y-axis only the left panel needs a y label
            if not (share_y and i == 1):
                ax.set_ylabel(y1_label, loc = "top", color = COLOURS[0], **lk)

            if opts.log_x:
                ax.set_xscale("log")
            if opts.log_y:
                ax.set_yscale("log")

            # Statistics box for this panel, headed by its label if one was given
            if lines or label:
                pf.add_stats_legend(ax, [(pf.box_text(label, lines), COLOURS[i], None)],
                                 opts.legend_size)

        # With a shared axis, changing one panel's limits changes both, so only do it once
        n_box_lines = max(len(lines1) + (1 if label1 else 0),
                          len(lines2) + (1 if label2 else 0))
        pf.add_headroom([ax_a] if share_y else [ax_a, ax_b], opts.log_y, n_box_lines)

        # User-given ranges. With a shared y-axis one range applies to both panels.
        if share_y:
            pf.apply_ranges(ax_a, opts.xrange, opts.yrange1 or opts.yrange2)
            pf.apply_ranges(ax_b, opts.xrange, None)
        else:
            pf.apply_ranges(ax_a, opts.xrange, opts.yrange1)
            pf.apply_ranges(ax_b, opts.xrange, opts.yrange2)

    # === SAVING ===
    pf.save_figure(fig, output_folder, stem, opts)


# =============================================================================
# Command line
# =============================================================================

def resolve_stats(choices):
    """Turn the --stats arguments into a list of statistic names."""
    if "none" in choices:
        return []
    if "all" in choices:
        return list(STAT_CHOICES)
    return list(choices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Plot histograms from a ROOT file using Matplotlib."
        )
    # Mandatory first file
    parser.add_argument("-f1", "--filename1", type = str, required = True,
                        help = "Path to the first ROOT file.")
    parser.add_argument("-H1", "--histogram_path1", type = str, required = False,
                        help = "Path to the histogram within the first ROOT file.")
    parser.add_argument("-d", "--directory", type = str, default = None,
                        help = "Plot every 1D histogram in this directory (recursively).")
    parser.add_argument("-o", "--output", type = str, default = "other",
                        help = "Output folder name inside outputs/.")

    # Optional second file
    parser.add_argument("-f2", "--filename2", type = str, required = False,
                        help = "Path to the second ROOT file.")
    parser.add_argument("-H2", "--histogram_path2", type = str, required = False,
                        help = "Path to the histogram within the second ROOT file.")

    # Optional Title, legend label and Axis overrides
    parser.add_argument("-t1", "--title1", type = str, default = None,
                        help = "Custom plot title for the first histogram (overrides ROOT title).")
    parser.add_argument("-t2", "--title2", type = str, default = None,
                        help = "Custom plot title for the second histogram (overrides ROOT title).")
    parser.add_argument("-l1", "--label1", type = str, default = None,
                        help = "Legend / statistics box label for the first histogram.")
    parser.add_argument("-l2", "--label2", type = str, default = None,
                        help = "Legend / statistics box label for the second histogram.")
    parser.add_argument("-x1", "--xlabel1", type = str, default = None,
                        help = "Custom X-axis label for the first histogram (overrides ROOT X title).")
    parser.add_argument("-x2", "--xlabel2", type = str, default = None,
                        help = "Custom X-axis label for the second histogram (overrides ROOT X title).")
    parser.add_argument("-y1", "--ylabel1", type = str, default = None,
                        help = "Custom Y-axis label for the first histogram (overrides ROOT Y title).")
    parser.add_argument("-y2", "--ylabel2", type = str, default = None,
                        help = "Custom Y-axis label for the second histogram (overrides ROOT Y title).")

    # Axis ranges
    parser.add_argument("--xrange", type = float, nargs = 2, default = None,
                        metavar = ("MIN", "MAX"),
                        help = "X-axis range, applied to every axis in the plot.")
    parser.add_argument("--yrange1", type = float, nargs = 2, default = None,
                        metavar = ("MIN", "MAX"),
                        help = "Y-axis range for the first histogram (or the shared y-axis).")
    parser.add_argument("--yrange2", type = float, nargs = 2, default = None,
                        metavar = ("MIN", "MAX"),
                        help = "Y-axis range for the second histogram (separate y-axes or "
                               "unshared side-by-side panels).")

    # Font sizes
    parser.add_argument("--title-size", type = float, default = None, metavar = "SIZE",
                        help = "Font size of the plot title. Default: ATLAS style default.")
    parser.add_argument("--label-size", type = float, default = None, metavar = "SIZE",
                        help = "Font size of the axis labels. Default: ATLAS style default.")
    parser.add_argument("--legend-size", type = float, default = DEFAULT_LEGEND_SIZE,
                        metavar = "SIZE",
                        help = "Font size of the legend / statistics box. Default: "
                               f"{DEFAULT_LEGEND_SIZE}.")

    # Optional log-scale axes
    parser.add_argument("-lx", "--logx", dest = "log_x", action = argparse.BooleanOptionalAction, default = False,
                        help = "Plot the x-axis in log scale (use --no-logx to disable).")
    parser.add_argument("-ly", "--logy", dest = "log_y", action = argparse.BooleanOptionalAction, default = False,
                        help = "Plot the y-axis in log scale (use --no-logy to disable).")

    # Normalisation
    parser.add_argument("--normalise", "--normalize", action = argparse.BooleanOptionalAction,
                        default = False,
                        help = "Normalise each histogram to unit area (efficiencies are left "
                               "unchanged). Use --no-normalise to disable.")

    # Layout of two-histogram plots
    parser.add_argument("--layout", choices = ["overlay", "side-by-side"], default = "overlay",
                        help = "Two-histogram plots: draw both on the same graph (overlay) "
                               "or in two panels next to each other (side-by-side).")
    parser.add_argument("--yaxes", choices = ["separate", "shared"], default = "separate",
                        help = "Overlay: separate = each histogram gets its own y-axis, "
                               "shared = one common y-axis. Side-by-side: shared = both "
                               "panels use the same y range.")

    # Statistics box
    parser.add_argument("--stats", nargs = "+", default = DEFAULT_STATS,
                        choices = STAT_CHOICES + ["all", "none"], metavar = "STAT",
                        help = "Statistics to show in the box. Choose any of: "
                               + ", ".join(STAT_CHOICES)
                               + ", or 'all' / 'none'. Default: "
                               + " ".join(DEFAULT_STATS) + ".")
    parser.add_argument("--stat-errors", action = argparse.BooleanOptionalAction, default = False,
                        help = "Show the uncertainty (±) on the mean and std dev.")

    # Report-friendly extras
    parser.add_argument("--no-title", action = "store_true",
                        help = "Omit the plot title (use the report caption instead).")
    parser.add_argument("--formats", nargs = "+", default = ["png"],
                        choices = ["png", "pdf", "svg"],
                        help = "Output file formats. Use pdf/svg for vector graphics in a report.")

    args = parser.parse_args()

    # Turn the --stats arguments into a plain list of statistic names.
    # The parsed arguments are passed to the plot functions as `opts`.
    args.stats = resolve_stats(args.stats)
    opts = args

    # If a directory was given, plot every histogram in it
    if args.directory is not None:

        directory = args.directory.strip("/")
        histogram_paths = find_histograms(args.filename1, directory)
        print(f"Found {len(histogram_paths)} histograms in '{directory}'")

        # Keep track of names used so far, since all plots go in the same folder
        used_names = set()

        for path in histogram_paths:
            histogram_name = os.path.basename(path)
            if histogram_name in used_names:
                print(f"Warning: {histogram_name} already plotted, it will be overwritten.")
            used_names.add(histogram_name)

            # Custom titles and labels are not passed, as they would apply to every plot
            plot_one_file(args.filename1, path, args.output, None, None, None, None, opts)

    elif args.histogram_path1 is None:
        parser.error("Provide either -H1 (single histogram) or -d (directory).")

    elif args.histogram_path2 is None:
        plot_one_file(args.filename1, args.histogram_path1, args.output,
                      args.title1, args.xlabel1, args.ylabel1, args.label1, opts)

    else:
        plot_two_files(args.filename1, args.filename2, args.histogram_path1,
                       args.histogram_path2, args.output, args.title1, args.title2,
                       args.xlabel1, args.ylabel1, args.xlabel2, args.ylabel2,
                       args.label1, args.label2, opts)