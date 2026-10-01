#!/usr/bin/env python3
import argparse
import os
import matplotlib.pyplot as plt
import mplhep as hep
import plotting_functions as pf

COLOURS = ("black", "#c1272d")
MARKERS = ("o", "s")

PLOT_KWARGS = {
    "histtype": "errorbar",
    "markersize": 4,
    "elinewidth": 1.2,
    "capsize": 2
}


def apply_common_axes_style(ax, args, xlabel=None, ylabel=None, yrange=None):
    """Applies repetitive styling to a given axis."""

    # Set axis labels and titles, using defaults if not overridden
    font_kw = {"fontsize": args.label_size} if args.label_size else {}

    if xlabel:
        ax.set_xlabel(xlabel, loc="right", **font_kw)
    if ylabel:
        ax.set_ylabel(ylabel, loc="top", **font_kw)

    if args.logx:
        ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    if args.xrange:
        ax.set_xlim(*args.xrange)
    if yrange:
        ax.set_ylim(*yrange)


def plot_single(filename, hist_path, args):
    """Generates a standard single-histogram plot."""

    # Load the histogram from the ROOT file
    hist = pf.load_histogram(filename, hist_path)
    if not hist:
        return

    # Determine whether to normalise the histogram and prepare it for plotting
    # Create a clone of the histogram to avoid modifying the original
    draw_hist = hist.Clone(hist.GetName() + "_norm") if args.normalise and hist.Integral() > 0 else hist

    # If normalisation is requested, scale the histogram to unit area
    if args.normalise and hist.Integral() > 0:
        draw_hist.Scale(1.0 / draw_hist.Integral())

    # Create the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

    # Format the legend label with statistics if requested
    lbl = pf.format_stats_label(hist, args.label1, args.stats, args.stat_errors)

    # Plot the histogram
    hep.histplot(draw_hist, ax=ax, color=COLOURS[0], marker=MARKERS[0], label=lbl, **PLOT_KWARGS)

    # Set the title and axis labels, using defaults from the histogram if not provided
    title_kw = {"fontsize": args.title_size} if args.title_size else {}

    # Set the title if not suppressed
    if not args.no_title:
        # pad=15 pushes the title up so it doesn't overlap the loc="top" y-axis label
        ax.set_title(args.title1 or hist.GetTitle(), pad=15, **title_kw)

    # Apply common styling to the axes, including labels and ranges
    apply_common_axes_style(
        ax, args,
        xlabel=args.xlabel1 or hist.GetXaxis().GetTitle() or "Variable",
        ylabel=args.ylabel1 or hist.GetYaxis().GetTitle() or "Events / Bin",
        yrange=args.yrange1
    )

    # loc="best" forces matplotlib to find an empty corner for the legend
    ax.legend(frameon=True, fontsize=args.legend_size, loc="best")

    # Save the plot to the specified output directory with the appropriate filename
    stem = os.path.basename(hist_path.strip("/"))
    if args.normalise:
        stem += "_norm"
    if args.logx:
        stem += "_logx"
    if args.logy:
        stem += "_logy"
    pf.save_plot(fig, args.output, stem, args.formats)


def plot_comparison(args):
    """Handles plotting two histograms, either overlaid or side-by-side."""

    # Load both histograms from the specified ROOT files and paths
    h1 = pf.load_histogram(args.filename1, args.histogram_path1)
    fn2 = args.filename2 or args.filename1
    h2 = pf.load_histogram(fn2, args.histogram_path2)

    if not h1 or not h2:
        return

    # Determine whether to normalise the histograms and prepare them for plotting
    # Create clones of the histograms to avoid modifying the originals
    draw_h1 = h1.Clone("h1_norm") if args.normalise and h1.Integral() > 0 else h1
    draw_h2 = h2.Clone("h2_norm") if args.normalise and h2.Integral() > 0 else h2

    # If normalisation is requested, scale both histograms to unit area
    if args.normalise:
        if h1.Integral() > 0:
            draw_h1.Scale(1.0 / draw_h1.Integral())
        if h2.Integral() > 0:
            draw_h2.Scale(1.0 / draw_h2.Integral())

    # Format the legend labels for both histograms, including statistics if requested
    lbl1 = pf.format_stats_label(h1, args.label1, args.stats, args.stat_errors)
    lbl2 = pf.format_stats_label(h2, args.label2, args.stats, args.stat_errors)

    # Set font sizes for titles and labels if specified in the arguments
    title_kw = {"fontsize": args.title_size} if args.title_size else {}
    label_kw = {"fontsize": args.label_size} if args.label_size else {}

    # Determine the base filename stem for saving the plot, based on the input histogram paths
    stem = f"{os.path.basename(args.histogram_path1)}_vs_{os.path.basename(args.histogram_path2)}"

    # ===== Layout 1: Side-by-Side Panels =====
    if args.layout == "side-by-side":

        # Create a figure with two subplots, sharing the y-axis if requested
        share_y = (args.yaxes == "shared")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=share_y, layout="constrained")

        # Plot the histograms on their respective axes
        hep.histplot(draw_h1, ax=ax1, color=COLOURS[0], marker=MARKERS[0], label=lbl1, **PLOT_KWARGS)
        hep.histplot(draw_h2, ax=ax2, color=COLOURS[1], marker=MARKERS[1], label=lbl2, **PLOT_KWARGS)

        # Set titles for each subplot if not suppressed
        if not args.no_title:
            ax1.set_title(args.title1 or h1.GetTitle(), pad=15, **title_kw)
            ax2.set_title(args.title2 or h2.GetTitle(), pad=15, **title_kw)

        # Apply common styling to both axes, including labels and ranges
        apply_common_axes_style(
            ax1, args,
            xlabel=args.xlabel1 or h1.GetXaxis().GetTitle() or "Variable",
            ylabel=args.ylabel1 or h1.GetYaxis().GetTitle() or "Events / Bin",
            yrange=args.yrange1
        )
        apply_common_axes_style(
            ax2, args,
            xlabel=args.xlabel2 or h2.GetXaxis().GetTitle() or "Variable",
            ylabel=args.ylabel2 or h2.GetYaxis().GetTitle() or "Events / Bin" if not share_y else None,
            yrange=args.yrange1 if share_y else args.yrange2
        )

        # Add legends to both subplots
        ax1.legend(frameon=True, fontsize=args.legend_size, loc="best")
        ax2.legend(frameon=True, fontsize=args.legend_size, loc="best")

        stem += "_sbs"

    # ===== Layout 2: Single Overlay =====
    else:
        # Create a single figure and axis for overlaying both histograms
        fig, ax1 = plt.subplots(figsize=(8, 6), layout="constrained")

        # Set the title for the overlay plot if not suppressed
        if not args.no_title:
            ax1.set_title(args.title1 or f"{h1.GetTitle()} & {h2.GetTitle()}", pad=15, **title_kw)

        # If the user requested separate y-axes
        if args.yaxes == "separate":

            # Create a twin axis sharing the same x-axis for the second histogram
            ax2 = ax1.twinx()

            # Plot both histograms on their respective axes with different colors and markers
            hep.histplot(draw_h1, ax=ax1, color=COLOURS[0], marker=MARKERS[0], label=lbl1, **PLOT_KWARGS)
            hep.histplot(draw_h2, ax=ax2, color=COLOURS[1], marker=MARKERS[1], label=lbl2, **PLOT_KWARGS)

            # Apply common styling to both axes, including labels and ranges
            apply_common_axes_style(ax1, args, xlabel=args.xlabel1 or h1.GetXaxis().GetTitle() or "Variable", yrange=args.yrange1)
            apply_common_axes_style(ax2, args, yrange=args.yrange2)

            # Set y-axis labels and colors for both axes, using defaults if not provided
            ax1.set_ylabel(args.ylabel1 or h1.GetYaxis().GetTitle() or "Events / Bin", color=COLOURS[0], loc="top", **label_kw)
            ax2.set_ylabel(args.ylabel2 or h2.GetYaxis().GetTitle() or "Events / Bin", color=COLOURS[1], loc="top", **label_kw)
            ax1.tick_params(axis="y", labelcolor=COLOURS[0])
            ax2.tick_params(axis="y", labelcolor=COLOURS[1])

            # Combine legends from both axes into a single legend on the first axis
            h_1, l_1 = ax1.get_legend_handles_labels()
            h_2, l_2 = ax2.get_legend_handles_labels()
            ax1.legend(h_1 + h_2, l_1 + l_2, frameon=True, fontsize=args.legend_size, loc="best")

        # Otherwise, if the user requested a shared y-axis, plot both histograms on the same axis
        else:

            # Plot both histograms on the same axis with different colors and markers
            hep.histplot(draw_h1, ax=ax1, color=COLOURS[0], marker=MARKERS[0], label=lbl1, **PLOT_KWARGS)
            hep.histplot(draw_h2, ax=ax1, color=COLOURS[1], marker=MARKERS[1], label=lbl2, **PLOT_KWARGS)

            # Apply common styling to the shared axis, including labels and ranges
            apply_common_axes_style(
                ax1, args,
                xlabel=args.xlabel1 or h1.GetXaxis().GetTitle() or "Variable",
                ylabel=args.ylabel1 or h1.GetYaxis().GetTitle() or "Events / Bin",
                yrange=args.yrange1
            )
            ax1.legend(frameon=True, fontsize=args.legend_size, loc="best")

    # Save the plot to the specified output directory with the appropriate filename
    if args.normalise:
        stem += "_norm"
    if args.logx:
        stem += "_logx"
    if args.logy:
        stem += "_logy"
    pf.save_plot(fig, args.output, stem, args.formats)


def parse_args():
    parser = argparse.ArgumentParser(description="Configurable ATLAS Histogram Plotter.")

    parser.add_argument("-f1", "--filename1", required=True, help="Path to first ROOT file")
    parser.add_argument("-H1", "--histogram_path1", default=None, help="Path to hist 1 in file 1")
    parser.add_argument("-f2", "--filename2", default=None, help="Path to second ROOT file (defaults to file 1)")
    parser.add_argument("-H2", "--histogram_path2", default=None, help="Path to hist 2 in file 2")
    parser.add_argument("-d", "--directory", default=None, help="Plot every 1D hist in this ROOT folder")
    parser.add_argument("-o", "--output", default="plots", help="Subfolder name inside outputs/")

    parser.add_argument("-t1", "--title1", default=None, help="Title for plot/panel 1")
    parser.add_argument("-t2", "--title2", default=None, help="Title for panel 2")
    parser.add_argument("-l1", "--label1", default=None, help="Legend header label for hist 1")
    parser.add_argument("-l2", "--label2", default=None, help="Legend header label for hist 2")
    parser.add_argument("-x1", "--xlabel1", default=None, help="X-axis title 1")
    parser.add_argument("-x2", "--xlabel2", default=None, help="X-axis title 2 (side-by-side mode)")
    parser.add_argument("-y1", "--ylabel1", default=None, help="Y-axis title 1")
    parser.add_argument("-y2", "--ylabel2", default=None, help="Y-axis title 2 (twin axis or side-by-side)")

    parser.add_argument("--title-size", type=float, default=None, help="Font size for plot titles")
    parser.add_argument("--label-size", type=float, default=None, help="Font size for axis labels")
    parser.add_argument("--legend-size", type=float, default=11, help="Font size for legend text")
    parser.add_argument("--no-title", action="store_true", help="Omit plot titles")

    parser.add_argument("--xrange", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--yrange1", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--yrange2", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("-lx", "--logx", action="store_true", default=False)
    parser.add_argument("-ly", "--logy", action="store_true", default=False)

    parser.add_argument("--normalise", action="store_true", default=False, help="Normalise histograms to unit area")
    parser.add_argument("--layout", choices=["overlay", "side-by-side"], default="overlay")
    parser.add_argument("--yaxes", choices=["separate", "shared"], default="separate")

    parser.add_argument(
        "--stats", nargs="*", default=["entries", "mean", "std"],
        choices=["entries", "integral", "mean", "std", "none"],
        help="Statistics to list inside the legend. Pass 'none' to disable."
    )
    parser.add_argument("--stat-errors", action="store_true", default=False, help="Show error bounds on mean/std")
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf"])

    args = parser.parse_args()

    if "none" in args.stats or not args.stats:
        args.stats = []

    return args


if __name__ == "__main__":
    # Parse command-line arguments and execute the appropriate plotting function based on the provided inputs
    args = parse_args()

    # If a directory is specified, find and plot all 1D histograms in that directory
    if args.directory:

        # Recursively find all 1D histograms in the specified ROOT file and directory
        paths = pf.find_all_histograms(args.filename1, args.directory)
        print(f"Found {len(paths)} histograms in '{args.directory}'")

        # If no histograms are found, print a message and exit
        for p in paths:
            plot_single(args.filename1, p, args)

    # If two histogram paths are specified, plot them for comparison
    elif args.histogram_path2:
        plot_comparison(args)

    # If only one histogram path is specified, plot that single histogram
    elif args.histogram_path1:
        plot_single(args.filename1, args.histogram_path1, args)

    # If neither histogram path nor directory is specified, print an error message
    else:
        print("Error: Specify either -H1, -d, or both -H1 and -H2.")