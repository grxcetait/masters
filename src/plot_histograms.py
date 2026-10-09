#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import custom plotting functions from the utils module
import utils.plotting_functions as pf

COLOURS = ("black", "#c1272d")
MARKERS = ("o", "s")

PLOT_KWARGS = {
    "histtype": "errorbar",
    "markersize": 4,
    "elinewidth": 1.2,
    "capsize": 2
}


def prepare_hist(h, name, normalise):
    """Returns a drawable copy of h; the original is never modified.

    The copy is optionally normalised to unit area, and bins with zero or
    negative content are replaced with NaN so they are not plotted.
    """
    d = h.Clone(name)
    d.SetDirectory(0)  # Detach from ROOT's current directory (avoids name-clash warnings)

    if normalise and h.Integral() > 0:

        # Make sure errors are stored so they scale with the content
        if d.GetSumw2N() == 0:
            d.Sumw2()

        d.Scale(1.0 / h.Integral())

    for b in range(1, d.GetNbinsX() + 1):
        if d.GetBinContent(b) <= 0:
            d.SetBinContent(b, float("nan"))
            d.SetBinError(b, 0.0)

    return d


def apply_common_axes_style(ax, args, xlabel=None, ylabel=None, yrange=None, logy=None):
    """Applies repetitive styling to a given axis.

    logy=None follows --logy; pass logy=False to force a linear y-axis (e.g. ratio panel).
    """

    # Set axis labels and titles, using defaults if not overridden
    font_kw = {"fontsize": args.label_size} if args.label_size else {}

    if xlabel:
        ax.set_xlabel(xlabel, loc="right", **font_kw)
    if ylabel:
        ax.set_ylabel(ylabel, loc="top", **font_kw)

    use_logy = args.logy if logy is None else logy

    if args.logx:
        ax.set_xscale("log")
    if use_logy:
        ax.set_yscale("log")
    if args.xrange and args.splitx is None:
        ax.set_xlim(*args.xrange)
    if yrange:
        ax.set_ylim(*yrange)


def plot_single(filename, hist_path, args):
    """Generates a standard single-histogram plot."""

    # Load the histogram from the ROOT file
    hist = pf.load_histogram(filename, hist_path)
    if not hist:
        return

    # Drawable copy: normalised if requested, empty bins masked (original untouched)
    draw_hist = prepare_hist(hist, hist.GetName() + "_draw", args.normalise)

    # Format the legend label with statistics (computed from the original histogram)
    lbl = pf.format_stats_label(hist, args.label1, args.stats, args.stat_errors)

    # Set font sizes for titles and labels if specified
    title_kw = {"fontsize": args.title_size} if args.title_size else {}
    label_kw = {"fontsize": args.label_size} if args.label_size else {}

    # Determine global X limits (fallback to histogram limits if args.xrange isn't passed)
    xmin = args.xrange[0] if args.xrange else hist.GetXaxis().GetXmin()
    xmax = args.xrange[1] if args.xrange else hist.GetXaxis().GetXmax()

    stem = os.path.basename(hist_path.strip("/"))

    # ===== Split X-Axis Layout =====
    if args.splitx is not None:

        # Create a figure with two vertically stacked subplots for the split X-axis layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), layout="constrained")

        # Plot the exact same histogram on BOTH axes
        hep.histplot(draw_hist, ax=ax1, color=COLOURS[0], marker=MARKERS[0], label=lbl,
                     linestyle="-" if args.connect else "None", **PLOT_KWARGS)
        hep.histplot(draw_hist, ax=ax2, color=COLOURS[0], marker=MARKERS[0],
                     linestyle="-" if args.connect else "None", **PLOT_KWARGS)  # No label to avoid duplicate legend

        # Set the title for the upper panel if not suppressed
        if not args.no_title:
            ax1.set_title(args.title1 or hist.GetTitle(), pad=15, **title_kw)

        # Apply the X-axis split
        ax1.set_xlim(xmin, args.splitx)
        ax2.set_xlim(args.splitx, xmax)

        # Apply common styling to both panels (yrange1 = upper, yrange2 = lower)
        apply_common_axes_style(ax1, args, yrange=args.yrange1)
        apply_common_axes_style(ax2, args,
                                xlabel=args.xlabel1 or hist.GetXaxis().GetTitle() or "Variable",
                                yrange=args.yrange2)

        # Manually add Y-labels to both panels
        ylabel = args.ylabel1 or hist.GetYaxis().GetTitle() or "Events / Bin"
        ax1.set_ylabel(ylabel, loc="top", **label_kw)
        ax2.set_ylabel(ylabel, loc="top", **label_kw)

        # Legend on the upper panel only
        ax1.legend(frameon=True, fontsize=args.legend_size, loc="best")

        stem += "_splitx"

    # ===== Standard Single Layout =====
    else:

        # Create a single panel for the histogram
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

        # Draw the histogram using mplhep's histplot function with the specified styling
        hep.histplot(draw_hist, ax=ax, color=COLOURS[0], marker=MARKERS[0], label=lbl,
                     linestyle="-" if args.connect else "None", **PLOT_KWARGS)

        # Set the title for the plot if not suppressed
        if not args.no_title:
            ax.set_title(args.title1 or hist.GetTitle(), pad=15, **title_kw)

        # Apply common styling to the axes, including titles, labels, ranges, and scales
        apply_common_axes_style(ax, args,
                                xlabel=args.xlabel1 or hist.GetXaxis().GetTitle() or "Variable",
                                ylabel=args.ylabel1 or hist.GetYaxis().GetTitle() or "Events / Bin",
                                yrange=args.yrange1)

        # Add legend to the plot
        ax.legend(frameon=True, fontsize=args.legend_size, loc="best")

    # Save the plot to the specified output directory with the appropriate filename
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

    # Drawable copies: normalised if requested, empty bins masked (originals untouched)
    draw_h1 = prepare_hist(h1, "h1_draw", args.normalise)
    draw_h2 = prepare_hist(h2, "h2_draw", args.normalise)

    # Format the legend labels for both histograms (statistics come from the originals)
    lbl1 = pf.format_stats_label(h1, args.label1, args.stats, args.stat_errors)
    lbl2 = pf.format_stats_label(h2, args.label2, args.stats, args.stat_errors)

    # Set font sizes for titles and labels if specified in the arguments
    title_kw = {"fontsize": args.title_size} if args.title_size else {}
    label_kw = {"fontsize": args.label_size} if args.label_size else {}

    # Determine the base filename stem for saving the plot, based on the input histogram paths
    stem = f"{os.path.basename(args.histogram_path1)}_vs_{os.path.basename(args.histogram_path2)}"

    # Warn when --ratio will be ignored
    if args.ratio and args.splitx is not None:
        print("Note: --ratio is ignored when --splitx is used.")
    elif args.ratio and args.layout == "side-by-side":
        print("Note: --ratio is ignored with --layout side-by-side.")
    elif args.ratio and args.yaxes == "separate":
        print("Note: --ratio needs --yaxes shared; ignoring --ratio.")

    # ===== Layout 0: Split X-Axis (both histograms overlaid on each panel) =====
    # Note: this uses a shared y-axis per panel, so --yaxes separate and --ratio are ignored here.
    if args.splitx is not None:

        xmin = args.xrange[0] if args.xrange else h1.GetXaxis().GetXmin()
        xmax = args.xrange[1] if args.xrange else h1.GetXaxis().GetXmax()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), layout="constrained")

        # Plot both histograms on both panels; only the upper panel carries legend labels
        for ax, labelled in ((ax1, True), (ax2, False)):
            kw1 = dict(PLOT_KWARGS, **({"label": lbl1} if labelled else {}))
            kw2 = dict(PLOT_KWARGS, **({"label": lbl2} if labelled else {}))
            hep.histplot(draw_h1, ax=ax, color=COLOURS[0], marker=MARKERS[0],
                         linestyle="-" if args.connect else "None", **kw1)
            hep.histplot(draw_h2, ax=ax, color=COLOURS[1], marker=MARKERS[1],
                         linestyle="-" if args.connect else "None", **kw2)

        if not args.no_title:
            ax1.set_title(args.title1 or f"{h1.GetTitle()} & {h2.GetTitle()}", pad=15, **title_kw)

        ax1.set_xlim(xmin, args.splitx)
        ax2.set_xlim(args.splitx, xmax)

        # Apply common styling to both panels (yrange1 = upper, yrange2 = lower)
        apply_common_axes_style(ax1, args, yrange=args.yrange1)
        apply_common_axes_style(ax2, args,
                                xlabel=args.xlabel1 or h1.GetXaxis().GetTitle() or "Variable",
                                yrange=args.yrange2)

        ylabel = args.ylabel1 or h1.GetYaxis().GetTitle() or "Events / Bin"
        ax1.set_ylabel(ylabel, loc="top", **label_kw)
        ax2.set_ylabel(ylabel, loc="top", **label_kw)

        ax1.legend(frameon=True, fontsize=args.legend_size, loc="best")

        stem += "_splitx"

    # ===== Layout 1: Side-by-Side Panels =====
    elif args.layout == "side-by-side":

        # Create a figure with two subplots, sharing the y-axis if requested
        share_y = (args.yaxes == "shared")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=share_y, layout="constrained")

        # Plot the histograms on their respective axes
        hep.histplot(draw_h1, ax=ax1, color=COLOURS[0], marker=MARKERS[0], label=lbl1,
                     linestyle="-" if args.connect else "None", **PLOT_KWARGS)
        hep.histplot(draw_h2, ax=ax2, color=COLOURS[1], marker=MARKERS[1], label=lbl2,
                     linestyle="-" if args.connect else "None", **PLOT_KWARGS)

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
            ylabel=(args.ylabel2 or h2.GetYaxis().GetTitle() or "Events / Bin") if not share_y else None,
            yrange=args.yrange1 if share_y else args.yrange2
        )

        # Add legends to both subplots
        ax1.legend(frameon=True, fontsize=args.legend_size, loc="best")
        ax2.legend(frameon=True, fontsize=args.legend_size, loc="best")

        stem += "_sbs"

    # ===== Layout 2: Single Overlay =====
    else:

        # The ratio panel is only available with a shared y-axis
        show_lower = args.ratio and args.yaxes != "separate"

        if show_lower:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")

        # If ratio is not requested, create a single panel for overlaying the histograms
        else:
            fig, ax1 = plt.subplots(figsize=(8, 6), layout="constrained")
            ax2 = None

        # Set the title for the main plot if not suppressed
        if not args.no_title:
            ax1.set_title(args.title1 or f"{h1.GetTitle()} & {h2.GetTitle()}", pad=15, **title_kw)

        # If the user requested separate y-axes (twin axes)
        if args.yaxes == "separate":

            # Create a twin y-axis for the second histogram
            ax_twin = ax1.twinx()

            # Plot the histograms on their respective axes
            hep.histplot(draw_h1, ax=ax1, color=COLOURS[0], marker=MARKERS[0], label=lbl1,
                         linestyle="-" if args.connect else "None", **PLOT_KWARGS)
            hep.histplot(draw_h2, ax=ax_twin, color=COLOURS[1], marker=MARKERS[1], label=lbl2,
                         linestyle="-" if args.connect else "None", **PLOT_KWARGS)

            # Apply common styling to both axes, including labels and ranges
            apply_common_axes_style(ax1, args, xlabel=args.xlabel1 or h1.GetXaxis().GetTitle() or "Variable", yrange=args.yrange1)
            apply_common_axes_style(ax_twin, args, yrange=args.yrange2)

            # Set y-axis labels for both axes, using defaults from the histograms if not provided
            ax1.set_ylabel(args.ylabel1 or h1.GetYaxis().GetTitle() or "Events / Bin", color=COLOURS[0], loc="top", **label_kw)
            ax_twin.set_ylabel(args.ylabel2 or h2.GetYaxis().GetTitle() or "Events / Bin", color=COLOURS[1], loc="top", **label_kw)

            # Set the tick label colours
            ax1.tick_params(axis="y", labelcolor=COLOURS[0])
            ax_twin.tick_params(axis="y", labelcolor=COLOURS[1])

            # Combine legends from both axes into a single legend on the main axis
            h_1, l_1 = ax1.get_legend_handles_labels()
            h_2, l_2 = ax_twin.get_legend_handles_labels()
            ax1.legend(h_1 + h_2, l_1 + l_2, frameon=True, fontsize=args.legend_size, loc="best")

        # Standard shared y-axis with optional Ratio panel
        else:

            # Plot both histograms on the same axis
            hep.histplot(draw_h1, ax=ax1, color=COLOURS[0], marker=MARKERS[0], label=lbl1,
                         linestyle="-" if args.connect else "None", **PLOT_KWARGS)
            hep.histplot(draw_h2, ax=ax1, color=COLOURS[1], marker=MARKERS[1], label=lbl2,
                         linestyle="-" if args.connect else "None", **PLOT_KWARGS)

            # Only label the x-axis on the main plot if there is no ratio plot
            main_xlabel = None if show_lower else (args.xlabel1 or h1.GetXaxis().GetTitle() or "Variable")
            apply_common_axes_style(ax1, args, xlabel=main_xlabel,
                                    ylabel=args.ylabel1 or h1.GetYaxis().GetTitle() or "Events / Bin",
                                    yrange=args.yrange1)
            ax1.legend(frameon=True, fontsize=args.legend_size, loc="best")

            # Draw the ratio panel
            if show_lower:

                # Plain ratio Hist1 / Hist2; ROOT propagates the statistical errors
                h_ratio = draw_h2.Clone("h_ratio")
                h_ratio.SetDirectory(0)
                h_ratio.Divide(draw_h1)

                # Mask bins where either histogram is empty, non-positive, or NaN
                for b in range(1, h_ratio.GetNbinsX() + 1):
                    c1 = draw_h1.GetBinContent(b)
                    c2 = draw_h2.GetBinContent(b)

                    if c1 != c1 or c2 != c2 or c1 <= 0 or c2 <= 0:
                        h_ratio.SetBinContent(b, float("nan"))
                        h_ratio.SetBinError(b, 0.0)

                hep.histplot(h_ratio, ax=ax2, color="black", marker="o",
                             linestyle="-" if args.connect else "none", **PLOT_KWARGS)

                # Baseline reference at ratio = 1
                ax2.axhline(1.0, color="black", linestyle="--", alpha=0.5, linewidth=1.2)

                # Ratio axis stays linear even when --logy is set for the main panel
                apply_common_axes_style(ax2, args, xlabel=args.xlabel1 or h1.GetXaxis().GetTitle() or "Variable",
                                        yrange=args.yrange2 or (0.5, 1.5), logy=False)
                ax2.set_ylabel(ylabel=args.ylabel2 or "Hist 2 / Hist 1", loc="center", **label_kw)

                stem += "_ratio"

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
    parser.add_argument("--yrange1", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="Y range for plot 1 (upper panel when using --splitx)")
    parser.add_argument("--yrange2", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="Y range for plot 2 (lower panel when using --splitx, ratio panel when using --ratio)")
    parser.add_argument("-lx", "--logx", action="store_true", default=False)
    parser.add_argument("-ly", "--logy", action="store_true", default=False)

    parser.add_argument("--normalise", action="store_true", default=False, help="Normalise histograms to unit area")
    parser.add_argument("--layout", choices=["overlay", "side-by-side"], default="overlay")
    parser.add_argument("--yaxes", choices=["separate", "shared"], default="separate")

    parser.add_argument("--splitx", type=float, default=None, help="X-axis value at which to split the graph vertically (upper panel for x < split, lower panel for x > split).")
    parser.add_argument("--connect", action="store_true", help="Draw lines connecting the data points.")
    parser.add_argument("--ratio", action="store_true", help="Draw a ratio plot (Hist1 / Hist2) in a lower panel. Only applies to overlay layout with --yaxes shared.")

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

        # Plot each histogram found
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