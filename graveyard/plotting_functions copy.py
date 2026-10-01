import argparse
import os

import matplotlib.pyplot as plt
import mplhep as hep
import ROOT as root
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# Colours / markers for the first and second histogram. The different marker
# shapes keep the two distinguishable in greyscale prints.
COLOURS = ("black", "#c1272d")
MARKERS = ("o", "s")

# Statistics that can be shown in the box, in the order they are displayed
STAT_CHOICES = ["entries", "integral", "mean", "std", "fwhm"]
DEFAULT_STATS = ["entries", "mean", "std", "fwhm"]

# Default legend / statistics box font size
DEFAULT_LEGEND_SIZE = 14


# ===== Helpers =====

def compute_fwhm(hist):
    """Full width at half maximum of a 1D histogram, or None if it can't be found.
    Uses linear interpolation between bin centres at the half-maximum crossings."""

    # Extract the bin data
    n = hist.GetNbinsX()
    contents = [hist.GetBinContent(i) for i in range(1, n + 1)]
    centres = [hist.GetBinCenter(i) for i in range(1, n + 1)]

    # Find the peak and half-maximum
    peak = max(contents)

    # If the histogram is empty or contains non-positive values, return None
    if peak <= 0:
        return None
    half = peak / 2
    i_peak = contents.index(peak)

    # Define a linear interpolation function
    def crossing(i_inside, i_outside):
        # Interpolate x where the content crosses `half` between the two bins
        y_in, y_out = contents[i_inside], contents[i_outside]
        x_in, x_out = centres[i_inside], centres[i_outside]
        return x_in + (half - y_in) / (y_out - y_in) * (x_out - x_in)

    # Walk left from the peak to the first bin below half maximum to find crossings
    left = None
    for i in range(i_peak, 0, -1):
        if contents[i - 1] < half:
            left = crossing(i, i - 1)
            break

    # Walk right from the peak to the first bin below half maximum to find crossings
    right = None
    for i in range(i_peak, n - 1):
        if contents[i + 1] < half:
            right = crossing(i, i + 1)
            break

    # If the distribution never drops below half maximum on one side, there is no FWHM
    if left is None or right is None:
        return None
    return right - left


def normalise_hist(hist):
    """Return a copy scaled so that the bin contents sum to 1. Errors are scaled
    consistently."""

    # Calculate the sum of the bin contents over the histogram range
    integral = hist.Integral()

    # Check if the histogram is empty or has a negative sum
    if integral <= 0:
        print("Warning: histogram has a non-positive integral, cannot normalise.")
        return hist

    # Create a clone and detatch histogram 
    normalised = hist.Clone(hist.GetName() + "_norm")
    normalised.SetDirectory(0)

    # Sumw2 must exist before scaling so that the errors are scaled as well
    if normalised.GetSumw2N() == 0:
        normalised.Sumw2()

    # Scale to unit area 
    normalised.Scale(1.0 / integral)
    return normalised


def prepare(hist, opts):
    """Return (histogram to draw, original histogram to take statistics from).
    Statistics always come from the original so entries/mean/etc. are never
    affected by normalisation."""
    if opts.normalise:

        # Check if the histogram is an efficiency
        if hist.GetYaxis().GetTitle() == "Efficiency":
            print("Note: efficiencies are not normalised.")
        else:
            return normalise_hist(hist), hist
    return hist, hist


def font_kw(size):
    """Keyword arguments for a font size. Empty if no size was given, so that the
    ATLAS style default is kept."""
    return {"fontsize": size} if size else {}


def box_text(label, lines):
    """Text of a legend / statistics box: the optional label, then the statistic lines."""
    return "\n".join(([label] if label else []) + lines)


def apply_ranges(ax, x_range_user, y_range_user):
    """Apply user-given axis ranges (None = leave the automatic range alone)."""
    if x_range_user:
        ax.set_xlim(*x_range_user)
    if y_range_user:
        ax.set_ylim(*y_range_user)


def x_range(hists, log_x):
    """Common x range of the histograms. A log axis cannot start at zero or below."""
    x_min = min(h.GetXaxis().GetXmin() for h in hists)
    x_max = max(h.GetXaxis().GetXmax() for h in hists)

    if log_x and x_min <= 0:
        positive_edges = [h.GetXaxis().GetBinUpEdge(i)
                          for h in hists for i in range(1, h.GetNbinsX() + 1)
                          if h.GetXaxis().GetBinUpEdge(i) > 0]
        x_min = min(positive_edges)

    return x_min, x_max


def output_folder_for(output):
    base_directory = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_directory, "outputs", output)
    os.makedirs(folder, exist_ok = True)
    return folder


def output_tags(opts, normalised, side_by_side = False):
    """Suffixes so that different variants of the same plot never overwrite each other."""
    tags = ""
    if normalised:
        tags += "_norm"
    if opts.log_x:
        tags += "_logx"
    if opts.log_y:
        tags += "_logy"
    if side_by_side:
        tags += "_sbs"
    return tags


def save_figure(fig, folder, stem, opts):
    for file_format in opts.formats:
        save_path = os.path.join(folder, f"{stem}.{file_format}")
        fig.savefig(save_path,dpi = 300, bbox_inches = "tight")
        print(f"Saved {file_format.upper()} to: {save_path}")
    plt.close(fig)


# ===== Drawing =====

def draw(ax, hist, colour, marker):
    hep.histplot(hist, ax = ax, histtype = "errorbar", yerr = True,
                 color = colour, marker = marker, markersize = 4,
                 elinewidth = 1.2, capsize = 2)


def stat_lines(hist, opts):
    """Build the short list of statistic lines requested by the user."""

    # If requested, don't show statistics
    if not opts.stats:
        return []

    def with_error(value, error):
        return f"{value:.3g} ± {error:.2g}" if opts.stat_errors else f"{value:.3g}"

    n_bins = hist.GetNbinsX()
    lines = []

    # Iterate in a fixed order so the box always looks the same
    for key in STAT_CHOICES:
        if key not in opts.stats:
            continue
        if key == "entries":
            lines.append(f"Entries = {int(hist.GetEntries()):,}")
        elif key == "integral":
            lines.append(f"Integral = {hist.Integral():.3g}")
        elif key == "mean":
            lines.append(f"Mean = {with_error(hist.GetMean(), hist.GetMeanError())}")
        elif key == "std":
            lines.append(f"Std dev = {with_error(hist.GetStdDev(), hist.GetStdDevError())}")
        elif key == "fwhm":
            fwhm = compute_fwhm(hist)
            lines.append(f"FWHM = {fwhm:.3g}" if fwhm is not None else "FWHM = n/a")

    return lines


def add_stats_legend(ax, blocks, font_size, loc = "upper right"):
    """Draw one compact box. Each block is (text, colour, marker); a marker of None
    gives a plain text block, otherwise a coloured marker identifies the histogram."""
    if not blocks:
        return

    has_markers = any(marker is not None for _, _, marker in blocks)

    handles, labels = [], []
    for text, colour, marker in blocks:
        if marker is None:
            handles.append(mpatches.Patch(color = "none"))
        else:
            handles.append(Line2D([0], [0], color = colour, marker = marker,
                                  markersize = 5, linestyle = "none"))
        labels.append(text)

    ax.legend(handles = handles, labels = labels, loc = loc,
              frameon = True, fancybox = False, edgecolor = "0.6",
              framealpha = 0.9, facecolor = "white",
              handlelength = 1.0 if has_markers else 0,
              handletextpad = 0.6 if has_markers else 0,
              borderpad = 0.6, labelspacing = 0.8,
              alignment = "left", prop = {"size": font_size})


def add_headroom(axes, log_y, n_lines):
    """Extend the y range upwards so the stats box never sits on the data.
    The more lines of text, the more room is made. Also makes linear axes start at zero.
    User-given y ranges are applied afterwards and override this."""
    fraction = min(0.7, 0.12 + 0.05 * n_lines)

    for ax in axes:
        y_low, y_high = ax.get_ylim()
        if log_y:
            # Add `fraction` of the current span, measured in decades
            ax.set_ylim(y_low, y_high * (y_high / y_low) ** fraction)
        else:
            y_low = 0 if y_low > 0 else y_low
            ax.set_ylim(y_low, y_high + fraction * (y_high - y_low))


def set_title(ax, text, opts):
    if text and not opts.no_title:
        ax.set_title(text, **font_kw(opts.title_size))