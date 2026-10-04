import os
import ROOT as root
import matplotlib.pyplot as plt
import mplhep as hep

# Use the standard ATLAS Matplotlib style (fonts, ticks, margins)
hep.style.use("ATLAS")

# Run ROOT in batch mode so it doesn't try to open X11 graphical windows
root.gROOT.SetBatch(True)


def load_histogram(filename, hist_path):
    """Safely loads a 1D TH1 or TEfficiency object from a ROOT file."""

    # Open the ROOT file and check for errors
    rfile = root.TFile.Open(filename)

    # Check if the file was opened successfully
    if not rfile or rfile.IsZombie():

        # Print an error message and return None if the file could not be opened
        print(f"Error: Could not open {filename}")
        return None

    # Try to retrieve the histogram or efficiency object from the specified path
    obj = rfile.Get(hist_path.strip("/"))

    # Check if the object was found and print an error message if not
    if not obj:
        print(f"Error: '{hist_path}' not found in {filename}")
        rfile.Close()
        return None

    # Check if the object is a TEfficiency
    if obj.InheritsFrom("TEfficiency"):

        # If it's a TEfficiency, we need to extract the total histogram and convert it to a TH1
        eff = obj
        total = eff.GetTotalHistogram()

        # Safety check: skip 2D or 3D efficiencies
        if total.GetDimension() != 1:
            print(f"Skipping {hist_path}: not a 1D efficiency.")
            rfile.Close()
            return None

        # Clone the histogram
        hist = total.Clone(eff.GetName() + "_eff")
        hist.SetDirectory(0)
        hist.Reset()

        # Put the proper title from the TEfficiency object
        hist.SetTitle(eff.GetTitle().split(";")[0])
        hist.GetYaxis().SetTitle("Efficiency")

        # Fill the histogram with efficiency values and errors
        for b in range(1, hist.GetNbinsX() + 1):

            # Only fill bins where the total histogram has content to avoid division by zero
            if total.GetBinContent(b) > 0:

                # Set the bin content to the efficiency value
                hist.SetBinContent(b, eff.GetEfficiency(b))

                # Calculate the error as the average of the lower and upper errors
                err = 0.5 * (eff.GetEfficiencyErrorLow(b) + eff.GetEfficiencyErrorUp(b))

                # Set the bin error to the calculated error
                hist.SetBinError(b, err)

        # Reset() and SetBinContent() leave the entry count meaningless, so restore it from the total histogram
        hist.SetEntries(total.GetEntries())

    else:

        # Standard TH1s: skip anything that is not a 1D histogram
        if not obj.InheritsFrom("TH1") or obj.GetDimension() != 1:
            print(f"Skipping {hist_path}: not a 1D histogram.")
            rfile.Close()
            return None

        hist = obj
        hist.SetDirectory(0)

    rfile.Close()
    return hist


def find_all_histograms(filename, directory=""):
    """Recursively scans a ROOT file or specific TDirectory to find all 1D histograms."""

    # Open the ROOT file and check for errors
    rfile = root.TFile.Open(filename)

    # Check if the file was opened successfully
    if not rfile or rfile.IsZombie():
        return []

    # If a specific directory is provided, navigate to it; otherwise, start from the root
    target_dir = rfile.Get(directory.strip("/")) if directory else rfile

    # Check if the directory was found and print an error message if not
    if not target_dir:
        print(f"Error: Directory '{directory}' not found in {filename}")
        rfile.Close()
        return []

    paths = []

    # Define a recursive function to walk through the directory structure
    def _walk(curr_dir, prefix):

        # Loop through all keys in the current directory
        for key in curr_dir.GetListOfKeys():

            # Get the name of the key and construct the full path
            name = key.GetName()

            # Construct the full path by appending the current prefix (if any) to the name
            full_path = f"{prefix}/{name}" if prefix else name

            # Get the class of the object associated with the key
            cls = root.TClass.GetClass(key.GetClassName())

            # Skip if the class is not found (e.g., corrupted or unknown object)
            if not cls:
                continue

            # Check if the object is a TDirectory, TH1 (but not TH2/TH3), or TEfficiency
            if cls.InheritsFrom("TDirectory"):
                _walk(curr_dir.Get(name), full_path)

            # Check if the object is a 1D histogram (TH1) but not a 2D or 3D histogram (TH2/TH3)
            elif cls.InheritsFrom("TH1") and not (cls.InheritsFrom("TH2") or cls.InheritsFrom("TH3")):
                paths.append(full_path)

            # Check if the object is a TEfficiency and ensure it is 1D before adding to the list
            elif cls.InheritsFrom("TEfficiency"):

                # Must open the efficiency object to check its dimension
                eff = curr_dir.Get(name)

                # Only add the path if the efficiency object is 1D
                if eff and eff.GetTotalHistogram().GetDimension() == 1:
                    paths.append(full_path)

    # Start the recursive walk from the target directory
    _walk(target_dir, directory.strip("/"))
    rfile.Close()

    # Return a sorted list of unique histogram paths found in the ROOT file
    return sorted(set(paths))


def format_stats_label(hist, label=None, stats_choices=None, stat_errors=False):
    """Builds a multi-line string for the legend, hiding header if 'none' is passed."""

    # If the label is explicitly set to "none" (case-insensitive), we skip adding a header line
    if label and label.lower() == "none":
        lines = []

    # If a label is provided, use it; otherwise, use the histogram's title or a default header
    else:
        header = label or hist.GetTitle() or "Histogram"
        lines = [header]

    # If no specific statistics are requested, return the header line(s) only
    if not stats_choices:
        return "\n".join(lines)

    # Loop through the requested statistics and format them accordingly
    for stat in stats_choices:
        if stat == "entries":
            lines.append(f"Entries = {int(hist.GetEntries()):,}")
        elif stat == "integral":
            lines.append(f"Integral = {hist.Integral():.3g}")
        elif stat == "mean":
            err = f" ± {hist.GetMeanError():.2g}" if stat_errors else ""
            lines.append(f"Mean = {hist.GetMean():.3g}{err}")
        elif stat == "std":
            err = f" ± {hist.GetStdDevError():.2g}" if stat_errors else ""
            lines.append(f"Std Dev = {hist.GetStdDev():.3g}{err}")

    # Join the lines with newline characters and return the formatted string
    return "\n".join(lines)


def save_plot(fig, output_subfolder, stem, formats=("png",)):
    """Handles standardizing the output directory structure and saving."""

    # Create the output directory if it doesn't exist
    out_dir = os.path.join("outputs", output_subfolder)
    os.makedirs(out_dir, exist_ok=True)

    # Loop through the specified formats and save the figure in each format
    for fmt in formats:

        # Construct the full file path for saving the figure
        filepath = os.path.join(out_dir, f"{stem}.{fmt}")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")

    plt.close(fig)