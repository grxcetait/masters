import os
import ROOT as root
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import numpy as np

# Use the standard ATLAS Matplotlib style (fonts, ticks, margins)
hep.style.use("ATLAS")

# Run ROOT in batch mode so it doesn't try to open X11 graphical windows
root.gROOT.SetBatch(True)

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


def rdataframe_to_awkward(df, suffixes):

    array = ak.from_rdataframe(df, columns = (suffixes))

    return array