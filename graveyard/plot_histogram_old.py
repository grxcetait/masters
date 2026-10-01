#!/usr/bin/env python3
# This tells the operating system which program to use to run the script. In this case, it is using Python 3.

# Import ROOT
import ROOT as root
import argparse
import os

# Run in batch mode to avoid opening GUI windows
root.gROOT.SetBatch(True)


def plot(filename, histogram_path, output):

    # Define datafiles output directory
    base_directory = os.path.dirname(os.path.abspath(__file__))
    outputs_folder = os.path.join(base_directory, "outputs", output)
    os.makedirs(outputs_folder, exist_ok=True)

    # Set the output filename as the histogram name
    histogram_name = os.path.basename(histogram_path)
    output_filename = f"{histogram_name}.png"
    save_path = os.path.join(outputs_folder, output_filename)

    # Open the ROOT file
    root_file = root.TFile.Open(filename)

    # Check if the file was successfully opened
    if not root_file or root_file.IsZombie():
        print("Error: Could not open ROOT file.")
        return

    # Retrieve the histogram from the ROOT file
    hist = root_file.Get(histogram_path)

    # Check if the histogram was successfully retrieved
    if not hist:
        print("Error: Could not retrieve histogram from the ROOT file.")
        root_file.Close()
        return

    # Set histogram scope and close the input file
    hist.SetDirectory(0)
    root_file.Close()

    # Create a canvas to draw on and draw the histogram
    canvas = root.TCanvas("canvas", "My Canvas", 800, 600)
    hist.Draw("AL")

    # Save the canvas as a PNG file
    canvas.SaveAs(save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    """Parse command line arguements"""
    parser = argparse.ArgumentParser(description = "Plot a histogram from a ROOT file.")
    parser.add_argument("-f", "--filename", type=str, required=True, help="The name of the ROOT file.")
    parser.add_argument("-H", "--histogram_path", type=str, required=True, help="The path to the histogram within the ROOT file (eg. /path/to/histogram).")
    parser.add_argument("-o", "--output", type=str, required=True, help="The name of the output folder.")
    args = parser.parse_args()

    # Call the function to plot the histogram
    plot(args.filename, args.histogram_path, args.output)