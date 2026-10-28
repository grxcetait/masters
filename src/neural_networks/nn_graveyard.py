#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path
import os

import uproot
import awkward as ak
import pandas as pd
import numpy as np

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# === ARGUEMENTS ====
filename = "CERNbox/initial_data/root_files/HHbbyy_200PU.root"
tree_name = "Events"
prefix = "HLTTrack_"

# Create a list of the branches we are interested in
track_branches = ["HLTTrack_pt", "HLTTrack_eta", "HLTTrack_d0", "HLTTrack_z0",
                  "HLTTrack_d0Err", "HLTTrack_z0Err", "TruthVertex_z", 
                  "TruthVertex_isValid"]

# Track selection (Run 1 vertex selection: |pt| > 0.5 GeV, |d0| < 4 mm, sig_d0 < 5 mm, sig_z0 < 10 mm)
pt_min = 1.5       # [GeV] 
d0_max = 4.0        # [mm]
sig_d0_max = 5.0    # [mm]
sig_z0_max = 10.0   # [mm]

# Grid along the beam axis
z_min = -200.0      # [mm]
z_max = 200.0       # [mm]
step = 0.1          # [mm] grid spacing, keep this smaller than res and tol

# Define model class
class VertexClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(6, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x


def load_dataset(track_variables):  #, truth_variables

    # Load in all events from the root file
    with uproot.open(f"{filename}:{tree_name}") as tree:
        events = tree.arrays(track_branches, entry_stop = 1)     # Only for 1 event for now 

    # Extract track branches
    pt = events[track_variables["pt"]]
    d0 = events[track_variables["d0"]]
    sig_d0 = events[track_variables["sig_d0"]]
    sig_z0 = events[track_variables["sig_z0"]]

    # Create a Boolean mask with the track selection criteria
    mask = (
        (pt > pt_min)
        & (np.abs(d0) < d0_max)
        & (sig_d0 > 0.0)
        & (sig_d0 < sig_d0_max)
        & (sig_z0 > 0.0)
        & (sig_z0 < sig_z0_max)
    )

    # Empty dictionary to hold dataset
    dataset = []

    for key, branch in track_variables:
        filtered_tracks = events[branch][mask]

        # Convert variances to normal uncertainties
        if key in ("sig_d0", "sig_z0"):
            filtered_tracks = np.sqrt(np.maximum(filtered_tracks, 0.0))

        dataset.append(ak.to_numpy(ak.flatten(filtered_tracks)))

    """
    # Apply mask, flatten and convert to numpy arrays
    for key, branch in track_variables.items():
        filtered_tracks = events[branch][mask]

        # Convert variances to normal uncertainties
        if key in ("sig_d0", "sig_z0"):
            filtered_tracks = np.sqrt(np.maximum(filtered_tracks, 0.0))

        dataset[key] = ak.to_numpy(ak.flatten(filtered_tracks))

    
    # Don't apply mask or flatten, but still convert to numpy arrays
    for key, branch in truth_variables.items():
        truth = events[branch]

        if key == "truth_z":
            dataset[key] = ak.to_numpy(truth).astype(np.float64)

        if key == "truth_valid":
            dataset[key] = ak.to_numpy(truth).astype(np.bool_)"""

    return dataset

def main():

    
    # Define track variables
    track_variables = {
        "pt": "HLTTrack_pt",
        "eta": "HLTTrack_eta",
        "d0": "HLTTrack_d0",
        "z0": "HLTTrack_z0",
        "sig_d0": "HLTTrack_d0Err",
        "sig_z0": "HLTTrack_z0Err",
    }

    """
    # Define track variables
    track_variables = [
            "HLTTrack_pt",
            "HLTTrack_eta",
            "HLTTrack_d0",
            "HLTTrack_z0",
            "HLTTrack_d0Err",
            "HLTTrack_z0Err",
    ]"""

    """
    # Define vertex variables
    vertex_variables = {
        #"truth_z": "TruthVertex_z",
        "truth_valid": "TruthVertex_isValid"
    }"""

    # Load dataset
    dataset = load_dataset(track_variables)      #, vertex_variables)

    # Split into input (X) and output (y) variables
    X = dataset[:, 0 : (len(track_variables))]
    y = dataset[:, (len(track_variables))]

    # Turn into tensors
    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32).reshape(-1, 1)

    print(X)
    print(y)
"""
    # Define model
    model = nn.Sequential(
        nn.Linear(6, 12),
        nn.ReLU(),
        nn.Linear(12, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    # Prepare for training
    loss_fn = nn.BCELoss() # binary cross entropy
    optimiser = optim.Adam(model.parameters(), lr = 0.001)

    # Define number of epochs and batches
    n_epochs = 100
    batch_size = 10

    # Build a training loop
    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i : i + batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i : i + batch_size]
            loss = loss_fn(Z0_pred, ybatch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print(f"Finished epoch {epoch}, latest loss {loss}")

    # Compute accuracy
    Zy_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean()
    print(f"Accurary {accuracy}")

    # Make class predictions with the model
    predictions = (model(X) > 0.5).int()
    for i in range(5):
        print("%s => %d (expected %d)") % X[i].tolist(0, predictions[i], Z0[i])"""


if __name__ == "__main__":

    main()