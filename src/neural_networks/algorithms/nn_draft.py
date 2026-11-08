#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
import os

import uproot
import awkward as ak
import pandas as pd
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

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

# Training settings
n_events_max = 1000
test_fraction = 0.2
n_epochs = 100
batch_size = 2048
lr = 1e-3
seed = 42

# === FUNCTIONS ====

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_data():

    # Load in all events from the root file
    with uproot.open(f"{filename}:{tree_name}") as tree:
        events = tree.arrays(track_branches, entry_stop = n_events_max)

    # Calculate the number of events
    n_events = len(events)

    # Extract variables 
    pt = events["HLTTrack_pt"]
    eta = events["HLTTrack_eta"]
    d0 = events["HLTTrack_d0"]
    z0 = events["HLTTrack_z0"]

    # Convert variances to normal uncertainties 
    # (because they're squared currently)
    sig_d0 = np.sqrt(np.maximum(events["HLTTrack_d0Err"], 0.0))
    sig_z0 = np.sqrt(np.maximum(events["HLTTrack_z0Err"], 0.0))

    # Truth information
    truth_z = ak.to_numpy(events["TruthVertex_z"]).astype(np.float64)
    truth_valid = ak.to_numpy(events["TruthVertex_isValid"]).astype(np.bool_)

    # Create a Boolean mask with the track selection criteria
    mask = (
        (pt > pt_min)
        & (np.abs(d0) < d0_max)
        & (sig_d0 > 0.0)
        & (sig_d0 < sig_d0_max)
        & (sig_z0 > 0.0)
        & (sig_z0 < sig_z0_max)
    )

    # Broadcast per-event quantities to per-track
    truth_z_per_track = ak.broadcast_arrays(truth_z, pt)[0]
    event_idx_per_track = ak.broadcast_arrays(np.arange(n_events), pt)[0]

    def flat(a):
        return ak.to_numpy(ak.flatten(a[mask]))

    #  One row per track, one column per parameter
    # Shape = (n_tracks, 6)
    X = np.column_stack([flat(pt), flat(eta), flat(d0), flat(z0), flat(sig_d0), flat(sig_z0)]) #.astype(np.float32))
    y = flat(truth_z_per_track) #.astype(np.float32)
    event_idx = flat(event_idx_per_track)

    return X, y, event_idx, n_events

# Define model class
class VertexRegressor(nn.Module):

    def __init__(self, input_size = 6):
        super().__init__()

        # Create 3 layers capable of handling the data
        self.layer_1 = nn.Linear(input_size, 12)
        self.layer_2 = nn.Linear(12, 8)
        self.layer_3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

# Training loop
def train_model(model, X, y, criterion, optimiser):

    # Create an empty list to store metric for each epoch
    metrics = []
    n = X.shape[0]
    device = X.device

    # Training loop
    for epoch in range(n_epochs):

        # Set model to train
        model.train()

        # Shuffle each epoch
        perm = torch.randperm(n, device = device)

        running_loss = torch.zeros((), device = device)

        for i in range(0, n, batch_size):

            # Obtain the batches
            idx = perm[i : i + batch_size]
            batch_X, batch_y = X[idx], y[idx]

            # Zero the gradients
            optimiser.zero_grad()

            # Run the model for this batch of data
            outputs = model(batch_X)

            # Loss backwards
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Step the optimiser (gradient descent?)
            optimiser.step()

            running_loss += loss.detach() * batch_X.size(0)

        # Calculate average training loss for the epoch
        train_loss = running_loss.item() / n 

        # Append epoch and training loss to metric list
        metrics.append({"Epoch": epoch + 1,
                        "Train loss": train_loss})

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f}")

    # Convert metrics list to DataFrame
    df_metrics = pd.DataFrame(metrics)

    return df_metrics
        
def evaluate_model(model, X, y, criterion):

    predictions = []
    actuals = []
    running_loss = 0.0

    n = X.shape[0]
    device = X.device

    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):

            # Shuffle each epoch
            perm = torch.randperm(n, device = device)
        
            # Obtain the batches
            idx = perm[i : i + batch_size]
            batch_X, batch_y = X[idx], y[idx]

            # Run the model for this batch of data
            outputs = model(batch_X)

            # Loss backwards
            loss = criterion(outputs, batch_y)
            #loss.backward()

            running_loss += loss.item() * batch_X.size(0)

            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(batch_y.squeeze().tolist())

        # Calculate average training loss for the epoch
        avg_loss = running_loss / n 

    return actuals, predictions, avg_loss

def main():

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    # Load data
    X, y, event_idx, n_events = load_data()

    # Split data into test and training by events
    all_events = np.arange(n_events)
    train_events, test_events = train_test_split(all_events, test_size = test_fraction, random_state = seed)

    # Create boolean masks for tracks belonging to those events
    train_mask = np.isin(event_idx, train_events)
    test_mask = np.isin(event_idx, test_events)

    # And send to device
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Initialise scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Feature normalisation using training data
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    # Turn into tensors
    X_train = torch.tensor(X_train, dtype = torch.float32, device = device)
    X_test = torch.tensor(X_test, dtype = torch.float32, device = device)
    y_train = torch.tensor(y_train, dtype = torch.float32, device = device)
    y_test = torch.tensor(y_test, dtype = torch.float32, device = device)

    # Initialise model
    model = VertexRegressor(input_size = X_train.shape[1]).to(device) 

    # Define the loss function and optimiser
    loss_function = nn.L1Loss()
    optimiser = optim.Adam(model.parameters(), lr = lr)

    # Train the model
    df_metrics = train_model(model, X_train, y_train, loss_function, optimiser)

    # Plot training loss over epochs
    #df_metrics.plot(x = "Epoch", y = "Train loss")
    #plt.show()

    # Evaluate on test set
    actuals, predictions, avg_loss = evaluate_model(model, X_test, y_test, loss_function)
    print(f"Test Loss: {avg_loss:.4f}")

if __name__ == "__main__":

    main()