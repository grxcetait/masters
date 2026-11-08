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

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import time
import numba 

# Set to a single core. Per event cost
numba.set_num_threads(1)

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


# === ARGUEMENTS ====
# Training settings
test_fraction = 0.15
validation_fraction = 0.15
n_epochs = 30
seed = 42
rng = np.random.default_rng(seed)

# Hyperparameters
lr = 1e-3
batch_size = 1024
layers = 3
neurons = 64

# Track selection (must match the runs whose results are read below)
pt_min = 1.5        # [GeV]
pt_max = np.inf     # [GeV]
eta_max = np.inf
d0_max = 4.0        # [mm]
sig_d0_max = 5.0    # [mm]
sig_z0_max = 10.0   # [mm]

# Gaussian density settings
n_sigma = 4.0
use_d0_weight = True

# Two-step settings
s_max = 3.0
n_cand = 30
min_tracks = 2

# Matching tolerance [mm]
tol = 0.3

# Saved per-event results, used to choose the events
base_dir = "outputs/sem1week4"
output_dir = "outputs/sem1week7"
selection_folder = f"{pt_min}ptmin_{pt_max}ptmax_{eta_max}etamax"
out_dir = os.path.join(output_dir)
two_step_npz = os.path.join(base_dir, "two_steps", selection_folder, f"vertex_candidates_s{s_max}.npz")

npz_filename = "vertex_candidates_s3_0.npz"
file = os.path.join(base_dir, "two_steps", selection_folder, f"vertex_candidates_s{s_max}.npz")

# === FUNCTIONS ====

def timed(fn, *args, n_repeats = 5, **kwargs):

    # Warm up as the first call includes numba compiling
    fn(*args, **kwargs)

    # Empty list to hold times
    times = []

    # Iterate through the number of repeats
    for _ in range(n_repeats):

        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    return out, min(times)

def get_device():

    # Use the GPU if there is one (CUDA or Apple MPS), otherwise the CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_features(d):

    # Obtain the candidate properties from the file
    z = d["cand_z"]
    height = d["cand_height"]
    ntrk = d["cand_ntrk"].astype(np.float64)
    sumpt = d["cand_sumpt"]
    sumpt2 = d["cand_sumpt2"]

    n_cand = z.shape[1]

    # Log transform the positive, log-tailed features so they are on a similar scale
    features = {
        "log_height" : np.log1p(height),
        "log_ntrk" : np.log1p(ntrk),
        "log_sumpt" : np.log1p(sumpt),
        "log_sumpt2" : np.log1p(sumpt2)
    }

    # Stack into one array of shape (events, candidates, features)
    names = list(features.keys())
    X = np.stack([features[k] for k in names], axis = -1).astype(np.float32)
    return X, names

def load_data():

    # Load in data from file
    d = np.load(file)

    # Obtain the candidates and true vertex
    cand_z = d["cand_z"]
    truth_z = d["truth_z"]

    # Empty candidate slots (events with fewer than n_cand candidates) have z = NaN and ntrk = 0
    empty = d["cand_ntrk"] == 0
    print(f"Empty candidate slots: {empty.sum()} (in {empty.any(axis = 1).sum()} events)")

    # Obtain the parameters and their names
    X, feature_names = get_features(d)
    X[empty] = 0.0

    # Finding which of the candidates is the true candidate (within tol)
    # Calculate dz between each candidate z and truth vertex
    dz = np.abs(cand_z - truth_z[:, None])

    # Set empty slots to infinity so they can never be the closest (NaN would break argmin)
    # Originally cand_z = NaN so dz would also be NaN. This is fixed now
    dz = np.where(empty, np.inf, dz)

    # Find the candidate with the smallest dz
    target = dz.argmin(axis = 1)

    # Rule out events where even the closest candidate is further than tol from the truth
    target[dz.min(axis = 1) > tol] = -1

    return X, target, cand_z, truth_z, feature_names, empty

# Define model class
class VertexRegressor(nn.Module):

    def __init__(self, input_size, layers, neurons):
        super(VertexRegressor, self).__init__()
        self.layers = nn.ModuleList()

        # Create specified number of layers
        for i in range(layers):
            self.layers.append(nn.Linear(input_size, neurons))
            input_size = neurons

        # Output layer gives one score per candidate
        self.out_layer = nn.Linear(input_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):

        # Pass each candidate through the hidden layers (same weights for every candidate)
        for layer in self.layers:
            x = self.relu(layer(x))

        # Obtain one score per candidate, shape (events, candidates)
        # Squeeze(-1) removes the loftover size-1 dimension, giving shape ^^
        x = self.out_layer(x).squeeze(-1)

        return x

# Training loop
def train_model(model, X, y, X_test, y_test, criterion, optimiser):

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
        # The test loss is only monitored, it is not used to make any choices
        train_loss = running_loss.item() / n 
        test_loss = loss_on(model, X_test, y_test)

        # Append epoch and training loss to metric list
        metrics.append({"Epoch": epoch + 1, "Train loss": train_loss, "Test loss": test_loss})

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    # Convert metrics list to DataFrame
    df_metrics = pd.DataFrame(metrics)

    return df_metrics

def loss_on(model, X, y, batch_size = 8192):

    # Sum the loss over events, skipping events where the truth wasn't found (target = -1)
    criterion = nn.CrossEntropyLoss(ignore_index = -1, reduction = "sum")

    # Put model into evaluation mode
    model.eval()
    total = 0.0

    # Do not keep track of gradients
    with torch.no_grad():

        # Iterate through batches of the data and add up the loss
        for i in range(0, X.shape[0], batch_size):
            outputs = model(X[i : i + batch_size])
            total += criterion(outputs, y[i : i + batch_size]).item()

    # Average over the events that were actually used
    return total / (y >= 0).sum().item()

def predict(model, X):

    # Put model into evaluation mode
    model.eval()
    chosen = []

    # Do not keep track of gradients
    with torch.no_grad():

        # Iterate through batches of the data
        for i in range(0, X.shape[0], batch_size):

            # Give one score per candidate
            # Higher score means the network thinks that candidate is more likely to be the primary vertex
            logits = model(X[i : i + batch_size])

            # Find the position of the highest score across the 30 candidates in each event
            chosen.append(logits.argmax(dim = 1).cpu())

    # Add to the chosen list
    return torch.cat(chosen).numpy()

def efficiency(model, X, target):
 
    # Fraction of events where the chosen candidate is the true one (target = -1 counts as wrong)
    return (predict(model, X) == target).mean()

def summarise(name, chosen, target, cand_z, truth_z):

    # If the target is >= 0, it has been found
    found = target >= 0

    # Check if the chosen candidate is the same as the target
    correct = chosen == target

    # Array the same length as chosen candidates
    rows = np.arange(len(chosen))

    # Calculate dz
    dz = np.abs(cand_z[rows, chosen] - truth_z)

    # Return dictionary of method and efficiencies 
    return {
        "Method" : name,
        "Eff (%)" : 100 * correct.mean(),
        "Eff if found (%)" : 100 * correct[found].mean(),
        "Median |dz| (mm)" : np.nanmedian(dz),
    }

def main():

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok = True)

    # Load data
    X, target, cand_z, truth_z, feature_names, empty = load_data()
    n_events, n_cand, n_feat = X.shape
    print(f"{n_events} events, {n_cand} candidates, {n_feat} features: {feature_names}")

    # Split data into test and training by events
    all_events = np.arange(n_events)
    training_idx, test_idx = train_test_split(all_events, test_size = test_fraction, random_state = seed)
    train_idx, val_idx = train_test_split(training_idx, test_size = validation_fraction / (1 - test_fraction), random_state = seed)

    # Events where the truth is not among the candidates can't be learnt from
    print(f"Number of events where the truth is not among the candidates: {np.sum(target[train_idx] < 0)} / {len(train_idx)}")
    train_idx = train_idx[target[train_idx] >= 0]

    # Feature normalisation for training candidates only
    scaler_X = StandardScaler()
    scaler_X.fit(X[train_idx][~empty[train_idx]])
    X_scaled = scaler_X.transform(X.reshape(-1, n_feat)).reshape(n_events, n_cand, n_feat)
    X_scaled[empty] = 0.0

    # Create the training, validation and test tensors (features and target)
    X_train = torch.tensor(X_scaled[train_idx], dtype = torch.float32, device = device)
    y_train = torch.tensor(target[train_idx], dtype = torch.long, device = device)
    X_val = torch.tensor(X_scaled[val_idx], dtype = torch.float32, device = device)
    t_val = target[val_idx]
    y_val = torch.tensor(target[val_idx], dtype = torch.long, device = device)
    X_test = torch.tensor(X_scaled[test_idx], dtype = torch.float32, device = device)
    y_test = torch.tensor(target[test_idx], dtype = torch.long, device = device)

    # Initialise model
    model = VertexRegressor(n_feat, layers, neurons).to(device) 

    # Set pytorch threads to 1 and to cpu so its fair with the two steps algorithm timings
    torch.set_num_threads(1)
    model_cpu = model.to("cpu")
    n_test = len(test_idx)

    # Define the loss function and optimiser
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)

    # Train the model and time how long it takes
    df_metrics = train_model(model, X_train, y_train, X_test, y_test, loss_function, optimiser) 

    # Evaluate on test events and compare with the classical choices
    d = np.load(file)

    # Candidate properties for the test events (what step 1 would hand over)
    d_test = {k: d[k][test_idx] for k in ["cand_z", "cand_height", "cand_ntrk", "cand_sumpt", "cand_sumpt2"]}
    empty_test = d_test["cand_ntrk"] == 0

    def nn_step2():
        X, _ = get_features(d_test)                                    # log transforms
        X = scaler_X.transform(X.reshape(-1, n_feat)).reshape(X.shape) # scaling
        X[empty_test] = 0.0
        return predict(model_cpu, torch.tensor(X, dtype = torch.float32))
    
    # Obtain the vertex chosen by the network
    chosen_nn, t = timed(nn_step2, n_repeats = 3)

    # Print results 
    print(f"Testing model: {1e6 * t / n_test:.3f} microseconds/event")

if __name__ == "__main__":

    main()