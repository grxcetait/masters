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

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


# === ARGUEMENTS ====
# Training settings
test_fraction = 0.2
n_epochs = 100
batch_size = 2048
lr = 1e-3
seed = 42

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

# Box (sum pT window) settings
box_res = 0.1       # [mm]
box_power = 1       # 1 = pT weighted

# Two-step settings
s_max = 3.0
n_cand = 30
min_tracks = 2

# Matching tolerance [mm]
tol = 0.3

# Saved per-event results, used to choose the events
base_dir = "outputs/sem1week6"
output_dir = "outputs/sem1week6"
selection_folder = f"{pt_min}ptmin_{pt_max}ptmax_{eta_max}etamax"
out_dir = os.path.join(output_dir)
#gaussian_csv = os.path.join(base_dir, "gaussian_weighting", selection_folder, "vertex_results_d0True_normTrue.csv")
#box_csv = os.path.join(base_dir, "sumpt", selection_folder, f"vertex_results_res{box_res}_pt{box_power}.csv")
two_step_npz = os.path.join(base_dir, "two_steps", selection_folder, f"vertex_candidates_s{s_max}.npz")

npz_filename = "vertex_candidates_s3_0.npz"
file = os.path.join(base_dir, "two_steps", selection_folder, f"vertex_candidates_s{s_max}.npz")

# === FUNCTIONS ====

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_features(d):
    
    z = d["cand_z"]
    height = d["cand_height"]
    ntrk = d["cand_ntrk"].astype(np.float64)
    sumpt = d["cand_sumpt"]
    sumpt2 = d["cand_sumpt2"]

    n_cand = z.shape[1]

    features = {
        "z" : z,
        "log_height" : np.log1p(height),
        "log_ntrk" : np.log1p(ntrk),
        "log_sumpt" : np.log1p(sumpt),
        "log_sumpt2" : np.log1p(sumpt2)
        #"height" : height,
        #"ntrk" : ntrk,
        #"sumpt" : sumpt,
        #"sumpt2" : sumpt2
    }

    names = list(features.keys())
    X = np.stack([features[k] for k in names], axis = -1).astype(np.float32)
    return X, names

def load_data():

    d = np.load(file)

    cand_z = d["cand_z"]
    truth_z = d["truth_z"]

    # Empty candidate slots (events with fewer than n_cand candidates) hae z = NaN and ntrk = 0
    empty = d["cand_ntrk"] == 0
    print(f"Empty candidate slots: {empty.sum()} (in empty.any(axis = 1).sum() events")

    X, feature_names = get_features(d)
    X[empty] = 0.0

    # Finding which of the candidates is the true candidate (within tol)
    # Calculate dz between each candidate z and truth vertex
    dz = np.abs(cand_z - truth_z[:, None])

    # Find the candidate with the smallest dz
    target = dz.argmin(axis = 1)

    # Rule out events where even the closest 
    target[dz.min(axis = 1) > tol] = -1

    return X, target, cand_z, truth_z, feature_names, empty

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
        return x.squeeze(-1)

def rank_within_event(a):
    """Rank of each candidate within its event (0 = largest value)."""
    order = np.argsort(-a, axis = 1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(a.shape[1])[None, :], axis = 1)
    return ranks

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

def predict(model, X):

    model.eval()
    chosen = []

    with torch.no_grad():

        for i in range(0, X.shape[0], batch_size):
            logits = model(X[i : i + batch_size])
            chosen.append(logits.argmax(dim = 1).cpu())

    return torch.cat(chosen).numpy()

def summarise(name, chosen, target, cand_z, truth_z):

    # If the target is >= 0, it has been found
    found = target >= 0

    # Check if the chosen candidate is the same as the target
    correct = chosen == target

    # Array the same length as chosen candidates
    rows = np.arange(len(chosen))

    # Calculate dz
    dz = np.abs(cand_z[rows, chosen] - truth_z)

    # 
    return {
        "Method" : name,
        "Eff (%)" : 100 * correct.mean(),
        "Eff if found (%)" : 100 * correct[found].mean(),
        "Median |dz| (mm)" : np.median(dz),
    }

def main():

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok = True)

    # Load data
    X, target, cand_z, truth_z, feature_names, empty = load_data()
    n_events, n_cand, n_feat = X.shape
    print(f"{n_events} events, {n_cand} candidates, {n_feat} features: {feature_names}")

    # Split data into test and training by events
    all_events = np.arange(n_events)
    train_idx, test_idx = train_test_split(all_events, test_size = test_fraction, random_state = seed)

    # Events where the truth is not among the candidates can't be learnt from
    print(f"Number of events where the truth is not among the candidates: {np.sum(target[train_idx] < 0)} / {len(train_idx)}")
    train_idx = train_idx[target[train_idx] >= 0]

    # Feature normalisation for training candidates only
    scaler_X = StandardScaler()
    scaler_X.fit(X[train_idx][~empty[train_idx]])
    X_scaled = scaler_X.transform(X.reshape(-1, n_feat)).reshape(n_events, n_cand, n_feat)

    X_train = torch.tensor(X_scaled[train_idx], dtype = torch.float32, device = device)
    y_train = torch.tensor(target[train_idx], dtype = torch.long, device = device)
    X_test = torch.tensor(X_scaled[test_idx], dtype = torch.float32, device = device)

    # Initialise model
    model = VertexRegressor(input_size = n_feat).to(device) 

    # Define the loss function and optimiser
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)

    # Train the model
    df_metrics = train_model(model, X_train, y_train, loss_function, optimiser)

    # Plot training loss over epochs
    #df_metrics.plot(x = "Epoch", y = "Train loss")
    #plt.show()

    # Evaluate on test events and compare with the classical choices
    d = np.load(file)
    chosen_nn = predict(model, X_test)
    t, cz, tz = target[test_idx], cand_z[test_idx], truth_z[test_idx]
    chosen_densest = np.zeros(len(test_idx), dtype = int) 
    chosen_sumpt = d["cand_sumpt"][test_idx].argmax(axis = 1)
    chosen_sumpt2 = d["cand_sumpt2"][test_idx].argmax(axis = 1)

    results = pd.DataFrame([
        summarise("Densest", chosen_densest, t, cz, tz),
        summarise("Sum pT", chosen_sumpt, t, cz, tz),
        summarise("Sum pT^2", chosen_sumpt2, t, cz, tz),
        summarise("Neural network", chosen_nn, t, cz, tz),
    ])
    print("\nTest set:")
    print(results.to_string(index = False, float_format = "%.3f"))

    # Evaluate on test set
    #print(f"Test Loss: {avg_loss:.4f}")
    #actuals, predictions, avg_loss = evaluate_model(model, X_test, y_test, loss_function)

if __name__ == "__main__":

    main()