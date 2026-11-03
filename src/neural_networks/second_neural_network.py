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
base_dir = "outputs/sem1week4"
output_dir = "outputs/sem1week7"
selection_folder = f"{pt_min}ptmin_{pt_max}ptmax_{eta_max}etamax"
out_dir = os.path.join(output_dir)
#gaussian_csv = os.path.join(base_dir, "gaussian_weighting", selection_folder, "vertex_results_d0True_normTrue.csv")
#box_csv = os.path.join(base_dir, "sumpt", selection_folder, f"vertex_results_res{box_res}_pt{box_power}.csv")
two_step_npz = os.path.join(base_dir, "two_steps", selection_folder, f"vertex_candidates_s{s_max}.npz")

npz_filename = "vertex_candidates_s3_0.npz"
file = os.path.join(base_dir, "two_steps", selection_folder, f"vertex_candidates_s{s_max}.npz")

# === FUNCTIONS ====

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

def feature_correlation(X, empty, target, feature_names):

    # Label each candidate: 1 if it is the matched true vertex, 0 otherwise
    #is_pv = np.zeros(empty.shape, dtype = float)
    #found = np.where(target >= 0)[0]
    #is_pv[found, target[found]] = 1.0

    # One row per real (non-empty) candidate
    df = pd.DataFrame(X[~empty], columns = feature_names)
    #df["is_PV"] = is_pv[~empty]

    # Spearman (rank) correlation: not affected by the log transforms or outliers
    corr = df.corr(method = "spearman")
    print("\nSpearman correlation:")
    print(corr.round(3).to_string())

    # Heatmap
    fig, ax = plt.subplots(figsize = (6, 5))
    im = ax.imshow(corr, cmap = "RdBu_r", vmin = -1, vmax = 1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation = 45, ha = "right")
    ax.set_yticks(range(len(corr)), corr.columns)

    # Write the correlation value in each cell
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha = "center", va = "center", fontsize = 8)

    # Save the figure to the output folder
    fig.colorbar(im, label = "Spearman correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_correlation.png"), dpi = 300)
    plt.close(fig)

    return corr

def drop_one_feature(X_scaled, empty, target, train_idx, test_idx, feature_names, device):

    # Targets are the same for every run, so only make them once
    y_train = torch.tensor(target[train_idx], dtype = torch.long, device = device)
    y_test = torch.tensor(target[test_idx], dtype = torch.long, device = device)

    # Empty list to hold results 
    rows = []

    # Iterate through each feature, dropping one at a time and running the model
    # None keeps all the features as a reference
    for dropped in [None] + feature_names:

        # Keep every feature except the dropped one
        keep = [k for k, name in enumerate(feature_names) if name != dropped]
        X_train = torch.tensor(X_scaled[train_idx][:, :, keep], dtype = torch.float32, device = device)
        X_test = torch.tensor(X_scaled[test_idx][:, :, keep], dtype = torch.float32, device = device)

        # Retrain from the same starting seed each time
        torch.manual_seed(seed)
        model = VertexRegressor(len(keep), layers, neurons).to(device)
        optimiser = optim.Adam(model.parameters(), lr = lr)
        train_model(model, X_train, y_train, X_test, y_test, nn.CrossEntropyLoss(), optimiser)

        # Store the test efficiency and loss without this feature
        rows.append({"Dropped": dropped or "none",
                     "Test eff (%)": 100 * efficiency(model, X_test, target[test_idx]),
                     "Test loss": loss_on(model, X_test, y_test)})

    # Change relative to the reference run with all features (first row)
    df = pd.DataFrame(rows)
    df["Eff change (%)"] = df["Test eff (%)"] - df["Test eff (%)"].iloc[0]
    df["Loss change"] = df["Test loss"] - df["Test loss"].iloc[0]
    print("\nDrop-one-feature retraining:")
    print(df.to_string(index = False, float_format = "%.4f"))
    df.to_csv(os.path.join(out_dir, "drop_one_feature.csv"), index = False)

    return df

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
    X_scaled[empty] = 0.0

    # Create the training and test tensore (features and target)
    X_train = torch.tensor(X_scaled[train_idx], dtype = torch.float32, device = device)
    y_train = torch.tensor(target[train_idx], dtype = torch.long, device = device)
    X_test = torch.tensor(X_scaled[test_idx], dtype = torch.float32, device = device)
    y_test = torch.tensor(target[test_idx], dtype = torch.long, device = device)

    # Initialise model
    model = VertexRegressor(n_feat, layers, neurons).to(device) 

    # Define the loss function and optimiser
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)

    # Train the model
    df_metrics = train_model(model, X_train, y_train, X_test, y_test, loss_function, optimiser)

    # Plot training and test loss over epochs
    fig, ax = plt.subplots()
    ax.plot(df_metrics["Epoch"], df_metrics["Train loss"], label = "Train loss")
    ax.plot(df_metrics["Epoch"], df_metrics["Test loss"], label = "Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()

    # Save the figure to the output folder
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_test_loss.png"), dpi = 300)
    plt.close(fig)

    # Evaluate on test events and compare with the classical choices
    d = np.load(file)
    
    # Obtain the vertex chosen by the network
    chosen_nn = predict(model, X_test)

    # Obtain the target, candidates and truth z per test event
    t, cz, tz = target[test_idx], cand_z[test_idx], truth_z[test_idx]

    # Choose the highest density, sumpt and sumpt2 per test event
    chosen_densest = np.zeros(len(test_idx), dtype = int) 
    chosen_sumpt = d["cand_sumpt"][test_idx].argmax(axis = 1)
    chosen_sumpt2 = d["cand_sumpt2"][test_idx].argmax(axis = 1)

    # Keep results in a dataframe
    results = pd.DataFrame([
        summarise("Densest", chosen_densest, t, cz, tz),
        summarise("Sum pT", chosen_sumpt, t, cz, tz),
        summarise("Sum pT^2", chosen_sumpt2, t, cz, tz),
        summarise("Neural network", chosen_nn, t, cz, tz),
    ])
    print("\nTest set:")
    print(results.to_string(index = False, float_format = "%.3f"))

    # Feature correlation and ranking
    feature_correlation(X, empty, target, feature_names)
    drop_one_feature(X_scaled, empty, target, train_idx, test_idx, feature_names, device)

if __name__ == "__main__":

    main()