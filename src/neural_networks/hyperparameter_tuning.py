#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
import os
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


# === ARGUEMENTS ====
# Splits
test_fraction = 0.15
validation_fraction = 0.15
seed = 42
rng = np.random.default_rng(seed)

# Search settings 
n_trials = 20           # number of random configs to try
max_epochs = 50        # upper limit per trial
patience = 10           # stop a trial after with many epochs without val improvement

# Define search space
# Learning rate is sampled log-uniformaly between two bounds, the rest are picked from the lists
param_dist = {
    "learning_rate": (1e-4, 1e-2),
    "batch_size": [128, 256, 512, 1024, 2048],
    "layers": [1, 2, 3, 4],
    "neurons": [8, 16, 32, 64, 128],
}

# Track selection (must match the runs whose results are read below)
pt_min = 1.5        # [GeV]
pt_max = np.inf     # [GeV]
eta_max = np.inf

# Two-step settings
s_max = 3.0

# Matching tolerance [mm]
tol = 0.3

# Saved per-event results, used to choose the events
base_dir = "outputs/sem1week4"
output_dir = "outputs/sem1week7"
selection_folder = f"{pt_min}ptmin_{pt_max}ptmax_{eta_max}etamax"
out_dir = os.path.join(output_dir)
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

    # Obtain the candidate properties from the file
    z = d["cand_z"]
    height = d["cand_height"]
    ntrk = d["cand_ntrk"].astype(np.float64)
    sumpt = d["cand_sumpt"]
    sumpt2 = d["cand_sumpt2"]

    n_cand = z.shape[1]

    # Log transform the positive, long-tailed features so they are on a similar scale
    features = {
        "z" : z,
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

        # Pass each candidate through the hidden layers (same weights for every candidate)
        for layer in self.layers:
            x = self.relu(layer(x))

        # Obtain one score per candidate, shape (events, candidates)
        # Squeeze(-1) removes the loftover size-1 dimension, giving shape ^^
        x = self.out_layer(x).squeeze(-1)

        return x

def efficiency(model, X, target, batch_size = 8192):

    # Fraction of events where the chosen candidate is the true one (target = -1 counts as wrong)
    return (predict(model, X, batch_size) == target).mean()

# Training loop
def train_model(model, X_train, y_train, X_val, t_val, y_val, lr, batch_size):

    # Define the loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr = lr)

    n = X_train.shape[0]
    device = X_train.device

    # Keep track of the best epoch so far for early stopping
    best_loss, best_eff, best_epoch, best_state, bad = np.inf, -1.0, 0, None, 0
    history = []

    # Training loop
    for epoch in range(max_epochs):

        # Set model to train
        model.train()

        # Shuffle each epoch
        perm = torch.randperm(n, device = device)

        running_loss = torch.zeros((), device = device)

        for i in range(0, n, batch_size):

            # Obtain the batches
            idx = perm[i : i + batch_size]
            batch_X, batch_y = X_train[idx], y_train[idx]

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
        val_loss = loss_on(model, X_val, y_val)
        val_eff = efficiency(model, X_val, t_val)

        # Append epoch and training loss to metric list
        history.append({"Epoch": epoch + 1, "Train loss": train_loss,
                        "Val loss": val_loss, "Val eff": val_eff})

        #print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val eff {100 * val_eff:.2f}%")

        # Early stopping on validation loss 
        # Save a copy of the weights whenever the validation loss improves
        if val_loss < best_loss:
            best_loss, best_epoch, bad = val_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # Otherwise, count the epochs without improvement and stop after patience 
        else:
            bad += 1
            if bad >= patience:
                break

    # Restore the weights from the best epoch and calculate its validation efficiency 
    model.load_state_dict(best_state)
    best_eff = efficiency(model, X_val, t_val)
    return best_loss, best_eff, best_epoch, pd.DataFrame(history)

def sample_config(rng):

    low, high = param_dist["learning_rate"]

    # Draw one random value for each hyperparameter
    # Learning rate is sampled uniformly in log10 so each order of magnitude is equally likely 
    return {
        "learning_rate": float(10 ** rng.uniform(np.log10(low), np.log10(high))),
        "batch_size": int(rng.choice(param_dist["batch_size"])),
        "layers": int(rng.choice(param_dist["layers"])),
        "neurons": int(rng.choice(param_dist["neurons"]))
    }

def predict(model, X, batch_size = 8192):

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

def report(results, n_top = 3):

    results = pd.DataFrame(results)

    # Print the top configuration (results are already sorted by validation loss)
    for rank, (_, row) in enumerate(results.head(n_top).iterrows(), start = 1):
        print(f"Rank {rank}: val loss {row['val_loss']:.4f} | val eff {row['val_eff (%)']:.2f}% "
              f"at epoch {int(row['best_epoch'])} | lr={row['learning_rate']:.2e}, "
              f"batch={int(row['batch_size'])}, layers={int(row['layers'])}, neurons={int(row['neurons'])}")

def loss_on(model, X, y, batch_size = 8192):

    # Sum the loss over events, skipping events where the truth wasn't found (target = -1)
    criterion = nn.CrossEntropyLoss(ignore_index = -1, reduction = "sum")

    # Put the model into evaluation mode
    model.eval()
    total = 0.0

    # Do not keep track of gradients
    with torch.no_grad():

        # Iterate through batches of the data and add up the loss
        for i in range(0, X.shape[0], batch_size):
            outputs = model(X[i : i + batch_size])
            total += criterion(outputs, y[i : i + batch_size]).item()

    # Average over the events that were actually used
    return total / (y >= 0).sum().item()

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

    # Split data into training, test and validation by events
    all_events = np.arange(n_events)
    training_idx, test_idx = train_test_split(all_events, test_size = test_fraction, random_state = seed)
    train_idx, val_idx = train_test_split(training_idx, test_size = validation_fraction, random_state = seed)

    # Events where the truth is not among the candidates can't be learnt from
    # They are only removed from training. Validation and test keep them (they count as failures)
    print(f"Number of events where the truth is not among the candidates: {np.sum(target[train_idx] < 0)} / {len(train_idx)}")
    train_idx = train_idx[target[train_idx] >= 0]

    # Feature normalisation for training candidates only (features)
    scaler_X = StandardScaler()
    scaler_X.fit(X[train_idx][~empty[train_idx]])
    X_scaled = scaler_X.transform(X.reshape(-1, n_feat)).reshape(n_events, n_cand, n_feat)
    X_scaled[empty] = 0.0

    # Create the training and test tensore (features and target)
    X_train = torch.tensor(X_scaled[train_idx], dtype = torch.float32, device = device)
    y_train = torch.tensor(target[train_idx], dtype = torch.long, device = device)
    X_val = torch.tensor(X_scaled[val_idx], dtype = torch.float32, device = device)
    t_val = target[val_idx]
    y_val = torch.tensor(target[val_idx], dtype = torch.long, device = device)    

    # Random search
    rows = []
    best = {"loss": np.inf}
    for trial in range(n_trials):

        # Draw a random configuration and build a fresh model for it
        cfg = sample_config(rng)
        torch.manual_seed(seed + trial)
        model = VertexRegressor(n_feat, cfg["layers"], cfg["neurons"]).to(device)

        # Train the model and time how long it takes
        t0 = time.time()
        val_loss, val_eff, best_epoch, history = train_model(model, X_train, y_train, X_val, t_val, y_val, 
                                                            cfg["learning_rate"], cfg["batch_size"])
        dt = time.time() - t0  

        # Print and store the results of this trial
        print(f"Trial {trial + 1:2d}/{n_trials} | val loss {val_loss:.4f} | val eff {100 * val_eff:.2f}% "
              f"(epoch {best_epoch}, {dt:.0f}s) | {cfg}")
        rows.append({**cfg, "val_loss": val_loss, "val_eff (%)": 100 * val_eff, 
                     "best_epoch": best_epoch, "time (s)": dt})

        # Keep the trial with the lowest validation loss
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "cfg": cfg, "model": model, "history": history}

    # Sort the trials by validation loss and save them 
    results = pd.DataFrame(rows).sort_values("val_loss")
    results.to_csv(os.path.join(out_dir, "random_search_results.csv"), index = False)
    print("\nTop configurations:")
    report(results)
 
    # Training curves of the best trial
    h = best["history"]
    best_ep = h["Val loss"].idxmin() + 1

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (7, 7))

    # Top: loss
    ax1.plot(h["Epoch"], h["Train loss"], label = "Train loss")
    ax1.plot(h["Epoch"], h["Val loss"], label = "Validation loss")
    ax1.axvline(best_ep, color = "grey", ls = "--", label = "Best epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.legend()

    # Bottom: validation efficiency
    ax2.plot(h["Epoch"], 100 * h["Val eff"], color = "tab:green", label = "Validation efficiency")
    ax2.axvline(best_ep, color = "grey", ls = "--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation efficiency (%)")
    ax2.legend()

    # Save the figure to the output folder
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "best_trial_training.png"), dpi = 300)


if __name__ == "__main__":

    main()