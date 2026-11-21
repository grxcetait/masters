#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
import os

import uproot
import awkward as ak

import numpy as np
import pandas as pd
import math
from scipy.special import erf

import matplotlib.pyplot as plt

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


# === ARGUMENTS ====
# Training settings
test_fraction = 0.15
validation_fraction = 0.15
pretrain_epochs = 10     # 0 = no pre-training
n_epochs = 30
seed = 42
n_events_max = 10000

# Hyperparameters
lr = 1e-3
lr_finetune = 3e-4       # smaller learning rate for step 2 after pre-training, so it doesn't undo step 1
batch_size = 32          # events per batch
layers = 3
neurons = 64
n_tracks_max = np.inf      # keep at most this many tracks per event (highest pT first)

# Track selection
pt_min = 1.5        # [GeV]
d0_max = 4.0        # [mm]
sig_d0_max = 5.0    # [mm]
sig_z0_max = 10.0   # [mm]

# Gaussian density settings
n_sigma = 4.0

# Grid along the beam axis
z_min = -200.0      # [mm]
z_max = 200.0       # [mm]
step = 0.1          # [mm]

# Matching tolerance [mm]: tracks within this distance of the truth vertex count as hard-scatter tracks,
# and a predicted vertex within this distance of the truth counts as found
tol = 0.3

# Window [mm] around the top-scoring track used to average z0 for the final prediction
z_window = 1.0

output_dir = "outputs/sem1week9"
out_dir = os.path.join(output_dir)

filename = "CERNbox/initial_data/root_files/HHbbyy_200PU.root"
tree_name = "Events"

# Create a list of the branches we are interested in
track_branches = ["HLTTrack_pt", "HLTTrack_eta", "HLTTrack_d0", "HLTTrack_z0",
                  "HLTTrack_d0Err", "HLTTrack_z0Err", "TruthVertex_z",
                  "TruthVertex_isValid"]

# === FUNCTIONS ====

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_parameters():

    # Load in all events from the root file
    with uproot.open(f"{filename}:{tree_name}") as tree:
        events = tree.arrays(track_branches, entry_stop = n_events_max)

    # Extract variables
    pt = events["HLTTrack_pt"]
    eta = events["HLTTrack_eta"]
    d0 = events["HLTTrack_d0"]
    z0 = events["HLTTrack_z0"]

    # Convert variances to normal uncertainties
    # (because they're squared currently)
    sig_d0 = np.sqrt(np.maximum(events["HLTTrack_d0Err"], 0.0))
    sig_z0 = np.sqrt(np.maximum(events["HLTTrack_z0Err"], 0.0))

    # Truth information: one hard-scatter vertex per event, NaN if it isn't valid
    truth_z = ak.to_numpy(events["TruthVertex_z"]).astype(np.float64)
    truth_valid = ak.to_numpy(events["TruthVertex_isValid"]).astype(bool)
    truth_z[~truth_valid] = np.nan

    # Create a Boolean mask with the track selection criteria
    mask = (
        (pt > pt_min)
        & (np.abs(d0) < d0_max)
        & (sig_d0 > 0.0)
        & (sig_d0 < sig_d0_max)
        & (sig_z0 > 0.0)
        & (sig_z0 < sig_z0_max)
    )

    # Apply the mask
    pt, eta, d0, z0 = pt[mask], eta[mask], d0[mask], z0[mask]
    sig_d0, sig_z0 = sig_d0[mask], sig_z0[mask]

    # Sort tracks by pT (highest first), so cutting at n_tracks_max drops the softest tracks
    order = ak.argsort(pt, ascending = False)

    # Put parameters into a dictionary (still one list of tracks per event)
    parameters = {
        "pt": pt[order],
        "eta": eta[order],
        "z0": z0[order],
        "d0": d0[order],
        "sig_z0": sig_z0[order],
        "sig_d0": sig_d0[order],
        }

    return parameters, truth_z

def pad(a, n_trk, fill = 0.0):

    # Events have different numbers of tracks, so pad every event up to n_trk tracks
    return ak.to_numpy(ak.fill_none(ak.pad_none(a, n_trk, clip = True), fill)).astype(np.float32)

def get_features(parameters, n_trk):

    # Unpack parameters
    p = parameters

    # Only log-transform the positive, long-tailed variables. Signed ones (eta, d0, z0) are left as they are,
    # because log1p of anything below -1 is NaN
    features = {
        "log_pt": np.log1p(p["pt"]),
        "eta": p["eta"],
        "d0": p["d0"],
        "z0": p["z0"],
        "log_sig_z0": np.log1p(p["sig_z0"]),
        "log_sig_d0": np.log1p(p["sig_d0"]),
    }
    names = list(features.keys())

    # Stack into one array of shape (events, tracks, features)
    # Padding is needed because numpy can't stack events with different numbers of tracks
    X = np.stack([pad(features[k], n_trk) for k in names], axis = -1)

    # True for real tracks, False for padding
    track_mask = pad(ak.ones_like(p["pt"]), n_trk) > 0

    return X, track_mask, names

def gaussian_density(z0, d0, sig_z0, sig_d0):

    # Your step-1 Gaussian track density for ONE event, on the grid,
    # then read off at each track's own z0 (one value per track)
    n_grid = int(round((z_max - z_min) / step)) + 1

    # Gaussian track density W(z)
    SQRT2 = math.sqrt(2.0)
    SQRT2PI = math.sqrt(2.0 * math.pi)
    weight = np.exp(-0.5 * (d0 / sig_d0) ** 2) / (SQRT2PI * sig_d0)
    density = np.zeros(n_grid)

    # Iterate through all the tracks in this event
    for j in range(len(weight)):
        width = n_sigma * sig_z0[j]

        low = max(int(math.ceil((z0[j] - width - z_min) / step)), 0)
        high = min(int(math.floor((z0[j] + width - z_min) / step)), n_grid - 1)
        if high < low:
            continue

        edges = z_min + (np.arange(low, high + 2) - 0.5) * step
        cdf = 0.5 * erf((edges - z0[j]) / (SQRT2 * sig_z0[j]))
        density[low:high + 1] += weight[j] * np.diff(cdf) / step

    # Grid point nearest each track's z0, then the density there
    k = np.clip(np.round((z0 - z_min) / step).astype(int), 0, n_grid - 1)

    return density[k]

def density_target(parameters, n_trk):

    # Gaussian density target: (events, tracks, 1)
    # Log-transformed because densities have a long tail
    n_events = len(parameters["pt"])
    y1 = np.zeros((n_events, n_trk, 1), dtype = np.float32)

    # Iterate through all events
    for e in range(n_events):

        # This event's tracks as numpy arrays (at most n_trk of them)
        z0 = ak.to_numpy(parameters["z0"][e][:n_trk]).astype(np.float64)
        d0 = ak.to_numpy(parameters["d0"][e][:n_trk]).astype(np.float64)
        sig_z0 = ak.to_numpy(parameters["sig_z0"][e][:n_trk]).astype(np.float64)
        sig_d0 = ak.to_numpy(parameters["sig_d0"][e][:n_trk]).astype(np.float64)

        # Calculate gaussian density at each track
        density = gaussian_density(z0, d0, sig_z0, sig_d0)

        # Log-transform the densities
        y1[e, :len(density), 0] = np.log1p(density)

    return y1

def hs_target(z0, track_mask, truth_z):

    # Hard-scatter tracks = real tracks within tol of the truth vertex
    dz = np.abs(z0 - truth_z[:, None])
    target = (dz < tol) & track_mask

    # Events with no hard-scatter track (or no valid truth) can't be learned from in phase 2
    usable = target.any(axis = 1)

    return target, usable

def load_data():

    # Load in parameters from the file
    parameters, truth_z = load_parameters()
    n_trk = min(int(ak.max(ak.num(parameters["pt"]))), n_tracks_max)

    # Obtain the features and their names
    X, track_mask, feature_names = get_features(parameters, n_trk)
    z0 = pad(parameters["z0"], n_trk)

    # Obtain the outputs for each set of training
    print("Computing step-1 densities...")
    y1 = density_target(parameters, n_trk)
    y2, usable = hs_target(z0, track_mask, truth_z)

    return X, track_mask, z0, feature_names, y1, y2, usable, truth_z

# Define model class
class VertexNetwork(nn.Module):

    def __init__(self, input_size, layers, neurons):
        super(VertexNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Create specified number of layers
        for i in range(layers):
            self.layers.append(nn.Linear(input_size, neurons))
            input_size = neurons

        self.relu = nn.ReLU()

        # Define output layers
        # Step 1: predicts the step 1 gaussian density at each track
        self.density = nn.Linear(neurons, 1)
        # Step 2: gives one score per track
        self.score = nn.Linear(neurons, 1)

    def forward(self, x, track_mask, step = "score"):

        # Pass each track through the hidden layers (same weights for every track)
        for layer in self.layers:
            x = self.relu(layer(x))

        if step == "density":

            # Shape (events, tracks, 1)
            return self.density(x)

        # Else step == "score"
        # Shape (events, tracks). Padded tracks get -inf so they can never be chosen
        return self.score(x).squeeze(-1).masked_fill(~track_mask, float("-inf"))

def density_loss(pred, target, track_mask):

    # Mean squared error over real tracks only
    return ((pred - target) ** 2)[track_mask].mean()

def score_loss(logits, target):

    # Probability the network puts on any hard-scatter track: -log( sum over HS tracks of softmax )
    log_p = torch.log_softmax(logits, dim = 1)
    log_p_hs = torch.logsumexp(log_p.masked_fill(~target, float("-inf")), dim = 1)
    return -log_p_hs.mean()

def run_epoch(model, X, M, Y, phase, optimiser = None):

    # Train if an optimiser is given, otherwise just measure the loss.
    # phase = "pretrain" uses the density head, phase = "train" uses the score head
    training = optimiser is not None
    model.train(training)
    n = X.shape[0]
    perm = torch.randperm(n, device = X.device) if training else torch.arange(n, device = X.device)
    running_loss = 0.0

    with torch.set_grad_enabled(training):
        for i in range(0, n, batch_size):

            # Obtain the batches
            idx = perm[i : i + batch_size]

            # Run the model for this batch of data
            if phase == "pretrain":
                loss = density_loss(model(X[idx], M[idx], step = "density"), Y[idx], M[idx])
            else:
                loss = score_loss(model(X[idx], M[idx], step = "score"), Y[idx])

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            running_loss += loss.item() * len(idx)

    return running_loss / n

def train_model(model, train, val, phase, epochs, learning_rate):

    # Generic loop used for both phase
    optimiser = optim.Adam(model.parameters(), lr = learning_rate)
    metrics = []

    for epoch in range(epochs):
        train_loss = run_epoch(model, *train, phase, optimiser)
        val_loss = run_epoch(model, *val, phase)
        metrics.append({"Epoch": epoch + 1, "Train loss": train_loss, "Validation loss": val_loss})
        print(f"[{phase}] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    return pd.DataFrame(metrics)

def plot_losses(df, ylabel, name):

    # Plot training and validation loss over epochs
    fig, ax = plt.subplots()
    ax.plot(df["Epoch"], df["Train loss"], label = "Train loss")
    ax.plot(df["Epoch"], df["Validation loss"], label = "Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()

    # Save the figure to the output folder
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi = 300)
    plt.close(fig)

def pv_predict(model, X, M, z0):

    # Put model into evaluation mode
    model.eval()
    z_top = []

    # Do not keep track of gradients
    with torch.no_grad():

        # Iterate through batches of the data
        for i in range(0, X.shape[0], batch_size):
            p = torch.softmax(model(X[i : i + batch_size], M[i : i + batch_size], step = "score"), dim = 1).cpu().numpy()
            z = z0[i : i + batch_size]

            # The highest-scoring track in each event
            top = p.argmax(axis = 1)
            z_best = z[np.arange(len(top)), top]
            z_top.append(z_best)

    return np.concatenate(z_top)

def summarise(name, z_pred, truth_z):

    # Only events with a valid truth vertex are counted
    ok = np.isfinite(truth_z)
    dz = np.abs(z_pred[ok] - truth_z[ok])

    # Return dictionary of method and efficiencies
    return {
        "Method" : name,
        f"Eff |dz|<{tol}mm (%)" : 100 * (dz < tol).mean(),
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
    X, track_mask, z0, feature_names, y1, y2, usable, truth_z = load_data()
    n_events, n_trk, n_feat = X.shape
    print(f"{n_events} events, up to {n_trk} tracks, {n_feat} features: {feature_names}")
    print(f"{(~usable).sum()} events have no track within {tol} mm of a valid truth vertex (skipped in phase 2)")

    # Split data into test and training by events
    all_events = np.arange(n_events)
    training_idx, test_idx = train_test_split(all_events, test_size = test_fraction, random_state = seed)
    train_idx, val_idx = train_test_split(training_idx, test_size = validation_fraction / (1 - test_fraction), random_state = seed)

    # Feature normalisation fitted on the real (non-padded) training tracks only
    # X[train_idx][track_mask[train_idx]] picks out the real tracks as a 2D (tracks, features) array
    scaler_X = StandardScaler()
    scaler_X.fit(X[train_idx][track_mask[train_idx]])
    X_scaled = scaler_X.transform(X.reshape(-1, n_feat)).reshape(n_events, n_trk, n_feat).astype(np.float32)
    X_scaled[~track_mask] = 0.0

    # Standardise the phase 1 targets too (also fitted on real training tracks only)
    scaler_y = StandardScaler()
    scaler_y.fit(y1[train_idx][track_mask[train_idx]])
    y1_scaled = scaler_y.transform(y1.reshape(-1, 1)).reshape(n_events, n_trk, 1).astype(np.float32)
    y1_scaled[~track_mask] = 0.0

    # Create the features, mask and target tensors for a set of events
    def tensors(idx, y, dtype):
        return (torch.tensor(X_scaled[idx], dtype = torch.float32, device = device),
                torch.tensor(track_mask[idx], dtype = torch.bool, device = device),
                torch.tensor(y[idx], dtype = dtype, device = device))

    # Initialise model (one model, used for both steps)
    model = VertexNetwork(n_feat, layers, neurons).to(device)

    # === Pre-training (step 1) ===
    if pretrain_epochs > 0:
        pre_train = tensors(train_idx, y1_scaled, torch.float32)
        pre_val = tensors(val_idx, y1_scaled, torch.float32)
        df1 = train_model(model, pre_train, pre_val, "pretrain", pretrain_epochs, lr)
        plot_losses(df1, "MSE (standardised log density)", "pretrain_loss.png")

    # === Training (step 2) ===
    # Only events with at least one hard-scatter track
    train = tensors(train_idx[usable[train_idx]], y2, torch.bool)
    val = tensors(val_idx[usable[val_idx]], y2, torch.bool)
    df2 = train_model(model, train, val, "train", n_epochs, lr_finetune if pretrain_epochs > 0 else lr)
    plot_losses(df2, "Loss (-log P(hard-scatter tracks))", "train_loss.png")

    # Predicted vertex z for every test event (all of them, including unusable ones)
    X_val, M_val, _ = tensors(val_idx, y2, torch.bool)
    z_top = pv_predict(model, X_val, M_val, z0[val_idx])

    # Keep results in a dataframe
    tag = f"pre-trained {pretrain_epochs} ep" if pretrain_epochs > 0 else "no pre-training"
    results = pd.DataFrame([
        summarise(f"NN ({tag})", z_top, truth_z[val_idx]),
    ])
    print("\nValidation set:")
    print(results.to_string(index = False, float_format = "%.3f"))

    # Save predictions to plot later on
    path = os.path.join(out_dir, f"nn_val_predictions_pretrain{pretrain_epochs}.npz")
    np.savez_compressed(path, event_id = val_idx, z_top = z_top, truth_z = truth_z[val_idx])
    torch.save(model.state_dict(), os.path.join(out_dir, f"vertex_network_pretrain{pretrain_epochs}.pt"))
    print(f"Saved: {Path(path).resolve()}")


if __name__ == "__main__":

    main()