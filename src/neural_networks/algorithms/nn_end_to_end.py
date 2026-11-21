#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
import time
from pathlib import Path
import os

import uproot
import awkward as ak

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Path routing to allow importing from the utils directory
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


# === ARGUMENTS ====
# Training settings
test_fraction = 0.15
validation_fraction = 0.15
n_epochs = 30
seed = 42
n_events_max = np.inf
evaluate_test = True    # set True only for the final run of a chosen configuration

# Hyperparameters
lr = 1e-3
batch_size = 32
layers = 3
neurons = 64

# Which feature groups to use (all off = the same inputs as your previous network)
use_neighbour_features = True   # tracks / sum pT / sum pT^2 within +- each window
neighbour_windows = [0.3, 1.0]  # half-widths of the z windows [mm]

# Timing settings
timings = False      # Set to True to time

# Track selection
pt_min = 1.5        # [GeV]
d0_max = 4.0        # [mm]
sig_d0_max = 5.0    # [mm]
sig_z0_max = 10.0   # [mm]

# Matching tolerance [mm]: tracks within this distance of the truth vertex count as hard-scatter tracks,
# and a predicted vertex within this distance of the truth counts as found
tol = 0.3

# Window [mm] around the top-scoring track used to average z0 for the final prediction
z_window = 1.0

output_dir = "outputs/sem1week10"
out_dir = os.path.join(output_dir)

filename = "CERNbox/initial_data/root_files/HHbbyy_200PU.root"
tree_name = "Events"

# Create a list of the branches we are interested in
track_branches = ["HLTTrack_pt", "HLTTrack_eta", "HLTTrack_d0", "HLTTrack_z0",
                  "HLTTrack_d0Err", "HLTTrack_z0Err",
                  "HLTTrack_chi2", "HLTTrack_ndof", "HLTTrack_rFirstHit",
                  "TruthVertex_z", "TruthVertex_isValid", "mu",
                  "HLTVertex_z", "HLTVertex_type"]

# === FUNCTIONS ====

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_parameters():

    # Load in events from the root file
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

    # Track quality: chi2 per degree of freedom (guard against ndof = 0) and first-hit radius
    chi2_ndof = events["HLTTrack_chi2"] / np.maximum(events["HLTTrack_ndof"], 1.0)
    r_first = events["HLTTrack_rFirstHit"]

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

    # Put parameters into a dictionary (still one list of tracks per event)
    parameters = {name: a[mask] for name, a in
                  [("pt", pt), ("eta", eta), ("z0", z0), ("d0", d0),
                   ("sig_z0", sig_z0), ("sig_d0", sig_d0),
                   ("chi2_ndof", chi2_ndof), ("r_first", r_first)]}

    return parameters, truth_z

def pad(a, n_trk, fill = 0.0):

    # Events have different numbers of tracks, so pad every event up to n_trk tracks
    return ak.to_numpy(ak.fill_none(ak.pad_none(a, n_trk, clip = True), fill)).astype(np.float32)

def neighbour_features(z0, pt):
    # For a event: for every track, how many other tracks (and how much pT, pT^2)

    # Sort by z0
    order = np.argsort(z0)
    zs = z0[order]

    # Cumulative sums of pt and pt2
    cumsum_pt = np.concatenate([[0.0], np.cumsum(pt[order])])
    cumsum_pt2 = np.concatenate([[0.0], np.cumsum(pt[order] ** 2)])

    # Empty array to hold data
    columns = []

    # Iterate through value of neighbour distance window
    for w in neighbour_windows:

        # Obtain the position of the first sorted track with z => z0 - w
        lo = np.searchsorted(zs, z0 - w, side = "left")

        # Obtain the position of the last sorted track with z <= `0 + w`
        hi = np.searchsorted(zs, z0 + w, side = "right")

        # Calculate the number of tracks inside the window
        # (minus itself) = number of neighbours
        n_near = (hi - lo) - 1

        # Calculate the sum pt and pt2 around the track
        # Subtract the track itself so these only describe its neighbours
        sum_pt = cumsum_pt[hi] - cumsum_pt[lo] - pt
        sum_pt2 = cumsum_pt2[hi] - cumsum_pt2[lo] - pt ** 2

        # Append data to the array
        columns += [np.log1p(n_near), np.log1p(np.maximum(sum_pt, 0.0)), np.log1p(np.maximum(sum_pt2, 0.0))]

    return np.stack(columns, axis = -1)

def get_features(parameters, n_trk):

    # Only log-transform the positive, long-tailed variables
    features = {
        "log_pt": np.log1p(parameters["pt"]),
        "eta": parameters["eta"],
        "d0": parameters["d0"],
        "z0": parameters["z0"],
        "log_sig_z0": np.log1p(parameters["sig_z0"]),
        "log_sig_d0": np.log1p(parameters["sig_d0"]),
    }
    names = list(features.keys())

    # Stack into one array of shape (events, tracks, features)
    # Padding is needed because numpy can't stack events with different numbers of tracks
    X_base = np.stack([pad(features[k], n_trk) for k in names], axis = -1)

    # True for real tracks, False for padding
    track_mask = pad(ak.ones_like(parameters["pt"]), n_trk) > 0

    # If not using neighbour feature then just return as it is
    if not use_neighbour_features:
        return X_base, track_mask, names

    # Neighbour features, computed event by event and written into a padded array
    n_events = len(parameters["pt"])
    n_neighbour_info = 3 * len(neighbour_windows) # Calculating number of nearby neighbours, sumpt and sumpt2 (3)
    X_near = np.zeros((n_events, n_trk, n_neighbour_info), dtype = np.float32)

    # Time how long this takes
    if timings: 
        start = time.perf_counter() 

    # Iterate through the number of events 
    for e in range(n_events):
        z0 = ak.to_numpy(parameters["z0"][e][:n_trk]).astype(np.float64)
        pt = ak.to_numpy(parameters["pt"][e][:n_trk]).astype(np.float64)
        X_near[e, :len(z0)] = neighbour_features(z0, pt)

    if timings: 
        # Print out how long it took
        per_event_ms = 1000 * (time.perf_counter() - start) / n_events
        print(f"Neighbour features: {per_event_ms:.3f} ms/event")

    # Add names of the neighbour parameters
    for w in neighbour_windows:
        names += [f"log_n_near_{w}mm", f"log_sumpt_near_{w}mm", f"log_sumpt2_near_{w}mm"]

    return np.concatenate([X_base, X_near], axis = -1), track_mask, names

def hs_target(z0, track_mask, truth_z):

    # Hard-scatter tracks = real tracks within tol of the truth vertex
    dz = np.abs(z0 - truth_z[:, None])
    target = (dz < tol) & track_mask

    # Events with no hard-scatter track (or no valid truth) can't be learned from
    usable = target.any(axis = 1)

    return target, usable

def load_data():

    # Load in parameters from the file
    parameters, truth_z = load_parameters()
    n_trk = int(ak.max(ak.num(parameters["pt"])))

    # Obtain the features and their names
    X, track_mask, feature_names = get_features(parameters, n_trk)
    z0 = pad(parameters["z0"], n_trk)
    pt = pad(parameters["pt"], n_trk)

    # Targets
    y, usable = hs_target(z0, track_mask, truth_z)

    return X, track_mask, z0, pt, feature_names, y, usable, truth_z


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

        # One score per track
        self.score = nn.Linear(neurons, 1)

    def forward(self, x, track_mask):

        # Pass each track through the hidden layers (same weights for every track)
        for layer in self.layers:
            x = self.relu(layer(x))

        # Shape (events, tracks). Padded tracks get -inf so they can never be chosen
        return self.score(x).squeeze(-1).masked_fill(~track_mask, float("-inf"))

def score_loss(logits, target):

    # Probability the network puts on any hard-scatter track: -log( sum over HS tracks of softmax )
    log_p = torch.log_softmax(logits, dim = 1)
    log_p_hs = torch.logsumexp(log_p.masked_fill(~target, float("-inf")), dim = 1)
    return -log_p_hs.mean()

def run_epoch(model, X, M, Y, optimiser = None):

    # Train if an optimiser is given, otherwise just measure the loss
    training = optimiser is not None
    model.train(training)
    n = X.shape[0]
    perm = torch.randperm(n, device = X.device) if training else torch.arange(n, device = X.device)
    running_loss = 0.0

    with torch.set_grad_enabled(training):
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = score_loss(model(X[idx], M[idx]), Y[idx])

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            running_loss += loss.item() * len(idx)

    return running_loss / n

def train_model(model, train, val, epochs, learning_rate):

    # Training loop. The validation loss is monitored and used to keep the best epoch
    optimiser = optim.Adam(model.parameters(), lr = learning_rate)
    metrics = []

    for epoch in range(epochs):
        train_loss = run_epoch(model, *train, optimiser)
        val_loss = run_epoch(model, *val)

        metrics.append({"Epoch": epoch + 1, "Train loss": train_loss, "Validation loss": val_loss})
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    return pd.DataFrame(metrics)

def plot_losses(df, name):

    # Plot training and validation loss over epochs
    fig, ax = plt.subplots()
    ax.plot(df["Epoch"], df["Train loss"], label = "Train loss")
    ax.plot(df["Epoch"], df["Validation loss"], label = "Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (-log P(hard-scatter tracks))")
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
            p = torch.softmax(model(X[i : i + batch_size], M[i : i + batch_size]), dim = 1).cpu().numpy()
            z = z0[i : i + batch_size]

            # The highest-scoring track in each event
            top = p.argmax(axis = 1)
            z_best = z[np.arange(len(top)), top]
            z_top.append(z_best)

    return np.concatenate(z_top)

def summarise(name, z_pred, truth_z):

    # Only events with a valid truth vertex are counted. A missing prediction (NaN) counts as not found
    ok = np.isfinite(truth_z)
    dz = np.abs(z_pred[ok] - truth_z[ok])

    return {
        "Method" : name,
        f"Eff |dz|<{tol}mm (%)" : 100 * np.mean(dz < tol),
        "Median |dz| (mm)" : np.nanmedian(dz),
    }

def main():

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok = True)

    # Load data
    X, track_mask, z0, pt, feature_names, y, usable, truth_z = load_data()
    n_events, n_trk, n_feat = X.shape
    print(f"{n_events} events, up to {n_trk} tracks, {n_feat} features: {feature_names}")
    print(f"{(~usable).sum()} events have no track within {tol} mm of a valid truth vertex (skipped in training)")

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

    # Create the features, mask and target tensors for a set of events
    def tensors(idx):
        return (torch.tensor(X_scaled[idx], dtype = torch.float32, device = device),
                torch.tensor(track_mask[idx], dtype = torch.bool, device = device),
                torch.tensor(y[idx], dtype = torch.bool, device = device))

    # Initialise and train the model (only on events with at least one hard-scatter track)
    model = VertexNetwork(n_feat, layers, neurons).to(device)
    train = tensors(train_idx[usable[train_idx]])
    val = tensors(val_idx[usable[val_idx]])
    df = train_model(model, train, val, n_epochs, lr)
    plot_losses(df, "train_loss.png")

    # Evaluate on ALL events of the chosen set (validation by default)
    eval_idx, set_name = (test_idx, "Test") if evaluate_test else (val_idx, "Validation")
    X_eval, M_eval, _ = tensors(eval_idx)

    # Time the trained model on the CPU
    if timings: 
        torch.set_num_threads(1)        # Set number of CPUs to 1
        model_cpu = model.to("cpu")
        X_cpu, M_cpu = X_eval.cpu(), M_eval.cpu()

        # Warm-up run (the first call includes one-off start-up costs)
        pv_predict(model_cpu, X_cpu[:batch_size], M_cpu[:batch_size], z0[eval_idx][:batch_size])

        start = time.perf_counter()
        z_top = pv_predict(model_cpu, X_cpu, M_cpu, z0[eval_idx])
        nn_ms = 1000 * (time.perf_counter() - start) / len(eval_idx)
        print(f"NN inference: {nn_ms:.3f} ms/event (1 CPU thread, batch of {batch_size})")

    # Else, just find the primary vertex 
    else:
        z_top = pv_predict(model, X_eval, M_eval, z0[eval_idx])

    # Keep results in a dataframe
    groups = []
    if use_neighbour_features: groups.append(f"neighbours {neighbour_windows} mm")
    tag = " + ".join(groups) if groups else "base features only"

    results = pd.DataFrame([
        summarise(f"NN top track z0 ({tag})", z_top, truth_z[eval_idx]),
    ])
    print(f"\n{set_name} set ({len(eval_idx)} events):")
    print(results.to_string(index = False, float_format = "%.3f"))

    # Save predictions
    suffix = "_".join(groups).replace(" ", "").replace("[", "").replace("]", "").replace(",", "-") or "base"
    path = os.path.join(out_dir, f"optionB_{set_name.lower()}_{suffix}.npz")
    np.savez_compressed(path, event_id = eval_idx, z_top = z_top, truth_z = truth_z[eval_idx])
    torch.save(model.state_dict(), os.path.join(out_dir, f"optionB_model_{suffix}.pt"))
    print(f"Saved: {Path(path).resolve()}")


if __name__ == "__main__":

    main()
