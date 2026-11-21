# Primary Vertex Finding at the ATLAS Trigger: Comparing Classical Algorithms and Neural Networks
**MPhys Project** · University of Edinburgh

> Finding the primary vertex in high pile-up proton-proton collisions using trigger-level tracks, comparing classical track-density vertex finding algorithms with two neural network approaches: a network that selects the hard-scatter vertex from a set of candidates, and an end-to-end network that predicts it directly from the tracks, with a focus on both efficiency and per-event speed.

## Background

At the LHC, every bunch crossing produces many simultaneous proton-proton collisions, known as pile-up. Among them, at most one is the interesting hard-scatter collision, and identifying where along the beam axis it happened, the primary vertex, is essential for reconstructing the event. At the High-Luminosity LHC, ATLAS will face around 200 pile-up collisions per bunch crossing, making it much harder to pick out the correct vertex.

This needs to happen at the trigger, where decisions must be made in a tiny fraction of a second. This project recreates classical vertex finding algorithms from the literature using High-Level Trigger (HLT) tracks, then develops neural networks to find the primary vertex, comparing how efficiently and how quickly each method finds it in simulated HH → bbγγ events with 200 pile-up.

The first network uses the output of the classical two-step algorithm, scoring each vertex candidate and choosing the most likely hard scatter. This is accurate, but finding the candidates in the first place is slow. The second network removes the candidate-finding step entirely. It takes the tracks directly and, with physics-motivated information about each track's neighbours, picks out the tracks belonging to the hard scatter and predicts the primary vertex position.

## Repository Structure

```
masters/
│
├── src/
│   ├── histograms/               # Plotting histograms and track parameters from ROOT files
│   │   ├── plot_histograms.py
│   │   └── plot_ttree_tracks.py
│   │
│   ├── vertex_finding/
│   │   ├── algorithms/           # Classical vertex finding algorithms
│   │   │   ├── box_sum_pt.py
│   │   │   ├── gaussian_weighting.py
│   │   │   └── two_steps.py
│   │   └── analysis/             # Efficiency tables, dz distributions, event displays and timing
│   │
│   └── neural_networks/
│       ├── algorithms/           # Neural network training and hyperparameter tuning
│       │   ├── nn_pv_finding.py
│       │   ├── hyperparameter_tuning.py
│       │   └── nn_draft.py
│       └── analysis/             # Feature analysis, timing, dz distributions and event displays
│
├── utils/                        # Shared helpers for reading TTrees and plotting
│   ├── navigating_ttrees.py
│   └── plotting_functions.py
│
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- [ROOT](https://root.cern/install/) with PyROOT (used for RDataFrame and reading histograms)
- Access to the input ROOT files on CERNbox

### Installation

```bash
# Clone the repository
git clone https://github.com/grxcetait/masters.git
cd masters

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install numpy pandas matplotlib mplhep uproot awkward numba torch scikit-learn
```

> **Note:** ROOT is not installed through pip. The easiest way to get it is through conda (`conda install -c conda-forge root`) or an LCG release on lxplus. It is only needed for the scripts in `src/histograms/`.

### Running the Scripts

Scripts are run from the repository root so that the input and output paths resolve correctly. For example:

```bash
# Run the two-step vertex finding algorithm
python src/vertex_finding/algorithms/two_steps.py

# Train the neural network on the two-step candidates
python src/neural_networks/algorithms/nn_pv_finding.py
```

> **Note:** The input data is not included in this repository. Scripts expect the ROOT files under `CERNbox/initial_data/root_files/`. Settings such as track selections, grid spacing and hyperparameters are set at the top of each script.

## Methods

| Stage | Technique |
|---|---|
| Track Selection | HLT tracks with cuts on pT, η, d0 and the d0 and z0 significances |
| Classical Vertex Finding | Box sum of pT, Gaussian-weighted track density, two-step candidate finding |
| Candidate Selection | Neural network scoring each vertex candidate from its track density, number of tracks, Σ pT and Σ pT² |
| End-to-End Vertex Finding | Neural network scoring each track from its parameters (pT, η, d0, z0 and their uncertainties) plus neighbour features: the number of tracks, Σ pT and Σ pT² within 0.3 mm and 1.0 mm windows in z |
| Optimisation | Random search hyperparameter tuning, feature ranking and correlations |
| Validation | Efficiency of matching the true vertex within a z tolerance, \|dz\| distributions, per-event timing |

## Key Results

*This project is ongoing. The results below are preliminary and will be updated as the analysis is completed.*

All results use HLT tracks with pT > 1.5 GeV and no η cut, with a vertex counted as found if it lies within 0.3 mm of the true vertex.

| Method | Efficiency | Time per event (1 CPU) |
|---|---|---|
| Box sum, track count (0.1 mm window) | 56.3% | – |
| Box sum, Σ pT (0.5 mm window) | 27.5% | – |
| Gaussian weighting | 68.7% | – |
| Two-step + choose highest Σ pT | 46.3% | ~1.8 ms |
| Two-step + candidate neural network | 86.3% | ~1.8 ms |
| End-to-end neural network | 92.4% | ~0.8 ms |

For the box sum, each window size from 0.1 mm to 2.0 mm was scanned and the best is shown for each weighting.

The simplest approach, sliding a box along z and taking the window with the most tracks, finds the vertex in 56.3% of events. Weighting the box by pT makes it worse, not better (27.5% for Σ pT and 7.1% for Σ pT²), because a single hard pile-up track can outweigh the hard-scatter vertex. Gaussian weighting replaces the box with each track's own Gaussian, so well-measured tracks count for more. This raises the efficiency to 68.7% and gives a precise vertex position (median |dz| = 0.030 mm).

The two-step algorithm finds the true vertex among its 30 candidates in 98.7% of events, but choosing the right one is hard. Ranking candidates by a single property fails in predictable ways: Σ pT is fooled by pile-up vertices with many soft tracks, and the track density by compact pile-up clusters. The candidate neural network combines all four candidate properties and reaches 86.3%. 

Timing showed that candidate finding dominates the cost of the two-step method, at ~1.8 ms per event compared with under 0.1 μs to choose a candidate and ~7 μs for the candidate network.

The end-to-end network removes candidate finding altogether. Using only each track's own parameters it reached 65%, showing that the network could not learn which tracks are near each other on its own. Adding the number of tracks, Σ pT and Σ pT² within 0.3 mm and 1.0 mm of each track raised this to 92.6% on the validation set, higher than the candidate network, at ~0.8 ms per event (0.43 ms for the neighbour features and 0.36 ms for inference).

## Acknowledgements

This project is carried out at the University of Edinburgh as part of the ATLAS group.
