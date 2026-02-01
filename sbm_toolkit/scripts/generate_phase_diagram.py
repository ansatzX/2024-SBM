#!/usr/bin/env python3
"""
SBM Phase Diagram Generation Script

Usage: python generate_phase_diagram.py [OPTIONS]

Options:
  --data-path PATH    Path to data directory (default: data/)
  --output-dir DIR    Output directory (default: results/)
  --help              Show this help message

Example:
  python generate_phase_diagram.py --data-path ../data/ --output-dir ./results/
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to find sbm_toolkit
SCRIPT_DIR = Path(__file__).parent
SBM_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SBM_DIR))

from sbm_toolkit.analysis import classify_dynamics
from sbm_toolkit.utils import load_array, save_pickle
from sbm_toolkit.visualization import plot_phase_diagram

# Default configuration
DATA_DIR = Path("data/")
OUTPUT_DIR = Path("results/")

# Parse arguments
args = sys.argv[1:]
i = 0
while i < len(args):
    arg = args[i]
    if arg == '--data-path' and i + 1 < len(args):
        DATA_DIR = Path(args[i + 1])
        i += 1
    elif arg == '--output-dir' and i + 1 < len(args):
        OUTPUT_DIR = Path(args[i + 1])
        i += 1
    elif arg == '--help':
        print(__doc__)
        sys.exit(0)
    i += 1

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 50)
print("SBM Phase Diagram Generation")
print("=" * 50)
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print()

# Load simulation data
print("Loading simulation data...")
data_path = DATA_DIR / "simulation_results.npz"

if data_path.exists():
    data = load_array(data_path)
    print(f"Data shape: {data.shape}")
else:
    print(f"ERROR: Data file not found: {data_path}")
    print("  Please ensure simulation_results.npz exists in data directory")
    sys.exit(1)

# Classify each trajectory
print("Classifying phases...")
classifications = {}

# Extract (s, alpha) grid parameters
n_samples = len(data)
s_vals = np.linspace(0, 1, 21)
alpha_vals = np.linspace(0, 1, 21)

for idx in range(n_samples):
    trajectory = data[idx]
    phase = classify_dynamics(trajectory)

    # Get s and alpha from grid
    s_idx = idx // 21
    alpha_idx = idx % 21

    if s_idx < len(s_vals) and alpha_idx < len(alpha_vals):
        s = round(float(s_vals[s_idx]), 2)
        alpha = round(float(alpha_vals[alpha_idx]), 2)
        classifications[(s, alpha)] = phase
    else:
        # Fallback: use index as key
        classifications[idx] = phase

# Save classifications
print("Saving classifications...")
save_pickle(classifications, OUTPUT_DIR / "classifications.pkl")
print(f"  Saved: {OUTPUT_DIR}/classifications.pkl")

# Plot phase diagram
print("Plotting phase diagram...")
fig = plot_phase_diagram(
    classifications,
    title="SBM Phase Diagram",
    save_path=OUTPUT_DIR / "phase_diagram.png"
)

print("=" * 50)
print("Done!")
print(f"  Phase diagram: {OUTPUT_DIR}/phase_diagram.png")
print("=" * 50)
