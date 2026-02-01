#!/bin/bash
# SBM Toolkit - Phase Diagram Generation Script
#
# Usage: ./generate_phase_diagram.sh
#

echo "========================================"
echo "SBM Phase Diagram Generation Guide"
echo "========================================"
echo ""
echo "This script shows the commands needed to generate a phase diagram."
echo ""
echo "1. Load simulation data"
echo "----------------------------------------"
echo "python -c \"
echo "  from sbm_toolkit.utils import load_array"
echo "  import numpy as np"
echo "  data = load_array('data/simulation_results.npz')"
echo "  print(f'Loaded data shape: {data.shape}')"
echo "\""
echo ""
echo "2. Extract dynamics features"
echo "----------------------------------------"
echo "python -c \"
echo "  from sbm_toolkit.analysis import extract_dynamics_features"
echo "  from sbm_toolkit.utils import load_array"
echo "  data = load_array('data/simulation_results.npz')"
echo "  features = extract_dynamics_features(data[0])"
echo "  print(f'Decay rate: {features[\"d\"]:.4f}, Final mean: {features[\"f\"]:.4f}')"
echo "\""
echo ""
echo "3. Classify phases"
echo "----------------------------------------"
echo "python -c \"
echo "  from sbm_toolkit.analysis import classify_dynamics, extract_dynamics_features"
echo "  from sbm_toolkit.utils import load_array, save_pickle"
echo "  from sbm_toolkit.visualization import plot_phase_diagram"
echo "  import numpy as np"
echo "  "
echo "  # Load data"
echo "  data = load_array('data/simulation_results.npz')"
echo "  "
echo "  # Classify each trajectory"
echo "  classifications = {}"
echo "  for i in range(len(data)):"
echo "      phase = classify_dynamics(data[i])"
echo "      classifications[(s, alpha)] = phase"
echo "  "
echo "  # Save"
echo "  save_pickle(classifications, 'classifications.pkl')"
echo "  "
echo "  # Plot"
echo "  fig = plot_phase_diagram(classifications, save_path='phase_diagram.png')"
echo "  fig.close()"
echo "\""
echo ""
echo "4. Plot phase diagram"
echo "----------------------------------------"
echo "The same commands as above in step 3"
echo ""
echo "========================================"
echo "Complete Python Script Example"
echo "========================================"
cat << 'EOF'
import numpy as np
from pathlib import Path
from sbm_toolkit.analysis import classify_dynamics, extract_dynamics_features
from sbm_toolkit.visualization import plot_phase_diagram
from sbm_toolkit.utils import load_array, save_pickle

# Load data
DATA_DIR = Path("data/")
OUTPUT_DIR = Path("results/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
data = load_array(DATA_DIR / "simulation_results.npz")
print(f"Data shape: {data.shape}")

# Classify each trajectory
print("Classifying...")
classifications = {}

# Generate (s, alpha) grid
s_vals = np.linspace(0, 1, 21)  # 21 values: 0.0 to 1.0
alpha_vals = np.linspace(0, 1, 21)

for idx, trajectory in enumerate(data):
    phase = classify_dynamics(trajectory)
    s = round(float(s_vals[idx // 21]), 2)
    alpha = round(float(alpha_vals[idx % 21]), 2)
    classifications[(s, alpha)] = phase

# Save
save_pickle(classifications, OUTPUT_DIR / "classifications.pkl")

# Plot
print("Plotting...")
fig = plot_phase_diagram(
    classifications,
    title="SBM Phase Diagram",
    save_path=OUTPUT_DIR / "phase_diagram.png"
)

print(f"Saved to: {OUTPUT_DIR}")
EOF
