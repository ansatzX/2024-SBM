# SBM Toolkit - Spin-Boson Model Analysis Toolkit

A lightweight toolkit for analyzing spin-bosonser model dynamics and classifying quantum phases (coherent, incoherent, pseudo-coherent).

## Installation

```bash
cd sbm_toolkit
pip install -e numpy matplotlib scipy pywt
```

## Directory Structure

```
sbm_toolkit/
├── __init__.py
├── analysis/
│   ├── __init__.py
│   └── dynamics_classifier.py
├── visualization/
│   └── __init__.py
└── utils/
    ├── __init__.py
    └── io.py
```

## Quick Start

### Classify a Single Trajectory

```python
from sbm_toolkit.analysis import classify_dynamics
import numpy as np

# Your trajectory data (list or numpy array of |S_z| values)
trajectory = np.linspace(0.9, 0.1, 200)

# Classify
phase = classify_dynamics(trajectory)
print(f"Phase: {phase}")  # Output: coherent, incoherent, or pseudo-coherent
```

### Generate Phase Diagram

```bash
# Using the provided script
bash sbm_toolkit/scripts/generate_phase_diagram.sh

# Or manually with Python
python -c "
from sbm_toolkit.analysis import classify_dynamics
from sbm_toolkit.visualization import plot_phase_diagram
from sbm_toolkit.utils import load_array, save_pickle
import numpy as np
from pathlib import Path

# Load your simulation data
data = load_array('data/simulation_results.npz')

# Classify each trajectory
classifications = {}
# Assuming 21x21 grid (s from 0.0 to 1.0, alpha from 0.0 to 1.0)
s_vals = np.linspace(0, 1, 21)
alpha_vals = np.linspace(0, 1, 21)

for idx, trajectory in enumerate(data):
    phase = classify_dynamics(trajectory)
    s = round(float(s_vals[idx // 21]), 2)
    alpha = round(float(alpha_vals[idx % 21]), 2)
    classifications[(s, alpha)] = phase

# Save and plot
save_pickle(classifications, 'classifications.pkl')
fig = plot_phase_diagram(classifications, save_path='phase_diagram.png')
"
```

## Core API

### Analysis Module (`sbm_toolkit.analysis`)

#### Phase Constants
- `DYNAMICS_COHERENT` = "coherent"`
- `DYNAMICS_INCOHERENT` = "incoherent"`
- `DYNAMICS_PSEUDO_COHERENT` = "pseudo-coherent"`

#### Main Functions

**`classify_dynamics(data, s=None, alpha=None)`**
- Classify trajectory into quantum phase
- Returns: "coherent", "incoherent", or "pseudo-coherent"
- Parameters:
  - `data`: Spin population trajectory (list or numpy array)
  - `s`: Optional coupling parameter (not used in classification)
  - `alpha`: Optional dephasing parameter (not used in classification)

**`extract_dynamics_features(data)`**
- Extract 13 dynamics features from trajectory
- Returns: Dictionary with keys:
  - `f`: final_mean (mean of last 10 points)
  - `d`: decay_rate (normalized decay rate)
  - `h`: half_life (steps to reach half initial value)
  - `v`: variance (overall variance)
  - `ve`: var_early (variance in first quarter)
  - `vm`: var_mid (variance in middle half)
  - `vl`: var_late (variance in last quarter)
  - `s`: std (standard deviation)
  - `p`: power_total (FFT power spectrum sum)
  - `g`: mean_abs_grad (mean absolute gradient)
  - `mg`: max_grad (maximum absolute gradient)
  - `pk`: n_peaks (number of peaks)
  - `et`: early_trend (linear trend in first portion)

### Visualization Module (`sbm_toolkit.visualization`)

**`plot_phase_diagram(classification_results, figsize=(10,8), title="SBM Phase Diagram", save_path=None)`**
- Plot phase diagram from classification results
- Parameters:
  - `classification_results`: Dict of {(s, alpha): phase}
  - `figsize`: Figure size in inches
  - `title`: Plot title
  - `save_path`: Optional path to save PNG

**`plot_spin_population(time_data, population_data, figsize=(10,6), title="Spin Population Dynamics", save_path=None)`**
- Plot spin population vs time

### Utils Module (`sbm_toolkit.utils`)

**`save_pickle(data, filepath)`**
- Save data to pickle file

**`load_pickle(filepath)`**
- Load data from pickle file

**`save_array(data, filepath)`**
- Save numpy array to .npz (compressed) format

**`load_array(filepath)`**
- Load numpy array from file

## Phase Classification Rules

The classifier uses **dynamics features only**, not parameter boundaries:

### Primary Discriminant: Decay Rate (`d`)

| Phase | Decay Rate Range | Physical Meaning |
|--------|-----------------|----------------|
| Coherent | d > 0.25 | Damped oscillatory behavior |
| Incoherent | 0.06 ≤ d ≤ 0.25 | Monotonic decay |
| Pseudo-coherent | d < 0.06 | Localized/preserved behavior |

### Secondary Discriminants (in boundary region 0.06 ≤ d ≤ 0.25)

| Feature | Description |
|---------|-------------|
| `power_total` | FFT power spectrum sum (high → incoherent) |
| `var_early` | Variance in first quarter |
| `max_grad` | Maximum gradient |
| `final_mean` | Mean of last 10 points |
| `variance` | Overall variance |

### Full Classification Logic

```
if d > 0.32:
    → coherent
elif d > 0.25 and f < 0.4:
    → coherent
elif d > 0.22 and v > 0.02:
    → coherent
elif d < 0.04:
    → pseudo-coherent
elif d < 0.06 and f > 0.8:
    → pseudo-coherent
elif 0.06 <= d <= 0.25:
    if f > 0.7:
        if p > 200 → incoherent
        elif p > 100 and mg > 0.3 → incoherent
        elif ve < 0.003 → incoherent
        else → pseudo-coherent
    elif f > 0.5:
        if v > 0.005 → incoherent
        elif p > 80 → incoherent
        elif ve > 0.01 → incoherent
        else → pseudo-coherent
    else:  # f < 0.5
        if d > 0.15 → coherent
        elif v > 0.004 → incoherent
        else → incoherent
```

## Performance

Tested on `labeled_data_v4.pkl` (369 samples):
- **Overall accuracy**: 85.09% (314/369)
- **Coherent**: 91.94% (57/62)
- **Pseudo-coherent**: 95.65% (242/253)
- **Incoherent**: 27.78% (15/54)

## References

The classifier was developed using data-driven analysis on labeled trajectories.
For more details, see `dynamics_classifier.py` source code.

## License

MIT License
