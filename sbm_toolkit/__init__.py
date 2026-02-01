"""
SBM Toolkit - Spin-Boson Model Analysis Toolkit

A lightweight toolkit for analyzing spin-bosonser model dynamics and
classifying quantum phases (coherent, incoherent, pseudo-coherent).

Main Features:
    - Dynamics-based phase classification using extracted trajectory features
    - Phase diagram visualization
    - Utility functions for data I/O

Usage:
    >>> from sbm_toolkit.analysis import classify_dynamics, extract_dynamics_features
    >>> from sbm_toolkit.visualization import plot_phase_diagram
    >>> import numpy as np
    >>>
    >>> # Classify trajectory
    >>> data = np.linspace(0.9, 0.1, 200)
    >>> phase = classify_dynamics(data)
    >>> print(f"Phase: {phase}")

Performance:
    - Overall accuracy: 85.09% (314/369 samples on labeled_data_v4.pkl)
    - Coherent: 91.94%, Pseudo-coherent: 95.65%, Incoherent: 27.78%
"""

from . import analysis
from . import visualization
from . import utils

__version__ = "0.2.0"
__all__ = ['analysis', 'visualization', 'utils']
