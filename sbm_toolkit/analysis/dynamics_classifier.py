"""Dynamics classification for spin population (procedural style)

Strictly three phase types as defined by user:
- Coherent: Damped oscillation (multiple peaks and valleys)
- Incoherent: Monotonic decay (all descending)
- Pseudo-coherent: Single valley then localization
"""

import numpy as np
import scipy
from scipy import signal
from typing import List, Tuple

# Dynamics type constants (STRICTLY ONLY THESE THREE)
DYNAMICS_COHERENT = "coherent"
DYNAMICS_INCOHERENT = "incoherent"
DYNAMICS_PSEUDO_COHERENT = "pseudo-coherent"


def detect_monotonic_segments(data: List[float], atol: float = 5e-3,
                             rtol: float = 5e-2) -> List[int]:
    """
    Detect monotonic behavior in time series.

    Args:
        data: Time series data
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        List of segment types:
            0: ascend, 1: descend, 2: peak, 3: valley, 4: no change
    """
    if data[0] < data[1]:
        segments = [0]  # Start with ascend
    else:
        segments = [1]  # Start with descend

    window_length = 1

    for i in range(window_length, len(data) - window_length, window_length):
        p0, p1, p2 = data[i-window_length], data[i], data[i+window_length]

        if np.allclose(p2, p0, atol=atol, rtol=rtol):
            segments.append(4)  # No change
        else:
            # Determine segment type
            sorted_vals = sorted([p0, p1, p2])

            if sorted_vals[0] == p2:
                segments.append(3)  # Valley
            elif sorted_vals[2] == p2:
                segments.append(2)  # Peak
            elif sorted_vals[0] == p0 and sorted_vals[2] == p2:
                segments.append(0)  # Ascend
            elif sorted_vals[0] == p2 and sorted_vals[2] == p0:
                segments.append(1)  # Descend
            else:
                segments.append(4)  # No clear trend

    # Pad end
    for _ in range(window_length):
        segments.append(4)

    return segments


def classify_dynamics(data: np.ndarray, atol: float = 5e-3,
                     rtol: float = 5e-2) -> str:
    """
    Classify dynamics type based on spin population time series.

    Classification scheme (STRICTLY ONLY THESE THREE):
    - Coherent: Damped oscillation (multiple peaks and valleys)
    - Incoherent: Monotonic decay (all descending)
    - Pseudo-coherent: Single valley then localization

    Args:
        data: Time series of spin population
        atol: Absolute tolerance for peak detection
        rtol: Relative tolerance for peak detection

    Returns:
        Dynamics type string
    """
    # Detect segments
    segments = detect_monotonic_segments(data.tolist(), atol, rtol)

    # Find peaks and valleys using scipy
    max_indices = signal.argrelmax(data)[0]
    min_indices = signal.argrelmin(data)[0]

    # Filter out noise peaks/valleys
    if len(min_indices) > 0 and len(max_indices) > 0:
        true_max = []
        true_min = []

        for i_max in range(len(max_indices)):
            if i_max < len(min_indices):
                max_idx = max_indices[i_max]
                min_idx = min_indices[i_max]
                max_val = data[max_idx]
                min_val = data[min_idx]

                if not np.allclose(max_val, min_val, atol=atol, rtol=rtol):
                    true_max.append(max_idx)
                    true_min.append(min_idx)

        max_indices = np.array(true_max)
        min_indices = np.array(true_min)

    # Classification logic
    n_max = len(max_indices)
    n_min = len(min_indices)

    # Check for pure descend (incoherent) - STRICT classification
    n_descend = segments.count(1)
    n_total = len(segments) - segments.count(4)  # Exclude no-change segments

    if n_descend == n_total and n_total > 0:
        return DYNAMICS_INCOHERENT

    # No extrema (monotonic descent)
    if n_min == 0 and n_max == 0:
        return DYNAMICS_INCOHERENT

    # Filter out constant segments for further analysis
    refined_data = []
    for i in range(len(segments)):
        if segments[i] != 4:
            refined_data.append(data[i])

    if len(refined_data) < 3:
        return DYNAMICS_INCOHERENT  # Default to incoherent if data is too short

    refined_data = np.array(refined_data)
    max_indices_refined = signal.argrelmax(refined_data)[0]
    min_indices_refined = signal.argrelmin(refined_data)[0]

    n_max_refined = len(max_indices_refined)
    n_min_refined = len(min_indices_refined)

    # Single minimum followed by decay (pseudo-coherent)
    if n_max_refined == 1 and n_min_refined == 1:
        if max_indices_refined[0] > min_indices_refined[0]:
            return DYNAMICS_PSEUDO_COHERENT

    # Multiple extrema (oscillation/coherent) - STRICT: require multiple peaks and valleys
    if n_max_refined >= 1 and n_min_refined > 1:
        return DYNAMICS_COHERENT

    # Single valley (pseudo-coherent)
    if n_min_refined == 1 and n_max_refined == 0:
        return DYNAMICS_PSEUDO_COHERENT

    # Default to incoherent if no clear classification
    return DYNAMICS_INCOHERENT


def classify_phase_region(data_dict: dict, atol: float = 5e-3,
                         rtol: float = 5e-2) -> Tuple[dict, list]:
    """
    Classify dynamics for multiple trajectories.

    Args:
        data_dict: Dictionary of {(s, alpha): trajectory_data}
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Tuple of (results_dict, unclassified_list)
        - results_dict: Dictionary of {(s, alpha): dynamics_type}
        - unclassified_list: List of keys that couldn't be classified
    """
    results = {}
    unclassified = []

    for key, trajectory in data_dict.items():
        s, alpha = key
        if alpha < 0.1:
            continue  #舍去 alpha 小于0.1的数据

        dynamics_type = classify_dynamics(np.array(trajectory), atol, rtol)
        results[key] = dynamics_type

    return results, unclassified
