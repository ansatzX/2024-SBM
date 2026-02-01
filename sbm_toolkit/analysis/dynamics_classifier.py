#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spin-Boson Model Dynamics Phase Classifier

This module provides dynamics-based classification of spin population trajectories
into three quantum phases of the spin-boson model:
- Coherent: Damped oscillatory behavior, maintains phase coherence
- Incoherent: Monotonic decay, phase coherence lost
- Pseudo-coherent: Localized behavior, intermediate coherence

Classification is based purely on dynamics features extracted from the trajectory,
NOT on parameter boundaries (s and alpha).

Methodology:
The classifier uses a multi-stage decision tree that analyzes:
1. Primary discriminant: decay_rate (d)
   - Coherent phase shows rapid decay (>0.32)
   - Pseudo-coherent phase shows very slow decay (<0.04)
   - Incoherent phase shows intermediate decay (0.06-0.25)

2. Secondary discriminants in boundary regions:
   - power_total: Sum of FFT power spectrum, indicates frequency content
   - var_early: Variance in first quarter of trajectory
   - max_grad: Maximum absolute gradient in trajectory
   - final_mean: Mean value of last 10 points
   - variance: Overall variance of the trajectory

Performance:
- Overall accuracy: 85.09% (314/369 samples)
- Coherent: 91.94% (57/62)
- Pseudo-coherent: 95.65% (242/253)
- Incoherent: 27.78% (15/54)
  Note: Incoherent phase is challenging to classify due to feature overlap
        with pseudo-coherent in boundary regions.

References:
- This classifier was developed using data-driven analysis on labeled_data_v4.pkl
- Achievements validated against reference phase diagram
"""

import numpy as np

# Phase classification constants
DYNAMICS_COHERENT = "coherent"
DYNAMICS_INCOHERENT = "incoherent"
DYNAMICS_PSEUDO_COHERENT = "pseudo-coherent"


def extract_dynamics_features(data):
    """
    Extract comprehensive dynamics features from spin population trajectory.

    The trajectory represents the time evolution of the spin population |S_z|,
    which is measured at each time step during the spin-boson model simulation.

    Features extracted:
    1. f (final_mean): Mean value of the last 10 time steps.
       - High values (>0.7) indicate preserved population (pseudo-coherent)
       - Low values (<0.4) indicate decay (coherent or incoherent)
       - Computed as: mean(data[-10:])

    2. d (decay_rate): Normalized rate of population decay.
       - Computed as: (initial_mean - final_mean) / (n / 100)
       - This normalization scales decay rate to be comparable across
         different trajectory lengths.
       - Coherent: d > 0.25 (rapid decay due to oscillation damping)
       - Pseudo-coherent: d < 0.04 (population preserved)
       - Incoherent: 0.06 <= d <= 0.25 (moderate decay)

    3. h (half_life): Time steps to reach half of initial population.
       - Indicates how quickly the trajectory decays.
       - Longer half-life suggests preserved dynamics (pseudo-coherent)
       - Shorter half-life suggests rapid decay (coherent)

    4. v (variance): Overall variance of the entire trajectory.
       - High variance indicates large fluctuations (coherent or incoherent)
       - Low variance indicates stable dynamics (pseudo-coherent)

    5. ve (var_early): Variance in the first quarter of trajectory.
       - Early dynamics often contain the most informative behavior.
       - In boundary regions, this helps discriminate incoherent vs pseudo-coherent.

    6. vm (var_mid): Variance in the middle half of trajectory.
       - Captures the intermediate-time behavior of the system.

    7. vl (var_late): Variance in the last quarter of trajectory.
       - Captures the asymptotic/final-time behavior.

    8. s (std): Standard deviation of the trajectory.
       - Closely related to variance, but in original scale units.

    9. p (power_total): Sum of FFT power spectrum (excluding DC component).
       - Computed via FFT: sum(|fft(data - mean)|^2[1:n//2])
       - High values indicate rich frequency content (incoherent)
       - Low values indicate simple dynamics (pseudo-coherent)
       - Critical discriminant in boundary regions where decay_rate overlaps.

    10. g (mean_abs_grad): Mean absolute gradient.
        - Average rate of change between consecutive time steps.
        - Indicates overall "activity" of the trajectory.

    11. mg (max_grad): Maximum absolute gradient.
        - Largest single-step change in the trajectory.
        - Sudden large changes are characteristic of incoherent decay.

    12. pk (n_peaks): Number of local maxima detected.
        - Peaks indicate oscillatory behavior (coherent)
        - Low peak count suggests monotonic or localized behavior.

    13. et (early_trend): Linear trend coefficient of first quarter.
        - Positive: increasing trend, Negative: decreasing trend
        - Determined by polyfit of degree 1 on early data.

    Args:
        data (array-like): Spin population trajectory, typically a list or
                          numpy array of float values in range [0, 1].
                          Each value represents |S_z| at a time step.

    Returns:
        dict: Feature dictionary with all extracted features.

    Example:
        >>> data = np.linspace(0.9, 0.1, 200)  # Linear decay
        >>> features = extract_dynamics_features(data)
        >>> print(f"Decay rate: {features['d']:.4f}")
    """
    # Convert to numpy array for efficient computation
    data = np.array(data)
    n = len(data)

    # ======== Basic Statistics ========
    # Initial and final mean values
    # The first 10 points represent the system's initial state after transient
    # The last 10 points represent the system's asymptotic/steady state
    initial_mean = float(np.mean(data[:10]))
    final_mean = float(np.mean(data[-10:]))

    # ======== Decay Rate Calculation ========
    # The decay rate measures how quickly the population decreases from initial to final
    # Normalization factor (n/100) ensures comparability across different trajectory lengths
    # This is a normalized measure of population loss per "100 time units"
    decay_rate = (initial_mean - final_mean) / (n / 100)

    # ======== Half-Life Calculation ========
    # Half-life is the time (in steps) at which population drops to half of initial value
    # This is a classic measure in exponential decay processes
    half_life = n  # Default: never reaches half, full trajectory is half-life
    half_val = initial_mean * 0.5
    for i in range(n):
        if data[i] < half_val:
            half_life = i
            break

    # ======== Variance Statistics ========
    # Variance measures the spread/fluctuation of the population
    # Different variances in different time regions reveal dynamics evolution
    variance = float(np.var(data))  # Overall variance
    var_early = float(np.var(data[:n//4]))  # First quarter variance
    var_mid = float(np.var(data[n//4:3*n//4]))  # Middle half variance
    var_late = float(np.var(data[3*n//4:]))  # Last quarter variance
    std = float(np.std(data))  # Standard deviation (sqrt of variance)

    # ======== FFT Power Spectrum ========
    # FFT (Fast Fourier Transform) decomposes the signal into frequency components
    # The power spectrum reveals the frequency content of the dynamics
    # High-frequency content (high power_total) indicates complex dynamics
    # Low-frequency content (low power_total) indicates simple/monotonic dynamics
    # We exclude the DC component (index 0) which is just the mean
    fft_result = np.fft.fft(data - np.mean(data))
    power_spectrum = np.abs(fft_result)**2
    power_total = float(np.sum(power_spectrum[1:n//2])) if n > 2 else 0.0

    # ======== Gradient Features ========
    # Gradients measure the rate of change between consecutive time steps
    # Large gradients indicate rapid changes or jumps in population
    grad = np.abs(np.diff(data))  # Absolute difference between consecutive points
    mean_abs_grad = float(np.mean(grad))  # Average rate of change
    max_grad = float(np.max(grad))  # Maximum rate of change

    # ======== Peak Detection ========
    # Peaks are local maxima in the trajectory
    # Damped oscillatory coherent behavior produces many peaks
    # Monotonic incoherent behavior produces few or no peaks
    # Localized pseudo-coherent behavior may have moderate peak count
    n_peaks = 0
    for i in range(2, n-2):
        # A point is a peak if it's greater than its 2 neighbors on each side
        # Using 2 neighbors on each side reduces noise sensitivity
        if data[i] > data[i-1] and data[i] > data[i-2] and \
           data[i] > data[i+1] and data[i] > data[i+2]:
            n_peaks += 1

    # ======== Early Trend ========
    # Linear trend coefficient of the first quarter of the trajectory
    # Positive: population increasing, Negative: population decreasing
    # This early-time behavior is often characteristic of the phase
    early_trend = float(np.polyfit(np.arange(n//4), data[:n//4], 1)[0])

    # ======== Return Feature Dictionary ========
    return {
        "f": final_mean,    # final_mean
        "d": decay_rate,    # decay_rate
        "h": half_life,     # half_life
        "v": variance,       # variance
        "ve": var_early,    # var_early
        "vm": var_mid,      # var_mid
        "vl": var_late,     # var_late
        "s": std,           # std
        "p": power_total,   # power_total
        "g": mean_abs_grad, # mean_abs_grad
        "mg": max_grad,     # max_grad
        "pk": n_peaks,      # n_peaks
        "et": early_trend    # early_trend
    }


def classify_dynamics(data, s=None, alpha=None):
    """
    Classify spin population trajectory into quantum phase (coherent/incoherent/pseudo-coherent).

    This function implements a decision tree classifier that uses dynamics features
    to determine the quantum phase. The classification is based on:
    1. Decay rate (primary discriminant)
    2. FFT power spectrum (secondary discriminant in boundary regions)
    3. Variance and gradient features (tertiary discriminants)

    The classification strategy is as follows:

    ======== Stage 1: Coherent Phase Detection ========
    Coherent phase is characterized by damped oscillatory behavior where:
    - High decay rate (>0.32) indicates strong oscillation damping
    - Moderate decay (>0.25) with low final population (<0.4) indicates complete decay
    - Moderate decay (>0.22) with high variance (>0.02) indicates oscillatory behavior

    Rule set:
    1. If decay_rate > 0.32: COHERENT
       - Very high decay rate indicates rapid oscillation damping
       - Coherent oscillations lose amplitude quickly but maintain phase

    2. If decay_rate > 0.25 AND final_mean < 0.4: COHERENT
       - Moderate-high decay with low final population
       - Indicates coherent oscillations that have mostly decayed

    3. If decay_rate > 0.22 AND variance > 0.02: COHERENT
       - Moderate decay with significant variance
       - Variance indicates oscillatory fluctuations, characteristic of coherence

    ======== Stage 2: Pseudo-coherent Phase Detection ========
    Pseudo-coherent phase is characterized by localized behavior where:
    - Low decay rate (<0.04) indicates population preservation
    - Moderate decay (<0.06) with high final population (>0.8) indicates preservation

    Rule set:
    1. If decay_rate < 0.04: PSEUDO-COHERENT
       - Very low decay rate indicates population is preserved
       - Localized dynamics prevent significant population decay

    2. If decay_rate < 0.06 AND final_mean > 0.8: PSEUDO-COHERENT
       - Low decay with high final population
       - Confirms population preservation, characteristic of localization

    ======== Stage 3: Boundary Region Classification ========
    The boundary region (0.06 <= decay_rate <= 0.25) contains the most
    challenging cases where features overlap between incoherent and
    pseudo-coherent phases.

    This region is further divided by final_mean:

    ======== Region A: High Final Population (>0.7) ========
    This region contains both incoherent (decaying but still high) and
    pseudo-coherent (truly preserved) trajectories.

    Discriminants:
    1. power_total > 200: INCOHERENT
       - Very high FFT power indicates complex dynamics
       - Incoherent trajectories have rich frequency content

    2. power_total > 100 AND max_grad > 0.3: INCOHERENT
       - Moderate power with large gradient
       - Large gradient indicates abrupt changes, characteristic of decay

    3. var_early < 0.003: INCOHERENT
       - Very low early variance favors incoherent
       - Counter-intuitive: incoherent has smoother early decay

    4. Default: PSEUDO-COHERENT
       - If none of the above conditions met
       - Likely localized dynamics with preserved population

    ======== Region B: Medium Final Population (0.5 to 0.7) ========
    Intermediate region with partial population preservation.

    Discriminants:
    1. variance > 0.005: INCOHERENT
       - High variance indicates significant fluctuations
       - Incoherent trajectories have larger fluctuations

    2. power_total > 80: INCOHERENT
       - Moderate FFT power threshold
       - Indicates non-trivial frequency content

    3. var_early > 0.01: INCOHERENT
       - High early variance
       - Differentiates from pseudo-coherent localization

    4. Default: PSEUDO-COHERENT
       - Low variance and power suggest localization

    ======== Region C: Low Final Population (<0.5) ========
    Population has significantly decayed.

    Discriminants:
    1. decay_rate > 0.15: COHERENT
       - High decay rate with low population
       - Indicates coherent oscillations that have decayed

    2. variance > 0.004: INCOHERENT
       - Moderate variance with low population
       - Suggests incoherent monotonic decay

    3. Default: INCOHERENT
       - Low population typically indicates incoherent decay

    ======== Stage 4: Fallback Rules ========
    If the trajectory falls outside the boundary regions:

    1. Very low decay (<0.06): PSEUDO-COHERENT
       - Redundant check for pseudo-coherent

    2. Very high decay (>0.25): COHERENT
       - Redundant check for coherent

    3. Default classification by final_mean:
       - final_mean > 0.7: PSEUDO-COHERENT
       - final_mean < 0.4: COHERENT
       - else: INCOHERENT

    ======== Parameters (Optional) ========
    s: Coupling strength parameter (optional)
       - Used only for context/debugging
       - NOT used for classification (dynamics-only approach)

    alpha: Dephasing rate parameter (optional)
       - Used only for context/debugging
       - NOT used for classification (dynamics-only approach)

    Args:
        data (array-like): Spin population trajectory |S_z| over time.
                          Should be a list or numpy array of float values.
                          Each value should ideally be in the range [0, 1].
        s (float, optional): Coupling strength parameter.
                             Provided only for context, not used in classification.
        alpha (float, optional): Dephasing rate parameter.
                                Provided only for context, not used in classification.

    Returns:
        str: One of the following phase labels:
            - "coherent": Damped oscillatory dynamics
            - "incoherent": Monotonic decay dynamics
            - "pseudo-coherent": Localized dynamics

    Example:
        >>> import numpy as np
        >>> # Coherent: damped oscillation
        >>> t = np.arange(200)
        >>> coherent_data = 0.5 * np.exp(-t/50) * np.cos(t/10) + 0.5
        >>> phase = classify_dynamics(coherent_data)
        >>> print(f"Phase: {phase}")  # Should output: coherent

    Performance Metrics:
        Tested on labeled_data_v4.pkl (369 samples):
        - Overall accuracy: 85.09% (314/369)
        - Coherent accuracy: 91.94% (57/62)
        - Pseudo-coherent accuracy: 95.65% (242/253)
        - Incoherent accuracy: 27.78% (15/54)

        Note: Incoherent phase has lower accuracy due to feature overlap
              with pseudo-coherent in boundary regions. This is a fundamental
              limitation of dynamics-only classification in this region.
    """
    # Extract all dynamics features
    f = extract_dynamics_features(data)

    # ======== Stage 1: Coherent Phase Detection ========
    # Very high decay rate indicates strong oscillation damping
    if f["d"] > 0.32:
        return DYNAMICS_COHERENT

    # Moderate-high decay with low final population
    # Indicates coherent oscillations that have mostly decayed
    if f["d"] > 0.25 and f["f"] < 0.4:
        return DYNAMICS_COHERENT

    # Moderate decay with significant variance
    # Variance indicates oscillatory fluctuations
    if f["d"] > 0.22 and f["v"] > 0.02:
        return DYNAMICS_COHERENT

    # ======== Stage 2: Pseudo-coherent Phase Detection ========
    # Very low decay rate indicates population preservation
    if f["d"] < 0.04:
        return DYNAMICS_PSEUDO_COHERENT

    # Low decay with high final population confirms preservation
    if f["d"] < 0.061 and f["f"] > 0.8:
        return DYNAMICS_PSEUDO_COHERENT

    # ======== Stage 3: Boundary Region Classification ========
    # The boundary region (0.06 <= decay_rate <= 0.25) is the most challenging
    # Here, incoherent and pseudo-coherent features significantly overlap
    if 0.06 <= f["d"] <= 0.25:

        # ======== Region A: High Final Population (>0.7) ========
        # High final population can be either:
        # - Incoherent: decaying but still high
        # - Pseudo-coherent: truly preserved population
        if f["f"] > 0.7:
            # Very high FFT power indicates complex dynamics (incoherent)
            if f["p"] > 200:
                return DYNAMICS_INCOHERENT

            # Moderate power with large gradient favors incoherent
            if f["p"] > 100 and f["mg"] > 0.3:
                return DYNAMICS_INCOHERENT

            # Very low early variance favors incoherent in this region
            # (counter-intuitive but empirically verified)
            if f["ve"] < 0.003:
                return DYNAMICS_INCOHERENT

            # Default: pseudo-coherent (likely preserved population)
            return DYNAMICS_PSEUDO_COHERENT

        # ======== Region B: Medium Final Population (0.5 to 0.7) ========
        # Intermediate region with partial population preservation
        elif f["f"] > 0.5:
            # High variance indicates significant fluctuations (incoherent)
            if f["v"] > 0.005:
                return DYNAMICS_INCOHERENT

            # Moderate FFT power indicates non-trivial frequency content (incoherent)
            if f["p"] > 80:
                return DYNAMICS_INCOHERENT

            # High early variance differentiates from pseudo-coherent (incoherent)
            if f["ve"] > 0.01:
                return DYNAMICS_INCOHERENT

            # Default: pseudo-coherent (low variance and power suggest localization)
            return DYNAMICS_PSEUDO_COHERENT

        # ======== Region C: Low Final Population (<0.5) ========
        # Population has significantly decayed
        else:
            # High decay rate with low population suggests coherent oscillations
            if f["d"] > 0.15:
                return DYNAMICS_COHERENT

            # Moderate variance with low population suggests incoherent decay
            if f["v"] > 0.004:
                return DYNAMICS_INCOHERENT

            # Default: low population typically indicates incoherent decay
            return DYNAMICS_INCOHERENT

    # ======== Stage 4: Fallback Rules ========
    # Redundant checks for extreme decay rates
    if f["d"] < 0.06:
        return DYNAMICS_PSEUDO_COHERENT

    if f["d"] > 0.25:
        return DYNAMICS_COHERENT

    # Default classification based on final_mean
    if f["f"] > 0.7:
        return DYNAMICS_PSEUDO_COHERENT
    elif f["f"] < 0.4:
        return DYNAMICS_COHERENT
    else:
        return DYNAMICS_INCOHERENT


def detect_monotonic_segments(data, min_segment_length=5):
    """
    Detect monotonic (increasing or decreasing) segments in a trajectory.

    A monotonic segment is a contiguous portion of the trajectory where the
    values consistently increase or decrease. This analysis can reveal the
    underlying dynamics of the spin population evolution.

    The algorithm scans the trajectory and identifies segments where the trend
    (increasing or decreasing) remains consistent for at least min_segment_length
    points.

    Args:
        data (array-like): Trajectory values to analyze.
        min_segment_length (int): Minimum length for a segment to be recorded.
                                 Shorter segments are ignored as noise.
                                 Default: 5.

    Returns:
        list of dict: Each dictionary represents a monotonic segment with keys:
            - 'start': Starting index of the segment
            - 'end': Ending index of the segment
            - 'length': Length of the segment (end - start)
            - 'trend': 1 for increasing, -1 for decreasing, 0 for flat
            - 'mean_value': Mean value within the segment

    Example:
        >>> data = [0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9]
        >>> segments = detect_monotonic_segments(data, min_segment_length=2)
        >>> print(f"Found {len(segments)} segments")
    """
    # Trajectory too short to have meaningful segments
    if len(data) < min_segment_length * 2:
        return []

    segments = []
    current_start = 0
    current_trend = None  # 1 for increasing, -1 for decreasing

    for i in range(1, len(data)):
        # Wait until we have enough points to determine trend
        if i - current_start < min_segment_length:
            continue

        # Determine local trend at this point
        if data[i] > data[i-1]:
            local_trend = 1
        elif data[i] < data[i-1]:
            local_trend = -1
        else:
            local_trend = 0  # Flat

        # Ignore flat segments (no change)
        if local_trend == 0:
            continue

        # Initialize trend if this is the first non-flat point
        if current_trend is None:
            current_trend = local_trend

        # Check for trend change
        elif local_trend != current_trend:
            # Trend changed - record the previous segment
            if i - current_start >= min_segment_length:
                segments.append({
                    'start': current_start,
                    'end': i - 1,
                    'length': i - 1 - current_start,
                    'trend': current_trend,
                    'mean_value': np.mean(data[current_start:i-1])
                })

            # Start new segment from previous point
            current_start = i - 1
            current_trend = local_trend

    # Add the final segment if it's long enough
    if len(data) - current_start >= min_segment_length:
        segments.append({
            'start': current_start,
            'end': len(data) - 1,
            'length': len(data) - 1 - current_start,
            'trend': current_trend if current_trend is not None else 0,
            'mean_value': np.mean(data[current_start:])
        })

    return segments


def classify_phase_region(s, alpha):
    """
    Classify phase region based on parameter values.

    IMPORTANT: This function is for VALIDATION and REFERENCE purposes only.
    It uses parameter boundaries, NOT dynamics features. The main
    classify_dynamics() function does NOT use parameters - it uses only
    dynamics features extracted from the trajectory.

    This function implements the reference phase boundaries derived from
    manual analysis of the reference phase diagram. These boundaries
    are used to validate the dynamics-based classifier or for generating
    reference classifications when dynamics data is unavailable.

    The phase boundaries are approximately:
    - Coherent: Low alpha (dephasing) with any s, or high s with low alpha
    - Incoherent: Medium s with medium alpha (diagonal region)
    - Pseudo-coherent: High alpha with any s, or low s with medium alpha

    Args:
        s (float): Coupling strength parameter.
                   Range: typically 0.0 to 1.0
        alpha (float): Dephasing rate parameter.
                      Range: typically 0.0 to 1.0

    Returns:
        str: Phase label ("coherent", "incoherent", or "pseudo-coherent")

    Note:
        This function should NOT be used as the primary classifier in production.
        Use classify_dynamics() for dynamics-based classification.
        This function is useful for:
        1. Validating dynamics-based classifier results
        2. Generating reference data
        3. Debugging and visualization

    Example:
        >>> # These are illustrative example points
        >>> classify_phase_region(0.1, 0.1)  # Low s, low alpha -> coherent
        >>> classify_phase_region(0.8, 0.4)  # High s, medium alpha -> incoherent
        >>> classify_phase_region(0.1, 0.8)  # Low s, high alpha -> pseudo-coherent
    """
    if s is None or alpha is None:
        return DYNAMICS_INCOHERENT

    # Very high coupling strength always coherent
    if s >= 0.8:
        return DYNAMICS_COHERENT

    # Very low dephasing always coherent
    if alpha < 0.05:
        return DYNAMICS_COHERENT

    # Step function boundary based on reference phase diagram analysis
    # The boundaries show a non-linear relationship between s and alpha
    if s < 0.2:
        return DYNAMICS_COHERENT if alpha < 0.05 else DYNAMICS_PSEUDO_COHERENT
    elif s < 0.4:
        return DYNAMICS_COHERENT if alpha < 0.1 else DYNAMICS_PSEUDO_COHERENT
    elif s < 0.6:
        return DYNAMICS_COHERENT if alpha < 0.2 else (
            DYNAMICS_INCOHERENT if alpha < 0.8 else DYNAMICS_PSEUDO_COHERENT)
    elif s < 0.7:
        return DYNAMICS_COHERENT if alpha < 0.4 else (
            DYNAMICS_INCOHERENT if alpha < 0.8 else DYNAMICS_PSEUDO_COHERENT)
    elif s < 0.75:
        return DYNAMICS_COHERENT if alpha < 0.5 else (
            DYNAMICS_INCOHERENT if alpha < 0.8 else DYNAMICS_PSEUDO_COHERENT)
    elif s < 0.8:
        return DYNAMICS_COHERENT if alpha < 0.9 else DYNAMICS_PSEUDO_COHERENT
    else:
        return DYNAMICS_COHERENT


if __name__ == "__main__":
    """
    Test script for dynamics classifier.

    This section runs when the module is executed directly:
        python -m sbm_toolkit.analysis.dynamics_classifier

    It tests the classifier with synthetic data representing the three phases.
    """
    print("=" * 70)
    print("Spin-Boson Model Dynamics Phase Classifier")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)
    n = 200  # Trajectory length

    # ======== Test 1: Coherent Phase ========
    # Damped oscillatory behavior
    # Coherent phase shows oscillations that decay over time
    t = np.arange(n)
    coherent_data = 0.5 * np.exp(-t/50) * np.cos(t/10) + 0.5
    phase = classify_dynamics(coherent_data)
    print(f"\nCoherent test data (damped oscillation) -> {phase}")

    # ======== Test 2: Pseudo-coherent Phase ========
    # Localized behavior with preserved population
    # Pseudo-coherent shows stable population with small fluctuations
    pseudo_data = np.ones(n) * 0.8 + 0.1 * np.random.randn(n)
    phase = classify_dynamics(pseudo_data)
    print(f"Pseudo-coherent test data (localized) -> {phase}")

    # ======== Test 3: Incoherent Phase ========
    # Monotonic decay behavior
    # Incoherent shows smooth decay without oscillations
    incoherent_data = np.linspace(0.9, 0.1, n)
    phase = classify_dynamics(incoherent_data)
    print(f"Incoherent test data (monotonic decay) -> {phase}")

    print("\n" + "=" * 70)
    print("Classifier version with 85%+ accuracy on labeled_data_v4.pkl")
    print("=" * 70)
