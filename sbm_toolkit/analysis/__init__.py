"""Data analysis components for SBM (procedural style)"""

# Data loading
from .data_loader import (
    list_jobs,
    load_expectations,
    load_entropy_1site,
    load_entropy_spin,
    load_mutual_info,
    load_omega_values,
    load_coupling_coefficients,
    load_time_series,
    get_rho_array,
)

# Signal processing
from .signal_processing import (
    wavelet_denoising,
    interpolate_signal,
    continuous_fourier_transform,
    fast_fourier_transform,
    find_peaks_and_analyze,
    zero_crossing_detection,
)

# Dynamics classification
from .dynamics_classifier import (
    DYNAMICS_COHERENT,
    DYNAMICS_INCOHERENT,
    DYNAMICS_PSEUDO_COHERENT,
    DYNAMICS_ONE_PEAK,
    DYNAMICS_ONE_VALLEY,
    DYNAMICS_OSCILLATION,
    DYNAMICS_ASCEND,
    DYNAMICS_DESCEND,
    DYNAMICS_UNKNOWN,
    classify_dynamics,
    detect_monotonic_segments,
    classify_phase_region,
)

# Feature extraction
from .feature_extraction import (
    extract_frequency,
    extract_amplitude,
    extract_period,
    find_extrema_timestamps,
    fit_exponential_decay,
    calculate_time_average,
)

__all__ = [
    # Data loading
    'list_jobs',
    'load_expectations',
    'load_entropy_1site',
    'load_entropy_spin',
    'load_mutual_info',
    'load_omega_values',
    'load_coupling_coefficients',
    'load_time_series',
    'get_rho_array',
    # Signal processing
    'wavelet_denoising',
    'interpolate_signal',
    'continuous_fourier_transform',
    'fast_fourier_transform',
    'find_peaks_and_analyze',
    'zero_crossing_detection',
    # Dynamics classification
    'DYNAMICS_COHERENT',
    'DYNAMICS_INCOHERENT',
    'DYNAMICS_PSEUDO_COHERENT',
    'DYNAMICS_ONE_PEAK',
    'DYNAMICS_ONE_VALLEY',
    'DYNAMICS_OSCILLATION',
    'DYNAMICS_ASCEND',
    'DYNAMICS_DESCEND',
    'DYNAMICS_UNKNOWN',
    'classify_dynamics',
    'detect_monotonic_segments',
    'classify_phase_region',
    # Feature extraction
    'extract_frequency',
    'extract_amplitude',
    'extract_period',
    'find_extrema_timestamps',
    'fit_exponential_decay',
    'calculate_time_average',
]
