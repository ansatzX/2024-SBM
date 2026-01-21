"""Feature extraction from time series data"""

import numpy as np
import scipy
from scipy import signal, optimize
from typing import Tuple, List, Dict
from .signal_processing import wavelet_denoising, interpolate_signal


def extract_frequency(time_data: np.ndarray, signal_data: np.ndarray,
                     dt: float = 0.1, interp_factor: int = 10) -> Tuple[float, List[float]]:
    """
    Extract dominant frequency from signal using peak detection.

    Args:
        time_data: Time values
        signal_data: Signal values
        dt: Time step
        interp_factor: Interpolation factor for smoother peak detection

    Returns:
        Tuple of (average_frequency, frequency_list)
    """
    # Interpolate and denoise
    n_points = len(signal_data) * interp_factor
    x_interp, y_interp = interpolate_signal(time_data, signal_data, n_points, kind='quadratic')
    y_denoised = wavelet_denoising(y_interp)

    # Find peaks (maxima)
    peak_indices = signal.argrelmax(y_denoised)[0]

    if len(peak_indices) < 2:
        return 0.0, []

    # Calculate periods between consecutive peaks
    periods = []
    for i in range(1, len(peak_indices)):
        period = x_interp[peak_indices[i]] - x_interp[peak_indices[i-1]]
        periods.append(period)

    if len(periods) == 0:
        return 0.0, []

    # Convert periods to frequencies
    frequencies = [1/p for p in periods]
    avg_frequency = 1 / np.mean(periods)

    return avg_frequency, frequencies


def extract_amplitude(time_data: np.ndarray, signal_data: np.ndarray,
                     dt: float = 0.1, interp_factor: int = 10) -> Tuple[float, List[float]]:
    """
    Extract amplitude from signal by measuring peak-to-trough distances.

    Args:
        time_data: Time values
        signal_data: Signal values
        dt: Time step
        interp_factor: Interpolation factor

    Returns:
        Tuple of (average_amplitude, amplitude_list)
    """
    # Interpolate and denoise
    n_points = len(signal_data) * interp_factor
    x_interp, y_interp = interpolate_signal(time_data, signal_data, n_points, kind='quadratic')
    y_denoised = wavelet_denoising(y_interp)

    # Find peaks and valleys
    max_indices = signal.argrelmax(y_denoised)[0]
    min_indices = signal.argrelmin(y_denoised)[0]

    if len(max_indices) == 0 or len(min_indices) == 0:
        return 0.0, []

    # Calculate amplitudes
    amplitudes = []
    for i in range(min(len(max_indices), len(min_indices))):
        max_val = y_denoised[max_indices[i]]
        min_val = y_denoised[min_indices[i]]
        amp = np.abs(max_val - min_val) / 2
        amplitudes.append(amp)

    if len(amplitudes) == 0:
        return 0.0, []

    avg_amplitude = np.mean(amplitudes)
    return avg_amplitude, amplitudes


def extract_period(time_data: np.ndarray, signal_data: np.ndarray,
                  dt: float = 0.1) -> Tuple[float, List[float]]:
    """
    Extract oscillation period from signal.

    Args:
        time_data: Time values
        signal_data: Signal values
        dt: Time step

    Returns:
        Tuple of (average_period, period_list)
    """
    avg_freq, freq_list = extract_frequency(time_data, signal_data, dt)

    if avg_freq == 0:
        return 0.0, []

    avg_period = 1 / avg_freq
    period_list = [1/f for f in freq_list] if len(freq_list) > 0 else []

    return avg_period, period_list


def find_extrema_timestamps(time_data: np.ndarray, signal_data: np.ndarray,
                           dt: float = 0.1, interp_factor: int = 10,
                           extrema_type: str = 'both') -> Dict[str, np.ndarray]:
    """
    Find timestamps of extrema (maxima, minima, or both).

    Args:
        time_data: Time values
        signal_data: Signal values
        dt: Time step
        interp_factor: Interpolation factor
        extrema_type: Type of extrema ('max', 'min', 'both')

    Returns:
        Dictionary with timestamps of extrema
    """
    # Interpolate and denoise
    n_points = len(signal_data) * interp_factor
    x_interp, y_interp = interpolate_signal(time_data, signal_data, n_points, kind='quadratic')
    y_denoised = wavelet_denoising(y_interp)

    result = {}

    if extrema_type in ['max', 'both']:
        max_indices = signal.argrelmax(y_denoised)[0]
        result['max_times'] = x_interp[max_indices]
        result['max_values'] = y_denoised[max_indices]

    if extrema_type in ['min', 'both']:
        min_indices = signal.argrelmin(y_denoised)[0]
        result['min_times'] = x_interp[min_indices]
        result['min_values'] = y_denoised[min_indices]

    if extrema_type == 'both':
        # Combine and sort
        all_indices = np.concatenate([max_indices, min_indices])
        all_times = x_interp[all_indices]
        all_values = y_denoised[all_indices]

        sorted_idx = np.argsort(all_times)
        result['all_times'] = all_times[sorted_idx]
        result['all_values'] = all_values[sorted_idx]

    return result


def fit_exponential_decay(time_data: np.ndarray, signal_data: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit exponential decay to envelope of oscillating signal.

    Fits: y = A * exp(-B * t^C)

    Args:
        time_data: Time values
        signal_data: Signal values

    Returns:
        Tuple of (A, B, C) parameters
    """
    def exp_decay(t, A, B, C):
        return A * np.exp(-B * (t**C))

    try:
        # Initial guess
        p0 = [signal_data[0], 0.1, 1.0]

        # Fit
        popt, _ = optimize.curve_fit(exp_decay, time_data, signal_data,
                                     p0=p0, maxfev=10000)
        return tuple(popt)

    except Exception:
        return 0.0, 0.0, 0.0


def calculate_time_average(signal_data: np.ndarray, start_idx: int = 0) -> float:
    """
    Calculate time-averaged value of signal.

    Args:
        signal_data: Signal values
        start_idx: Starting index for averaging (to skip transients)

    Returns:
        Time-averaged value
    """
    return np.mean(signal_data[start_idx:])
