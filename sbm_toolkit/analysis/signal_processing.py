"""Signal processing utilities"""

import numpy as np
import pywt
import scipy
from scipy import signal, integrate, interpolate, fft
from typing import Tuple, Optional


def wavelet_denoising(signal_data: np.ndarray, wavelet: str = 'db4',
                     level: int = 4, cutoff_index: int = 4) -> np.ndarray:
    """
    Denoise signal using wavelet decomposition.

    Args:
        signal_data: Input signal
        wavelet: Wavelet type (default: 'db4')
        level: Decomposition level
        cutoff_index: Index to cut high-frequency components

    Returns:
        Denoised signal
    """
    # Wavelet decomposition
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)

    # Zero out high-frequency components
    coeff_denoising = [
        coeffs[i] if i < cutoff_index else np.zeros_like(coeffs[i])
        for i in range(len(coeffs))
    ]

    # Reconstruct signal
    reconstructed_signal = pywt.waverec(coeff_denoising, wavelet)

    # Ensure same length as input
    if len(reconstructed_signal) != len(signal_data):
        reconstructed_signal = reconstructed_signal[:len(signal_data)]

    return reconstructed_signal


def interpolate_signal(x: np.ndarray, y: np.ndarray, n_points: int,
                      kind: str = 'cubic') -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate signal to higher resolution.

    Args:
        x: Input x values
        y: Input y values
        n_points: Number of interpolation points
        kind: Interpolation type ('linear', 'quadratic', 'cubic')

    Returns:
        Tuple of (x_interpolated, y_interpolated)
    """
    x_uniform = np.linspace(x.min(), x.max(), n_points)
    interpolator = interpolate.interp1d(x, y, kind=kind)
    y_uniform = interpolator(x_uniform)

    return x_uniform, y_uniform


def fast_fourier_transform(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Fast Fourier Transform on signal.

    Args:
        x: Time values
        y: Signal values

    Returns:
        Tuple of (frequencies, fft_amplitudes)
    """
    N = len(x)
    T = (x.max() - x.min()) / N  # Sample time

    yf = fft.fft(y)[:N//2]
    xf = fft.fftfreq(N, T)[:N//2]

    return xf, yf


def continuous_fourier_transform(x: np.ndarray, y: np.ndarray,
                                n_freqs: int = 1000,
                                freq_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform continuous Fourier transform on signal.

    Args:
        x: Time values
        y: Signal values
        n_freqs: Number of frequency points
        freq_range: Frequency range (min, max). If None, use (0, 5)

    Returns:
        Tuple of (frequencies, cft_amplitudes)
    """
    if freq_range is None:
        freq_range = (0, 5)

    xf = np.linspace(freq_range[0], freq_range[1], n_freqs)
    yf = []

    for omega in xf:
        integral = integrate.trapezoid(
            y * np.exp(-1j * 2 * np.pi * omega * x), x
        )
        yf.append(integral)

    return xf, np.array(yf)


def find_peaks_and_analyze(xf: np.ndarray, yf: np.ndarray, N: int,
                          plot: bool = False) -> Tuple[float, float, int]:
    """
    Find and analyze peaks in Fourier transform.

    Args:
        xf: Frequency values
        yf: Complex Fourier amplitudes
        N: Number of data points
        plot: Whether to plot results

    Returns:
        Tuple of (dominant_frequency, amplitude, error_code)
        error_code: 0=success, 1=exception, 2=no peaks
    """
    fft_amp = 2.0 / N * np.abs(yf)
    peak_indices, _ = signal.find_peaks(fft_amp)

    if len(peak_indices) == 0:
        return 0.0, 0.0, 2  # No peaks found

    try:
        amp = 2.0 / N * np.abs(yf)

        max_indices = signal.argrelmax(amp)[0]
        min_indices = signal.argrelmin(amp)[0]

        if len(max_indices) == 0 or len(min_indices) < 2:
            return 0.0, 0.0, 2

        peaks = []
        for i_peak in range(len(max_indices)):
            peak_index = max_indices[i_peak]

            # Find surrounding minima
            left_mins = min_indices[min_indices < peak_index]
            right_mins = min_indices[min_indices > peak_index]

            if len(left_mins) == 0 or len(right_mins) == 0:
                continue

            left_index = left_mins[-1]
            right_index = right_mins[0]

            # Calculate baseline and peak height
            baseline = (amp[left_index] + amp[right_index]) / 2
            value = amp[peak_index] - baseline
            peaks.append(value)

        if len(peaks) == 0:
            return 0.0, 0.0, 2

        # Find dominant peak
        sorted_peaks = sorted(peaks, reverse=True)
        peak_idx = peaks.index(sorted_peaks[0])
        index = max_indices[peak_idx]

        freq = xf[index]
        amplitude = peaks[peak_idx]

        if plot:
            import matplotlib.pyplot as plt
            plt.xlim(0.1, 5)
            plt.scatter(freq, amplitude, color='red')
            plt.annotate(text=f'{freq}_{amplitude:.02f}',
                        xy=(xf[index], amp[index]),
                        xytext=(xf[index], amp[index]))
            plt.plot(xf, amp)

        return freq, amplitude, 0

    except Exception as e:
        return 0.0, 0.0, 1  # Exception occurred


def zero_crossing_detection(signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect zero crossings in signal after denoising.

    Args:
        signal_data: Input signal

    Returns:
        Tuple of (crossing_positions, crossing_values)
    """
    clean_signal = wavelet_denoising(signal_data)

    min_indices = signal.argrelmin(clean_signal)[0]
    max_indices = signal.argrelmax(clean_signal)[0]

    zero_crossing_signal = []
    zero_crossing_pos = []

    for i in range(min([len(min_indices), len(max_indices)])):
        max_index = max_indices[i]
        min_index = min_indices[i]

        max_signal = signal_data[max_index]
        min_signal = signal_data[min_index]

        zero_crossing_value = (max_signal + min_signal) / 2
        zero_crossing_location = (max_index + min_index) / 2

        zero_crossing_pos.append(zero_crossing_location)
        zero_crossing_signal.append(zero_crossing_value)

    return np.array(zero_crossing_pos), np.array(zero_crossing_signal)
