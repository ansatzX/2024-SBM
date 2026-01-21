"""Spectrum analysis plotting utilities"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_spectrum_analysis(freq_data: np.ndarray, amplitude_data: np.ndarray,
                          figsize: Tuple[int, int] = (10, 6),
                          title: str = "Frequency Spectrum",
                          xlabel: str = "Frequency",
                          ylabel: str = "Amplitude",
                          xlim: Optional[Tuple[float, float]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot frequency spectrum.

    Args:
        freq_data: Frequency values
        amplitude_data: Amplitude values
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        xlim: X-axis limits
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(freq_data, amplitude_data, '-', linewidth=2)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)

    if xlim:
        ax.set_xlim(xlim)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_omega_vs_feature(omega_data: np.ndarray, feature_data: np.ndarray,
                         feature_name: str = "Feature",
                         figsize: Tuple[int, int] = (10, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot feature vs bosonic mode frequency.

    Args:
        omega_data: Mode frequencies
        feature_data: Feature values (e.g., entropies, amplitudes)
        feature_name: Name of the feature
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(omega_data, feature_data, alpha=0.6, s=50)
    ax.set_xlabel(r"$\omega$ (bosonic frequency)", fontsize=14)
    ax.set_ylabel(feature_name, fontsize=14)
    ax.set_title(f"{feature_name} vs Mode Frequency", fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
