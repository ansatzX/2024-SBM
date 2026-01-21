"""Time evolution plotting utilities"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


def plot_spin_population(time_data: np.ndarray, population_data: np.ndarray,
                        figsize: Tuple[int, int] = (10, 6),
                        title: str = "Spin Population Dynamics",
                        xlabel: str = "Time (a.u.)",
                        ylabel: str = r"$\langle\sigma_z\rangle$",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot spin population vs time.

    Args:
        time_data: Time values
        population_data: Population values
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time_data, population_data, '-', linewidth=2)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_time_series(time_data: np.ndarray, data_dict: dict,
                    figsize: Tuple[int, int] = (12, 6),
                    title: str = "Time Series",
                    xlabel: str = "Time (a.u.)",
                    ylabel: str = "Value",
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot multiple time series on the same plot.

    Args:
        time_data: Time values
        data_dict: Dictionary of {label: data_array}
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, data in data_dict.items():
        ax.plot(time_data, data, '-', label=label, linewidth=2)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_entropy_evolution(time_data: np.ndarray, entropy_data: np.ndarray,
                          figsize: Tuple[int, int] = (10, 6),
                          title: str = "Entropy Evolution",
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot entropy evolution over time.

    Args:
        time_data: Time values
        entropy_data: Entropy values
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time_data, entropy_data, '-', linewidth=2, color='darkblue')
    ax.set_xlabel("Time (a.u.)", fontsize=14)
    ax.set_ylabel(r"$S$", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
