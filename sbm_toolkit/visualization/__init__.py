"""Visualization tools for SBM analysis"""

import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import numpy as np

# Dynamics type constants
DYNAMICS_COHERENT = "coherent"
DYNAMICS_INCOHERENT = "incoherent"
DYNAMICS_PSEUDO_COHERENT = "pseudo-coherent"

# Phase colors for consistent visualization
PHASE_COLORS = {
    DYNAMICS_COHERENT: "#1f77b4",
    DYNAMICS_INCOHERENT: "#2ca02c",
    DYNAMICS_PSEUDO_COHERENT: "#ff7f0e",
}


def plot_phase_diagram(classification_results: Dict[Tuple[float, float], str],
                      figsize: Tuple[int, int] = (10, 8),
                      title: str = "SBM Phase Diagram",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot phase diagram from classification results.

    Args:
        classification_results: Dictionary of {(s, alpha): dynamics_type}
        figsize: Figure size (width, height) in inches
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Separate points by dynamics type
    coherent_points = []
    incoherent_points = []
    pseudo_coherent_points = []

    for (s, alpha), dynamics_type in classification_results.items():
        if dynamics_type == DYNAMICS_COHERENT:
            coherent_points.append((s, alpha))
        elif dynamics_type == DYNAMICS_INCOHERENT:
            incoherent_points.append((s, alpha))
        else:
            pseudo_coherent_points.append((s, alpha))

    # Plot each phase with distinct markers and colors
    if pseudo_coherent_points:
        s_vals = [p[0] for p in pseudo_coherent_points]
        alpha_vals = [p[1] for p in pseudo_coherent_points]
        ax.scatter(s_vals, alpha_vals, marker='s', s=80, alpha=0.7,
                   label='Pseudo-coherent', color=PHASE_COLORS[DYNAMICS_PSEUDO_COHERENT],
                   edgecolors='black', linewidth=0.5)

    if coherent_points:
        s_vals = [p[0] for p in coherent_points]
        alpha_vals = [p[1] for p in coherent_points]
        ax.scatter(s_vals, alpha_vals, marker='^', s=80, alpha=0.7,
                   label='Coherent', color=PHASE_COLORS[DYNAMICS_COHERENT],
                   edgecolors='black', linewidth=0.5)

    if incoherent_points:
        s_vals = [p[0] for p in incoherent_points]
        alpha_vals = [p[1] for p in incoherent_points]
        ax.scatter(s_vals, alpha_vals, marker='o', s=80, alpha=0.7,
                   label='Incoherent', color=PHASE_COLORS[DYNAMICS_INCOHERENT],
                   edgecolors='black', linewidth=0.5)

    # Configure axes
    ax.set_xlabel(r'$s$ (tunneling)', fontsize=14)
    ax.set_ylabel(r'$\alpha$ (coupling)', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_spin_population(time_data: np.ndarray, population_data: np.ndarray,
                        figsize: Tuple[int, int] = (10, 6),
                        title: str = "Spin Population Dynamics",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot spin population vs time.

    Args:
        time_data: Time values
        population_data: Population values |S_z|
        figsize: Figure size (width, height) in inches
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time_data, population_data, '-', linewidth=1.5, color='#1f77b4')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(r"$\langle S_z \rangle$", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


__all__ = [
    # Phase diagram
    'plot_phase_diagram',
    # Time evolution
    'plot_spin_population',
    # Constants
    'DYNAMICS_COHERENT',
    'DYNAMICS_INCOHERENT',
    'DYNAMICS_PSEUDO_COHERENT',
    'PHASE_COLORS',
]
