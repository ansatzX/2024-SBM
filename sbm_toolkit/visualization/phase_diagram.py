"""Phase diagram plotting utilities (procedural style)"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# Dynamics type constants (same as in dynamics_classifier)
DYNAMICS_COHERENT = "coherent"
DYNAMICS_INCOHERENT = "incoherent"
DYNAMICS_PSEUDO_COHERENT = "pseudo-coherent"


def plot_phase_diagram(classification_results: Dict[Tuple[float, float], str],
                      figsize: Tuple[int, int] = (10, 8),
                      title: str = "SBM Phase Diagram",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot phase diagram from classification results.

    Args:
        classification_results: Dictionary of {(s, alpha): dynamics_type}
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure

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
        else:  # Pseudo-coherent
            pseudo_coherent_points.append((s, alpha))

    # Plot points
    if pseudo_coherent_points:
        s_vals = [p[0] for p in pseudo_coherent_points]
        alpha_vals = [p[1] for p in pseudo_coherent_points]
        ax.scatter(s_vals, alpha_vals, marker='x', s=50, label='pseudo-coherent', color='orange')

    if coherent_points:
        s_vals = [p[0] for p in coherent_points]
        alpha_vals = [p[1] for p in coherent_points]
        ax.scatter(s_vals, alpha_vals, marker='+', s=50, label='coherent', color='blue')

    if incoherent_points:
        s_vals = [p[0] for p in incoherent_points]
        alpha_vals = [p[1] for p in incoherent_points]
        ax.scatter(s_vals, alpha_vals, marker='o', s=50, label='incoherent', color='red')

    ax.set_xlabel(r'$s$', fontsize=14)
    ax.set_ylabel(r'$\alpha$', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_entropy_heatmap(entropy_data: Dict[Tuple[float, float], float],
                        figsize: Tuple[int, int] = (10, 8),
                        title: str = "Spin Entropy Heatmap",
                        save_path: Optional[str] = None,
                        cmap: str = 'viridis') -> plt.Figure:
    """
    Plot entropy values as a heatmap.

    Args:
        entropy_data: Dictionary of {(s, alpha): entropy_value}
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap name

    Returns:
        Matplotlib figure object
    """
    # Extract unique s and alpha values
    s_values = sorted(set(k[0] for k in entropy_data.keys()))
    alpha_values = sorted(set(k[1] for k in entropy_data.keys()))

    # Create mesh grid
    s_mesh, alpha_mesh = np.meshgrid(s_values, alpha_values)

    # Fill entropy values
    entropy_mesh = np.zeros_like(s_mesh)
    for i in range(len(alpha_values)):
        for j in range(len(s_values)):
            key = (s_mesh[i, j], alpha_mesh[i, j])
            entropy_mesh[i, j] = entropy_data.get(key, np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    contour = ax.contourf(s_mesh, alpha_mesh, entropy_mesh, 500, cmap=cmap)
    cbar = plt.colorbar(contour, ax=ax, format='%.2f')
    cbar.set_label(r'$S_{\mathrm{stable}}$', fontsize=14)

    ax.set_xlabel(r'$s$', fontsize=14)
    ax.set_ylabel(r'$\alpha$', fontsize=14)
    ax.set_title(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
