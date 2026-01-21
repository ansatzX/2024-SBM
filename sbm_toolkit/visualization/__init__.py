"""Visualization tools for SBM analysis (procedural style)"""

# Phase diagram
from .phase_diagram import (
    plot_phase_diagram,
    plot_entropy_heatmap,
)

# Time evolution
from .time_evolution import (
    plot_time_series,
    plot_spin_population,
    plot_entropy_evolution,
)

# Spectrum
from .spectrum import (
    plot_spectrum_analysis,
    plot_omega_vs_feature,
)

__all__ = [
    # Phase diagram
    'plot_phase_diagram',
    'plot_entropy_heatmap',
    # Time evolution
    'plot_time_series',
    'plot_spin_population',
    'plot_entropy_evolution',
    # Spectrum
    'plot_spectrum_analysis',
    'plot_omega_vs_feature',
]
