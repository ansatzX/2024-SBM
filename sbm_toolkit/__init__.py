"""
SBM Toolkit - Spin-Boson Model Simulation and Analysis Toolkit

A comprehensive package for simulating and analyzing spin-boson model dynamics
using tree tensor network methods.

Procedural style - no coupling, low dependency.
"""

__version__ = "0.1.0"
__author__ = "Cunxi Gong"

from . import analysis
from . import visualization
from . import utils

__all__ = ['analysis', 'visualization', 'utils']
