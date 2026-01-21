"""Data loading utilities for SBM simulation results (procedural style)"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from ..utils.io import load_pickle


def list_jobs(data_folder: Union[str, Path]) -> List[str]:
    """
    List all job folders in data directory.

    Args:
        data_folder: Path to folder containing simulation results

    Returns:
        List of job folder names
    """
    data_folder = Path(data_folder)
    return sorted([
        d.name for d in data_folder.iterdir()
        if d.is_dir() and d.name.startswith('traj_')
    ])


def load_expectations(data_folder: Union[str, Path], job_name: str) -> np.ndarray:
    """
    Load expectation values (typically sigma_z).

    Args:
        data_folder: Path to folder containing simulation results
        job_name: Job folder name

    Returns:
        Array of expectation values
    """
    data_folder = Path(data_folder)
    filepath = data_folder / job_name / 'expectations.pickle'
    return np.array(load_pickle(filepath))


def load_entropy_1site(data_folder: Union[str, Path], job_name: str, step: int) -> Dict:
    """
    Load single-site entropies for a specific step.

    Args:
        data_folder: Path to folder containing simulation results
        job_name: Job folder name
        step: Time step number

    Returns:
        Dictionary of {dof_name: entropy_value}
    """
    data_folder = Path(data_folder)
    filepath = data_folder / job_name / f'{step:04d}_step_entropy_1site.pickle'
    return load_pickle(filepath)


def load_entropy_spin(data_folder: Union[str, Path], job_name: str, step: int) -> float:
    """
    Load spin entropy for a specific step.

    Args:
        data_folder: Path to folder containing simulation results
        job_name: Job folder name
        step: Time step number

    Returns:
        Spin entropy value
    """
    data_folder = Path(data_folder)
    filepath = data_folder / job_name / f'{step:04d}_step_entropy_spin.pickle'
    entropy_dict = load_pickle(filepath)
    return entropy_dict['spin']


def load_mutual_info(data_folder: Union[str, Path], job_name: str, step: int) -> Dict:
    """
    Load mutual information for a specific step.

    Args:
        data_folder: Path to folder containing simulation results
        job_name: Job folder name
        step: Time step number

    Returns:
        Dictionary of mutual information values
    """
    data_folder = Path(data_folder)
    filepath = data_folder / job_name / f'{step:04d}_step_mutual_infos.pickle'
    return load_pickle(filepath)


def load_omega_values(data_folder: Union[str, Path], job_name: str) -> np.ndarray:
    """
    Load bosonic mode frequencies.

    Args:
        data_folder: Path to folder containing simulation results
        job_name: Job folder name

    Returns:
        Array of frequency values

    Raises:
        FileNotFoundError: If omega file doesn't exist
    """
    data_folder = Path(data_folder)
    filepath = data_folder / job_name / 'sdf_wang1_omega.pickle'

    if filepath.exists():
        return load_pickle(filepath)
    else:
        raise FileNotFoundError(f"Omega file not found: {filepath}")


def load_coupling_coefficients(data_folder: Union[str, Path], job_name: str) -> np.ndarray:
    """
    Load coupling coefficients.

    Args:
        data_folder: Path to folder containing simulation results
        job_name: Job folder name

    Returns:
        Array of coupling coefficients
    """
    data_folder = Path(data_folder)
    filepath = data_folder / job_name / 'sdf_wang1_c.pickle'
    return load_pickle(filepath)


def load_time_series(data_folder: Union[str, Path], job_name: str, nsteps: int,
                     data_type: str = 'entropy_1site') -> List[Dict]:
    """
    Load time series of data.

    Args:
        data_folder: Path to folder containing simulation results
        job_name: Job folder name
        nsteps: Number of time steps
        data_type: Type of data ('entropy_1site', 'mutual_info', etc.)

    Returns:
        List of data dictionaries for each time step
    """
    data_series = []

    for step in range(nsteps):
        if data_type == 'entropy_1site':
            data = load_entropy_1site(data_folder, job_name, step)
        elif data_type == 'mutual_info':
            data = load_mutual_info(data_folder, job_name, step)
        elif data_type == 'entropy_spin':
            data = load_entropy_spin(data_folder, job_name, step)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        data_series.append(data)

    return data_series


