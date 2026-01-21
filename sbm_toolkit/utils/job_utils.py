"""Job naming and parsing utilities"""

import re
from typing import Tuple


def generate_job_name(s: float, alpha: float, Omega: int = 1, omega_c: int = 10,
                     nmodes: int = 1000, bond_dims: int = 20,
                     td_method: int = 0, rho_type: int = 0) -> str:
    """
    Generate standardized job name.

    Args:
        s: Spectral exponent
        alpha: Coupling strength
        Omega: Tunneling splitting
        omega_c: Cutoff frequency
        nmodes: Number of modes
        bond_dims: Bond dimension
        td_method: Time evolution method
        rho_type: Discretization type

    Returns:
        Formatted job name string
    """
    return (f"traj_s{s:.2f}_alpha{alpha:.2f}_Omega{Omega}_omega_c{omega_c}_"
            f"nmodes{nmodes}_bond_dims{bond_dims}_td_method_{td_method}_rho_type_{rho_type}")


def parse_job_name(job_name: str) -> Tuple[float, float, int, int, int, int, int, int]:
    """
    Parse job name to extract parameters.

    Args:
        job_name: Job name string

    Returns:
        Tuple of (s, alpha, Omega, omega_c, nmodes, bond_dims, td_method, rho_type)
    """
    # Extract s
    s_match = re.search(r's(\d+\.\d+)', job_name)
    s = float(s_match.group(1)) if s_match else 0.0

    # Extract alpha
    alpha_match = re.search(r'alpha(\d+\.\d+)', job_name)
    alpha = float(alpha_match.group(1)) if alpha_match else 0.0

    # Extract Omega
    Omega_match = re.search(r'Omega(\d+)', job_name)
    Omega = int(Omega_match.group(1)) if Omega_match else 1

    # Extract omega_c
    omega_c_match = re.search(r'omega_c(\d+)', job_name)
    omega_c = int(omega_c_match.group(1)) if omega_c_match else 10

    # Extract nmodes
    nmodes_match = re.search(r'nmodes(\d+)', job_name)
    nmodes = int(nmodes_match.group(1)) if nmodes_match else 1000

    # Extract bond_dims
    bond_dims_match = re.search(r'bond_dims(\d+)', job_name)
    bond_dims = int(bond_dims_match.group(1)) if bond_dims_match else 20

    # Extract td_method
    td_method_match = re.search(r'td_method_(\d+)', job_name)
    td_method = int(td_method_match.group(1)) if td_method_match else 0

    # Extract rho_type
    rho_type_match = re.search(r'rho_type_(\d+)', job_name)
    rho_type = int(rho_type_match.group(1)) if rho_type_match else 0

    return s, alpha, Omega, omega_c, nmodes, bond_dims, td_method, rho_type
