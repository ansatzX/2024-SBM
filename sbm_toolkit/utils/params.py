"""Parameter management for SBM simulations (procedural style)"""

from typing import Tuple


def translate_param(s: float, alpha: float, omega_c: float, Omega: int = 1) -> Tuple[float, float, float]:
    """
    Translate parameters from paper convention to renormalizer convention.

    Based on: Phys. Rev. Lett. 129, 120406 (2022)

    Args:
        s: Spectral exponent
        alpha: Coupling strength
        omega_c: Cutoff frequency
        Omega: Tunneling splitting

    Returns:
        Tuple of (s_reno, alpha_reno, omega_c_reno)
    """
    s_reno = s
    alpha_reno = 4 * alpha  # Translation from Wang1 to PRL convention
    omega_c_reno = omega_c * Omega

    return s_reno, alpha_reno, omega_c_reno


# def get_reno_s(s: float) -> float:
#     """
#     Get renormalized s parameter.

#     Args:
#         s: Spectral exponent

#     Returns:
#         Renormalized s
#     """
#     return s


# def get_reno_alpha(alpha: float) -> float:
#     """
#     Get renormalized alpha parameter.

#     Args:
#         alpha: Coupling strength

#     Returns:
#         Renormalized alpha (4*alpha for Wang1 scheme)
#     """
#     return 4 * alpha


# def get_reno_omega_c(omega_c: float, Omega: int) -> float:
#     """
#     Get renormalized omega_c parameter.

#     Args:
#         omega_c: Cutoff frequency
#         Omega: Tunneling splitting

#     Returns:
#         Renormalized omega_c
#     """
#     return omega_c * Omega


# def get_delta(Omega: int) -> float:
#     """
#     Get tunneling amplitude.

#     Args:
#         Omega: Tunneling splitting energy

#     Returns:
#         Tunneling amplitude (Omega/2)
#     """
#     return Omega * 0.5


# def validate_params(s: float, alpha: float, nmodes: int, bond_dims: int) -> None:
#     """
#     Validate simulation parameters.

#     Args:
#         s: Spectral exponent
#         alpha: Coupling strength
#         nmodes: Number of modes
#         bond_dims: Bond dimension

#     Raises:
#         ValueError: If any parameter is invalid
#     """
#     if s <= 0:
#         raise ValueError(f"s must be positive, got {s}")
#     if alpha <= 0:
#         raise ValueError(f"alpha must be positive, got {alpha}")
#     if nmodes <= 0:
#         raise ValueError(f"nmodes must be positive, got {nmodes}")
#     if bond_dims <= 0:
#         raise ValueError(f"bond_dims must be positive, got {bond_dims}")
