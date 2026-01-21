"""File I/O utilities"""

import pickle
import numpy as np
from pathlib import Path
from typing import Any, Union


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """
    Save data to pickle file.

    Args:
        data: Data to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_array(data: np.ndarray, filepath: Union[str, Path], compressed: bool = True) -> None:
    """
    Save numpy array to file.

    Args:
        data: Numpy array to save
        filepath: Path to save file
        compressed: Whether to use compression (npz format)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if compressed:
        np.savez_compressed(filepath, data=data)
    else:
        np.save(filepath, data)


def load_array(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load numpy array from file.

    Args:
        filepath: Path to array file

    Returns:
        Loaded array
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npz':
        return np.load(filepath)['data']
    else:
        return np.load(filepath)


def save_xvg(data: np.ndarray, filepath: Union[str, Path], header: str = "") -> None:
    """
    Save data in GROMACS XVG format.

    Args:
        data: Data to save (N x M array, first column is x-axis)
        filepath: Path to save file
        header: Optional header text
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        for row in data:
            f.write("  ".join(map(str, row)) + "\n")
