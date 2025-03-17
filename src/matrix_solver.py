import time

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as spla

from src.config import NUM_MODES
from src.grid_discretization import (
    initialize_grid_vector,
    initialize_tridiagonal_matrix,
)


def _get_frequency(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Get the frequencies of the eigenvalues based on:
    lambda = sqrt(-k)

    Params
    -------
    - eigenvalues (np.ndarray): eigenvalues of the matrix

    Returns
    --------
    - frequencies (np.ndarray): frequencies of the eigenvalues
    """
    frequencies = np.sqrt(-eigenvalues.real)
    return frequencies


def solve_eigenvalues(
    matrix: np.ndarray | sp.sparse._csr.csr_matrix,
    model: str = "None",
    num_modes: int = NUM_MODES,
    side: str = "SM",
) -> tuple:
    """
    Solve the eigenvalues of a matrix using the specified model

    Params
    -------
    - matrix (np.ndarray | sp.sparse._csr.csr_matrix): matrix to solve the eigenvalues for
    - model (str): model to use for solving the eigenvalues. Choose from 'None' or 'h'. Default is 'None'
    - num_modes (int): number of modes to solve. Default is NUM_MODES
    - side (str): side of the eigenvalues to solve. Choose from
        - LM: Largest (in magnitude) eigenvalues.
        - SM: Smallest (in magnitude) eigenvalues.
        - LA: Largest (algebraic) eigenvalues.
        - SA: Smallest (algebraic) eigenvalues.
        - BE: Half (k/2) from each end of the spectrum.

    Returns
    --------
    - frequencies (np.ndarray): frequencies of the eigenvalues
    - eigenvectors (np.ndarray): eigenvectors of the eigen
    """
    start_time = time.time()

    assert side in ["LM", "SM", "LA", "SA", "BE"], (
        "Invalid side. Choose from LM, SM, LA, SA, BE"
    )

    if model == "None":
        eigenvalues, eigenvectors = (
            la.eig(matrix)
            if isinstance(matrix, np.ndarray)
            else spla.eigs(matrix, k=num_modes, which=side)
        )
    elif model == "h":
        eigenvalues, eigenvectors = (
            la.eigh(matrix)
            if isinstance(matrix, np.ndarray)
            else spla.eigsh(matrix, k=num_modes, which=side)
        )
    else:
        raise ValueError("Invalid model. Choose from 'None' or 'h'")

    time_output = (
        f"Time taken to solve the eigenvalues: {time.time() - start_time:.2f} seconds"
    )
    if not isinstance(matrix, np.ndarray):
        time_output += " using sparse solver"
    print(time_output)

    frequencies = _get_frequency(eigenvalues)
    return (frequencies, eigenvectors)


def get_frequencies(
    N: int,
    dx: float,
    L_list: np.ndarray,
    shape: str,
) -> np.ndarray:
    """
    Get the frequencies of the eigenvalues for a list of L values

    Params
    -------
    - N (int): size of the grid
    - dx (float): step size of the grid
    - L_list (np.ndarray): list of L values
    - shape (str): shape of the grid. Choose from 'square', 'rectangle', 'circle'

    Returns
    --------
    - frequencies_list (np.ndarray): frequencies of the eigenvalues for each L value
    """
    assert shape in ["square", "rectangle", "circle"], (
        "Invalid shape. Choose from 'square', 'rectangle', 'circle"
    )

    assert L_list.shape[0] == NUM_MODES, "Size of L_list should be equal to NUM_MODES"
    frequencies_list = np.zeros((len(L_list), NUM_MODES))
    assert frequencies_list.shape[0] == L_list.shape[0], (
        "Size of frequencies_list should be equal to L_list"
    )

    for i, L in enumerate(L_list):
        v = initialize_grid_vector(N, L, shape=shape)
        M = initialize_tridiagonal_matrix(v, sparse=True)
        M = M * dx**2

        frequencies, _ = solve_eigenvalues(M, model="h", num_modes=NUM_MODES)
        frequencies_list[i] = frequencies[:NUM_MODES]

    assert frequencies_list != 0
    return frequencies_list
