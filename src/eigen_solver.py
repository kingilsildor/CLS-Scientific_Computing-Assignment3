import time

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as spla

from src.config import NUM_EIGENVALUES


def _calc_frequency(eigenvalues: np.ndarray) -> np.ndarray:
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
    frequencies = np.sqrt(-eigenvalues)
    return frequencies


def solve_eigenvalues(
    matrix: np.ndarray | sp.sparse._csr.csr_matrix,
    model: str = "None",
    num_eigen: int = NUM_EIGENVALUES,
) -> tuple:
    """
    Solve the eigenvalues of a matrix using the specified model

    Params
    -------
    - matrix (np.ndarray | sp.sparse._csr.csr_matrix): matrix to solve the eigenvalues for
    - model (str): model to use for solving the eigenvalues. Choose from 'None' or 'h'. Default is 'None'
    - num_modes (int): number of modes to solve. Default is NUM_MODES

    Returns
    --------
    - frequencies (np.ndarray): frequencies of the eigenvalues
    - eigenvectors (np.ndarray): eigenvectors of the eigen
    """
    start_time = time.time()

    # Model uses the SM (smallest magnitude) eigenvalues
    if model == "None":
        print("Solving the eigenvalues using regular solver for matrix")
        eigenvalues, eigenvectors = (
            la.eig(matrix)
            if isinstance(matrix, np.ndarray)
            else spla.eigs(matrix, k=num_eigen, which="SM")
        )
    elif model == "h":
        print("Solving the eigenvalue using hermitian solver")
        eigenvalues, eigenvectors = (
            la.eigh(matrix)
            if isinstance(matrix, np.ndarray)
            else spla.eigsh(matrix, k=6, which="SM")
        )
    else:
        raise ValueError("Invalid model. Choose from 'None' or 'h'")

    time_output = (
        f"Time taken to solve the eigenvalues: {time.time() - start_time:.2f} seconds"
    )
    if not isinstance(matrix, np.ndarray):
        time_output += " using sparse solver"
    N = int(np.sqrt(matrix.shape[0]))
    time_output += f" with matrix of size {int(matrix.shape[0])}, N={N}"
    print(time_output)

    idx = np.argsort(np.abs(eigenvalues.real))
    eigenfrequencies = _calc_frequency(eigenvalues)
    return (eigenfrequencies[idx], eigenvectors[:, idx])


def time_dependent_solution(
    c: float, frequency: float, t: float, A: int = 1, B: int = 1
) -> np.ndarray:
    """
    Calculate the time-dependent solution of the wave equation

    Params
    -------
    - c (float): speed of sound
    - frequency (float): frequency of the eigenmode
    - t (float): time
    - A (int): amplitude of the cosine. Default is 1
    - B (int): amplitude of the sine. Default is 1

    Returns
    --------
    - T (np.ndarray): time-dependent solution of the wave equation
    """
    return A * np.cos(c * frequency * t) + B * np.sin(c * frequency * t)
