import numpy as np

from script.create_plot import plot_eigenmodus
from src.config import NUM_MODES
from src.eigen_solver import solve_eigenvalues
from src.grid_discretization import (
    initialize_grid_vector,
    initialize_tridiagonal_matrix,
)


def get_frequencies_list(L_list: np.ndarray, shape: str) -> np.ndarray:
    """
    Get the frequencies list for the given L_list

    Params
    -------
    - L_list (np.ndarray): list of L values
    - shape (str): shape of the grid

    Returns
    --------
    - frequencies_list (np.ndarray): list of frequencies
    """
    N = len(L_list)
    frequencies_list = np.zeros((N, NUM_MODES), dtype=float)

    for i, L in enumerate(L_list):
        v = initialize_grid_vector(L, shape)
        M = initialize_tridiagonal_matrix(v, L)
        frequencies, eigenvectors = solve_eigenvalues(M, num_eigen=NUM_MODES)

        frequencies_list[i] = frequencies.real
        eigenmode = (
            eigenvectors[:, 0].reshape(L, L if shape != "rectangle" else 2 * L).real
        )
        plot_eigenmodus(L, eigenmode, shape, save_img=True)
    return frequencies_list
