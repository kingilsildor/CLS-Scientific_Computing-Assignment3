import numpy as np
import scipy as sp
from numba import njit
from scipy.sparse import csr_matrix

from src.config import DIAGONAL_VALUE, OFF_DIAGONAL_VALUE


def initialize_grid(N: int, value: int | float = 0.0) -> np.ndarray:
    """
    Initialize a grid of size N x N with a given value

    Params
    -------
    - N (int): size of the grid
    - value (int | float): value to fill the grid with. Default is 0.0

    Returns
    --------
    - grid (np.ndarray): grid of size N x N
    """
    if isinstance(N, int):
        value = float(value)
    if not isinstance(value, float):
        raise ValueError("Value should be an integer or a float")

    grid = np.full((N, N), fill_value=value, dtype=float)
    assert grid.shape == (N, N)
    return grid


@njit
def _fill_neighbours(A, N) -> np.ndarray:
    """
    Helper function to fill the neighbours of a given node in a grid

    Params
    -------
    - A (np.ndarray): grid of size n x n
    - n (int): size of the grid
    """
    for i in range(N * N):
        if (i + 1) % N != 0:
            A[i, i + 1] = OFF_DIAGONAL_VALUE
            A[i + 1, i] = OFF_DIAGONAL_VALUE
        if i + N < N * N:
            A[i, i + N] = OFF_DIAGONAL_VALUE
            A[i + N, i] = OFF_DIAGONAL_VALUE


def initialize_tridiagonal_matrix(N: int) -> sp.sparse._csr.csr_matrix:
    """
    Initialize a tridiagonal matrix of size N x N with given diagonal and off-diagonal values

    Params
    -------
    - N (int): size of the matrix

    Returns
    --------
    - matrix (sp.sparse._csr.csr_matrix): tridiagonal matrix of size N x N
    """
    if N % 2 != 0:
        N += 1

    matrix = np.zeros((N * N, N * N))
    np.fill_diagonal(matrix, DIAGONAL_VALUE)
    _fill_neighbours(matrix, N)

    assert matrix.shape == (N * N, N * N)

    sparse_matrix = csr_matrix(matrix)
    return sparse_matrix


def initialize_grid_vector(
    N: int, L: int, value: int | float = 1.0, shape: str = "square"
) -> np.ndarray:
    """
    Initialize a grid of size N x N with a given value in a specific shape

    Params
    -------
    - N (int): size of the grid
    - L (int): size of the shape
    - value (int | float): value to fill the grid with. Default is 1.0
    - shape (str): shape of the grid. Default is 'square'. Choose from 'square', 'rectangle', 'circle'

    Returns
    --------
    - vector (np.ndarray): grid of size N x N as a vector of size N^2 x 1
    """
    if N % 2 != 0:
        N += 1

    if isinstance(value, int):
        value = float(value)
    if not isinstance(value, float):
        raise ValueError("Value should be an integer or a float")

    matrix = np.zeros((N, N), dtype=float)

    if shape == "square":
        start = (N - L) // 2
        end = start + L
        matrix[start:end, start:end] = value
    elif shape == "rectangle":
        start_x = (N - L) // 2
        end_x = start_x + L
        start_y = (N - 2 * L) // 2
        end_y = start_y + 2 * L
        matrix[start_x:end_x, start_y:end_y] = value
    elif shape == "circle":
        x, y = np.ogrid[:N, :N]
        mask = (x - N // 2) ** 2 + (y - N // 2) ** 2 <= (L // 2) ** 2
        matrix[mask] = value
    else:
        raise ValueError("Invalid shape. Choose from 'square', 'rectangle', 'circle'")

    vector = matrix.flatten().reshape(-1, 1)
    assert vector.shape == (N * N, 1)
    return vector


if __name__ == "__main__":
    N = 100
    matrix = initialize_tridiagonal_matrix(N)
    vector = initialize_grid_vector(N, 20, 1.0, "circle")
    print(matrix.shape, vector.shape)
