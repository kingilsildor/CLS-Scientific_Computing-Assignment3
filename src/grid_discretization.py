import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix


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


def initialize_tridiagonal_matrix(
    N: int, diagonal_value: int | float = -4.0, off_diagonal_value: int | float = 1.0
) -> sp.sparse._csr.csr_matrix:
    """
    Initialize a tridiagonal matrix of size N x N with given diagonal and off-diagonal values

    Params
    -------
    - N (int): size of the matrix
    - diagonal_value (int | float): value to fill the diagonal with. Default is -4.0
    - off_diagonal_value (int | float): value to fill the off-diagonal with. Default is 1.0

    Returns
    --------
    - matrix (sp.sparse._csr.csr_matrix): tridiagonal matrix of size N x N
    """
    if N % 2 != 0:
        N += 1

    if isinstance(diagonal_value, int):
        diagonal_value = float(diagonal_value)
        off_diagonal_value = float(off_diagonal_value)
    if not isinstance(diagonal_value, float) or not isinstance(
        off_diagonal_value, float
    ):
        raise ValueError("Values should be integers or floats")

    matrix = np.zeros((N, N), dtype=float)
    np.fill_diagonal(matrix, diagonal_value)
    np.fill_diagonal(matrix[1:], off_diagonal_value)
    np.fill_diagonal(matrix[:, 1:], off_diagonal_value)
    assert matrix.shape == (N, N)

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
    N = 5
    matrix = initialize_tridiagonal_matrix(N, -4.0, 1.0)
    print(matrix.shape)

    vector = initialize_grid_vector(N, 2, shape="square")
    print(vector.shape)

    assert matrix.shape[0] ** 2 == vector.shape[0]
