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
def _fill_neighbours(M, N) -> np.ndarray:
    """
    Fill the neighbors of a given matrix A

    Params
    -------
    - new_A (np.ndarray): matrix to fill the neighbors for
    - A (np.ndarray): matrix to find the neighbors for

    Returns
    --------
    - new_A (np.ndarray): matrix with neighbors filled
    """
    matrix = np.zeros((M * N, M * N))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for x in range(M):
        for y in range(N):
            index = x * N + y

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if 0 <= nx < M and 0 <= ny < N:
                    neighbor_index = nx * N + ny
                    matrix[index, neighbor_index] = OFF_DIAGONAL_VALUE
    return matrix


# TODO fix circle


def initialize_tridiagonal_matrix(
    vector: np.ndarray, L: int, sparse: bool = True
) -> np.ndarray | sp.sparse._csr.csr_matrix:
    """
    Initialize a tridiagonal matrix from a given vector

    Params
    -------
    - vector (np.ndarray): vector to initialize the tridiagonal matrix from
    - L (int): size of the grid
    - sparse (bool): whether to return a sparse matrix. Default is True

    Returns
    --------
    - stencil_matrix (np.ndarray | sp.sparse._csr.csr_matrix): tridiagonal matrix of the 5-point stencil
    """
    N = vector.shape[0]
    rows, cols = L, int(N / L)

    stencil_matrix = _fill_neighbours(rows, cols)
    np.fill_diagonal(stencil_matrix, DIAGONAL_VALUE)

    if sparse:
        stencil_matrix = csr_matrix(stencil_matrix)

    # Multiply by the spacial step size
    h = 1 / L
    stencil_matrix /= h**2
    return stencil_matrix


def initialize_grid_vector(L: int, shape: str = "square") -> np.ndarray:
    """
    Initialize a grid of size L x L with a given value in a specific shape

    Params
    -------
    - L (int): size of the shape
    - shape (str): shape of the grid. Default is 'square'. Choose from 'square', 'rectangle', 'circle'

    Returns
    --------
    - vector (np.ndarray): grid of size N x N as a vector of size N^2 x 1
    """

    if shape == "square":
        matrix = np.ones((L, L))
    elif shape == "rectangle":
        matrix = np.ones((L, 2 * L))
    elif shape == "circle":
        matrix = np.zeros((L, L))

        y, x = np.ogrid[:L, :L]
        center = (L - 1) / 2
        radius = L / 2
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
        matrix[mask] = 1
    else:
        raise ValueError("Invalid shape. Choose from 'square', 'rectangle', 'circle'")

    vector = matrix.flatten().reshape(-1, 1)
    return vector


if __name__ == "__main__":
    N = 100
    matrix = initialize_tridiagonal_matrix(N)
    vector = initialize_grid_vector(N, 20, 1.0, "circle")
