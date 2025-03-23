import numpy as np
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
def _fill_neighbours(rows: int, cols: int, mask: np.ndarray) -> np.ndarray:
    """
    Fill the neighbors of a given matrix A only if the neighbor cell is part of the shape.

    Params
    -------
    - rows (int): Number of rows in the grid
    - cols (int): Number of columns in the grid
    - mask (np.ndarray): Flattened vector indicating valid cells (1 for valid, 0 for invalid)

    Returns
    --------
    - matrix (np.ndarray): Matrix with neighbors filled based on the mask
    """
    matrix = np.zeros((rows * cols, rows * cols))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for x in range(rows):
        for y in range(cols):
            index = x * cols + y

            if mask[index] == 0:
                continue

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if 0 <= nx < rows and 0 <= ny < cols:
                    neighbor_index = nx * cols + ny

                    if mask[neighbor_index] == 1:
                        matrix[index, neighbor_index] = OFF_DIAGONAL_VALUE

    return matrix


def initialize_tridiagonal_matrix(
    vector: np.ndarray, L: int, sparse: bool = True
) -> np.ndarray | csr_matrix:
    """
    Initialize a tridiagonal matrix from a given vector with a mask for valid cells.

    Params
    -------
    - vector (np.ndarray): Vector to initialize the tridiagonal matrix from
    - L (int): Size of the grid
    - sparse (bool): Whether to return a sparse matrix. Default is True

    Returns
    --------
    - stencil_matrix (np.ndarray | csr_matrix): Tridiagonal matrix of the 5-point stencil
    """
    N = vector.shape[0]
    rows, cols = L, int(N / L)

    mask = vector.flatten()

    stencil_matrix = _fill_neighbours(rows, cols, mask)
    np.fill_diagonal(stencil_matrix, DIAGONAL_VALUE)

    if sparse:
        stencil_matrix = csr_matrix(stencil_matrix)

    # Multiply by the spatial step size
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
