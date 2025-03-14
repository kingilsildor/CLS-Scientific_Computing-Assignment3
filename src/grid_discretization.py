import numpy as np
import scipy as sp
from numba import njit
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


@njit
def _fill_neighbours(new_A, A) -> np.ndarray:
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
    N = A.shape[0]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for x in range(N):
        for y in range(N):
            if A[x, y] == 1:
                current_index = x * N + y

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N and A[nx, ny] == 1:
                        neighbor_index = nx * N + ny
                        new_A[current_index, neighbor_index] = 1
    return new_A


def initialize_tridiagonal_matrix(
    vector: np.ndarray, sparse: bool = True
) -> np.ndarray | sp.sparse._csr.csr_matrix:
    """
    Initialize a tridiagonal matrix from a given vector

    Params
    -------
    - vector (np.ndarray): vector to initialize the tridiagonal matrix from
    - sparse (bool): whether to return a sparse matrix. Default is True

    Returns
    --------
    - matrix (np.ndarray | sp.sparse._csr.csr_matrix): tridiagonal matrix of the 5-point stencil
    """
    N = vector.shape[0]
    side = int(np.sqrt(N))
    matrix = vector.flatten().reshape(side, side)

    stencil_matrix = -4 * np.eye(N, dtype=int)
    stencil_matrix = _fill_neighbours(stencil_matrix, matrix)

    if sparse:
        stencil_matrix = csr_matrix(stencil_matrix)

    return stencil_matrix


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
    if isinstance(value, int):
        value = float(value)
    if not isinstance(value, float):
        raise ValueError("Value should be an integer or a float")

    matrix = np.zeros((N, N))

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
