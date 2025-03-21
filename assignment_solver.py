import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix

FILL_VALUE = 1


def create_grid(L):
    grid = np.zeros((L, L))
    center = (L // 2, L // 2)
    radius = L / 2

    for i in range(L):
        for j in range(L):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if distance <= radius:
                grid[i, j] = FILL_VALUE
    return grid


# @njit
def create_matrix(L, vector):
    grid = create_grid(L)
    matrix = np.zeros((L * L, L * L))
    np.fill_diagonal(matrix, -4)

    for x in range(L):
        for y in range(L):
            if grid[x, y] == 1:
                idx = x * L + y

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < L and 0 <= ny < L and grid[nx, ny] == 1:
                        neighbor_idx = nx * L + ny
                        matrix[idx, neighbor_idx] = 1
    return matrix


def solve_eigenvalues(matrix, num_eigenvalues):
    matrix /= 0.01**2
    eigenvalues, eigenvectors = spla.eigs(matrix, k=50)

    idx = np.argsort(np.abs(eigenvalues))[:num_eigenvalues]
    eigenfrequencies = np.sqrt(-eigenvalues)
    return eigenfrequencies[idx], eigenvectors[:, idx]


def plot_eigenmodus(eigenvectors, num_modes, L):
    for i in range(num_modes):
        v = eigenvectors[:, i]
        max_abs = np.max(np.abs(v))

        plt.imshow(v.real.reshape(L, L), cmap="RdBu", vmin=-max_abs, vmax=max_abs)
        plt.show()


def main():
    L = 40
    num_eigenvalues = 10
    vector = create_grid(L)
    matrix = create_matrix(L, vector)
    matrix = csr_matrix(matrix)

    _, eigenvectors = solve_eigenvalues(matrix, num_eigenvalues)
    plot_eigenmodus(eigenvectors, 2, L)


if __name__ == "__main__":
    main()
