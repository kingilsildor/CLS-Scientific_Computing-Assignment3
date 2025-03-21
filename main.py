import numpy as np

from src.config import *
from src.eigen_solver import solve_eigenvalues
from src.grid_discretization import (
    initialize_grid_vector,
    initialize_tridiagonal_matrix,
)

PRINT = 4


def create_grid(L, shape, dx=0.01):
    v = initialize_grid_vector(L, shape=shape)
    m = initialize_tridiagonal_matrix(v, L, sparse=True)
    m = m * dx**2
    return m


def eigenvalues(m):
    frequencies, eigenvectors = solve_eigenvalues(m, num_eigen=5)
    return frequencies[:PRINT], eigenvectors[:, :PRINT]


def main():
    t_list = np.linspace(0, 100, 200)

    shape = "rectangle"
    L = 60
    m = create_grid(L, shape)
    frequencies, eigenvectors = eigenvalues(m)

    # eigenmode = eigenvectors[:, -1].reshape(L, L).real
    # frequency = frequencies[-1]

    # print(frequency, eigenmode)
    # plot_eigenmode_animation(1.0, eigenmode, frequency, t_list, shape)


if __name__ == "__main__":
    main()
