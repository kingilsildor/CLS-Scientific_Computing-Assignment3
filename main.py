import numpy as np

from script.create_plot import plot_eigenmode_animation, plot_multiple_eigenmodes
from src.config import *
from src.eigen_solver import solve_eigenvalues
from src.grid_discretization import (
    initialize_grid_vector,
    initialize_tridiagonal_matrix,
)

import matplotlib.pyplot as plt

PRINT = 4


def create_grid(L, shape, dx=0.01):
    v = initialize_grid_vector(L, shape=shape)
    m = initialize_tridiagonal_matrix(v, L, sparse=True)
    h = 1/L
    m /= h**2
    return m


def eigenvalues(m):
    frequencies, eigenvectors = solve_eigenvalues(m, num_eigen=6)
    return frequencies[:PRINT], eigenvectors[:, :PRINT]


def main():
    t_list = np.linspace(0, 10, 100)

    shape = "square"
    L = 50
    m = create_grid(L, shape)
    np.savetxt('matrix.txt', m.toarray())
    frequencies, eigenvectors = eigenvalues(m)

    EIGENMODE_VALUE = 2
    eigenmode = eigenvectors[:, EIGENMODE_VALUE].reshape(L, L)
    frequency = frequencies[EIGENMODE_VALUE]

    # plot_multiple_eigenmodes(4, frequencies, eigenvectors, L, shape, save_img=True)

    plot_eigenmode_animation(1.0, eigenmode, frequency, t_list, shape)


if __name__ == "__main__":
    main()
