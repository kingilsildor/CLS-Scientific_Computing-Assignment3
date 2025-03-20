from script.create_plot import plot_multiple_eigenmode
from src.config import *
from src.eigen_solver import solve_eigenvalues
from src.grid_discretization import (
    initialize_grid_vector,
    initialize_tridiagonal_matrix,
)

PRINT = 4


def create_grid(L, shape, dx=0.01):
    v = initialize_grid_vector(L, L, shape=shape)
    m = initialize_tridiagonal_matrix(v, L, sparse=True)
    m = m * dx**2
    return m


def eigenvalues(m):
    frequencies, eigenvectors = solve_eigenvalues(m, num_eigen=5)
    return frequencies[:PRINT], eigenvectors[:, :PRINT]


def main():
    shape = "rectangle"
    L = 60
    m = create_grid(L, shape)
    frequencies, eigenvectors = eigenvalues(m)
    print(frequencies)
    plot_multiple_eigenmode(PRINT, frequencies, eigenvectors, L, shape=shape)


if __name__ == "__main__":
    main()
