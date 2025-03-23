import matplotlib.pyplot as plt

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
    return m


def eigenvalues(m):
    frequencies, eigenvectors = solve_eigenvalues(m, num_eigen=6)
    return frequencies[:PRINT], eigenvectors[:, :PRINT]


def main():
    shape = "circle"
    m = create_grid(5, shape)
    plt.imshow(m.toarray())
    plt.show()


if __name__ == "__main__":
    main()
