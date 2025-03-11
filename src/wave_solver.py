# Solves the wave equation eigenvalue problem
from src.grid_discretization import (
    initialize_grid_vector,
    initialize_tridiagonal_matrix,
)

N = 20
M = initialize_tridiagonal_matrix(N)
v = initialize_grid_vector(N=N, L=5, shape="square")
print(M, v)
