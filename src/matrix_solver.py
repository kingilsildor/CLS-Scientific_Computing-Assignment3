import numpy as np
from scipy.sparse.linalg import eigs

from grid_discretization import initialize_grid_vector, initialize_tridiagonal_matrix

N = 10
v = initialize_grid_vector(N, 4, shape="square")
M = initialize_tridiagonal_matrix(N)
K = 1
delta_x = 0.01

M = (1 / K * 1 / delta_x**2) * M

num_eigenvalues = 5
eigenvalues, eigenvectors = eigs(M, k=num_eigenvalues, which="SM")

# Output results
print("Eigenvalues:\n", np.real(eigenvalues))
print(
    "Eigenvectors (each column corresponds to an eigenvalue):\n", np.real(eigenvectors)
)
