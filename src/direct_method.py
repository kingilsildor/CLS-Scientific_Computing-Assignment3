import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from src.grid_discretization import initialize_tridiagonal_matrix, initialize_grid_vector

def direct_method(L: int, L_domain: float = 4.0, source_pos: tuple = (0.6, 1.2)):
    """
    Create a direct solver for a matrix in a discretized grid of a circle

    Params
    -------
    - L (int): size of the grid
    - L_domain (float): diameter of the circle. Default is 4.0
    - source_pos (tuple): position of the source. Default is (0.6, 1.2)

    Returns
    --------
    - c_grid (np.ndarray): grid of size L x L as a vector of size L^2 x 1
    - masked_c_grid (np.ma.masked_array): masked grid of size L x L
    """

    # Initialise the domain (the circle mask)
    domain_vector = initialize_grid_vector(L, shape="circle")
    domain_mask = domain_vector.reshape(L, L)
    circle_mask = np.abs(domain_mask) > 1e-12

    # Construct stencil matrix
    M = initialize_tridiagonal_matrix(domain_vector, L, sparse=True).tolil()

    # Apply boundary conditions
    for i in range(L * L):
        if domain_vector[i] == 0:
            M[i, :] = 0 
            M[i, i] = 1 
    
    M = M.tocsr()
    
    # create right-hand side vector'
    
    b = np.zeros(L * L, dtype=float)

    # Convert source position to grid index
    source_x, source_y = source_pos
    h = L_domain / L
    i = int(L / 2 + (source_y * (L / L_domain)))  # Row index
    j = int(L / 2 + (source_x * (L / L_domain)))  # Column index
    source_index = i * L + j
    
    if domain_vector[source_index] != 0:
        b[source_index] = 1.0
        print(f"b[{source_index}] successfully set to {b[source_index]}")
    else:
        raise ValueError("Source position is not in the domain")

    # Solve system using direct method

    c = spla.spsolve(M, -1*b)
    c_grid = c.reshape(L, L)
    #create mask for plotting

    masked_c_grid = np.ma.masked_where(~circle_mask, c_grid)

    return c_grid, masked_c_grid

def plot_concentration(L: int, source_pos: tuple = (0.6, 1.2)):
    """
    Plot the concentration grid

    Params
    -------
    - L (int): size of the grid
    - c_grid (np.ndarray): grid of size L x L as a vector of size L^2 x 1
    - masked_c_grid (np.ma.masked_array): masked grid of size L x L
    """

    c_grid, masked_c_grid = direct_method(L, source_pos=source_pos)

    plt.figure(figsize=(6, 5))
    plt.imshow(masked_c_grid, extent=(-2, 2, -2, 2), origin="lower", cmap="coolwarm")
    plt.colorbar(label="Concentration c(x, y)")
    plt.title("Steady-State Diffusion in Circular Domain")
    plt.show()

