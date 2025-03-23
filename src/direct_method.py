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
    
    # Correct scaling due do differing ways step size is handled
    h_actual = L_domain / L
    h_module = 1 / L
    scaling_correction = (h_module / h_actual)**2

    # Construct stencil matrix
    M = initialize_tridiagonal_matrix(domain_vector, L, sparse=True).tolil()

    # Correct scaling
    M *= scaling_correction 

    # Apply Dirichlet boundary conditions
    for i in range(L * L):
        if domain_vector[i] == 0:
            M[i, :] = 0 
            M[i, i] = 1 
    
    M = M.tocsr()
    
    # create right-hand side vector'
    
    b = np.zeros(L * L, dtype=float)

    # Convert source position to grid index
    
    source_x, source_y = source_pos

    i = int((source_y + L_domain / 2) * (L / L_domain))
    j = int((source_x + L_domain / 2) * (L / L_domain))
    source_index = i * L + j
    
    if domain_vector[source_index] != 0:
        b[source_index] = 1 / (h_actual**2)
        print(f"b[{source_index}] successfully set to {b[source_index]}")
    else:
        raise ValueError("Source position is not in the domain")

    # Solve system using direct method

    c = spla.spsolve(M, -b)
    c_grid = c.reshape(L, L)


    # Total concentration in the disk
    masked_sum = np.sum(c_grid[circle_mask]) * h_actual**2
    print("Total concentration in disk:", masked_sum)

    #create mask for plotting
    
    masked_c_grid = np.ma.masked_where(~circle_mask, c_grid)

    return c_grid, masked_c_grid

def compare_concentrations(L=150, source_positions=[(0.6, 1.2), (0.0, 0.0)], titles=None):
    """
    Plot and compare concentration distributions for different source positions.
    
    Params
    -------
    - L (int): Grid size
    - source_positions (list of tuples): List of (x, y) source positions
    - titles (list of str): Optional titles for each subplot
    """
    if titles is None:
        titles = [f"Source at {pos}" for pos in source_positions]

    fig, axes = plt.subplots(1, len(source_positions), figsize=(12, 5))

    for ax, pos, title in zip(axes, source_positions, titles):
        c_grid, masked_c_grid = direct_method(L, source_pos=pos)
        im = ax.imshow(
            masked_c_grid,
            extent=(-2, 2, -2, 2),
            origin="lower",
            cmap="coolwarm"
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="c(x, y)")

    plt.suptitle(f"Steady-State Diffusion Comparison (L = {L})", fontsize=14)
    plt.tight_layout()
    plt.show()