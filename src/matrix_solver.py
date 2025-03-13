import time

import numpy as np
import scipy as sp
import scipy.linalg as la
import scipy.sparse.linalg as spla


def solve_eigenvalues(
    matrix: np.ndarray | sp.sparse._csr.csr_matrix,
    model: str = "None",
) -> tuple:
    """
    Solve the eigenvalues of a matrix using the specified model

    Params
    -------
    - matrix (np.ndarray | sp.sparse._csr.csr_matrix): matrix to solve the eigenvalues for
    - model (str): model to use for solving the eigenvalues. Choose from 'None' or 'h'. Default is 'None'

    Returns
    --------
    - eigenoutput (Tuple): tuple of eigenvalues and eigenvectors
    """
    start_time = time.time()
    eigenoutput = None
    if model == "None":
        eigenoutput = (
            la.eig(matrix) if isinstance(matrix, np.ndarray) else spla.eigs(matrix)
        )
    elif model == "h":
        eigenoutput = (
            la.eigh(matrix) if isinstance(matrix, np.ndarray) else spla.eigsh(matrix)
        )
    else:
        raise ValueError("Invalid model. Choose from 'None' or 'h'")

    time_output = (
        f"Time taken to solve the eigenvalues: {time.time() - start_time:.2f} seconds"
    )
    if not isinstance(matrix, np.ndarray):
        time_output += " using sparse solver"
    print(time_output)

    return eigenoutput
