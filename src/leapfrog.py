import numpy as np


def leapfrog(
    T: int,
    F_x,
    k: float,
    delta_t: float = 0.01,
    m: float = 1,
    initial_v=None,
    F_t=None,
):
    """
    Run the Leapfrog method for a given time period T, force equation F_x, spring constant k and time step delta_t

    Params
    -------
    - T (int): the time period to run the method for
    - F_x (function): the force equation
    - k (float): the spring constant
    - delta_t (float): the time step size
    - m (float): the mass of the object
    - initial_v (str): the initial velocity, can be either none or "zero" if we want to set v_1/2 = 0. Default is None
    - F_t (function): the external force. Default is None

    Returns
    --------
    - x_list (np.ndarray): the list of positions over time
    - v_list (np.ndarray): the list of velocities over time at half steps
    """
    # Get number of timesteps and initialise arrays
    num_time_steps = int(T / delta_t)

    x_list = np.zeros(num_time_steps + 1)
    v_list = np.zeros(num_time_steps + 1)

    # Initial conditions
    x_list[0] = 1
    if initial_v == "zero":
        v_list[0] = 0
    else:
        v_list[0] = delta_t * F_x(x_list[0], k) / (2 * m)

    # If external force is not provided, set it to zero
    if F_t is None:

        def F_t(t):
            return 0

    # Run the method
    for i in range(num_time_steps):
        x_list[i + 1] = x_list[i] + delta_t * v_list[i]
        v_list[i + 1] = (
            v_list[i] + delta_t * (F_x(x_list[i + 1], k) + F_t((i + 1) * delta_t)) / m
        )

    return x_list, v_list


def exact_position(t: float, k: float, m: float):
    """
    Calculate the exact position of the object at time t given the spring constant k and mass m
    """
    A = 1
    omega = np.sqrt(k / m)
    phi = np.pi / 2

    return A * np.sin(omega * t + phi)


def exact_velocity(t: float, k: float, m: float):
    """
    Calculate the exact velocity of the object at time t given the spring constant k and mass m
    """
    A = 1
    omega = np.sqrt(k / m)
    phi = np.pi / 2

    return A * omega * np.cos(omega * t + phi)
