import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    FIG_DPI,
    FIG_LABEL_SIZE,
    FIG_LEGEND_SIZE,
    FIG_SIZE,
    FIG_TICK_SIZE,
    FIG_TITLE_SIZE,
)


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


def plot_leapfrog_various_k(
    positions: np.ndarray,
    velocities: np.ndarray,
    k_values: list,
    T: int,
    delta_t: float = 0.01,
    save: bool = False,
):
    """
    Plot the position and velocity over time for various spring constants k

    Params
    -------
    - positions (np.ndarray): the list of positions over time for each k
    - velocities (np.ndarray): the list of velocities over time for each k
    - k_values (list): the list of spring constants
    - T (int): the time period the method was run for
    - delta_t (float): the time step size. Default is 0.01
    - save (bool): whether to save the plot
    """
    t_position = np.linspace(0, T, len(positions[0]))
    t_velocity = np.linspace(delta_t / 2, T + delta_t / 2, len(positions[0]))
    colors = ["r", "g", "y"]

    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
    fig.suptitle("Position and Velocity over Time", fontsize=FIG_TITLE_SIZE)

    for i, k in enumerate(k_values):
        axes[0].plot(t_position, positions[i], label=f"k = {k}", color=colors[i])
    axes[0].set_ylabel("Position", fontsize=FIG_LABEL_SIZE)
    axes[0].tick_params(axis="y", labelsize=FIG_TICK_SIZE)
    axes[0].legend(fontsize=FIG_LEGEND_SIZE)
    axes[0].grid()

    for i, k in enumerate(k_values):
        axes[1].plot(t_velocity, velocities[i], label=f"k = {k}", color=colors[i])

    axes[1].set_xlabel("Time", fontsize=FIG_LABEL_SIZE)
    axes[1].set_ylabel("Velocity", fontsize=FIG_LABEL_SIZE)
    axes[1].tick_params(axis="both", labelsize=FIG_TICK_SIZE)
    axes[1].legend(fontsize=FIG_LEGEND_SIZE)
    axes[1].grid()

    plt.tight_layout()
    if save:
        plt.savefig(
            "results/leapfrog_position_velocity.png", dpi=FIG_DPI, bbox_inches="tight"
        )
    plt.show()


def plot_leapfrog_errors(
    positions: np.ndarray,
    velocities: np.ndarray,
    k: float,
    T: int,
    delta_t: float = 0.01,
    m: float = 1,
    save: bool = False,
):
    """
    Plot the error in position and velocity over time for the Leapfrog method for two different initial velocities

    Params
    -------
    - positions (np.ndarray): the list of positions over time for different initial velocities
    - velocities (np.ndarray): the list of velocities over time for different initial velocities
    - k (int): the spring constant
    - T (int): the time period the method was run for
    - delta_t (float): the time step size. Default is 0.01
    - m (float): the mass of the object. Default is 1
    - save (bool): whether to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
    fig.suptitle("Error in Position and Velocity over Time", fontsize=FIG_TITLE_SIZE)

    # Plot position errors
    t_position = np.linspace(0, T, len(positions[0]))

    axes[0].plot(
        t_position,
        exact_position(t_position, k, m) - positions[0],
        label=r"$v_{1/2} = \Delta t \cdot F(x_0) / 2m$",
        color="y",
    )
    axes[0].plot(
        t_position,
        exact_position(t_position, k, m) - positions[1],
        label=r"$v_{1/2} = 0$",
        color="g",
    )
    axes[0].set_ylabel("Error in Position", fontsize=FIG_LABEL_SIZE)
    axes[0].tick_params(axis="both", labelsize=FIG_TICK_SIZE)
    axes[0].legend(fontsize=FIG_LEGEND_SIZE)
    axes[0].grid()

    # Plot velocity errors
    t_velocity = np.linspace(delta_t / 2, T + delta_t / 2, len(positions[0]))

    axes[1].plot(
        t_velocity,
        exact_velocity(t_velocity, k, m) - velocities[0],
        label=r"$v_{1/2} = \Delta t \cdot F(x_0) / 2m$",
        color="y",
    )
    axes[1].plot(
        t_velocity,
        exact_velocity(t_velocity, k, m) - velocities[1],
        label=r"$v_{1/2} = 0$",
        color="g",
    )
    axes[1].set_xlabel("Time", fontsize=FIG_LABEL_SIZE)
    axes[1].set_ylabel("Error in Velocity", fontsize=FIG_LABEL_SIZE)
    axes[1].tick_params(axis="both", labelsize=FIG_TICK_SIZE)
    axes[1].legend(fontsize=FIG_LEGEND_SIZE)
    axes[1].grid()

    plt.tight_layout()
    if save:
        plt.savefig("results/leapfrog_errors.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_leapfrog_errors_start_end(
    positions: np.ndarray,
    k: float,
    T: int,
    delta_t: float = 0.01,
    m: float = 1,
    save: bool = False,
):
    """
    Plot the position over time for the Leapfrog method for two different initial velocities for the first and last 5 seconds

    Params
    -------
    - positions (np.ndarray): the list of positions over time for different initial velocities
    - k (int): the spring constant
    - T (int): the time period the method was run for
    - delta_t (float): the time step size. Default is 0.01
    - m (float): the mass of the object. Default is 1
    - save (bool): whether to save the plot
    """
    t_position = np.linspace(0, T, len(positions[0]))
    idx = int(5 / delta_t)

    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE)
    fig.suptitle("Position vs Time (First and Last 5 Seconds)", fontsize=FIG_TITLE_SIZE)

    # Plot first 5 seconds of position
    axes[0].plot(
        t_position[:idx],
        positions[0][:idx],
        label=r"Calculated ($v_{1/2}$ Average)",
        color="y",
    )
    axes[0].plot(
        t_position[:idx],
        positions[1][:idx],
        label=r"Calculated ($v_{1/2} = 0$)",
        color="g",
    )
    axes[0].plot(
        t_position[:idx],
        exact_position(t_position, k, m)[:idx],
        label="Exact",
        color="r",
    )
    axes[0].set_ylabel("Position", fontsize=FIG_LABEL_SIZE)
    axes[0].tick_params(axis="both", labelsize=FIG_TICK_SIZE)
    axes[0].legend(fontsize=FIG_LEGEND_SIZE)
    axes[0].grid()

    # Plot last 5 seconds of position
    axes[1].plot(
        t_position[-idx:],
        positions[0][-idx:],
        label=r"Calculated ($v_{1/2}$ Average)",
        color="y",
    )
    axes[1].plot(
        t_position[-idx:],
        positions[1][-idx:],
        label=r"Calculated ($v_{1/2} = 0$)",
        color="g",
    )
    axes[1].plot(
        t_position[-idx:],
        exact_position(t_position, k, m)[-idx:],
        label="Exact",
        color="r",
    )
    axes[1].set_xlabel("Time", fontsize=FIG_LABEL_SIZE)
    axes[1].set_ylabel("Position", fontsize=FIG_LABEL_SIZE)
    axes[1].tick_params(axis="both", labelsize=FIG_TICK_SIZE)
    axes[1].legend(fontsize=FIG_LEGEND_SIZE)
    axes[1].grid()

    plt.tight_layout()
    if save:
        plt.savefig("results/leapfrog_start_end.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_leapfrog_driving_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    omegas: list,
    T: int,
    delta_t: float = 0.01,
    k: float = 1,
    m: float = 1,
    save: bool = False,
):
    """
    Plot the position, velocity and phase space for the Leapfrog method with a driving force for three different driving frequencies

    Params
    -------
    - positions (np.ndarray): the list of positions over time for different driving frequencies
    - velocities (np.ndarray): the list of velocities over time for different driving frequencies
    - omegas (list): the list of driving frequencies
    - T (int): the time period the method was run for
    - delta_t (float): the time step size. Default is 0.01
    - k (float): the spring constant. Default is 1
    - m (float): the mass of the object. Default is 1
    - save (bool): whether to save the plot
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    colors = ["g", "y", "r"]
    omega = (k / m) ** 0.5

    t_position = np.linspace(0, T, len(positions[0]))
    t_velocity = np.linspace(delta_t / 2, T + delta_t / 2, len(positions[0]))

    # Plot the positions
    for i, position in enumerate(positions):
        axes[0][i].plot(
            t_position,
            exact_position(t_position, k, m),
            color=colors[1],
            label="Without driving force",
        )
        axes[0][i].plot(
            t_position, position, color=colors[0], label="With driving force"
        )
        axes[0][i].set_title(
            rf"Position vs Time $\omega_{{drive}} = {omegas[i] / omega:.1f}\omega$"
        )
        axes[0][i].set_xlabel("Time")
        axes[0][i].legend()
        axes[0][i].grid()

    # Plot the velocities
    for i, velocity in enumerate(velocities):
        axes[1][i].plot(
            t_velocity,
            exact_velocity(t_position, k, m),
            color=colors[1],
            label="Without driving force",
        )
        axes[1][i].plot(
            t_velocity, velocity, color=colors[0], label="With driving force"
        )
        axes[1][i].set_title(
            rf"Velocity vs Time $\omega_{{drive}} = {omegas[i] / omega:.1f}\omega$"
        )
        axes[1][i].set_xlabel("Time")
        axes[1][i].legend()
        axes[1][i].grid()

    # Plot the phase space
    for i, position in enumerate(positions):
        axes[2][i].plot(positions[i], velocities[i], color=colors[i])
        axes[2][i].set_title(
            rf"Phase Space $\omega_{{drive}} = {omegas[i] / omega:.1f}\omega$"
        )
        axes[2][i].set_xlabel("Position")
        axes[2][i].grid()

    axes[0][0].set_ylabel("Position")
    axes[1][0].set_ylabel("Velocity")
    axes[2][0].set_ylabel("Velocity")

    plt.tight_layout()
    if save:
        plt.savefig(
            "results/leapfrog_driving_force.png", dpi=FIG_DPI, bbox_inches="tight"
        )
    plt.show()


def plot_leapfrog_phase_plots(
    positions: np.ndarray,
    velocities: np.ndarray,
    omegas: list,
    k: float = 1,
    m: float = 1,
    save: bool = False,
):
    """
    Plot the phase diagrams for the Leapfrog method for different driving frequencies

    Params
    -------
    - positions (np.ndarray): the list of positions over time for different driving frequencies
    - velocities (np.ndarray): the list of velocities over time for different driving frequencies
    - omegas (list): the list of driving frequencies
    - k (float): the spring constant. Default is 1
    - m (float): the mass of the object. Default is 1
    - save (bool): whether to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(
        "Phase Diagrams for Different Driving Frequencies", fontsize=FIG_TITLE_SIZE
    )
    colors = ["g", "y", "r", "goldenrod"]
    omega = (k / m) ** 0.5

    for i, ax in enumerate(axes.flat):
        ax.plot(positions[i], velocities[i], color=colors[i])
        ax.set_xlabel("Position", fontsize=FIG_LABEL_SIZE)
        ax.set_ylabel("Velocity", fontsize=FIG_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=FIG_TICK_SIZE)
        ax.set_title(
            rf"$\omega_{{drive}} = {omegas[i] / omega:.1f}\omega$",
            fontsize=FIG_LABEL_SIZE,
        )
        ax.grid()

    plt.tight_layout()
    if save:
        plt.savefig(
            "results/leapfrog_phase_plots.png", dpi=FIG_DPI, bbox_inches="tight"
        )
    plt.show()
