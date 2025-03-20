import matplotlib.pyplot as plt
import numpy as np

from src.config import FIG_DPI, FIG_SIZE


def leapfrog(T, F_x, k, delta_t=0.01, m=1, initial_v=None, F_t=None):
    # Get number of timesteps and initials arrays
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


def plot_leapfrog_various_k(positions, velocities, k_values, T, save=False):
    t = np.linspace(0, T, len(positions[0]))
    colors = ["r", "g", "y"]

    # Positions
    plt.figure(figsize=FIG_SIZE)

    for i, k in enumerate(k_values):
        plt.plot(t, positions[i], label=f"k = {k}", color=colors[i])

    plt.title("Position vs Time")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    if save:
        plt.savefig("results/leapfrog_position.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()

    # Velocities
    plt.figure(figsize=FIG_SIZE)

    for i, k in enumerate(k_values):
        plt.plot(t, velocities[i], label=f"k = {k}", color=colors[i])

    plt.title("Velocity vs Time")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.legend()
    if save:
        plt.savefig("results/leapfrog_velocity.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.show()


def plot_leapfrog_errors(positions, velocities, k, T, delta_t=0.01, m=1, save=False):
    # Plot position errors
    t_position = np.linspace(0, T, len(positions[0]))

    def exact_position(t, k, m):
        A = 1
        omega = np.sqrt(k / m)
        phi = np.pi / 2

        return A * np.sin(omega * t + phi)

    plt.figure(figsize=FIG_SIZE)
    plt.plot(
        t_position,
        exact_position(t_position, k, m) - positions[0],
        label=r"$v_{1/2} = \Delta t \cdot F(x_0) / 2m$",
        color="y",
    )
    plt.plot(
        t_position,
        exact_position(t_position, k, m) - positions[1],
        label=r"$v_{1/2} = 0$",
        color="g",
    )
    plt.title("Error in Position vs Time")
    plt.xlabel("Time")
    plt.ylabel("Error in Position")
    plt.legend()
    if save:
        plt.savefig(
            "results/leapfrog_position_errors.png", dpi=FIG_DPI, bbox_inches="tight"
        )
    plt.show()

    # Plot velocity errors
    t_velocity = np.linspace(delta_t / 2, T + delta_t / 2, len(positions[0]))

    def exact_velocity(t, k, m):
        A = 1
        omega = np.sqrt(k / m)
        phi = np.pi / 2

        return A * omega * np.cos(omega * t + phi)

    plt.figure(figsize=FIG_SIZE)
    plt.plot(
        t_velocity,
        exact_velocity(t_velocity, k, m) - velocities[0],
        label=r"$v_{1/2} = \Delta t \cdot F(x_0) / 2m$",
        color="y",
    )
    plt.plot(
        t_velocity,
        exact_velocity(t_velocity, k, m) - velocities[1],
        label=r"$v_{1/2} = 0$",
        color="g",
    )
    plt.title("Error in Velocity vs Time")
    plt.xlabel("Time")
    plt.ylabel("Error in Velocity")
    plt.legend()
    if save:
        plt.savefig(
            "results/leapfrog_velocity_errors.png", dpi=FIG_DPI, bbox_inches="tight"
        )
    plt.show()
