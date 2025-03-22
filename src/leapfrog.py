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


def leapfrog(T, F_x, k, delta_t=0.01, m=1, initial_v=None, F_t=None):
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


def exact_position(t, k, m):
    A = 1
    omega = np.sqrt(k / m)
    phi = np.pi / 2

    return A * np.sin(omega * t + phi)


def exact_velocity(t, k, m):
    A = 1
    omega = np.sqrt(k / m)
    phi = np.pi / 2

    return A * omega * np.cos(omega * t + phi)


def plot_leapfrog_various_k(
    positions, velocities, k_values, T, delta_t=0.01, save=False
):
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


def plot_leapfrog_errors(positions, velocities, k, T, delta_t=0.01, m=1, save=False):
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


def plot_leapfrog_errors_start_end(positions, k, T, delta_t=0.01, m=1, save=False):
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
    positions, velocities, omegas, T, delta_t=0.01, k=1, m=1, save=False
):
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


def plot_leapfrog_phase_plots(positions, velocities, omegas, k=1, m=1, save=False):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(
        "Phase Diagrams for Different Driving Frequencies", fontsize=FIG_TITLE_SIZE
    )
    colors = ["g", "y", "r", "gold"]
    omega = (k / m) ** 0.5

    for i, ax in enumerate(axes.flat):
        ax.plot(
            positions[i],
            velocities[i],
            color=colors[i],
            label=rf"$\omega_{{drive}} = {omegas[i] / omega:.1f}\omega$",
        )
        ax.set_xlabel("Position", fontsize=FIG_LABEL_SIZE)
        ax.set_ylabel("Velocity", fontsize=FIG_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=FIG_TICK_SIZE)
        ax.legend(fontsize=FIG_LEGEND_SIZE)
        ax.grid()

    plt.tight_layout()
    if save:
        plt.savefig(
            "results/leapfrog_phase_plots.png", dpi=FIG_DPI, bbox_inches="tight"
        )
    plt.show()
