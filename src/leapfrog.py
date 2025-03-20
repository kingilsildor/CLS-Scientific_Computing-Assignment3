import matplotlib.pyplot as plt
import numpy as np


def leapfrog(T, F_x, k, delta_t=0.01, F_t=None, m=1, initial_v=None):
    # Get number of timesteps and initials arrays
    num_time_steps = int(T // delta_t)

    x_list = np.zeros(num_time_steps + 1)
    v_list = np.zeros(num_time_steps + 1)

    # Initial conditions
    x_list[0] = 1
    if initial_v == "zero":
        v_list[0] = 0
    else:
        v_list[0] = delta_t * F_x(x_list[0], k) / m

    # If external force is not provided, set it to zero
    if F_t is None:

        def F_t(t, A, omega):
            return 0

    # Run the method
    for i in range(num_time_steps):
        x_list[i + 1] = x_list[i] + delta_t * v_list[i]
        v_list[i + 1] = (
            v_list[i]
            + delta_t * (F_x(x_list[i + 1], k) + F_t((i + 1) * delta_t, 0.1, 0.2)) / m
        )

    return x_list, v_list


def plot_leapfrog(x_list, v_list, T, k, m=1):
    # Get exact anlytical solution
    t = np.linspace(0, T, len(x_list))
    A = 1
    omega = np.sqrt(k / m)
    phi = np.pi / 2
    exact_sol = A * np.sin(omega * t + phi)

    plt.plot(t, exact_sol, label="Exact solution")
    plt.plot(t, x_list, label="Leapfrog solution")
    plt.title("Position vs Time")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    plt.show()

    # plt.plot(np.abs(exact_sol - x_list))
    plt.plot(exact_sol - x_list)
    plt.title("Error in Position vs Time")
    plt.xlabel("Time")
    plt.ylabel("Error in Position")
    plt.show()

    plt.plot(t, v_list)
    plt.title("Velocity vs Time")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.show()


def plot_leapfrog_various_k(positions, velocities, k_values, T, save=False):
    t = np.linspace(0, T, len(positions[0]))
    colors = ["r", "g", "y"]

    # Positions
    for i, k in enumerate(k_values):
        plt.plot(t, positions[i], label=f"k = {k}", color=colors[i])

    plt.title("Position vs Time")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend()
    if save:
        plt.savefig("results/leapfrog_position.png")
    plt.show()

    # Velocities
    for i, k in enumerate(k_values):
        plt.plot(t, velocities[i], label=f"k = {k}", color=colors[i])

    plt.title("Velocity vs Time")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.legend()
    if save:
        plt.savefig("results/leapfrog_velocity.png")
    plt.show()
