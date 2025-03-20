import matplotlib.pyplot as plt
import numpy as np

# TODO: delete after we start using np and plt in the code
blah = np.sum([1, 2, 3])
fig = plt.plot()


def leapfrog(T, F_x, delta_t=0.01, F_t=None, m=1):
    num_time_steps = int(T // delta_t)

    x_list = np.zeros(num_time_steps + 1)
    v_list = np.zeros(num_time_steps + 1)

    x_list[0] = 1  # Initial position of the mass at t = 0
    v_list[0] = (
        delta_t * F_x(x_list[0]) / m
    )  # Initial velocity of the mass at t = 1/2 * delta_t
    # v_list[0] = 0

    for i in range(num_time_steps):
        x_list[i + 1] = x_list[i] + delta_t * v_list[i]
        v_list[i + 1] = v_list[i] + delta_t * F_x(x_list[i + 1]) / m

    return x_list, v_list


def plot_leapfrog(x_list, v_list, T, k, m=1, detail="Position"):
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
    plt.show()

    plt.plot(t, v_list)
    plt.title("Velocity vs Time")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.show()
