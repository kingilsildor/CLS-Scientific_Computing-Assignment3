import glob
import os

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.config import FIG_DIR, FIG_DPI, FIG_SIZE, SCALER
from src.eigen_solver import time_dependent_solution


def create_seaborn_heatmap(arr: np.ndarray, ax: matplotlib.axes._axes.Axes) -> None:
    """
    Create a heatmap using seaborn library.

    Params
    -------
    - arr (np.ndarray): array to plot
    - ax (matplotlib.axes._axes.Axes): axis to plot the heatmap
    """
    sns.heatmap(
        arr,
        ax=ax,
        cmap="viridis",
        cbar=False,
        annot=True,
        square=True,
        annot_kws={"size": 8},
    )
    ax.set_xticks([])
    ax.set_yticks([])


def plot_eigenmodus(
    plot_amount: int,
    N: int,
    L: int,
    frequencies: np.ndarray,
    eigenvectors: np.ndarray,
    shape: str,
    save_img: bool = False,
) -> None:
    """
    Plot the eigenmodes of the matrix and save the image if specified.

    Params
    -------
    - plot_amount (int): amount of eigenmodes to plot
    - N (int): size of the grid
    - L (int): size of the shape
    - frequencies (np.ndarray): frequencies of the eigenvalues
    - eigenvectors (np.ndarray): eigenvectors of the eigenvalues
    - save_img (bool): flag to save the image. Default is False
    """
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    N_eigenvectors = eigenvectors.shape[1]
    if plot_amount > N_eigenvectors:
        raise ValueError(
            f"Plot amount should be less than or equal to {N_eigenvectors}"
        )

    for i in range(plot_amount):
        plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        v = eigenvectors[:, i].T.real
        plt.imshow(v.reshape(N, N), cmap="RdBu")
        plt.colorbar()
        plt.title(
            f"Eigenmode {i + 1} with frequency ${frequencies[i]:.4f}$\n {shape} shape of $L={L}$ and $N={N}$"
        )
        plt.tight_layout()

        if save_img:
            title = f"eigenmode_{shape}_{i + 1}.png"
            plt.savefig(f"{FIG_DIR}{title}")
        else:
            plt.show()
        plt.close()


def plot_eigenfrequency(
    L_list, frequency_list, N: int, shape: str, save_img: bool = False
) -> None:
    """
    Plot the eigenfrequencies of the matrix and save the image if specified.

    Params
    -------
    - L_list (np.ndarray): list of L values
    - frequency_list (np.ndarray): list of frequencies
    - N (int): size of the grid
    - shape (str): shape of the grid
    - save_img (bool): flag to save the image. Default is False
    """
    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    for i, _ in enumerate(L_list):
        plt.plot(L_list, frequency_list[:, i], label=f"Eigenmode {i + 1}")
    plt.xlabel("$L$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"Eigenfrequencies {shape} shape with $N={N}$")
    plt.tight_layout()

    if save_img:
        if not os.path.exists(FIG_DIR):
            os.makedirs(FIG_DIR)
        title = f"eigenfrequencies_{shape}.png"
        plt.savefig(f"{FIG_DIR}{title}")
    else:
        plt.show()
    plt.close()


def plot_eigenmode_animation(
    c,
    eigenmode,
    eigenfrequency,
    timepoints,
    shape: str,
    duration: int = 10,
    delete_img: bool = True,
) -> None:
    """
    Plot the eigenmode animation and save the gif. Delete the images if specified.

    Params
    -------
    - c (float): speed of the wave
    - eigenmode (np.ndarray): eigenmode of the matrix
    - eigenfrequency (float): frequency of the eigenmode
    - timepoints (np.ndarray): timepoints to plot
    - shape (str): shape of the grid
    - duration (int): duration of each frame in ms. Default is 10
    - delete_img (bool): flag to delete the images. Default is True
    """
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    for i, t in enumerate(timepoints):
        plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)

        u = time_dependent_solution(c, eigenmode, eigenfrequency, t)

        plt.imshow(u.real, cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(
            f"t = {timepoints[i]:.1f} for frequency {eigenfrequency:.4f} and shape {shape}"
        )
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}frame_{i:03d}.png")
        plt.close()

    images = [
        imageio.imread(f"{FIG_DIR}frame_{i:03d}.png") for i in range(len(timepoints))
    ]
    imageio.mimsave(f"{FIG_DIR}wave.gif", images, duration=duration)

    if delete_img:
        images = glob.glob(f"{FIG_DIR}frame_*.png")
        for image in images:
            os.remove(image)
        assert not glob.glob(f"{FIG_DIR}frame_*.png")


def plot_multiple_eigenmodes(
    amount: int,
    frequencies: np.ndarray,
    eigenvectors: np.ndarray,
    L: int,
    shape: str,
    save_img: bool = False,
) -> None:
    """
    Plot multiple eigenmodes in a single plot.

    Params
    -------
    - amount (int): amount of eigenmodes to plot
    - frequencies (np.ndarray): frequencies of the eigenvalues
    - eigenvectors (np.ndarray): eigenvectors of the eigenvalues
    - L (int): size of the shape
    - shape (str): shape of the grid
    - save_img (bool): flag to save the image. Default is False
    """
    N_eigenvectors = eigenvectors.shape[1]
    if amount > N_eigenvectors:
        raise ValueError(
            f"Plot amount should be less than or equal to {N_eigenvectors}"
        )

    x, y = FIG_SIZE
    x = x * 2
    fig, axs = plt.subplots(1, amount, figsize=(x, y), dpi=FIG_DPI)
    for i in range(amount):
        if shape == "rectangle":
            v = eigenvectors[:, i].real.reshape(L, 2 * L)
        else:
            v = eigenvectors[:, i].real.reshape(L, L)
        img = axs[i].imshow(v, cmap="RdBu", origin="lower", aspect="auto")
        if shape == "rectangle":
            axs[i].set_aspect(1.5)
        fig.colorbar(img, ax=axs[i])

        current_frequency = frequencies[i] * SCALER
        axs[i].set_title(f"Eigenmode {i + 1}\nFrequency: {current_frequency:.2f} kHz")

    plt.tight_layout()

    if save_img:
        if not os.path.exists(FIG_DIR):
            os.makedirs(FIG_DIR)
        title = f"{shape}_eigenmodes.png"
        plt.savefig(f"{FIG_DIR}{title}")
    else:
        plt.show()
    plt.close()
