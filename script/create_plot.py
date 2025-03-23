import glob
import os

import imageio
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src.config import (
    FIG_DIR,
    FIG_DPI,
    FIG_LABEL_SIZE,
    FIG_SIZE,
    FIG_TICK_SIZE,
    FIG_TITLE_SIZE,
    SELECT_MODE,
)
from src.eigen_solver import time_dependent_solution


def create_seaborn_heatmap(
    arr: np.ndarray,
    ax: plt.Axes,
    annot: bool = False,
    normalize: bool = False,
    cmap: str = "viridis",
) -> None:
    """
    Create a heatmap using seaborn library with 1 always mapped to yellow.

    Params
    -------
    - arr (np.ndarray): array to plot
    - ax (matplotlib.axes._axes.Axes): axis to plot the heatmap
    - annot (bool): flag to annotate the heatmap. Default is False
    - normalize (bool): flag to normalize the heatmap. Default is False
    - cmap (str): colormap for the heatmap. Default is "viridis"
    """
    norm = mcolors.Normalize(vmin=min(0, arr.min()), vmax=max(1, arr.max()))

    sns.heatmap(
        np.round(arr, 2),
        ax=ax,
        cmap=cmap,
        cbar=False,
        annot=annot,
        square=True,
        annot_kws={"size": 8},
        norm=norm if normalize else None,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def plot_eigenmodus(
    L: int,
    eigenmode: np.ndarray,
    shape: str,
    save_img: bool = False,
) -> None:
    """
    Plot the eigenmodes of the matrix and save the image if specified.

    Params
    -------
    - L (int): size of the grid
    - eigenmode (np.ndarray): eigenmode of the matrix
    - shape (str): shape of the grid
    - save_img (bool): flag to save the image. Default is False
    """
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    eigenmode = eigenmode.real
    plt.imshow(eigenmode, cmap="RdBu")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    if save_img:
        title = f"eigenmode_{shape}_{L}.png"
        plt.savefig(f"{FIG_DIR}{title}", bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def _add_image_with_line(
    ax: plt.Axes,
    image_path: str,
    xy: tuple,
    offset: tuple = (0.5, 0.5),
    zoom: float = 0.03,
) -> None:
    """
    Add an image to the plot with a line connecting the image to the point.

    Params
    -------
    - ax (matplotlib.axes._axes.Axes): axis to plot the image
    - image_path (str): path to the image
    - xy (tuple): point to plot the image
    - offset (tuple): offset for the image. Default is (0.5, 0.5)
    - zoom (float): zoom level for the image. Default is 0.03
    """
    img = mpimg.imread(image_path)
    xy_offset = (xy[0] + offset[0], xy[1] + offset[1])

    imagebox = OffsetImage(img, zoom=zoom)

    ab = AnnotationBbox(imagebox, xy_offset, frameon=True, boxcoords="data", pad=0.0)
    ax.add_artist(ab)

    ax.plot([xy[0], xy_offset[0]], [xy[1], xy_offset[1]], linestyle="--", color="gray")


def _add_images_with_lines(ax: plt.Axes, image_points: list, image_paths: list) -> None:
    """
    Add multiple images with lines connecting them to the plot.

    Params
    -------
    - ax (matplotlib.axes._axes.Axes): axis to plot the images
    - image_points (list): list of points to plot the images
    - image_paths (list): list of paths to the images
    """
    # Manually offset values for the images
    offset_one = [(25, 0.05)]
    offset_rest = [(10, -0.07) for _ in range(len(image_points) - 1)]
    offset_list = offset_one + offset_rest

    for point, img_path, offset in zip(image_points, image_paths, offset_list):
        _add_image_with_line(ax, img_path, point, offset=offset)


def plot_eigenfrequency(
    L_list, frequencies, shape: str, save_img: bool = False
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
    assert frequencies.shape[0] == L_list.shape[0]

    _, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)

    ax.plot(L_list, frequencies, label="Mean Eigenfrequencies", marker="o")
    ax.set_xlabel("$L$", fontsize=FIG_LABEL_SIZE)
    ax.set_ylabel("Frequency ($\\lambda$) in kHz", fontsize=FIG_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=FIG_TICK_SIZE)
    ax.set_title(f"Eigenfrequencies for a {shape} shape", fontsize=FIG_TITLE_SIZE)

    image_points = [(L_list[i], frequencies[i]) for i in range(len(L_list))]
    image_paths = [f"{FIG_DIR}eigenmode_{shape}_{L}.png" for L in L_list]
    _add_images_with_lines(ax, image_points, image_paths)

    ax.legend()
    plt.tight_layout()

    if save_img:
        if not os.path.exists(FIG_DIR):
            os.makedirs(FIG_DIR)
        title = f"eigenfrequencies_{shape}.png"
        plt.savefig(f"{FIG_DIR}{title}", bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_eigenmode_animation(
    c,
    eigenmode,
    eigenfrequency,
    timepoints,
    shape: str,
    duration: int = 100,
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

    min = -np.max(eigenmode.real)
    max = np.max(eigenmode.real)

    for i, t in enumerate(timepoints):
        plt.figure(figsize=FIG_SIZE)
        T = time_dependent_solution(c, eigenfrequency, t)
        u = eigenmode * T

        plt.imshow(u.real, cmap="RdBu", vmin=min, vmax=max)
        plt.colorbar()
        plt.xlabel("x", fontsize=FIG_LABEL_SIZE)
        plt.ylabel("y", fontsize=FIG_LABEL_SIZE)
        plt.title(
            f"t = {timepoints[i]:.1f} for Eigenmodus {SELECT_MODE}\nFrequency ($\\lambda$)={eigenfrequency.real:.2f} kHz",
            fontsize=FIG_TITLE_SIZE,
        )

        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}frame_{i:03d}.png", bbox_inches="tight", dpi=FIG_DPI)
        plt.close()

    # Create the gif
    images = [
        imageio.imread(f"{FIG_DIR}frame_{i:03d}.png") for i in range(len(timepoints))
    ]
    imageio.mimsave(f"{FIG_DIR}{shape}_wave.gif", images, duration=duration)

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
    shift: int = 0,
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
    - shift (int): shift the eigenmode index. Default is 0
    - save_img (bool): flag to save the image. Default is False
    """
    N_eigenvectors = eigenvectors.shape[1]
    if amount > N_eigenvectors:
        raise ValueError(
            f"Plot amount should be less than or equal to {N_eigenvectors}"
        )

    x, y = FIG_SIZE
    x = x * 3
    fig, ax = plt.subplots(1, amount, figsize=(x, y))
    for i in range(amount):
        if shape == "rectangle":
            v = eigenvectors[:, i].real.reshape(L, 2 * L)
        else:
            v = eigenvectors[:, i].real.reshape(L, L)
        img = ax[i].imshow(v, cmap="RdBu", origin="lower", aspect="equal")
        if shape == "rectangle":
            ax[i].set_aspect(1.5)
        fig.colorbar(img, ax=ax[i])

        ax[i].tick_params(axis="both", labelsize=FIG_TICK_SIZE * 2)
        ax[i].set_title(
            f"Eigenmode {i + 1 + shift}\nFrequency ($\\lambda$)={frequencies[i].real:.2f} kHz",
            fontsize=FIG_TITLE_SIZE * 1.4,
        )

    plt.tight_layout()

    if save_img:
        if not os.path.exists(FIG_DIR):
            os.makedirs(FIG_DIR)
        title = f"{shape}_eigenmodes.png"
        plt.savefig(f"{FIG_DIR}{title}", bbox_inches="tight", dpi=FIG_DPI)
    else:
        plt.show()
    plt.close()
