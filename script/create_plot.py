import matplotlib
import numpy as np
import seaborn as sns


def seaborn_heatmap(arr: np.ndarray, ax: matplotlib.axes._axes.Axes) -> None:
    """
    Create a heatmap using seaborn library.

    Params
    -------
    - arr (np.ndarray): array to plot
    - ax (matplotlib.axes._axes.Axes): axis to plot the heatmap
    """
    sns.heatmap(arr, ax=ax, cmap="viridis", cbar=False, annot=True, square=True)
    ax.set_xticks([])
    ax.set_yticks([])
