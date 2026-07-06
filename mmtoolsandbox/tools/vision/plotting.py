# Copyright © 2026 Apple Inc.

"""
A collection of plotting tools.

A Matplotlib backend interface with explicit parameters for common chart types.

The `plot` function serves as a unified interface for generating various types of charts
(line, bar, scatter, histogram, boxplot, pie, heatmap).
"""

import io

from matplotlib import pyplot as plt
from PIL import Image

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.image_id import ImageResult
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.vision.common import store_image


def _save_plot_to_db(fig: plt.Figure, dpi: int = 100) -> ImageResult:  # type: ignore[name-defined]
    """Convert a Matplotlib figure to a PIL Image and store it in the database.

    Args:
        fig: The Matplotlib figure to save.
        dpi: The resolution in dots per inch.

    Returns:
        The ImageResult containing the ID of the stored image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    image = Image.open(buf)
    return store_image(image)


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
def plot(
    type: str,
    title: str = "",
    x: list[float] | None = None,
    y: list[float] | None = None,
    categories: list[str] | None = None,
    data: list[float] | None = None,
    heights: list[float] | None = None,
    groups: dict | None = None,  # type: ignore[type-arg]
    sizes: list[float] | None = None,
    labels: list[str] | None = None,
    matrix: dict | None = None,  # type: ignore[type-arg]
    xlabel: str = "",
    ylabel: str = "",
    color: str | None = None,
    marker: str = "",
    line_style: str = "solid",
    linewidth: float = 1.5,
    point_size: float = 50.0,
    num_bins: int = 10,
    bar_label_rotation: float = 0.0,
    show_means: bool = False,
    explode: list[float] | None = None,
    start_angle: float = 0.0,
    color_map: str = "viridis",
    show_colorbar: bool = True,
    figsize: list[float] | None = None,
    grid: bool = False,
    xlim: list[float] | None = None,
    ylim: list[float] | None = None,
    dpi: int = 100,
) -> ImageResult:
    """Create a plot of the specified type using Matplotlib.

    This function unifies multiple plotting capabilities into a single interface.
    It supports the following chart types:
    - "line": 2D line plot.
    - "bar": Vertical bar chart.
    - "scatter": Scatter plot.
    - "histogram": Histogram of data distribution.
    - "boxplot": Box plot for grouped data.
    - "pie": Pie chart.
    - "heatmap": 2D heatmap from a matrix.

    Examples:
        # Line plot
        plot(type="line", x=[1, 2, 3], y=[4, 5, 6], title="Line Plot", grid=True)

        # Bar plot
        plot(type="bar", categories=["A", "B"], heights=[10, 20], title="Bar Chart", figsize=[8, 6])

        # Scatter plot
        plot(type="scatter", x=[1, 2, 3], y=[4, 5, 6], color="red", title="Scatter Plot", xlim=[0, 5])

    Args:
        type (str): The type of plot to generate.
            - "line": Requires `x`, `y`. Optional: `color` (default "blue"), `line_style` (default "solid"), `marker` (default "").
            - "bar": Requires `categories`, `heights`. Optional: `color` (default "blue"), `bar_label_rotation` (default 0).
            - "scatter": Requires `x`, `y`. Optional: `color` (default "blue"), `point_size` (default 50), `marker` (default "o").
            - "histogram": Requires `data`. Optional: `num_bins` (default 10), `color` (default "blue").
            - "boxplot": Requires `groups`. Optional: `color` (default "blue"), `show_means` (default False).
            - "pie": Requires `sizes`, `labels`. Optional: `explode`, `start_angle` (default 0).
            - "heatmap": Requires `matrix`. Optional: `color_map` (default "viridis"), `show_colorbar` (default True).
        title (str): The title of the plot.
        x (list[float] | None): X-axis coordinates (line, scatter).
        y (list[float] | None): Y-axis coordinates (line, scatter).
        categories (list[str] | None): Categories for bar chart.
        data (list[float] | None): Values to bin for histogram.
        heights (list[float] | None): Heights of each bar for bar chart.
        groups (dict | None): Dictionary of group names and values for boxplot.
        sizes (list[float] | None): Slice proportions for pie chart.
        labels (list[str] | None): Labels for pie chart slices.
        matrix (dict | None): Dictionary of rows for heatmap.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        color (str | None): Color for the plot elements.
            - Named colors: "blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"
            - Hex color codes, e.g., "#1f77b4"
        marker (str): Marker style for line and scatter plots.
            - Common options: ".", ",", "o", "s", "^", "v", "<", ">", "x", "+", "*", "D", "d", "p", "h", "H", "X"
        line_style (str): Line style for line plots.
            - Options: "solid", "dashed", "dashdot", "dotted"
        linewidth (float): Width of the plot lines.
        point_size (float): Marker size for scatter plots (points^2).
        num_bins (int): Number of bins for histogram.
        bar_label_rotation (float): Rotation angle (degrees) for x labels in bar chart.
        show_means (bool): Whether to display mean markers in boxplot.
        explode (list[float] | None): Offset fractions for pie chart slices.
        start_angle (float): Starting angle for pie chart.
        color_map (str): Colormap for heatmap.
            - Options: "viridis", "plasma", "inferno", "magma", "cividis"
        show_colorbar (bool): Whether to display color legend for heatmap.
        figsize (list[float] | None): Figure size in inches [width, height].
        grid (bool): Whether to show grid lines.
        xlim (list[float] | None): X-axis limits [min, max].
        ylim (list[float] | None): Y-axis limits [min, max].
        dpi (int): Resolution of the saved image in dots per inch.

    Returns:
        ImageResult: The ID of the created plot image.
    """
    if figsize:
        fig, ax = plt.subplots(figsize=tuple(figsize))
    else:
        fig, ax = plt.subplots()

    if grid:
        ax.grid(True)

    if xlim and len(xlim) == 2:
        ax.set_xlim((xlim[0], xlim[1]))
    if ylim and len(ylim) == 2:
        ax.set_ylim((ylim[0], ylim[1]))

    if type == "line":
        if x is None or y is None:
            raise ValueError("Arguments 'x' and 'y' are required for line plot.")
        ax.plot(
            x,
            y,
            color=color or "blue",
            linestyle=line_style,
            linewidth=linewidth,
            marker=marker,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif type == "bar":
        if categories is None or heights is None:
            raise ValueError(
                "Arguments 'categories' and 'heights' are required for bar plot."
            )
        ax.bar(categories, heights, color=color or "blue")
        ax.tick_params(axis="x", rotation=bar_label_rotation)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif type == "scatter":
        if x is None or y is None:
            raise ValueError("Arguments 'x' and 'y' are required for scatter plot.")
        # Default marker for scatter if not provided could be 'o'
        scatter_marker = marker if marker else "o"
        ax.scatter(x, y, c=color or "blue", s=point_size, marker=scatter_marker)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif type == "histogram":
        if data is None:
            raise ValueError("Argument 'data' is required for histogram.")
        ax.hist(data, bins=num_bins, color=color or "blue")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif type == "boxplot":
        if groups is None:
            raise ValueError("Argument 'groups' is required for boxplot.")
        plot_data = list(groups.values())
        plot_labels = list(groups.keys())
        result = ax.boxplot(
            plot_data, tick_labels=plot_labels, patch_artist=True, showmeans=show_means
        )
        box_color = color or "blue"
        for box in result["boxes"]:
            box.set_facecolor(box_color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    elif type == "pie":
        if sizes is None or labels is None:
            raise ValueError(
                "Arguments 'sizes' and 'labels' are required for pie chart."
            )
        ax.pie(sizes, labels=labels, explode=explode, startangle=start_angle)

    elif type == "heatmap":
        if matrix is None:
            raise ValueError("Argument 'matrix' is required for heatmap.")
        plot_data = list(matrix.values())
        plot_labels = list(matrix.keys())
        im = ax.imshow(plot_data, cmap=color_map)
        ax.set_yticks(range(len(plot_labels)))
        ax.set_yticklabels(plot_labels)
        if show_colorbar:
            fig.colorbar(im, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    else:
        raise ValueError(f"Unknown plot type: {type}")

    ax.set_title(title)
    return _save_plot_to_db(fig, dpi=dpi)
