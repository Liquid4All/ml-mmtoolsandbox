# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.tools.vision.plotting"""

from typing import Iterator

import pytest

from mmtoolsandbox.common.execution_context import (
    ExecutionContext,
    get_current_context,
    new_context,
)
from mmtoolsandbox.common.image_id import ImageResult
from mmtoolsandbox.common.tool_conversion import convert_to_openai_tool
from mmtoolsandbox.tools.vision.common import load_image
from mmtoolsandbox.tools.vision.plotting import plot


@pytest.fixture
def context() -> Iterator[ExecutionContext]:
    """Context for testing plotting tools."""
    test_context = ExecutionContext()
    with new_context(test_context):
        yield get_current_context()


def test_plotting_line(context: ExecutionContext) -> None:
    result = plot(
        type="line",
        x=[2018, 2019, 2020, 2021],
        y=[10, 15, 12, 18],
        color="blue",
        line_style="dashed",
        marker="o",
        title="GDP Trend",
        xlabel="Year",
        ylabel="GDP (Trillion USD)",
    )

    assert isinstance(result, ImageResult)

    # Verify image is stored
    image = load_image(result.image_id)
    assert image is not None
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_plotting_bar(context: ExecutionContext) -> None:
    result = plot(
        type="bar",
        categories=["Shanghai", "New York"],
        heights=[4.3, 3.8],
        color="orange",
        bar_label_rotation=0,
        title="GDP Comparison",
        xlabel="City",
        ylabel="GDP (Trillion USD)",
    )

    assert isinstance(result, ImageResult)

    image = load_image(result.image_id)
    assert image is not None
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_plotting_scatter(context: ExecutionContext) -> None:
    result = plot(
        type="scatter",
        x=[1, 2, 3, 4],
        y=[10, 15, 13, 17],
        color="green",
        point_size=80,
        marker="s",
        title="Investment vs Growth",
        xlabel="Investment",
        ylabel="Growth Rate",
    )

    assert isinstance(result, ImageResult)

    image = load_image(result.image_id)
    assert image is not None
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_plotting_histogram(context: ExecutionContext) -> None:
    result = plot(
        type="histogram",
        data=[2.3, 2.5, 3.1, 2.8, 3.3, 2.9],
        num_bins=5,
        color="purple",
        title="GDP Growth Distribution",
        xlabel="Growth Rate",
        ylabel="Frequency",
    )

    assert isinstance(result, ImageResult)

    image = load_image(result.image_id)
    assert image is not None
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_plotting_boxplot(context: ExecutionContext) -> None:
    result = plot(
        type="boxplot",
        groups={"City A": [2, 3, 4], "City B": [3, 5, 6]},
        color="cyan",
        show_means=True,
        title="City GDP Distribution",
        xlabel="City",
        ylabel="GDP",
    )

    assert isinstance(result, ImageResult)

    image = load_image(result.image_id)
    assert image is not None
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_plotting_pie(context: ExecutionContext) -> None:
    result = plot(
        type="pie",
        sizes=[50, 30, 20],
        labels=["Industry", "Services", "Agriculture"],
        explode=[0.1, 0, 0],
        start_angle=90,
        title="Economic Sector Share",
    )

    assert isinstance(result, ImageResult)

    image = load_image(result.image_id)
    assert image is not None
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_plotting_heatmap(context: ExecutionContext) -> None:
    result = plot(
        type="heatmap",
        matrix={"Row 1": [1, 2], "Row 2": [3, 4]},
        color_map="viridis",
        show_colorbar=True,
        title="Correlation Matrix",
        xlabel="Feature X",
        ylabel="Feature Y",
    )

    assert isinstance(result, ImageResult)

    image = load_image(result.image_id)
    assert image is not None
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_tool_conversion() -> None:
    convert_to_openai_tool(tool=plot)
