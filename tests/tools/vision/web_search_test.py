# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.tools.vision.web_search"""

import os
from typing import Iterator

import pytest

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import ExecutionContext, new_context
from mmtoolsandbox.tools.vision.web_search import (
    web_search_serper,
)

if "SERPER_API_KEY" not in os.environ:
    pytest.skip(
        "Tests intended for local mac development with search API access only. "
        "Disabled by default due to API capacity constraint. "
        "Enable with SERPER_API_KEY.",
        allow_module_level=True,
    )


@pytest.fixture(scope="function", autouse=True)
def execution_context() -> Iterator[None]:
    """Autouse fixture which will setup and teardown execution context before and after each test function"""
    test_context = ExecutionContext()
    test_context.add_to_database(
        namespace=DatabaseNamespace.SETTING,
        rows=[
            {
                "wifi": True,
                "location_service": True,
                "low_battery_mode": False,
                "locale": "en_US",
                "latitude": 37.334606,
                "longitude": -122.009102,
            },
        ],
    )
    with new_context(test_context):
        yield


def test_web_search_serper() -> None:
    results = web_search_serper(query="Apple")
    assert len(results) > 0
    first_result = results[0]
    assert "title" in first_result
    assert "link" in first_result
