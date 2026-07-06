# Copyright © 2026 Apple Inc.

"""Unit tests for mmtoolsandbox.tools.setting"""

from typing import Iterator

import pytest

from mmtoolsandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    new_context,
)
from mmtoolsandbox.tools.tool_sandbox.setting import (
    get_boolean_settings,
    get_cellular_service_status,
    get_location_service_status,
    get_low_battery_mode_status,
    get_wifi_status,
    set_boolean_settings,
    set_cellular_service_status,
    set_location_service_status,
    set_low_battery_mode_status,
    set_wifi_status,
)


@pytest.fixture(scope="function", autouse=True)
def execution_context() -> Iterator[None]:
    """Autouse fixture which will setup and teardown execution context before and after each test function

    Returns:

    """
    test_context = ExecutionContext()
    test_context.add_to_database(
        namespace=DatabaseNamespace.SETTING,
        rows=[
            {
                "cellular": False,
                "wifi": False,
                "location_service": False,
                "low_battery_mode": False,
            },
        ],
    )
    with new_context(test_context):
        yield


def test_set_location_service_status() -> None:
    with pytest.raises(ValueError):
        set_location_service_status(on=False)
    set_location_service_status(on=True)
    assert get_location_service_status()


def test_get_location_service_status() -> None:
    assert not get_location_service_status()


def test_set_cellular_service_status() -> None:
    with pytest.raises(ValueError):
        set_cellular_service_status(on=False)
    set_cellular_service_status(on=True)
    assert get_cellular_service_status()


def test_get_cellular_service_status() -> None:
    assert not get_cellular_service_status()


def test_set_wifi_status() -> None:
    with pytest.raises(ValueError):
        set_wifi_status(on=False)
    set_wifi_status(on=True)
    assert get_wifi_status()


def test_get_wifi_status() -> None:
    assert not get_wifi_status()


def test_set_boolean_settings() -> None:
    with pytest.raises(ValueError):
        set_boolean_settings(setting_name="cellular", on=False)
    set_boolean_settings(setting_name="cellular", on=True)
    assert get_boolean_settings("cellular")
    with pytest.raises(KeyError):
        set_boolean_settings("latitude", on=True)


def test_get_boolean_settings() -> None:
    assert not get_boolean_settings("cellular")
    with pytest.raises(KeyError):
        get_boolean_settings("latitude")


def test_set_low_battery_mode_status() -> None:
    with pytest.raises(ValueError):
        set_low_battery_mode_status(on=False)
    set_low_battery_mode_status(on=True)
    assert get_low_battery_mode_status()
    # Turning on all others should fail
    with pytest.raises(PermissionError):
        set_location_service_status(on=True)
    with pytest.raises(PermissionError):
        set_cellular_service_status(on=True)
    with pytest.raises(PermissionError):
        set_wifi_status(on=True)
    # Turn off again, turn on a few things
    set_low_battery_mode_status(on=False)
    set_location_service_status(on=True)
    set_wifi_status(on=True)
    assert get_location_service_status()
    assert not get_cellular_service_status()
    assert get_wifi_status()
    # Turning on low battery should set all others to false
    set_low_battery_mode_status(on=True)
    assert not get_location_service_status()
    assert not get_cellular_service_status()
    assert not get_wifi_status()


def test_get_low_battery_mode_status() -> None:
    assert not get_low_battery_mode_status()
