# Copyright © 2026 Apple Inc.

import pytest

from mmtoolsandbox.datasets.registry import populate_dataset_registry


@pytest.fixture(scope="session", autouse=True)
def initialize_dataset_registry() -> None:
    # This fixture populates the dataset registry for all tests.
    #
    # If each test case was responsible for populating the registry,
    # running tests in parallel would lead to KeyErrors as some threads
    # might try to look up the registry before it has been fully populated.
    populate_dataset_registry()
