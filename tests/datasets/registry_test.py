# Copyright © 2026 Apple Inc.

import pytest

from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.datasets.registry import load_dataset


@pytest.mark.parametrize("dataset_name", DatasetName)
def test_loading_all_datasets(dataset_name: DatasetName) -> None:
    load_dataset(dataset_name, config={})
