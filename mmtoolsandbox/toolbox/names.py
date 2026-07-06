# Copyright © 2026 Apple Inc.

"""Define the names of the available toolboxes."""

from typing_extensions import TypeAlias

from mmtoolsandbox.datasets.names import DatasetName

# At least for now the toolbox names are the same as the dataset names meaning that
# there is a 1:1 mapping between toolboxes and datasets.
ToolboxName: TypeAlias = DatasetName
