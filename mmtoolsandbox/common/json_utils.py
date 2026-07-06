# Copyright © 2026 Apple Inc.

import json
from typing import Any, runtime_checkable

from typing_extensions import Protocol

from mmtoolsandbox.common.image_id import ImageResult


@runtime_checkable
class SupportsModelDump(Protocol):
    """Pydantic v2 support"""

    def model_dump(self) -> dict[str, Any]: ...


@runtime_checkable
class SupportsDict(Protocol):
    """Pydantic v1 support"""

    def dict(self) -> dict[str, Any]: ...


class MMToolSandboxJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to add support for other data types.

    The tool tracing requires the tool arguments and result to be JSON serializable.
    Tools can use user-defined classes so it is not guaranteed that the arguments or
    result is serializable with the standard JSON encoder. Thus, we define our own
    encoder here where we can handle custom classes and then fall back to the standard
    encoder.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, SupportsModelDump):
            return o.model_dump()
        elif isinstance(o, SupportsDict):
            # `swai_reference_agent_builtins.search_entities` returns `list[Entity]`
            # where `Entity` is a pydantic model. Apparently, the standard
            # `json.JSONEncoder` does not know how to handle them so we manually convert
            # them to a dict here.
            return o.dict()
        elif isinstance(o, ImageResult):
            return str(o)
        elif isinstance(o, set):
            return list(o)

        # Use the standard JSON encoder.
        return super().default(o)
