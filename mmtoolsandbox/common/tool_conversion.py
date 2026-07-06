# Copyright © 2026 Apple Inc.

"""Convert Python tool functions into OpenAI function-calling schema.

Adapted from https://github.com/langchain-ai/langchain/blob/
90184255f880a26cbdffd7b764deae9be3242ece/libs/core/langchain_core/utils/function_calling.py#L276
"""

from __future__ import annotations

import inspect
import uuid
from types import UnionType
from typing import (
    Any,
    Callable,
    Literal,
    Union,
    get_args,
    get_origin,
)

import docstring_parser
from pydantic import BaseModel
from strenum import StrEnum

from mmtoolsandbox.common.utils import NotGiven


def generate_tool_call_id() -> str:
    """Generate a tool call ID.

    Some models (e.g. Gemini, Mistral) do not generate tool call IDs when requesting a
    tool call. For consistency, we just generate a tool ID in those cases.

    Returns:
        A valid tool call ID.
    """
    uuid_str = str(uuid.uuid4())
    # Note that `uuid4` includes hyphens, which are not allowed in a variable name. We
    # may however decide downstream to use the tool call ID in the Python code for
    # making the function call, e.g.:
    #   tool_call_code = f"{tool_id}_parameters = {tool_call_dict_str}\n"
    # We also prepend `call_` just like the OpenAI API does to prevent the variable
    # name from being interpreted as a literal.
    return "call_" + uuid_str.replace("-", "_")


def default_parameter_processing(
    params: list[inspect.Parameter],
) -> list[inspect.Parameter]:
    """Modify the function arg spec of tool to a format more friendly for convert_to_openai_tool.
    In specific

    1. Remove Optional
    2. Remove NotGiven from Union types

    Args:
        params:   params to be processed

    Returns:
        Modified tool params
    """
    processed_params = []
    # Iterate through annotations. Note that this only unpacks top level annotation. Doesn't do so recursively
    for param in params:
        parameter_type = param.annotation
        type_origin = get_origin(parameter_type)
        type_args = get_args(parameter_type)

        # The OpenAI API function format does not support union or optional types. So if
        # possible we simplify those types below.
        if type_origin not in (Union, UnionType):
            processed_params.append(param)
            continue

        # For an `Optional[str]`, `type_origin`` will be `Union`/`UnionType` and `type_args`` will
        # be `(str, None)`. In that case, remove the `None` or `NotGiven` from the union
        # and extract the plain type.
        real_types = [t for t in type_args if t not in (type(None), NotGiven)]
        if len(real_types) == 1:
            param = param.replace(annotation=type_args[0])
        elif all(type in (int, float) for type in type_args):
            # Special case: `Union[int, float]` can be represented as float based on
            # PEP-0484:
            #   "when an argument is annotated as having type float, an argument of type
            #   int is acceptable"
            # , see https://peps.python.org/pep-0484/#the-numeric-tower .
            param = param.replace(annotation=float)

        processed_params.append(param)

    return processed_params


def _extract_docstring(tool: Callable[..., Any]) -> docstring_parser.Docstring:
    """Parse the docstring of *tool* in Google style."""
    docstring_text = inspect.getdoc(tool)
    return docstring_parser.parse(
        docstring_text or "",
        # We explicitly specify the `GOOGLE` docstring style since otherwise
        # `docstring_parse` can infer the wrong style if e.g. the docstring is not
        # actually following the `GOOGLE` style (e.g. missing a semicolon after an
        # exception type).
        style=docstring_parser.DocstringStyle.GOOGLE,
    )


def _tool_parameters(tool: Callable[..., Any]) -> list[inspect.Parameter]:
    """Return the processed parameter list for *tool*."""
    params = list(inspect.signature(tool).parameters.values())
    return default_parameter_processing(params)


def format_returns_section(docstring: docstring_parser.Docstring) -> str:
    """Format the Returns section from a parsed docstring into a readable string.

    Handles both simple returns (e.g. "The result.") and structured
    Success/Failure returns from AppWorld tools.
    """
    if not docstring.many_returns:
        return ""

    parts = []
    for ret in docstring.many_returns:
        type_name = ret.type_name or ""
        desc = (ret.description or "").strip()
        if type_name and desc:
            parts.append(f"{type_name}: {desc}")
        elif type_name:
            parts.append(type_name)
        elif desc:
            parts.append(desc)

    if not parts:
        return ""

    content = "\n".join(parts)
    if "\n" in content:
        return f"Returns:\n{content}"
    return f"Returns: {content}"


PYTHON_TO_JSON_TYPES = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
}


def _get_python_function_name(function: Callable[..., Any]) -> str:
    """Get the name of a Python function."""
    return function.__name__


# TODO: List rn only allows for python literals. Properly address this through recursion.
def _get_python_function_arguments(
    function: Callable[..., Any],
    parsed_parameters: list[docstring_parser.DocstringParam],
) -> dict[str, Any]:
    """Get JsonSchema describing a Python functions arguments.

    Assumes all function arguments are of primitive types (int, float, str, bool) or
    are subclasses of pydantic.BaseModel.
    """
    # Note that by default, the `docstring_parser` preserves whitespaces that are used
    # for visually aligning parameters. This docstring here:
    #
    #   Args:
    #       arg_name:   My description
    #
    # will result in:
    #   p.arg_name = "arg_name"
    #   p.description = "  My description"
    # Thus, we need to strip whitespaces from the left.
    arg_name_to_description = {
        p.arg_name: "" if p.description is None else p.description.strip()
        for p in parsed_parameters
    }

    properties: dict[str, Any] = {}
    parameters = _tool_parameters(function)
    for param in parameters:
        arg = param.name
        arg_type = param.annotation
        if arg == "return":
            continue

        if inspect.isclass(arg_type) and issubclass(arg_type, (list, tuple)):
            # The annotation is a bare list/tuple, not bound to a type, like list[str].
            raise TypeError(
                f"{param.name}: Must provide the type contained in the list/tuple."
            )

        origin = get_origin(arg_type)
        type_args = get_args(arg_type)
        if origin is not None:
            if origin not in (list, tuple, Literal):
                raise TypeError(f"{param.name}: Can not handle type `{origin}`.")
            if not type_args:
                raise TypeError(
                    f"{param.name}: Must must be bound to a type in `{origin}`."
                )
            # Handle lists/tuples.
            if origin in (list, tuple):
                if len(type_args) > 1:
                    raise TypeError(
                        f"{param.name}: Must provide a single type argument for `{origin}`, instead got `{type_args}`."
                    )
                assert len(type_args) == 1  # Already validated, just for typechecker.

                if hasattr(type_args[0], "__supertype__") and hasattr(
                    type_args[0].__supertype__, "__name__"
                ):
                    # Type alias (using NewType) type e.g. NewType("ImageId", int)
                    # Note: The values are implictly converted when executing the call.
                    json_type = PYTHON_TO_JSON_TYPES[
                        type_args[0].__supertype__.__name__
                    ]
                else:
                    json_type = PYTHON_TO_JSON_TYPES[type_args[0].__name__]

                properties[arg] = {
                    "type": "array",
                    "items": {
                        "type": json_type,
                    },
                }
            elif origin == Literal:
                if not all(type(type_args[0]) is type(x) for x in type_args):
                    raise TypeError(
                        f"{param.name}: Literal must contain identical type for all literals. Found {type_args}"
                    )
                properties[arg] = {
                    "type": PYTHON_TO_JSON_TYPES[type(type_args[0]).__name__],
                    "enum": list(type_args),
                }
        elif isinstance(arg_type, type) and issubclass(arg_type, BaseModel):
            # Mypy error:
            # "type" has no attribute "schema"
            properties[arg] = arg_type.schema()
        elif isinstance(arg_type, type) and issubclass(arg_type, StrEnum):
            properties[arg] = {
                "type": PYTHON_TO_JSON_TYPES["str"],
                "enum": [str(x) for x in arg_type],
            }
        elif (
            hasattr(arg_type, "__name__")
            and getattr(arg_type, "__name__") in PYTHON_TO_JSON_TYPES
        ):
            properties[arg] = {"type": PYTHON_TO_JSON_TYPES[arg_type.__name__]}
        elif (
            hasattr(arg_type, "__supertype__")
            and hasattr(arg_type.__supertype__, "__name__")
            and arg_type.__supertype__.__name__ in PYTHON_TO_JSON_TYPES
        ):
            # Type alias (using NewType) type e.g. NewType("ImageId", int)
            # Note: The values are implictly converted when executing the call.
            properties[arg] = {
                "type": PYTHON_TO_JSON_TYPES[arg_type.__supertype__.__name__]
            }
        else:
            properties[arg] = {"type": "object"}

        description = arg_name_to_description.get(arg)
        if description is not None:
            if arg not in properties:
                properties[arg] = {}
            properties[arg]["description"] = arg_name_to_description[arg]
    return properties


def _get_python_function_required_args(function: Callable[..., Any]) -> list[str]:
    """Get the required arguments for a Python function."""
    params = _tool_parameters(function)
    required = [p.name for p in params if p.default == p.empty]
    is_class = type(function) is type
    if is_class and required[0] == "self":
        required = required[1:]
    return required


def convert_python_function_to_openai_function(
    name: str,
    function: Callable[..., Any],
) -> dict[str, Any]:
    """Convert a Python function to an OpenAI function-calling API compatible dict.

    Assumes the Python function has type hints and a docstring with a description. If
    the docstring has Google Python style argument descriptions, these will be included
    as well.

    Args:
        name:     The tool name (typically ``function.__name__``).
        function: A Python function.

    Returns:
        An OpenAI compatible function declaration object.
    """
    docstring = _extract_docstring(function)
    description = "" if docstring.description is None else docstring.description.strip()

    return_info = format_returns_section(docstring)
    if return_info:
        description = f"{description}\n\n{return_info}" if description else return_info

    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": _get_python_function_arguments(function, docstring.params),
            "required": _get_python_function_required_args(function),
        },
    }


def convert_to_openai_tool(
    *,
    tool: Callable[..., Any],
    name: str | None = None,
) -> dict[str, Any]:
    """Convert a raw function/class to an OpenAI tool.

    Args:
        tool: A Python function.
        name: The tool name (defaults to ``tool.__name__``).

    Returns:
        A dict version of the passed in tool which is compatible with the OpenAI
        tool-calling API.
    """
    name = _get_python_function_name(tool) if name is None else name
    function = convert_python_function_to_openai_function(name, tool)
    return {"type": "function", "function": function}


def convert_to_openai_tools(
    *,
    name_to_tool: dict[str, Callable[..., Any]],
) -> list[dict[str, Any]]:
    return [
        convert_to_openai_tool(tool=tool, name=name)
        for name, tool in name_to_tool.items()
    ]
