# Copyright © 2026 Apple Inc.

from enum import auto
from typing import Literal, Optional

import pytest
from strenum import StrEnum

from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.common.tool_conversion import (
    convert_python_function_to_openai_function,
    convert_to_openai_tool,
)
from mmtoolsandbox.common.tool_registry import get_all_tools
from mmtoolsandbox.toolbox.names import ToolboxName


@pytest.mark.parametrize("toolbox_name", ToolboxName)
def test_converting_all_tools(toolbox_name: ToolboxName) -> None:
    """Ensure that all our tools can be converted to the OpenAI tool format."""
    name_to_tool = get_all_tools(toolbox_name)
    try:
        for tool in name_to_tool.values():
            openai_tool = convert_to_openai_tool(tool=tool)
            assert "function" == openai_tool["type"]
            function = openai_tool["function"]
            assert function is not None
            assert function["name"] is not None
            assert function["description"] is not None
            assert "object" == function["parameters"]["type"]

            # Ensure that each parameter has a type and a description.
            for pname, param in function["parameters"]["properties"].items():
                assert param["type"] is not None
                assert "description" in param, f"Param `{pname}` missing docstring"
                assert param["description"], f"Param `{pname}` has empty docstring"
                if param["type"] == "array":
                    items = param.get("items", None)
                    assert isinstance(items, dict), param
                    assert items.get("type", None) is not None
                if "enum" in param:
                    assert isinstance(param["enum"], list)

            # Ensure that all required parameters are part of the properties entry.
            for required_param_name in function["parameters"]["required"]:
                assert (
                    function["parameters"]["properties"][required_param_name]
                    is not None
                )
    except AssertionError as e:
        raise AssertionError(f"Test failed for {tool.__name__}: {str(e)}") from e


def test_convert_python_function_to_openai_function() -> None:
    def compute(a: int, b: Optional[int] = None) -> int:
        """Computes something.

        Args:
            a: A number.
            b: An optional number.

        Returns:
            The number.

        Raises:
            ValueError: If less than or more than 1 self entry was found

        """
        return a

    NAME = "foo"
    expected_json_schema = {
        "name": NAME,
        "description": "Computes something.\n\nReturns: The number.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "A number."},
                "b": {"type": "integer", "description": "An optional number."},
            },
            "required": ["a"],
        },
    }
    assert expected_json_schema == convert_python_function_to_openai_function(
        name=NAME, function=compute
    )

    # Whitespaces in the docstring should not matter.
    def compute_wo_newlines(a: int, b: Optional[int] = None) -> int:
        """Computes something.
        Args:
            a: A number.
            b: An optional number.
        Returns:
            The number.
        Raises:
            ValueError: If less than or more than 1 self entry was found
        """
        return a

    assert expected_json_schema == convert_python_function_to_openai_function(
        name=NAME, function=compute_wo_newlines
    )


def test_convert_python_function_to_openai_function_one_line_docstring() -> None:
    def with_oneliner() -> None:
        """Does nothing."""
        ...

    NAME = "foo"
    assert convert_python_function_to_openai_function(
        name=NAME, function=with_oneliner
    ) == {
        "name": NAME,
        "description": "Does nothing.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def test_convert_to_openai_tool_listlike() -> None:
    def testfun1(values: list[str]) -> str:
        return "OK"

    assert convert_to_openai_tool(tool=testfun1) == {
        "type": "function",
        "function": {
            "name": "testfun1",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["values"],
            },
        },
    }

    def testfun2(values: Optional[tuple[int]] = None) -> str:
        return "OK"

    assert convert_to_openai_tool(tool=testfun2) == {
        "type": "function",
        "function": {
            "name": "testfun2",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "integer"}}
                },
                "required": [],
            },
        },
    }


def test_convert_to_openai_tool_literal() -> None:
    def testfun1(value: Literal["Value1", "Value2"]) -> str:
        return "OK"

    assert convert_to_openai_tool(tool=testfun1) == {
        "type": "function",
        "function": {
            "name": "testfun1",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "enum": ["Value1", "Value2"]}
                },
                "required": ["value"],
            },
        },
    }


def test_convert_to_openai_tool_enum() -> None:
    class Value(StrEnum):
        Value1 = auto()
        Value2 = auto()

    def testfun1(value: Value) -> str:
        return "OK"

    assert convert_to_openai_tool(tool=testfun1) == {
        "type": "function",
        "function": {
            "name": "testfun1",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "enum": ["Value1", "Value2"]}
                },
                "required": ["value"],
            },
        },
    }


def test_convert_to_openai_tool_image_id() -> None:
    def testfun1(image_id: ImageId) -> ImageId:
        return ImageId(image_id + 1)

    assert convert_to_openai_tool(tool=testfun1) == {
        "type": "function",
        "function": {
            "name": "testfun1",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {"image_id": {"type": "integer"}},
                "required": ["image_id"],
            },
        },
    }
