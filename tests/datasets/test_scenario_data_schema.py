# Copyright © 2026 Apple Inc.

"""Tests for ScenarioDataSchema in mmtoolsandbox.datasets.scenarios."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from mmtoolsandbox.common.execution_context import RoleType, ScenarioCategories
from mmtoolsandbox.common.image_id import ImageId
from mmtoolsandbox.common.scenario import Scenario, ScenarioExtension
from mmtoolsandbox.datasets.scenarios import ScenarioDataSchema


@pytest.fixture
def mock_base_scenario() -> Scenario:
    mock = MagicMock(spec=Scenario)
    mock.next_image_id.return_value = ImageId(0)
    return mock


@pytest.fixture
def base_scenarios(mock_base_scenario: Scenario) -> dict[str, Scenario]:
    return {"base": mock_base_scenario}


# ---------------------------------------------------------------------------
# from_dict / to_dict round-trip
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_minimal_fields(self) -> None:
        data = {"name": "test_scenario", "base_scenario": "base"}
        schema = ScenarioDataSchema.from_dict(data)

        assert schema.name == "test_scenario"
        assert schema.base_scenario == "base"
        assert schema.messages == []
        assert schema.milestones == []
        assert schema.image_paths is None
        assert schema.tool_allow_list is None

    def test_all_core_fields(self) -> None:
        data = {
            "name": "full_scenario",
            "base_scenario": "base",
            "image_paths": ["/img/0.png", "/img/1.png"],
            "messages": [
                {"sender": "USER", "recipient": "AGENT", "content": "Hello"},
            ],
            "tool_allow_list": ["search_contacts", "add_reminder"],
            "milestones": [{"type": "check"}],
            "task_completion_criteria": "reminder created",
        }
        schema = ScenarioDataSchema.from_dict(data)

        assert schema.image_paths == ["/img/0.png", "/img/1.png"]
        assert len(schema.messages) == 1
        assert schema.tool_allow_list == ["search_contacts", "add_reminder"]
        assert schema.task_completion_criteria == "reminder created"

    def test_appworld_extended_fields(self) -> None:
        data = {
            "name": "appworld_scenario",
            "base_scenario": "base",
            "appworld_entities": {"gmail.emails": [{"subject": "test"}]},
            "agentsandbox_entities": {"REMINDER": [{"content": "buy milk"}]},
            "device_state_id": "user@example.com",
            "appworld_base_task": "task_001",
            "reference_time": "2026-04-17T10:00:00Z",
            "description": "Test scenario",
            "difficulty": "easy",
            "entity_diff_specs": [{"table": "gmail.emails", "operation": "create"}],
            "categories": ["VISION_SYNTHETIC"],
            "max_messages": 30,
        }
        schema = ScenarioDataSchema.from_dict(data)

        assert schema.appworld_entities == {"gmail.emails": [{"subject": "test"}]}
        assert schema.device_state_id == "user@example.com"
        assert schema.reference_time == "2026-04-17T10:00:00Z"
        assert schema.max_messages == 30
        assert schema.entity_diff_specs is not None


class TestToDict:
    def test_minimal_round_trip(self) -> None:
        data: Dict[str, Any] = {"name": "s1", "base_scenario": "base"}
        schema = ScenarioDataSchema.from_dict(data)
        result = schema.to_dict()

        assert result["name"] == "s1"
        assert result["base_scenario"] == "base"
        # Optional fields omitted when None
        assert "image_paths" not in result
        assert "tool_allow_list" not in result

    def test_extended_fields_round_trip(self) -> None:
        data: Dict[str, Any] = {
            "name": "s2",
            "base_scenario": "base",
            "appworld_entities": {"todoist.tasks": [{"title": "Buy milk"}]},
            "device_state_id": "user@test.com",
            "max_messages": 40,
            "categories": ["VISION_SYNTHETIC"],
        }
        schema = ScenarioDataSchema.from_dict(data)
        result = schema.to_dict()

        assert result["appworld_entities"] == {"todoist.tasks": [{"title": "Buy milk"}]}
        assert result["device_state_id"] == "user@test.com"
        assert result["max_messages"] == 40


# ---------------------------------------------------------------------------
# to_scenario_extension
# ---------------------------------------------------------------------------


class TestToScenarioExtension:
    def test_basic_conversion(self, base_scenarios: dict[str, Scenario]) -> None:
        data: Dict[str, Any] = {
            "name": "test_ext",
            "base_scenario": "base",
            "messages": [
                {"sender": "USER", "recipient": "AGENT", "content": "Do something"},
            ],
            "tool_allow_list": ["search_contacts"],
        }
        schema = ScenarioDataSchema.from_dict(data)
        ext = schema.to_scenario_extension(base_scenarios)

        assert ext.name == "test_ext"
        assert ext.tool_allow_list == ["search_contacts"]
        assert len(ext.messages) == 1
        assert ext.messages[0]["sender"] == RoleType.USER
        assert ext.messages[0]["recipient"] == RoleType.AGENT

    def test_image_ids_resolved(self, base_scenarios: dict[str, Scenario]) -> None:
        data: Dict[str, Any] = {
            "name": "img_scenario",
            "base_scenario": "base",
            "messages": [
                {
                    "sender": "USER",
                    "recipient": "AGENT",
                    "content": "Look at this",
                    "image_ids": [0, 1],
                },
            ],
        }
        schema = ScenarioDataSchema.from_dict(data)
        ext = schema.to_scenario_extension(base_scenarios)

        assert ext.messages[0]["image_ids"] == [ImageId(0), ImageId(1)]

    def test_categories_resolved(self, base_scenarios: dict[str, Scenario]) -> None:
        data: Dict[str, Any] = {
            "name": "cat_scenario",
            "base_scenario": "base",
            "categories": ["VISION_SYNTHETIC"],
        }
        schema = ScenarioDataSchema.from_dict(data)
        ext = schema.to_scenario_extension(base_scenarios)

        assert ScenarioCategories.VISION_SYNTHETIC in ext.categories

    def test_unknown_categories_skipped(
        self, base_scenarios: dict[str, Scenario]
    ) -> None:
        data: Dict[str, Any] = {
            "name": "skip_cat",
            "base_scenario": "base",
            "categories": ["NONEXISTENT_CATEGORY", "VISION_SYNTHETIC"],
        }
        schema = ScenarioDataSchema.from_dict(data)
        ext = schema.to_scenario_extension(base_scenarios)

        assert len(ext.categories) == 1
        assert ScenarioCategories.VISION_SYNTHETIC in ext.categories

    def test_runtime_metadata_from_extended_fields(
        self, base_scenarios: dict[str, Scenario]
    ) -> None:
        data: Dict[str, Any] = {
            "name": "meta_scenario",
            "base_scenario": "base",
            "appworld_entities": {"gmail.emails": []},
            "device_state_id": "user@test.com",
            "description": "A test",
        }
        schema = ScenarioDataSchema.from_dict(data)
        ext = schema.to_scenario_extension(base_scenarios)

        assert ext.runtime_metadata is not None
        assert ext.runtime_metadata["appworld_entities"] == {"gmail.emails": []}
        assert ext.runtime_metadata["device_state_id"] == "user@test.com"
        assert ext.runtime_metadata["description"] == "A test"

    def test_missing_base_scenario_raises(
        self, base_scenarios: dict[str, Scenario]
    ) -> None:
        data: Dict[str, Any] = {
            "name": "bad_base",
            "base_scenario": "nonexistent",
        }
        schema = ScenarioDataSchema.from_dict(data)

        with pytest.raises(ValueError, match="not found"):
            schema.to_scenario_extension(base_scenarios)


# ---------------------------------------------------------------------------
# from_scenario_extension
# ---------------------------------------------------------------------------


class TestFromScenarioExtension:
    def test_round_trip(self, base_scenarios: dict[str, Scenario]) -> None:
        original_data: Dict[str, Any] = {
            "name": "round_trip",
            "base_scenario": "base",
            "messages": [
                {"sender": "USER", "recipient": "AGENT", "content": "Hello"},
            ],
            "tool_allow_list": ["search_contacts"],
        }
        schema = ScenarioDataSchema.from_dict(original_data)
        ext = schema.to_scenario_extension(base_scenarios)
        restored = ScenarioDataSchema.from_scenario_extension(ext, base_scenarios)

        assert restored.name == "round_trip"
        assert restored.base_scenario == "base"
        assert restored.tool_allow_list == ["search_contacts"]
        assert len(restored.messages) == 1

    def test_missing_base_raises(self) -> None:
        ext = MagicMock(spec=ScenarioExtension)
        ext.base_scenario = MagicMock()

        with pytest.raises(ValueError, match="not found"):
            ScenarioDataSchema.from_scenario_extension(ext, {})
