# Copyright © 2026 Apple Inc.

"""Tests for prompt_templates.py composer and sections."""

import re

import pytest

from mmtoolsandbox.common.prompt_templates import (
    compose_agent_prompt,
    compose_user_prompt,
)

# ---------------------------------------------------------------------------
# Section duplication tests — every headed section (## TITLE) must appear
# at most once in any prompt, regardless of mode/flag combination.
# ---------------------------------------------------------------------------

# All valid compose_agent_prompt flag combinations that produce a prompt.
_ALL_VALID_CONFIGS: list[dict[str, object]] = [
    # Standard modes
    {},
    {"support_images": True},
    {"auto_login": True},
    {"auto_login": True, "support_images": True},
    # Search-only
    {"enable_tool_search": True},
    {"enable_tool_search": True, "support_images": True},
    {"enable_tool_search": True, "auto_login": True, "support_images": True},
    # Hybrid (search + coding)
    {"enable_tool_search": True, "enable_coding_tool": True},
    {"enable_tool_search": True, "enable_coding_tool": True, "support_images": True},
    {
        "enable_tool_search": True,
        "enable_coding_tool": True,
        "auto_login": True,
        "support_images": True,
    },
    # Pure code exec
    {"pure_code_exec": True},
    {"pure_code_exec": True, "support_images": True},
    {"pure_code_exec": True, "auto_login": True},
    {"pure_code_exec": True, "auto_login": True, "support_images": True},
    # UI variants
    {"enable_ui": True},
    {"enable_ui": True, "support_images": True},
    {"enable_ui": True, "auto_login": True},
    {"enable_ui": True, "auto_login": True, "support_images": True},
    {"enable_ui": True, "enable_tool_search": True, "enable_coding_tool": True},
    {
        "enable_ui": True,
        "enable_tool_search": True,
        "enable_coding_tool": True,
        "auto_login": True,
        "support_images": True,
    },
    # Reasoning variants
    {"enable_reasoning": "standard"},
    {"enable_reasoning": "extended"},
    {"enable_reasoning": "standard", "support_images": True},
    {"pure_code_exec": True, "auto_login": True, "enable_reasoning": "standard"},
    {
        "pure_code_exec": True,
        "auto_login": True,
        "support_images": True,
        "enable_reasoning": "extended",
    },
    {
        "enable_tool_search": True,
        "enable_coding_tool": True,
        "enable_reasoning": "extended",
        "support_images": True,
    },
]


class TestNoDuplicateSections:
    """Every ## heading must appear at most once in any prompt."""

    @pytest.mark.parametrize(
        "cfg", _ALL_VALID_CONFIGS, ids=lambda c: str(c) or "default"
    )
    def test_no_duplicate_headings(self, cfg: dict[str, object]) -> None:
        prompt = compose_agent_prompt(**cfg)  # type: ignore[arg-type]
        # Extract all ## headings
        headings = re.findall(r"^##\s+.+$", prompt, re.MULTILINE)
        duplicates = [h for h in headings if headings.count(h) > 1]
        assert not duplicates, (
            f"Duplicate section headings in prompt for config {cfg}: {duplicates}"
        )

    @pytest.mark.parametrize(
        "cfg", _ALL_VALID_CONFIGS, ids=lambda c: str(c) or "default"
    )
    def test_images_section_at_most_once(self, cfg: dict[str, object]) -> None:
        prompt = compose_agent_prompt(**cfg)  # type: ignore[arg-type]
        count = prompt.count("## IMAGES")
        assert count <= 1, f"IMAGES section appears {count} times for config {cfg}"

    @pytest.mark.parametrize(
        "cfg", _ALL_VALID_CONFIGS, ids=lambda c: str(c) or "default"
    )
    def test_code_environment_at_most_once(self, cfg: dict[str, object]) -> None:
        prompt = compose_agent_prompt(**cfg)  # type: ignore[arg-type]
        count = prompt.count("PYTHON EXECUTION ENVIRONMENT")
        assert count <= 1, (
            f"CODE_ENVIRONMENT section appears {count} times for config {cfg}"
        )

    @pytest.mark.parametrize(
        "cfg", _ALL_VALID_CONFIGS, ids=lambda c: str(c) or "default"
    )
    def test_reasoning_section_at_most_once(self, cfg: dict[str, object]) -> None:
        prompt = compose_agent_prompt(**cfg)  # type: ignore[arg-type]
        count = prompt.count("## Reasoning Format")
        assert count <= 1, f"Reasoning section appears {count} times for config {cfg}"

    @pytest.mark.parametrize(
        "cfg", _ALL_VALID_CONFIGS, ids=lambda c: str(c) or "default"
    )
    def test_auth_section_at_most_once(self, cfg: dict[str, object]) -> None:
        prompt = compose_agent_prompt(**cfg)  # type: ignore[arg-type]
        count = prompt.count("## AUTHENTICATION")
        assert count <= 1, (
            f"AUTHENTICATION section appears {count} times for config {cfg}"
        )

    @pytest.mark.parametrize(
        "cfg", _ALL_VALID_CONFIGS, ids=lambda c: str(c) or "default"
    )
    def test_rules_section_at_most_once(self, cfg: dict[str, object]) -> None:
        prompt = compose_agent_prompt(**cfg)  # type: ignore[arg-type]
        count = prompt.count("## RULES")
        assert count <= 1, f"RULES section appears {count} times for config {cfg}"


# ---------------------------------------------------------------------------
# Mode selection tests
# ---------------------------------------------------------------------------


class TestStandardMode:
    """Standard mode (no flags) — minimal prompt, no discovery, no code."""

    def test_has_tool_discovery(self) -> None:
        prompt = compose_agent_prompt()
        assert "## TOOL DISCOVERY" in prompt

    def test_no_code(self) -> None:
        prompt = compose_agent_prompt()
        assert "execute_code" not in prompt
        assert "## CODE FORMAT" not in prompt

    def test_has_auth(self) -> None:
        prompt = compose_agent_prompt()
        assert "## AUTHENTICATION" in prompt
        assert "## SPECIAL APPS" in prompt

    def test_no_images_by_default(self) -> None:
        prompt = compose_agent_prompt()
        assert "## IMAGES" not in prompt

    def test_images_when_flagged(self) -> None:
        prompt = compose_agent_prompt(support_images=True)
        assert "## IMAGES" in prompt


class TestSearchOnlyMode:
    """Search-only mode — discovery, no code."""

    def test_has_discovery(self) -> None:
        prompt = compose_agent_prompt(enable_tool_search=True)
        assert "api_docs_search_api_docs" in prompt

    def test_no_execute_code(self) -> None:
        prompt = compose_agent_prompt(enable_tool_search=True)
        assert "execute_code" not in prompt

    def test_no_code_format(self) -> None:
        prompt = compose_agent_prompt(enable_tool_search=True)
        assert "## CODE FORMAT" not in prompt


class TestHybridMode:
    """Hybrid mode — discovery + execute_code via function-calling API."""

    def test_has_discovery_and_execute_code(self) -> None:
        prompt = compose_agent_prompt(enable_tool_search=True, enable_coding_tool=True)
        assert "execute_code" in prompt
        assert "api_docs_search_api_docs" in prompt

    def test_no_raw_code_blocks(self) -> None:
        prompt = compose_agent_prompt(enable_tool_search=True, enable_coding_tool=True)
        assert "## CODE FORMAT" not in prompt
        assert "## HOW YOU WORK" not in prompt


class TestPureCodeExecMode:
    """Pure code-exec mode — raw ```python blocks."""

    def test_has_response_format(self) -> None:
        prompt = compose_agent_prompt(pure_code_exec=True)
        assert "## RESPONSE FORMAT" in prompt

    def test_has_how_you_work(self) -> None:
        prompt = compose_agent_prompt(pure_code_exec=True)
        assert "## HOW YOU WORK" in prompt

    def test_has_code_environment(self) -> None:
        prompt = compose_agent_prompt(pure_code_exec=True)
        assert "PYTHON EXECUTION ENVIRONMENT" in prompt

    def test_has_discovery(self) -> None:
        prompt = compose_agent_prompt(pure_code_exec=True)
        assert "api_docs_search_api_docs" in prompt


# ---------------------------------------------------------------------------
# AppWorld flag tests
# ---------------------------------------------------------------------------


class TestAuthAndDiscovery:
    def test_has_auth_and_special_apps(self) -> None:
        prompt = compose_agent_prompt()
        assert "## AUTHENTICATION" in prompt
        assert "## SPECIAL APPS" in prompt

    def test_with_images(self) -> None:
        prompt = compose_agent_prompt(support_images=True)
        assert "## IMAGES" in prompt

    def test_manual_login(self) -> None:
        prompt = compose_agent_prompt()
        assert "ALWAYS log in" in prompt

    def test_auto_login(self) -> None:
        prompt = compose_agent_prompt(auto_login=True)
        assert "ALREADY logged in" in prompt
        assert "ALWAYS log in" not in prompt


# ---------------------------------------------------------------------------
# UI flag tests
# ---------------------------------------------------------------------------


class TestUIFlag:
    """UI rendering sections."""

    def test_has_ui_rendering(self) -> None:
        prompt = compose_agent_prompt(enable_ui=True)
        assert "## UI RENDERING" in prompt

    def test_has_ui_image_handling(self) -> None:
        prompt = compose_agent_prompt(enable_ui=True)
        assert "IMAGE HANDLING IN UI" in prompt

    def test_ui_with_images(self) -> None:
        prompt = compose_agent_prompt(enable_ui=True, support_images=True)
        assert "## IMAGES" in prompt

    def test_not_with_pure_code_exec(self) -> None:
        with pytest.raises(ValueError, match="not compatible"):
            compose_agent_prompt(pure_code_exec=True, enable_ui=True)

    def test_works_with_hybrid(self) -> None:
        prompt = compose_agent_prompt(
            enable_tool_search=True,
            enable_coding_tool=True,
            enable_ui=True,
            support_images=True,
        )
        assert "## UI RENDERING" in prompt
        assert "## IMAGES" in prompt
        assert "## TOOL DISCOVERY" in prompt

    def test_absent_when_not_flagged(self) -> None:
        prompt = compose_agent_prompt()
        assert "## UI RENDERING" not in prompt


# ---------------------------------------------------------------------------
# Reasoning flag tests
# ---------------------------------------------------------------------------


class TestReasoningFlag:
    """Reasoning augment sections."""

    def test_standard_reasoning(self) -> None:
        prompt = compose_agent_prompt(enable_reasoning="standard")
        assert "<think>" in prompt
        assert "Reasoning Format" in prompt

    def test_extended_reasoning(self) -> None:
        prompt = compose_agent_prompt(enable_reasoning="extended")
        assert "Reasoning Format (Extended)" in prompt
        assert "Reflect" in prompt

    def test_no_reasoning_by_default(self) -> None:
        prompt = compose_agent_prompt()
        assert "<think>" not in prompt


# ---------------------------------------------------------------------------
# User prompt tests
# ---------------------------------------------------------------------------


class TestComposeUserPrompt:
    def test_base_instruction(self) -> None:
        prompt = compose_user_prompt("Ask User B to find restaurants.")
        assert "Ask User B to find restaurants." in prompt
        assert "ui_user_interact" not in prompt

    def test_with_ui(self) -> None:
        prompt = compose_user_prompt("Base instruction", enable_ui=True)
        assert "Base instruction" in prompt
        assert "ui_user_interact" in prompt

    def test_empty_base(self) -> None:
        prompt = compose_user_prompt(enable_ui=True)
        assert "ui_user_interact" in prompt
