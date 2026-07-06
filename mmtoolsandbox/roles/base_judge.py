# Copyright © 2026 Apple Inc.

"""Shared judge rubrics, data structures, and constants.

This module is the single source of truth for:
- System prompts (agent judge, UI judge, user simulator judge)
- Criteria tuples (used for aggregation key naming)
- JudgeResult / CriterionResult data classes (used by all judge backends)
- Infrastructure constants (_MAX_IMAGES, _INFRASTRUCTURE_NAMESPACES)

Backend-specific API logic lives in openai_judge.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mmtoolsandbox.common.databases import DatabaseNamespace

# ---------------------------------------------------------------------------
# Judge registry — maps CLI names to (module_path, class_name) for lazy import.
# Avoids importing all judge backends at module level.
# ---------------------------------------------------------------------------

_JUDGE_REGISTRY: dict[str, tuple[str, str]] = {
    "GPT_5_4_2026_03_05_Judge": (
        "mmtoolsandbox.roles.openai_judge",
        "GPT_5_4_2026_03_05_Judge",
    ),
    "Claude_4_5_Sonnet_Judge": (
        "mmtoolsandbox.roles.anthropic_judge",
        "Claude_4_5_Sonnet_Judge",
    ),
}


def get_judge_instance(judge_name: str) -> Any:
    """Instantiate a judge by CLI name.

    Args:
        judge_name: Key from ``_JUDGE_REGISTRY``.

    Returns:
        An instantiated judge object (OpenAIAPIJudge or subclass).

    Raises:
        ValueError: If ``judge_name`` is not in the registry.
    """
    import importlib

    if judge_name not in _JUDGE_REGISTRY:
        raise ValueError(
            f"Unknown judge '{judge_name}'. Available: {sorted(_JUDGE_REGISTRY.keys())}"
        )
    module_path, class_name = _JUDGE_REGISTRY[judge_name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


# ---------------------------------------------------------------------------
# Judge output schema — single source of truth for prompt, parsing, and viz
# ---------------------------------------------------------------------------

_RUBRIC_CRITERIA = (
    "task_completion",
    "instruction_following",
    "tool_use_validity",
    "no_side_effects",
    "information_accuracy",
)

_UI_RUBRIC_CRITERIA = (
    "ui_task_focus",
    "ui_structure_and_patterns",
    "ui_action_clarity",
    "ui_feedback_and_flow",
)

_USER_RUBRIC_CRITERIA = (
    "request_fidelity",
    "conversational_naturalness",
    "grounded_consistency",
    "tool_channel_correctness",
)


@dataclass
class CriterionResult:
    """Result for a single rubric criterion.

    Attributes:
        criterion: Name of the rubric criterion (e.g. ``"task_completion"``).
        analysis: Brief reasoning grounded in trusted evidence.
        passed: Whether the criterion passed. Serialized as ``"pass"`` in JSON.
        evidence: One-sentence summary of supporting evidence.
    """

    criterion: str
    analysis: str
    # Note: "pass" is a Python keyword; we use "passed" internally but
    # serialize to "pass" in JSON for the LLM schema.
    passed: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize with 'pass' key (matching the LLM output schema)."""
        return {
            "criterion": self.criterion,
            "analysis": self.analysis,
            "pass": self.passed,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CriterionResult:
        """Deserialize a ``CriterionResult`` from a dictionary.

        Tolerates missing fields by falling back to safe defaults.

        Args:
            data: Dictionary with keys ``criterion``, ``analysis``, ``pass``,
                and ``evidence``.

        Returns:
            A ``CriterionResult`` instance.
        """
        return cls(
            criterion=data.get("criterion", "unknown"),
            analysis=data.get("analysis", ""),
            passed=bool(data.get("pass", False)),
            evidence=data.get("evidence", ""),
        )


@dataclass
class JudgeResult:
    """Structured result from the rubric judge.

    Attributes:
        criteria_evaluation: List of per-criterion results.
        result: Overall pass/fail (True only if all criteria pass).
        reasoning: Brief summary of the overall judgment.
    """

    criteria_evaluation: list[CriterionResult]
    result: bool
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage in EvaluationResult.judge_result."""
        return {
            "criteria_evaluation": [c.to_dict() for c in self.criteria_evaluation],
            "result": self.result,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JudgeResult:
        """Parse judge model JSON output. Tolerates missing/extra fields."""
        criteria = [
            CriterionResult.from_dict(c) for c in data.get("criteria_evaluation", [])
        ]
        return cls(
            criteria_evaluation=criteria,
            result=bool(data.get("result", False)),
            reasoning=data.get("reasoning", ""),
        )


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_INFRASTRUCTURE_NAMESPACES: frozenset[DatabaseNamespace] = frozenset(
    {
        DatabaseNamespace.SANDBOX,
        DatabaseNamespace.IMAGE,
    }
)

_MAX_IMAGES = 10

# ---------------------------------------------------------------------------
# Agent task completion rubric (5 criteria)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating whether an AI Agent successfully completed a task.\n\n"
    "## TRUST MODEL\n"
    "- TRUSTED evidence: tool call results, execution environment outputs, database state changes, images. "
    "Base your judgments ONLY on these.\n"
    '- UNTRUSTED: the agent\'s natural language claims ("I\'ve completed the task", "Done!"). '
    "The agent may sound confident while having failed. Always cross-check claims against tool results "
    "and database state.\n\n"
    "## INPUT FORMAT\n"
    "You will receive:\n"
    "1. Task Completion Criteria — the specific conditions for success.\n"
    "2. Conversation — the full USER↔AGENT message thread.\n"
    "3. Agent Actions — chronological AGENT↔ENVIRONMENT interactions: tool calls, results, and errors.\n"
    "4. Database Changes — the actual changes to the system's databases. This is ground truth.\n"
    "5. Relevant Images — user-provided images and output artifacts (if any).\n\n"
    "## RUBRIC (5 criteria, each pass/fail)\n"
    "Evaluate each criterion independently using ONLY trusted evidence:\n\n"
    "1. **task_completion**: Did the agent perform the primary action requested by the user?\n"
    "   - Pass: The core action was attempted AND the tool call succeeded.\n"
    "   - Fail: The action was not attempted, or the tool call returned an error.\n\n"
    "2. **instruction_following**: Were ALL specific details in the criteria satisfied?\n"
    "   - If the criteria lists N items, count each one. All N must be present.\n"
    "   - Check exact values: names, dates, phone numbers, amounts, calendar names.\n"
    "   - Additional correct information beyond what the criteria requires is acceptable.\n"
    "     Only fail for extra content if the criteria explicitly forbids it.\n"
    "   - For text content (email bodies, note contents), accept reasonable paraphrasing\n"
    "     and minor formatting differences (e.g., presence or absence of quotes around\n"
    "     titles, slight rewording). Only fail if the meaning or key data points are\n"
    "     wrong or missing.\n"
    "   - Pass: Every required detail is present. Fail: Any required detail is missing or wrong.\n\n"
    "3. **tool_use_validity**: Did the agent use tools in a way that was valid, sufficient, and "
    "consistent with the task constraints?\n"
    "   - Do NOT require a specific tool or exact tool sequence unless the criteria explicitly mandate it.\n"
    "   - Alternative valid tool-use strategies should pass if they satisfy the task and do not violate constraints.\n"
    "   - Pass: Tool usage is valid and sufficient. Fail: Tool usage is invalid, insufficient, or violates constraints.\n\n"
    "4. **no_side_effects**: Did the agent avoid unintended actions?\n"
    "   - Check: no extra database writes, no wrong recipients, no forbidden tool calls.\n"
    '   - If criteria says info comes "from the image", the agent should NOT have called a search tool '
    "for that same information.\n"
    "   - If an API requires a field to be filled (e.g., a contacts API requires email\n"
    "     as a mandatory parameter), the agent providing a placeholder value for that\n"
    "     required field is NOT a side effect. Only flag actions that are truly optional\n"
    "     and unrelated to the task.\n"
    "   - Pass: Only necessary actions taken. Fail: Unintended actions detected.\n\n"
    "5. **information_accuracy**: Is the final answer or artifact factually correct?\n"
    "   - Cross-check the agent's response against tool results, database state, and images.\n"
    "   - For numerical values, accept differences within 1% due to rounding.\n"
    "   - Pass: Information is correct. Fail: Information is wrong or fabricated.\n\n"
    "## FAILURE MODE CHECKLIST\n"
    "Before scoring, explicitly verify:\n"
    "1. COMPLETENESS: If criteria lists N items, count each one in tool calls or database state.\n"
    "2. EXECUTION vs CLAIMS: Count actual successful tool calls, not agent's summary.\n"
    "3. UNINTENDED ACTIONS: Check for tools called beyond what was necessary.\n"
    "4. TOOL ERRORS: Check every tool result for error indicators.\n"
    "5. CORRECT TARGETS: Verify exact recipients, phone numbers, calendar names, dates.\n"
    "6. PRECONDITION CHECK: If criteria says 'delete X' or 'update X', verify that X\n"
    "   actually exists in the database state before penalizing the agent for not\n"
    "   finding it. If X does not exist, the agent correctly finding nothing is NOT\n"
    "   a failure.\n\n"
    "## EVALUATION STEPS\n"
    "1. Read the criteria and list every checkable requirement.\n"
    "2. Review agent actions: valid tools? reasonable arguments? success results?\n"
    "3. Review database changes: do they reflect the expected outcome?\n"
    "4. For each of the 5 rubric criteria, first write a brief analysis grounded in trusted "
    "evidence, then determine pass/fail.\n"
    "5. Determine overall result: pass ONLY if ALL 5 criteria pass.\n\n"
    "## OUTPUT FORMAT\n"
    "Output a JSON object with no markdown formatting, no code blocks:\n"
    "{\n"
    '  "criteria_evaluation": [\n'
    '    {"criterion": "task_completion", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "instruction_following", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "tool_use_validity", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "no_side_effects", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "information_accuracy", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"}\n'
    "  ],\n"
    '  "result": boolean,\n'
    '  "reasoning": "Brief summary of overall judgment."\n'
    "}"
)

# ---------------------------------------------------------------------------
# UI quality rubric (4 criteria)
# ---------------------------------------------------------------------------

_UI_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating the quality of AI-generated interactive UI screens.\n\n"
    "## TRUST MODEL\n"
    "You have two complementary evidence sources — use BOTH together:\n"
    "- Component tree JSON (render_ui_screen arguments): objective, structured data showing "
    "component types, nesting, and labels. Good for detecting structural problems "
    "(text dumps, flat trees, missing buttons).\n"
    "- Rendered UI screenshots: show the visual result including content quality, "
    "readability, and layout. Good for detecting content problems the JSON hides "
    "(garbage text, broken images, overflow/truncation, walls of text split across "
    "multiple Text components).\n"
    "- Neither alone is sufficient. A well-structured JSON tree can render poorly "
    "(broken image placeholders, overflowing content). A clean-looking screenshot can "
    "hide structural laziness (one big Text component with all data as a string).\n"
    "- Evaluate ONLY what the agent controls (component structure, labels, information "
    "choices) — do NOT penalize for renderer-level visual quality (font rendering, "
    "pixel spacing, colors).\n\n"
    "## INPUT FORMAT\n"
    "You will receive:\n"
    "1. UI Requirements — functional descriptions of what the UI should enable "
    "the user to do in each round (e.g., 'select which item to purchase', "
    "'confirm the expense split'). These describe the FUNCTION, not a specific "
    "layout — the agent may use any component pattern that achieves the goal.\n"
    "2. UI Structure Summary — per-render structural facts: component counts by type, "
    "nesting depth, and button labels with action bindings. Use this to detect "
    "structural problems (text dumps, flat trees, missing buttons) without parsing "
    "raw JSON.\n"
    "3. UI Tool Trace — chronological UI-related tool calls and results.\n"
    "4. User Interactions — ui_user_interact calls showing what the user clicked.\n"
    "5. UI Screenshots — rendered UI images only (no user-provided images). Use these "
    "to evaluate visual quality: readability, content correctness, layout clarity.\n\n"
    "## STRUCTURAL ANALYSIS\n"
    "Before scoring, check the UI Structure Summary for each render:\n"
    "- A render with depth <= 2 and only Text components is a text dump — FAIL.\n"
    "- A render with 'no buttons' when the task requires user action — FAIL.\n"
    "- A render with only 1 Button for a list of N items (instead of per-item "
    "buttons) may indicate poor action clarity.\n"
    "- Cross-check the summary against the screenshots: does the visual result "
    "match what the structure promises?\n\n"
    "## RUBRIC (4 criteria, each pass/fail)\n"
    "Evaluate each criterion independently:\n\n"
    "1. **ui_task_focus**: Does the screen make the primary goal and action immediately clear?\n"
    "   - Pass: The most important information is visually prominent. The primary action is "
    "obvious within a quick glance. The screen is free of unnecessary clutter. Secondary "
    "details are de-emphasized rather than competing for attention.\n"
    "   - Fail: Too much information competing for attention. No clear focal point. The user "
    "cannot quickly determine what the main action is.\n"
    "   - Anti-pattern: More than 5 top-level peer components with no grouping hierarchy.\n\n"
    "2. **ui_structure_and_patterns**: Is the UI organized with appropriate patterns, "
    "logical grouping, and clear hierarchy?\n"
    "   - Pass: Used appropriate UI pattern for the task (List for comparisons, Card for "
    "details, Form for input). Related elements are grouped together. Clear hierarchy via "
    "headings, body text, and captions. Content is scannable — concise labels, not walls "
    "of text.\n"
    "   - Fail: Used a single Text component with all information as plain text inside a "
    "Card. No structural hierarchy. Items that should be separate (e.g., 5 restaurants) "
    "are merged into one block.\n"
    "   - Anti-patterns from the UI Structure Summary:\n"
    "     - Only Text components with depth <= 2: a text dump, not a real UI.\n"
    "     - N items that should be separate but the summary shows only 1 Card and "
    "1 Text: items are concatenated into one block.\n\n"
    "3. **ui_action_clarity**: Do interactive elements clearly communicate what they do?\n"
    "   - Pass: Buttons have specific verb labels ('Book Now', 'View Trails', 'Save Recipe' "
    "— not generic 'Click' or 'Submit'). Primary action is distinguishable from secondary "
    "actions. In a list/comparison, each item has its own action button. Form fields have "
    "clear labels.\n"
    "   - Fail: Vague button labels ('OK', 'Submit' without context). Missing action "
    "buttons (e.g., a comparison list with no way to select individual items). All buttons "
    "look identical with no primary/secondary distinction.\n"
    "   - Anti-patterns from the UI Structure Summary:\n"
    "     - 'no buttons' when the task requires user action.\n"
    "     - A list of N items but only 1 Button globally instead of per-item buttons.\n"
    "     - Button labels from the summary with generic names ('?', 'ok').\n\n"
    "4. **ui_feedback_and_flow**: After user interaction, does the UI clearly communicate "
    "the new state and guide the user to the next step?\n"
    "   - Pass: After a button click, the resulting screen confirms what happened. State "
    "changes are visible. Multi-step flows have clear progression.\n"
    "   - Fail: After interaction, the UI doesn't change or shows no indication that "
    "anything happened. No confirmation of submitted data.\n"
    "   - NOTE: If there are no user interactions in the trajectory, score PASS by default.\n\n"
    "## EVALUATION STEPS\n"
    "1. Read the UI Structure Summary for each render. Check component counts, "
    "depth, and button presence. Flag text dumps and missing interactions.\n"
    "2. Check the UI screenshots for visual quality: is the content meaningful, "
    "readable, and well-laid-out? Are images rendered (not broken)? Is content "
    "clipped or overflowing?\n"
    "3. Cross-check: does the screenshot match what the structure summary promises? "
    "A summary showing multiple Card components should produce a visually "
    "structured screen, not a wall of text.\n"
    "4. Check button labels from the summary — are they task-specific or generic?\n"
    "5. If the user interacted, compare pre- and post-interaction screens for feedback.\n"
    "6. For each of the 4 criteria, write a brief analysis grounded in both the "
    "structure summary and screenshots, then determine pass/fail.\n"
    "7. Determine overall result: pass ONLY if ALL 4 criteria pass.\n\n"
    "## OUTPUT FORMAT\n"
    "Output a JSON object with no markdown formatting, no code blocks:\n"
    "{\n"
    '  "criteria_evaluation": [\n'
    '    {"criterion": "ui_task_focus", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "ui_structure_and_patterns", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "ui_action_clarity", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "ui_feedback_and_flow", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"}\n'
    "  ],\n"
    '  "result": boolean,\n'
    '  "reasoning": "Brief summary of overall UI quality judgment."\n'
    "}"
)

# ---------------------------------------------------------------------------
# User simulator quality rubric (4 criteria)
# ---------------------------------------------------------------------------

_USER_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating the quality of an AI-powered User Simulator in a "
    "multi-turn agent evaluation scenario.\n\n"
    "## CONTEXT\n"
    "In this setup, two LLMs interact: an AGENT (being tested) and a USER SIMULATOR (being "
    "evaluated here). The user simulator receives hidden instructions (SYSTEM→USER) that tell "
    "it what to ask the agent. Your job is to evaluate how well the user simulator performed "
    "its role — NOT how well the agent performed.\n\n"
    "## TRUST MODEL\n"
    "- The SYSTEM→USER instruction is the GROUND TRUTH for the user's behavior.\n"
    "- TRUSTED evidence: user tool calls, the SYSTEM→USER instruction, images, conversation "
    "history, and the User Available Tools list.\n"
    "- NOT the user's responsibility: the agent's execution quality, tool errors, or "
    "incorrect values extracted by the agent from images.\n\n"
    "## INPUT FORMAT\n"
    "You will receive:\n"
    "1. User Instruction — the hidden SYSTEM→USER instruction (ground truth).\n"
    "2. User Available Tools — the tools actually provided to the user in this scenario.\n"
    "3. Conversation — the full USER↔AGENT message thread.\n"
    "4. User Tool Calls — the user's tool calls in chronological order.\n"
    "5. Relevant Images — images available in the scenario.\n"
    "6. Scenario limits — max_messages budget and actual turn_count.\n"
    "7. Scenario metadata — challenge_type, require_disambiguation, num_user_rounds, "
    "image_arrival.\n\n"
    "## SCENARIO CONTEXT\n"
    "The scenario metadata describes the evaluation design:\n"
    "- **challenge_type**: How the user's intent evolves across rounds.\n"
    "  - `none`: Straightforward multi-round delegation. No corrections or changes.\n"
    "  - `error_correction`: The user intentionally gives wrong information in an early\n"
    "    round, then corrects it later. This is BY DESIGN — the wrong value is NOT a\n"
    "    consistency failure or request fidelity issue. If the agent proactively\n"
    "    corrects the error before the user's correction round, the user may either\n"
    "    deliver the scripted correction anyway (redundant but acceptable) OR\n"
    "    acknowledge the agent's fix and move on — both are valid behaviors.\n"
    "  - `goal_change`: The user changes their mind mid-conversation (e.g., 'delete the\n"
    "    note, send an email instead'). This is BY DESIGN — not a contradiction.\n"
    "  - `state_mutation`: Later rounds modify state created by earlier rounds.\n"
    "- **require_disambiguation**: When true, the script expects the agent to ask clarifying\n"
    "  questions. The user should WAIT for the agent to ask, then answer. If the agent does\n"
    "  not ask, the user is NOT required to force the disambiguation.\n"
    "- **num_user_rounds**: The total number of scripted rounds.\n"
    "- **image_arrival**: When images are delivered (`upfront`, `progressive`, `mixed`, `late`).\n"
    "- **max_messages / turn_count**: If turn_count equals max_messages, the conversation was\n"
    "  terminated by the framework, not by the user. Do NOT penalize the user for anything\n"
    "  related to the premature ending.\n\n"
    "## RUBRIC (4 criteria, each pass/fail)\n"
    "Evaluate each criterion independently:\n\n"
    "1. **request_fidelity**: Did the user deliver the complete intended task?\n"
    "   - The user must communicate ALL task-critical requests and details from the "
    "SYSTEM→USER instruction to the agent.\n"
    "   - Required clarifications must be provided when the conversation calls for them.\n"
    "   - Information may be revealed incrementally but must arrive in time for the agent "
    "to act.\n"
    "   - Evaluate ONLY against the SYSTEM→USER instruction. The user does not have access "
    "to any other evaluation criteria, so only hold them accountable for information in "
    "their instruction.\n"
    "   - Natural elaboration beyond the instruction is acceptable. When the agent asks "
    "follow-up questions, the user may add reasonable details — this is normal conversation, "
    "not a failure.\n"
    "   - IMPORTANT: Only judge whether the user COMMUNICATED the right requests — NOT whether "
    "the agent executed them correctly. If the user asked for the right thing but the agent "
    "made mistakes, that is the agent's fault.\n"
    "   - In visual tool-calling scenarios, the user is NOT expected to dictate exact values "
    "from images. Extracting information from images is the AGENT's responsibility.\n"
    "   - When the agent cannot complete the task, the user should NOT be penalized for "
    "accepting a reasonable fallback or ending the conversation.\n"
    "   - Compare each round's scripted Query against the actual USER→AGENT message. "
    "If the user drops key task details from the Query (e.g., sends images without "
    "specifying what to extract), this is a failure.\n"
    "   - Use the Round Query Delivery Check in the evidence. If a round shows "
    "'WARNING — actual message is N% the length of scripted Query', carefully "
    "compare the scripted Query and actual message to identify missing details. "
    "A short message that omits specific task details from the Query is a failure.\n"
    "   - HOWEVER: If a round's Query content appears in message i+N rather than "
    "message i (preceded by brief acknowledgments, quick conversational exchanges, "
    "or agent clarification responses), this is ACCEPTABLE — treat it as a PASS. "
    "Evaluate whether the Query content was delivered ANYWHERE in the conversation "
    "before the agent acts on it, not whether it arrived in a specific message slot. "
    "A short 'thanks' or 'got it' message followed by the full query in the next "
    "turn is normal human conversation, not a failure.\n"
    "   - If the user re-sends images or re-delivers a query the agent already processed "
    "successfully, this is a failure — the user is wasting turns with unscripted "
    "redundant requests.\n"
    "   - AGENT-PREEMPTED CORRECTIONS: In error_correction scenarios, if the agent "
    "independently corrects an error BEFORE the user's scripted correction round, "
    "the user is NOT required to re-deliver the now-redundant correction. The user "
    "acknowledging the agent's proactive fix (e.g., 'Good catch!') is correct "
    "behavior — treat this as a PASS for that round. Forcing a redundant correction "
    "would itself violate the 'no unscripted redundant requests' rule above.\n"
    "   - CRITICAL: Only the **Query** field in each round is a mandatory deliverable. "
    "The **Instructions** field describes expected agent behavior and how the user "
    "should REACT — it is NOT a list of mandatory user actions. Language like "
    "'the assistant should ask', 'the assistant will likely find', or 'when it "
    "asks' describes what the agent might do, not what the user must do. "
    "If the agent doesn't trigger the expected behavior, the user should simply "
    "proceed to the next round.\n"
    "   - If the agent fails (stuck in loops, empty responses, exhausts max_messages), "
    "do NOT penalize the user for undelivered later rounds. The user cannot "
    "deliver Round N+1 until the agent finishes Round N.\n"
    "   - The user is the SUPERVISOR who gives instructions. If the user poses questions "
    "that should come from the agent (e.g., 'Which contact should I use?', 'Could "
    "you clarify which file?'), or claims to BE the assistant, this is a role swap. "
    "This is a failure.\n"
    "   - Do NOT penalize the user for `end_conversation` timing. Whether or when "
    "the user called `end_conversation` is excluded from this evaluation — it is "
    "a framework mechanism not visible in the evidence. Never fail request_fidelity "
    "because the user 'did not end the conversation.'\n"
    "   - Pass: User communicated the task clearly per their instruction and collaborated "
    "reasonably. Fail: User failed to communicate requests from their instruction, provided "
    "values that contradict their instruction, or actively undermined the task.\n\n"
    "2. **conversational_naturalness**: Do the USER→AGENT messages sound like a real human?\n"
    "   - Casual, conversational tone. Concise messages (1-3 sentences typical).\n"
    "   - Natural multi-turn rhythm: clarifications, follow-ups, reactions to progress.\n"
    "   - Consistent persona across turns. No internal tool logic leaking into chat.\n"
    "   - IMPORTANT: Only evaluate USER→AGENT messages in the Conversation section.\n"
    "     Tool calls in the User Tool Calls section (USER→EXECUTION_ENVIRONMENT) are\n"
    "     a separate channel and are fine — do NOT penalize tool usage there.\n"
    "     However, if a USER→AGENT message contains tool-call syntax, function names,\n"
    "     or API calls (e.g., 'search_notes(...)', 'Notes_search_notes'), this IS a\n"
    "     failure — the user should communicate in plain natural language only in\n"
    "     their messages to the agent.\n"
    "   - For single-turn tasks, a single complete message is acceptable.\n"
    "   - Pass: Natural, human-like tone and flow. "
    "Fail: Robotic, templated, overly formal, or exposing control logic in chat.\n\n"
    "3. **grounded_consistency**: Is the user consistent and epistemically grounded?\n"
    "   - Facts, preferences, and constraints must remain stable across all turns.\n"
    "   - The user may only claim knowledge from: (a) the SYSTEM→USER instruction, and "
    "(b) what they observed in conversation.\n"
    "   - Natural elaboration is NOT a contradiction. If the user starts with a vague "
    "request ('save this recipe') and later adds details ('add it to Recipes folder'), "
    "that is normal conversational refinement, not inconsistency.\n"
    "   - Only flag contradictions where the user reverses a previously stated fact "
    "(e.g., 'Send to John' then later 'I said send to Sarah').\n"
    "   - The user should NOT claim knowledge of system state (e.g., listing database "
    "entries, note names, contact lists) unless that information was explicitly "
    "provided in the SYSTEM→USER instruction or revealed by the agent in "
    "conversation.\n"
    "   - Pass: Consistent facts and grounded claims. "
    "Fail: Reversed facts, hallucinated knowledge, or references to nonexistent events.\n\n"
    "4. **tool_channel_correctness**: Did the user use available tools correctly?\n"
    "   - Evaluate all tools in User Available Tools EXCEPT `end_conversation`. "
    "The `end_conversation` tool is handled by the framework and is NOT part "
    "of this evaluation.\n"
    "   - For image delivery: check whether images ACTUALLY ARRIVED by looking for "
    "[image_ids=N] annotations on USER→AGENT messages in the Conversation. "
    "If a round's script says [provide: image_N] and the corresponding "
    "USER→AGENT message has [image_ids=N], the images were delivered — "
    "regardless of whether the tool call appears in the User Tool Calls section.\n"
    "   - IMPORTANT: If the User Tool Calls section shows 'No user tool calls' but "
    "conversation messages DO have [image_ids=N] annotations, the images were "
    "delivered by the framework automatically. This is NOT a tool_channel_correctness "
    "failure. The presence of [image_ids=N] in the Conversation is the definitive "
    "evidence of image delivery.\n"
    "   - If a round requires images but the USER→AGENT message has NO [image_ids=N] "
    "annotation, the images were NOT delivered — even if the user's text says "
    "'here is the image.' This is a failure.\n"
    "   - Use the Image Delivery Verification section in the evidence to check "
    "whether all required images were delivered. If it says 'MISSING IDs [N]', "
    "this is a tool_channel_correctness failure.\n"
    "   - `ui_user_interact`: only expected if listed in User Available Tools.\n"
    "   - Pass: Required images delivered, available tools used correctly, no "
    "hallucinated actions. "
    "Fail: Required images missing, or user describes tool actions without calling them.\n\n"
    "## EVALUATION STEPS\n"
    "1. Read the SYSTEM→USER instruction. List every requirement the user was given.\n"
    "2. Trace through the conversation: was each requirement delivered to the agent?\n"
    "3. Check user messages for naturalness and persona consistency.\n"
    "4. Check for cross-turn contradictions or hallucinated knowledge.\n"
    "5. Check User Available Tools. Review tool calls: correct usage? valid arguments?\n"
    "6. ROLE CHECK: Verify the user always speaks as the supervisor giving instructions, "
    "never as the assistant asking for clarification or claiming to be an assistant. "
    "If the user says things like 'Which contact should I use?', 'Could you clarify "
    "which file?', or identifies itself as the assistant, this is a role swap — "
    "the user is acting as the assistant instead of the supervisor. This is a "
    "request_fidelity failure.\n"
    "7. For each of the 4 criteria, write a brief analysis, then determine pass/fail.\n"
    "8. Determine overall result: pass ONLY if ALL 4 criteria pass.\n\n"
    "## OUTPUT FORMAT\n"
    "Output a JSON object with no markdown formatting, no code blocks:\n"
    "{\n"
    '  "criteria_evaluation": [\n'
    '    {"criterion": "request_fidelity", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "conversational_naturalness", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "grounded_consistency", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"},\n'
    '    {"criterion": "tool_channel_correctness", "analysis": "brief reasoning", "pass": boolean, "evidence": "one sentence"}\n'
    "  ],\n"
    '  "result": boolean,\n'
    '  "reasoning": "Brief summary of overall user simulator quality judgment."\n'
    "}"
)

# ---------------------------------------------------------------------------
# UI-mode addendum for the user simulator judge
# ---------------------------------------------------------------------------

_USER_UI_ADDENDUM = (
    "\n\n## UI INTERACTION RULES\n"
    "This scenario uses interactive UI screens. The following rules supplement "
    "the criteria above.\n\n"
    "### request_fidelity addendum\n"
    "When Instructions say 'use `ui_user_interact` to click X' AND the "
    "Conversation contains a `<ui_state_server>` Interactive Elements block "
    "listing the matching element (or a functionally equivalent one), that "
    "interaction IS a mandatory action — the user must call `ui_user_interact`. "
    "However, text fallback is correct behavior (NOT a failure) in ANY of "
    "these cases:\n"
    "- The Conversation has NO `<ui_state_server>` Interactive Elements block "
    "(the agent did not render a UI or rendered one without interactive "
    "elements).\n"
    "- The `<ui_state_server>` block lists interactive elements that do NOT "
    "match the expected action (e.g., the Instructions say 'click Create "
    "Tasks' but the only buttons are 'View Document'). The user cannot click "
    "a button that does not exist.\n"
    "- The agent executed the action directly without presenting a UI for "
    "user confirmation. The user accepting the agent's completed action is "
    "correct — the user is not required to force a UI interaction that the "
    "agent bypassed.\n\n"
    "### tool_channel_correctness addendum\n"
    "When `ui_user_interact` is in User Available Tools, check the "
    "Conversation for `<ui_state_server>` Interactive Elements blocks from "
    "the agent. The user should use `ui_user_interact` ONLY when the agent "
    "renders interactive elements that match or are functionally equivalent "
    "to the action described in the Instructions. In all other cases (no UI "
    "rendered, no interactive elements, or mismatched elements), text fallback "
    "is correct — do NOT penalize the user for failing to interact with UI "
    "elements that were absent or inappropriate."
)


def get_user_system_prompt(enable_ui: bool = False) -> str:
    """Return the user judge system prompt, with UI addendum if needed."""
    if enable_ui:
        return _USER_SYSTEM_PROMPT + _USER_UI_ADDENDUM
    return _USER_SYSTEM_PROMPT
