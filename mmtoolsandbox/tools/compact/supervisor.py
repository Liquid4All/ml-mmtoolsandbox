# Copyright © 2026 Apple Inc.

# mypy: disable-error-code="no-any-return"

"""COMPACT Supervisor tools — merges show endpoints (4->1)."""

from __future__ import annotations

from typing import Any, Literal

from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.tools.compact import mark_compact_tools_absorbed_by


def _get(name: str) -> Any:
    import mmtoolsandbox.tools.appworld.supervisor as m

    return getattr(m, name)


# ---------------------------------------------------------------------------
# Strategy 5: Supervisor show (4 -> 1)
# ---------------------------------------------------------------------------


@register_as_tool(
    toolboxes={DatasetName.COMPACT},
    database_namespaces=set(),
    visible_to=(RoleType.AGENT,),
)
def supervisor_show(
    entity_type: Literal["profile", "addresses", "payment_cards", "passwords"],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Show supervisor's profile, addresses, payment cards, or account passwords.

    Entity types:
        profile: Show your supervisor's profile information.
        addresses: Show your supervisor's addresses.
        payment_cards: Show your supervisor's payment cards.
        passwords: Show your supervisor's app account passwords.

    Args:
        entity_type: Which supervisor information to retrieve.

    Returns:
        Supervisor information (dict for profile, list for others).

    Raises:
        ConnectionError: If network is unavailable.
    """
    func_map = {
        "profile": "supervisor_show_profile",
        "addresses": "supervisor_show_addresses",
        "payment_cards": "supervisor_show_payment_cards",
        "passwords": "supervisor_show_account_passwords",
    }
    return _get(func_map[entity_type])()


mark_compact_tools_absorbed_by(
    "supervisor_show",
    "supervisor_show_profile",
    "supervisor_show_addresses",
    "supervisor_show_payment_cards",
    "supervisor_show_account_passwords",
)
