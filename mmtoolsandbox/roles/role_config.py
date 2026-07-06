# Copyright © 2026 Apple Inc.

from argparse import ArgumentParser, Namespace
from typing import Protocol, Type, runtime_checkable

from mmtoolsandbox.roles.base_role import BaseRole


class RoleConfig:
    """Roles can implement this config for passing the parsed CLI args to the init."""


@runtime_checkable
class ConfigMixin(Protocol):
    @classmethod
    def from_config(cls, config: RoleConfig) -> BaseRole:
        """Instantiate BaseRole from the agent-specific RoleConfig."""

    @classmethod
    def extend_cli_parser(cls, parser: ArgumentParser) -> None:
        """Allow BaseRole to add new CLI arguments to the given parser."""

    @classmethod
    def validate_cli_args(
        cls, args: Namespace, agent_type: Type[BaseRole], user_type: Type[BaseRole]
    ) -> None:
        """Allow BaseRole to validate CLI arguments it has set."""

    @classmethod
    def make_config(cls, args: Namespace) -> RoleConfig:
        """Create a RoleConfig object from the CLI arguments."""
