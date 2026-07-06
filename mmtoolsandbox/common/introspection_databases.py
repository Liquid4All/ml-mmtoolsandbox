# Copyright © 2026 Apple Inc.

from __future__ import annotations

import copy
from enum import auto
from typing import Any

import polars as pl
from strenum import StrEnum


class IntrospectionDatabaseNamespace(StrEnum):
    """Namespace for introspection databases.

    Any field ending with "_json" must be JSON encoded.

    These databases can be used to store introspection data that
      - is useful to store as part of the `ExecutionContext`
      - but is not needed for milestone evaluation

    Examples could be raw LLM output, runtime measurements, etc.
    """

    # Database for miscellaneous introspection data. If there is a specific use case or
    # measurement type add a new entry.
    MISC = auto()


class IntrospectionDatabaseManager:
    db_schemas = {
        IntrospectionDatabaseNamespace.MISC: {
            "sandbox_message_index": pl.Int32,
            "description": pl.String,
            "content": pl.String,
        },
    }

    def __init__(self) -> None:
        self._dbs = {
            namespace: pl.DataFrame(schema=self.db_schemas[namespace])
            for namespace in self.db_schemas
        }

    def add(
        self,
        namespace: IntrospectionDatabaseNamespace,
        rows: list[dict[str, Any]],
        sandbox_message_index: int,
    ) -> None:
        """Add multiple rows to an introspection database.

        Args:
            namespace:  Database namespace
            rows:  List of rows to be added. Each item must be a dict mapping from
                   column name to value.
            sandbox_message_index:  The value for the MMToolSandbox message index. The
                                    `ExecutionContext` contains a database storing the
                                    MMToolSandbox messages (i.e. the whole conversation).
                                    Rows in the introspection databases are associated
                                    with these MMToolSandbox messages through its index.

        Raises:
            KeyError:   When provided column names in rows do not match the schema.
        """
        # Check that the given rows match the schema.
        rows_column_names = {x for row in rows for x in row.keys()}
        schema_column_names = set(self.db_schemas[namespace].keys())
        extra_column_names = rows_column_names - schema_column_names
        if len(extra_column_names) > 0:
            raise KeyError(
                f"Failed to add rows to the {namespace} introspection database. The "
                "following column names do not exist in the schema: "
                f"{sorted(extra_column_names)}\nThe schema defines these columns: "
                f"{sorted(schema_column_names)}."
            )

        # Add sandbox_message_index to rows
        rows = copy.deepcopy(rows)
        for row in rows:
            row["sandbox_message_index"] = sandbox_message_index

        df_to_add = pl.DataFrame(rows, schema=self.db_schemas[namespace])
        self._dbs[namespace] = self._dbs[namespace].vstack(df_to_add)

    def get_database(self, name: IntrospectionDatabaseNamespace) -> pl.DataFrame:
        return self._dbs[name]

    def get_populated_databases(self) -> list[IntrospectionDatabaseNamespace]:
        return [namespace for namespace, db in self._dbs.items() if not db.is_empty()]

    def serialize_to_dict(self) -> dict[str, Any]:
        # We skip empty databases since the introspection manager maintains the superset
        # of all existing introspection databases, but only some of them might be in use
        # at any given time.
        return {
            namespace: db.to_dicts()
            for namespace, db in self._dbs.items()
            if not db.is_empty()
        }

    @classmethod
    def deserialize_from_dict(
        cls, name_to_rows: dict[str, Any]
    ) -> IntrospectionDatabaseManager:
        dbs: dict[IntrospectionDatabaseNamespace, pl.DataFrame] = {}
        # When `name_to_rows` was read from a JSON file or string the introspection
        # database namespace will be a string instead of the `IntrospectionDatabase`
        # enum type. If it is already an enum type then converting it to the enum below
        # should be a no-op.
        for namespace_str, rows in name_to_rows.items():
            namespace = IntrospectionDatabaseNamespace(namespace_str)
            dbs[namespace] = pl.from_dicts(
                rows, schema=cls.db_schemas[IntrospectionDatabaseNamespace(namespace)]
            )

        manager = cls()
        manager._dbs = dbs
        return manager

    def get_database_as_json_object(self, name: IntrospectionDatabaseNamespace) -> Any:
        """Create introspection JSON object."""
        db = self.get_database(name)
        json_object = [row for row in db.iter_rows(named=True)]
        return json_object


def get_introspection_pretty_print_file_name(
    db_name: IntrospectionDatabaseNamespace,
) -> str:
    """Get the file name to use for the pretty print of the introspection database."""
    return f"introspection_{db_name}_pretty_print.txt"


def get_introspection_json_file_name(
    db_name: IntrospectionDatabaseNamespace,
) -> str:
    """Get the file name to use for the JSON dump of the introspection database."""
    return f"introspection_{db_name}.json"
