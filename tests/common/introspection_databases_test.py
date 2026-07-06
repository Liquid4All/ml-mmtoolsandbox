# Copyright © 2026 Apple Inc.

import json

import polars as pl

from mmtoolsandbox.common.introspection_databases import (
    IntrospectionDatabaseManager,
    IntrospectionDatabaseNamespace,
)


def test_introspection_database_manager() -> None:
    manager = IntrospectionDatabaseManager()

    # The databases are initially empty so `get_populated_databases` should return
    # nothing.
    assert 0 == len(manager.get_populated_databases())

    # Now add some data and check that the database has been populated.
    manager.add(
        IntrospectionDatabaseNamespace.MISC,
        rows=[{"description": "Foo", "content": "Bar"}],
        sandbox_message_index=123,
    )
    assert [IntrospectionDatabaseNamespace.MISC] == manager.get_populated_databases()
    df = manager.get_database(IntrospectionDatabaseNamespace.MISC)
    assert df.equals(
        pl.DataFrame(
            {"sandbox_message_index": 123, "description": "Foo", "content": "Bar"}
        )
    )


def test_serialization_roundtrip() -> None:
    manager = IntrospectionDatabaseManager()
    manager.add(
        IntrospectionDatabaseNamespace.MISC,
        rows=[{"description": "Foo", "content": "Bar"}],
        sandbox_message_index=123,
    )
    serialized_dict = manager.serialize_to_dict()
    reconstructed_manager = IntrospectionDatabaseManager.deserialize_from_dict(
        serialized_dict
    )
    assert serialized_dict == reconstructed_manager.serialize_to_dict()

    # Also check that deserialization works when converting the dict to JSON first.
    serialized_json = json.dumps(serialized_dict)
    reconstructed_manager = IntrospectionDatabaseManager.deserialize_from_dict(
        json.loads(serialized_json)
    )
    assert serialized_dict == reconstructed_manager.serialize_to_dict()
