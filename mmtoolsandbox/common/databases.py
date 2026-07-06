# Copyright © 2026 Apple Inc.

from enum import auto

from strenum import StrEnum


class DatabaseNamespace(StrEnum):
    """Namespace for each database."""

    SANDBOX = auto()
    IMAGE = auto()  # Image storage and management for multimodal conversations.
    SETTING = auto()
    CONTACT = auto()
    MESSAGING = auto()
    REMINDER = auto()
    NOTES = auto()
    CALENDARS = auto()  # Represents calendar categories instead of events.
    CALENDAR_EVENTS = auto()
    CALENDAR_EVENTS_RECURRENCE_STAGING = (
        auto()
    )  # Staging database for recurring calendar events.
