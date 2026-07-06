# Copyright © 2026 Apple Inc.

"""Validation utilities for common data types"""

from __future__ import annotations

import collections.abc
import datetime
import inspect
from enum import Enum
from functools import wraps
from types import UnionType
from typing import (
    Any,
    Callable,
    Literal,
    NoReturn,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

# `ccy` provides type hints for Python 3.10, but not for 3.9 and 3.11. We allow the
# unused-ignore in case someone is using Python 3.10.
import ccy  # type: ignore[import-untyped,unused-ignore]
import iso639
import iso3166
import phonenumbers

from mmtoolsandbox.common.i18n import Locale
from mmtoolsandbox.common.utils import NOT_GIVEN, NotGiven

T = TypeVar("T")
Numeric = TypeVar("Numeric", int, float)


def validate_currency_code(currency_code: str) -> Any:
    """Validation for 3 letter currency code conforming to ISO 4217

    Args:
        currency_code:  String to be validated as currency code

    Returns:

    Raises:
        ValueError: When currency_code is not a known currency code

    """
    try:
        ccy.currency(currency_code)
    except KeyError:
        raise ValueError(
            f"{currency_code} is not a known 3 letter ISO 4217 currency code"
        )


def validate_type(value: Any, value_name: str, expected_type: T) -> None:
    """Validate that `value` is of type `expected_type`.

    Args:
      value:         Value to validate the type of.
      value_name:    Name of value to show in error message.
      expected_type: Expected type of value.

    Raises:
      TypeError if `value` is not of the expected type(s).
    """
    expected_types = set(
        get_args(expected_type)
        if get_origin(expected_type) in (Union, UnionType)
        else (expected_type,)
    )
    expected_types = {get_origin(t) or t for t in expected_types}
    if type(value) in expected_types:
        return

    if collections.abc.Sequence in expected_types:
        # typing.Sequence seem to be an alias to collections.abc.Sequence
        raise NotImplementedError("Sequence not supported. Use `list` type instead.")

    if get_origin(expected_type) is Literal:
        if value in get_args(expected_type):
            return
        else:
            raise TypeError(
                f"Enum Parameter '{value_name}' is not one of the required values {get_args(expected_type)}."
            )

    # Handle enum
    if inspect.isclass(expected_type) and issubclass(expected_type, Enum):
        if value in list(expected_type):
            return
        else:
            raise TypeError(
                f"Enum Parameter '{value_name}' is not one of the required values {tuple(str(x) for x in expected_type)}."
            )

    # We allow the same upcasting that occurs naturally in Python (e.g. upcasting an
    # `int` to a `float` when performing an addition). Also based on PEP 484:
    #    "when an argument is annotated as having type float, an argument of type "
    #    "int is acceptable;"
    # , see https://peps.python.org/pep-0484/#the-numeric-tower .
    if isinstance(value, int) and float in expected_types:
        return

    # Booleans are a subclass of integers.
    if isinstance(value, bool) and (int in expected_types or float in expected_types):
        return

    raise_type_error(value_name=value_name, value=value, expected_type=expected_type)


def raise_type_error(value_name: str, value: Any, expected_type: Any) -> NoReturn:
    """Raised a TypeError with a custom error message."""
    raise TypeError(
        f"Parameter '{value_name}' is of type '{type(value)}', but expected "
        f"'{expected_type}'."
    )


def typechecked(function: Callable[..., T]) -> Callable[..., T]:
    """Validate the parameter types of a function.

    Args:
      function: Function to check types of.
    Raises:
      TypeError if any parameter is not of the expected type(s).
    """

    @wraps(function)
    def typechecker(*args: Any, **kwargs: Any) -> Any:
        params = inspect.signature(function).parameters
        # Check positional arguments.
        for arg, param in zip(args, params.values()):
            validate_type(arg, param.name, param.annotation)
        # Check keyword arguments.
        for name, kwarg in kwargs.items():
            param = params[name]
            validate_type(kwarg, param.name, param.annotation)
        return function(*args, **kwargs)

    return typechecker


def validate_range(
    value: Numeric,
    value_name: str,
    *,
    min_val: Numeric | None = None,
    max_val: Numeric | None = None,
) -> None:
    """Validate that `value` is between the optional min and max values (inclusive).

    Args:
      value:      Value to validate the range of.
      value_name: Name of value to show in error message.
      min_val:    Optional minimum of value range to validate.
      max_val:    Optional maximum of value range to validate.
    Raises:
      ValueError if `value` is outside the expected range.
    """
    if min_val is not None and value < min_val:
        raise ValueError(
            f"Parameter '{value_name}' is smaller than its valid range minimum of {min_val}."
        )
    if max_val is not None and value > max_val:
        raise ValueError(
            f"Parameter '{value_name}' is larger than its valid range maximum of {max_val}."
        )


def validate_type_range(
    value: Any,
    value_name: str,
    expected_type: T,
    *,
    min_val: Numeric | None = None,
    max_val: Numeric | None = None,
) -> None:
    """Validate parameter type and range.

    Args:
      value:         Value to validate the type and range of.
      value_name:    Name of value to show in error message.
      expected_type: Expected type of value.
      min_val:       Optional minimum of value range to validate.
      max_val:       Optional maximum of value range to validate.
    Raises:
      TypeError if `value` is not of the expected type(s).
      ValueError if `value` is outside the expected range.
    """
    validate_type(value, value_name, expected_type)
    # Don't check the range if the value is not numeric (e.g. NotGiven, None).
    if isinstance(value, (float, int)):
        validate_range(
            cast(Numeric, value), value_name, min_val=min_val, max_val=max_val
        )


def validate_timestamp(timestamp: T, name: str, expected_type: T) -> None:
    """Validate the range of timestamps so a unit mistake is caught (s vs ms/us/ns).

    Args:
        timestamp: Timestamp to validate.
        timestamp_name: Name of variable to validate.
        expected_type: Expected type of timestamp.
    Raises:
        TypeError if timestamp is not of the expected type(s).
        ValueError if timestamp is outside the expected range.
    """
    TS_JAN_1_1980 = 315529200.0  # in seconds
    TS_JAN_1_2050 = 2524604400.0  # in seconds
    validate_type_range(
        timestamp, name, expected_type, min_val=TS_JAN_1_1980, max_val=TS_JAN_1_2050
    )


def validate_latitude(latitude: T, name: str, expected_type: T) -> None:
    """Validate the type and range of latitude in degrees.

    Args:
        latitude: Latitude value to validate.
        name: Name of variable to validate.
        expected_type: Expected type of latitude.
    Raises:
        TypeError if latitude is not of the expected type(s).
        ValueError if latitude is outside the expected range.
    """
    MIN_LAT = -90.0  # degrees
    MAX_LAT = 90.0  # degrees
    validate_type_range(latitude, name, expected_type, min_val=MIN_LAT, max_val=MAX_LAT)


def validate_longitude(longitude: T, name: str, expected_type: T) -> None:
    """Validate the type and range of longitude in degrees.

    Args:
        longitude: Longitude value to validate.
        name: Name of variable to validate.
        expected_type: Expected type of longitude.
    Raises:
      TypeError if longitude is not of the expected type(s).
      ValueError if longitude is outside the expected range.
    """
    MIN_LON = -180.0  # degrees
    MAX_LON = 180.0  # degrees
    validate_type_range(
        longitude, name, expected_type, min_val=MIN_LON, max_val=MAX_LON
    )


def validate_phone_number(phone_number: str | NotGiven) -> None:
    """Validate if a string is a phone number.

    Args:
        phone_number: Phone number to be validated. If it is NOT_GIVEN, validation is ignored

    Raises:
        NumberParseException:  If phone number cannot be parsed.
    """
    if phone_number is not NOT_GIVEN:
        assert not isinstance(phone_number, NotGiven)
        phonenumbers.parse(phone_number)


def validate_iso_639_set_1_language_code(language_code: str) -> None:
    """Validate if a string is an ISO 639 Set 1 language code.

    Args:
        language_code: Code to be validated

    Raises:
        LanguageNotFoundError if code is not an ISO 639 Set 1 language code.
    """
    iso639.Language.from_part1(language_code)


def validate_iso3166_alpha2_country_code(country_code: str) -> None:
    """Validate if a string is an ISO 3166 alpha-2 country code.

    Args:
        country_code: Code to be validated

    Raises:
        ValueError if code is not an ISO 3166 alpha-2 country code.
    """
    if country_code.upper() not in iso3166.countries_by_alpha2:
        raise ValueError(f"{country_code} is not an ISO 3166 alpha-2 country code.")


def validate_iso_8601_date_time_str(datetime_str: str) -> None:
    """Validate if a string is an ISO 8601 datetime string.

    Args:
        datetime_str:   String to be validated

    Raises:
        ValueError if string is not an ISO 8601 datetime string.
    """
    datetime.datetime.fromisoformat(datetime_str)


def validate_iso_8601_date_time_str_with_timezone(datetime_str: str) -> None:
    """Validate if a string is an ISO 8601 datetime string with UTC timezone offset.

    Args:
        datetime_str:   String to be validated

    Raises:
        ValueError if string is not an ISO 8601 datetime string with UTC timezone offset.
    """
    current_datetime = datetime.datetime.fromisoformat(datetime_str)
    if not current_datetime.tzinfo:
        raise ValueError(
            f"{datetime_str} is not an ISO 8601 datetime string with UTC timezone offset. "
            f"Example format: 2024-12-09T15:00-08:00"
        )


def validate_locale(locale_str: str) -> None:
    """Validate if a locale string is a recognized locale defined in mmtoolsandbox.common.i18n.Locale

    Args:
        locale_str: Locale string to be validated.
    """
    locale_list = list(locale.name for locale in Locale)
    if locale_str not in locale_list:
        raise ValueError(
            f"{locale_str} is not a recognized locale. "
            f"Possible options are {locale_list}"
        )
