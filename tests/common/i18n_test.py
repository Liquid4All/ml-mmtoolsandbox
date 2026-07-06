# Copyright © 2026 Apple Inc.

from typing import cast

import holidays
import pytest

from mmtoolsandbox.common.constants import REPO_DIR
from mmtoolsandbox.common.i18n import (
    HolidayNamePredicate,
    Locale,
    Localizer,
)
from mmtoolsandbox.common.validators import (
    validate_iso3166_alpha2_country_code,
    validate_iso_639_set_1_language_code,
)

MOCK_LOCALIZED_DATA_FILE = (
    REPO_DIR / "tests" / "common" / "i18n_test_data" / "localized_test_data.json"
)


@pytest.fixture(params=["en_US", "de_DE", "es_ES"])
def locale(request: pytest.FixtureRequest) -> str:
    return cast(str, request.param)


@pytest.fixture
def localizer(locale: str) -> Localizer:
    return Localizer.from_path(MOCK_LOCALIZED_DATA_FILE, Locale[locale])


EXPECTED_OUTPUTS = {
    "de_DE": {
        "a common string": "eine gemeinsame Zeichenfolge",
        -122.009102: -122.009103,
        42: 43,
        True: False,
    },
    "es_ES": {
        "a common string": "una cadena común",
        -122.009102: -122.009104,
        42: 44,
        True: True,
    },
    "en_US": {
        "a common string": "a common string",
        -122.009102: -122.009102,
        42: 42,
        True: True,
    },
}


def test_locale_info(locale: str) -> None:
    locale_info = Locale[locale].value
    validate_iso_639_set_1_language_code(locale_info.language)
    validate_iso3166_alpha2_country_code(locale_info.country)


@pytest.mark.parametrize("key", ["a common string", -122.009102, 42, True])
def test_localize(localizer: Localizer, locale: str, key: str) -> None:
    assert localizer(key) == EXPECTED_OUTPUTS[locale][key]


def test_localize_with_scope() -> None:
    localizer = Localizer.from_path(MOCK_LOCALIZED_DATA_FILE, locale=Locale.de_DE)

    localized_str = localizer("a common string", scope="some_other_scope")
    expected_str = "eine übliche Zeichenfolge"
    assert localized_str == expected_str

    localized_str = localizer("a common string", scope="scope_does_not_exist")
    expected_str = "eine gemeinsame Zeichenfolge"  # string from the default scope
    assert localized_str == expected_str


def test_filter_holidays_by_name_predicate() -> None:
    test_holidays = [
        (
            -1,  # dummy similarity value, not used
            date,
            name,
        )
        for date, name in holidays.country_holidays(
            country="SE",
            language="sv",
            years=2025,
        ).items()
    ]

    # Verify Söndag is in the results
    assert [h for h in test_holidays if "Söndag" == h[2]]

    # Test exclude=True - should filter OUT "Söndag" holidays
    predicate = HolidayNamePredicate("Söndag", exclude=True)
    filtered_exclude = [h for h in test_holidays if predicate(h)]
    assert all("Söndag" != h[2] for h in filtered_exclude)

    # Test exclude=False - should filter IN only "Söndag" holidays
    predicate = HolidayNamePredicate("Söndag", exclude=False)
    filtered_include = [h for h in test_holidays if predicate(h)]
    assert all("Söndag" == h[2] for h in filtered_include)

    # Test regex pattern - exclude holidays starting with "S"
    predicate = HolidayNamePredicate(r"S.*", exclude=True, regex=True)
    filtered_regex = [h for h in test_holidays if predicate(h)]
    assert all(not h[2].startswith("S") for h in filtered_regex)
