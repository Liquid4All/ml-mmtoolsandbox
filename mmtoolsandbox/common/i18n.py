# Copyright © 2026 Apple Inc.

from __future__ import annotations

import datetime
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    TypeAlias,
    TypeVar,
    cast,
)

DEFAULT_SCOPE: str = "default"

LocalizableType: TypeAlias = str | int | float | bool

T = TypeVar("T", bound=LocalizableType)


HolidayPredicate: TypeAlias = Callable[[tuple[float, datetime.date, str]], bool]


@dataclass(frozen=True)
class HolidayNamePredicate:
    """Predicate for filtering holidays by name.

    Args:
        pattern: String pattern to match against holiday names
        exclude: If True, drop matching holidays from the output; if False, include only matching holidays
        regex:   If True, treat pattern as a regular expression; if False, use exact matching
    """

    pattern: str
    exclude: bool = True
    regex: bool = False

    def __call__(self, holiday: tuple[float, datetime.date, str]) -> bool:
        _, _, name = holiday

        if re.match(self.pattern, name) if self.regex else self.pattern == name:
            return not self.exclude

        return self.exclude


@dataclass(frozen=True, kw_only=True)
class LocaleInfo:
    """
    Contains locale-specific information, including country and language.

    Attributes:
        language:                           ISO 639-1 country code
        country:                            ISO 3166-1 alpha-2 country code
        _holidays_language_code:            Language code for the holidays library, see https://holidays.readthedocs.io/en/latest/#available-countries.
                                            Defaults to language
        use_char_tokenization_for_rouge_l:  Whether to use character-level tokenization for rouge-L string similarity comparison. If False, use RougeScorer default which splits on whitespace
    """

    language: str
    country: str
    _holidays_language_code: str | None = None
    use_char_tokenization_for_rouge_l: bool = False
    filter_holidays_predicate: HolidayPredicate | None = None

    @property
    def holidays_language_code(self) -> str:
        return self._holidays_language_code or self.language


class Locale(Enum):
    en_US = LocaleInfo(
        language="en",
        country="US",
    )
    ar_SA = LocaleInfo(
        language="ar",
        country="SA",
    )
    da_DK = LocaleInfo(
        language="da",
        country="DK",
    )
    de_DE = LocaleInfo(
        language="de",
        country="DE",
    )
    es_ES = LocaleInfo(
        language="es",
        country="ES",
    )
    es_MX = LocaleInfo(
        language="es",
        country="MX",
    )
    fi_FI = LocaleInfo(
        language="fi",
        country="FI",
    )
    fr_FR = LocaleInfo(
        language="fr",
        country="FR",
    )
    he_IL = LocaleInfo(
        language="he",
        country="IL",
    )
    it_IT = LocaleInfo(
        language="it",
        country="IT",
    )
    ja_JP = LocaleInfo(
        language="ja",
        country="JP",
        use_char_tokenization_for_rouge_l=True,
    )
    ko_KR = LocaleInfo(
        language="ko",
        country="KR",
        use_char_tokenization_for_rouge_l=True,
    )
    ms_MY = LocaleInfo(
        language="ms",
        country="MY",
        _holidays_language_code="ms_MY",
    )
    nb_NO = LocaleInfo(
        language="nb",
        country="NO",
        _holidays_language_code="no",
    )
    nl_NL = LocaleInfo(
        language="nl",
        country="NL",
    )
    pt_BR = LocaleInfo(
        language="pt",
        country="BR",
        _holidays_language_code="pt_BR",
    )
    ru_RU = LocaleInfo(
        language="ru",
        country="RU",
    )
    sv_SE = LocaleInfo(
        language="sv",
        country="SE",
        filter_holidays_predicate=HolidayNamePredicate(pattern="Söndag"),
    )
    th_TH = LocaleInfo(
        language="th",
        country="TH",
        use_char_tokenization_for_rouge_l=True,
    )
    tr_TR = LocaleInfo(
        language="tr",
        country="TR",
    )
    vi_VN = LocaleInfo(
        language="vi",
        country="VN",
    )
    zh_CN = LocaleInfo(
        language="zh",
        country="CN",
        _holidays_language_code="zh_CN",
        use_char_tokenization_for_rouge_l=True,
    )
    zh_HK = LocaleInfo(
        language="zh",
        country="HK",
        _holidays_language_code="zh_HK",
        use_char_tokenization_for_rouge_l=True,
    )
    zh_TW = LocaleInfo(
        language="zh",
        country="TW",
        _holidays_language_code="zh_TW",
        use_char_tokenization_for_rouge_l=True,
    )


@dataclass(frozen=True)
class Localizer:
    """Localizer is a pattern to localize values."""

    localized_data: dict[str, dict[str, LocalizableType]] | None = None

    def __call__(self, localizable_val: T, *, scope: str = DEFAULT_SCOPE) -> T:
        """
        Args:
            localizable_val:  The localizable value for the original language
            scope: The scope in which to look up the localized value.
                   Scopes enable localizing values differently based on the context.
                   If the value is not found in the specified scope, the default scope is used as a fallback.

        Returns:
            The localized value.
        """
        if self.localized_data is None:
            return localizable_val

        localized_values = self.localized_data[str(localizable_val)]

        if scope in localized_values:
            localized_value = localized_values[scope]
        else:
            localized_value = localized_values[DEFAULT_SCOPE]

        # NOTE: casting the return value from LocalizableType to T is a hack
        # to ensure that the return type is the same as the input type.
        # Generics don't seem to allow attributes of type Optional[dict[str, dict[str, T]]]
        # where T is not a concrete type
        return cast(T, localized_value)

    @classmethod
    def from_path(
        cls,
        data_file_path: Path | str,
        locale: Locale,
        lookup_key_locale: Locale = Locale.en_US,
    ) -> "Localizer":
        """Initialize localizer using data from the given path for the given locale.

        Args:
            data_file_path: File path to load scenarios data from.
            locale: Locale for which to load scenarios data.
            lookup_key_locale: Locale of the strings used as lookup keys. Defaults to en_US.

        Returns:
            Localizer instance that can be used to localize values using the data loaded from the given path.
        """
        if locale == lookup_key_locale:
            # return a dummy localizer which acts as an identity function
            # that returns the lookup key as the localized value
            return Localizer()

        with open(data_file_path) as stream:
            scenarios_data: dict[str, Any] = json.load(stream)

        # Check if locale has any translations
        has_translations = any(
            locale.name in locale_values
            for scope_values in scenarios_data.values()
            for locale_values in scope_values.values()
        )

        if not has_translations:
            # Return DefaultLocalizer for unsupported locales (fallback to English)
            logging.warning(
                f"No translations found for locale '{locale.name}'. "
                f"Falling back to English (en_US) for localized data."
            )
            return DefaultLocalizer

        localised_scenarios_data: dict[str, Any] = dict()
        for original_val, scope_values in scenarios_data.items():
            localised_scenarios_data[original_val] = dict()
            for scope, locale_values in scope_values.items():
                if locale.name in locale_values.keys():
                    localised_scenarios_data[original_val][scope] = locale_values[
                        locale.name
                    ]

        return cls(localised_scenarios_data)


DefaultLocalizer: Localizer = Localizer()


class CharacterTokenizer:
    """Character-level tokenizer that splits text into individual characters.

    This tokenizer is designed for languages where whitespace-based tokenization
    doesn't work well, particularly CJK (Chinese, Japanese, Korean) languages where
    words are not separated by spaces. It normalizes text to lowercase and treats
    each character as a separate token.
    """

    def tokenize(self, text: str) -> list[str]:
        return list(text.lower())


class WhitespaceTokenizer:
    """Whitespace-based tokenizer similar to rouge_scorer's whitespace tokenizer.

    This tokenizer splits text on whitespace boundaries while preserving Unicode
    word characters. It normalizes text to lowercase and replaces non-alphanumeric
    characters with spaces, then splits on whitespace.

    Key difference from rouge_scorer: preserves non-ASCII word characters
    (like accented letters: é, ü, ä, etc.) instead of replacing them with spaces.
    This makes it suitable for languages with diacritics and special characters.
    """

    _SPACES_RE = re.compile(r"\s+")
    _NON_ALPHANUM_RE = re.compile(r"\W+")

    def tokenize(self, text: str) -> list[str]:
        text = text.lower()

        text = self._NON_ALPHANUM_RE.sub(" ", text)
        tokens = self._SPACES_RE.split(text)

        return tokens


def get_rouge_scorer_kwargs(locale: Locale) -> dict[str, Any]:
    if locale == Locale.en_US:
        scorer_kwargs = {"use_stemmer": False, "tokenizer": WhitespaceTokenizer()}
    elif locale.value.use_char_tokenization_for_rouge_l:
        scorer_kwargs = {"use_stemmer": False, "tokenizer": CharacterTokenizer()}
    else:
        scorer_kwargs = {"use_stemmer": False, "tokenizer": WhitespaceTokenizer()}

    return scorer_kwargs
