# Copyright © 2026 Apple Inc.

"""Utilities for dataset loading"""

import requests


def assert_service_reachable(url: str) -> None:
    """Assert that a URL is reachable. Necessary precondition for some dataset / toolboxes.

    Args:
        url:    URL link.
    """
    # We won't know exactly which request would be defined, so just try all of them
    for request_method in [
        requests.head,
        requests.options,
        requests.get,
        requests.post,
    ]:
        try:
            response = request_method(url, timeout=2)  # type: ignore[operator]
            if response.ok:
                return
        except Exception:
            pass
    raise ConnectionError(
        f"Unable to reach {url}. Please double check your VPN connection or Secret."
    )
