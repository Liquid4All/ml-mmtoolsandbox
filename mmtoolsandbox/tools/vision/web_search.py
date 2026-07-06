# Copyright © 2026 Apple Inc.

"""
A collection of tools which simulates common functions used for web search domain.
"""

import json
import os
from typing import Any, cast

import requests
from typeguard import typechecked

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.common.execution_context import RoleType
from mmtoolsandbox.common.utils import register_as_tool
from mmtoolsandbox.toolbox.names import ToolboxName
from mmtoolsandbox.tools.tool_sandbox.setting import get_wifi_status


@register_as_tool(
    toolboxes={ToolboxName.FULL},
    visible_to=(RoleType.AGENT,),
    database_namespaces={DatabaseNamespace.IMAGE},
)
@typechecked
def web_search_serper(
    query: str,
) -> list[dict[str, Any]]:
    """Performs real time websearch based on textual query using Google Search via Serper API.

    Similar to google search. Returns a list of search results, including a short summary snippet, webpage url etc.

    Args:
        query:  Concise search query targeted for a search engine.

    Returns:
        A list of search results, including a short summary snippet, webpage url etc.
    """
    if not get_wifi_status():
        raise ConnectionError("Wifi is not enabled")
    if "SERPER_API_KEY" not in os.environ:
        raise PermissionError(
            "Please provide 'SERPER_API_KEY' in environment variable. "
            "You can obtain an API key from https://serper.dev/"
        )

    proxies = None
    if "EGRESS_HTTP_PROXY" in os.environ and "EGRESS_HTTPS_PROXY" in os.environ:
        proxies = {
            "http": os.environ["EGRESS_HTTP_PROXY"],
            "https": os.environ["EGRESS_HTTPS_PROXY"],
        }

    response = requests.post(
        url="https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json",
        },
        data=json.dumps({"q": query}),
        timeout=20,
        proxies=proxies,
    )
    response_json = response.json()

    try:
        results = response_json["organic"]
    except KeyError:
        raise requests.RequestException(response_json)

    return cast(list[dict[str, Any]], results)
