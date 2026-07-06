# Copyright © 2026 Apple Inc.

"""Helper functions to scan available registered toolboxes, dump domains and available
tools into json files.

Example:
```
python3 mmtoolsandbox/toolbox/scan_registered_toolbox.py --toolboxes FULL --output_file domain_descriptions.json
```

To find registered toolbox names, see here: mmtoolsandbox/toolbox/names.py
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import defaultdict
from datetime import datetime
from typing import (
    Any,
    Callable,
)

from mmtoolsandbox.common.databases import DatabaseNamespace
from mmtoolsandbox.datasets.names import DatasetName
from mmtoolsandbox.toolbox.loading import load_toolbox
from mmtoolsandbox.toolbox.names import ToolboxName


def get_tool_signature(tool: Callable[..., Any]) -> str:
    """Format tool information as a function signature string.

    Args:
        tool: The tool function to inspect.

    Returns:
        A string representation of the tool's signature, including the docstring.
    """
    name = tool.__name__
    try:
        signature = str(inspect.signature(tool))
    except ValueError:
        # Fallback if signature cannot be inspected
        signature = "()"

    docstring = inspect.getdoc(tool) or ""
    # Indent docstring
    indented_docstring = "\n".join(
        f"    {line}" if line else "" for line in docstring.splitlines()
    )

    return f'def {name}{signature}:\n    """{indented_docstring}"""'


def get_tool_description(tool: Callable[..., Any]) -> str:
    """Extract the first paragraph of the docstring as the description.

    Args:
        tool: The tool function to inspect.

    Returns:
        The first paragraph of the tool's docstring, or a default description if none exists.
    """
    docstring = inspect.getdoc(tool) or ""
    if not docstring:
        return f"Use the {tool.__name__} tool."

    # Split by double newlines to get paragraphs
    paragraphs = docstring.strip().split("\n\n")
    # Return the first paragraph, replacing single newlines with spaces
    return paragraphs[0].replace("\n", " ")


def compute_signature_hash(signatures: list[str]) -> str:
    """Compute hash of function signatures for cache validation.

    Args:
        signatures: A list of function signature strings.

    Returns:
        A SHA-256 hash of the sorted and concatenated signatures.
    """
    # Sort signatures for consistent hashing
    sorted_sigs = sorted(signatures)
    combined = "\n".join(sorted_sigs)
    return hashlib.sha256(combined.encode()).hexdigest()


def scan_toolboxes(
    toolbox_names: list[str],
    output_file: str,
    exclude_domains: set[str] | None = None,
    domain_overrides: dict[str, str] | None = None,
) -> None:
    """Scan toolboxes and dump domain descriptions to a JSON file.

    Args:
        toolbox_names: A list of toolbox names to scan.
        output_file: The path to the output JSON file.
        exclude_domains: Optional set of domain names to exclude from the output.
        domain_overrides: Optional mapping of tool_name -> domain_name to override
            the default domain assignment for specific tools.

    Raises:
        ValueError: If a domain is found in multiple toolboxes.
    """
    if exclude_domains is None:
        exclude_domains = set()
    if domain_overrides is None:
        domain_overrides = {}
    appworld_apps: list[str] = []
    try:
        from mmtoolsandbox.tools.appworld import get_appworld_apps

        appworld_apps = get_appworld_apps()
    except ImportError:
        print("Warning: Could not import AppWorld tools registry")

    # Dictionary to store tools by domain
    # Structure: domain_name -> {tool_name -> (signature, description)}
    domain_tools: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)

    # Track which toolbox owns which domain to detect duplicates across toolboxes
    # domain_name -> toolbox_name
    domain_owners: dict[str, str] = {}

    for name_str in toolbox_names:
        try:
            # Convert string to ToolboxName enum
            toolbox_name_enum: ToolboxName = DatasetName[name_str]
        except KeyError:
            print(
                f"Warning: Toolbox '{name_str}' not found in ToolboxName enum. Skipping."
            )
            continue

        print(f"Loading toolbox: {name_str}...")
        try:
            toolbox = load_toolbox(toolbox_name_enum, config={})
        except Exception as e:
            print(f"Error loading toolbox '{name_str}': {e}")
            continue

        print(f"  Found {len(toolbox.tools)} tools")

        for tool in toolbox.tools:
            tool_name = tool.__name__

            # Get namespaces (domains) for the tool
            try:
                namespaces: set[DatabaseNamespace] = getattr(
                    tool, "database_namespaces", set()
                )
            except AttributeError:
                print(
                    f"  Warning: Tool '{tool_name}' has no 'database_namespaces' attribute."
                )
                namespaces = set()

            signature = get_tool_signature(tool)
            description = get_tool_description(tool)

            # Check if this tool has an explicit domain override
            domain_key: str | None = None
            if tool_name in domain_overrides:
                domain_key = domain_overrides[tool_name]
                if tool_name not in domain_tools[domain_key]:
                    if domain_key not in domain_owners:
                        domain_owners[domain_key] = name_str
                    domain_tools[domain_key][tool_name] = (signature, description)
                continue

            # If no namespace, extract domain from tool name prefix (e.g., spotify_login -> SPOTIFY)
            if not namespaces:
                # Try to match against known AppWorld app names
                domain_key = None
                for app in appworld_apps:
                    if tool_name.startswith(f"{app}_"):
                        domain_key = app.upper()
                        break

                # Fallback: extract first part before underscore
                if domain_key is None:
                    if "_" in tool_name:
                        domain_key = tool_name.split("_")[0].upper()
                    else:
                        domain_key = name_str  # Fallback to toolbox name

                # Skip if tool already registered (same tool may appear in multiple toolboxes)
                if tool_name in domain_tools[domain_key]:
                    continue
                if domain_key not in domain_owners:
                    domain_owners[domain_key] = name_str
                domain_tools[domain_key][tool_name] = (signature, description)
                continue

            for ns in namespaces:
                domain_key = ns.name
                # Skip if tool already registered (same tool may appear in multiple toolboxes)
                if tool_name in domain_tools[domain_key]:
                    continue
                if domain_key not in domain_owners:
                    domain_owners[domain_key] = name_str

                domain_tools[domain_key][tool_name] = (signature, description)

    # Construct the final output structure
    output_data = {}
    current_time = datetime.now().isoformat()

    for domain, tools_map in domain_tools.items():
        # Skip excluded domains
        if domain in exclude_domains:
            print(f"  Excluding domain: {domain}")
            continue

        # Sort tools by name for consistency
        sorted_tool_names = sorted(tools_map.keys())

        signatures = []
        descriptions = []
        function_names = []

        for tool_name in sorted_tool_names:
            sig, desc = tools_map[tool_name]
            function_names.append(tool_name)
            signatures.append(sig)
            descriptions.append(desc)

        output_data[domain] = {
            "description": descriptions,
            "signature_hash": compute_signature_hash(signatures),
            "signatures": signatures,
            "function_names": function_names,
            "cached_at": current_time,
            "toolbox": domain_owners[domain],
        }

    # Write to file
    print(f"Writing results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan domains and tools in toolboxes.")
    parser.add_argument(
        "--toolboxes",
        type=str,
        nargs="+",
        required=True,
        help="List of toolbox names (e.g., FULL MEDIUM)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSON file",
    )
    parser.add_argument(
        "--exclude-domains",
        type=str,
        nargs="*",
        default=[],
        help="Domain names to exclude from the output (e.g., SANDBOX GET CONVERT)",
    )
    parser.add_argument(
        "--domain-overrides",
        type=str,
        nargs="*",
        default=[],
        help="Override domain for specific tools as tool_name:DOMAIN pairs "
        "(e.g., web_search_serper:SEARCH)",
    )

    args = parser.parse_args()

    # Support both space-separated (via nargs='+') and comma-separated (legacy/alternative)
    toolbox_names = []
    for item in args.toolboxes:
        toolbox_names.extend([name.strip() for name in item.split(",") if name.strip()])

    exclude_domains = set(args.exclude_domains)

    domain_overrides: dict[str, str] = {}
    for entry in args.domain_overrides:
        if ":" not in entry:
            print(
                f"Warning: Invalid domain override '{entry}', expected tool_name:DOMAIN"
            )
            continue
        tool_name, domain = entry.split(":", 1)
        domain_overrides[tool_name] = domain

    scan_toolboxes(toolbox_names, args.output_file, exclude_domains, domain_overrides)


if __name__ == "__main__":
    main()
