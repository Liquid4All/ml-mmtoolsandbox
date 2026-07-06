# Copyright © 2026 Apple Inc.

from __future__ import annotations

import subprocess


def get_git_sha() -> str | None:
    """Get the git SHA of the `HEAD` branch."""
    # From https://stackoverflow.com/a/21901260
    # Note that there are some 3rd party Python modules for interacting with git. I have
    # tried `pygit2` and `GitPython`, but both failed to get the commit associated with
    # `HEAD` for me.
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.PIPE,  # < avoid spamming the console
            )
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        # The mmtoolsandbox script was not executed from within the git repository so we
        # cannot figure out the git SHA.
        return None


def has_local_changes() -> bool:
    # From https://stackoverflow.com/a/3878934 . `git diff --exit-code` will return 0 if
    # there are no local changes. The `--quiet` suppresses printing to stdout. Note that
    # this approach does not detect untracked files, but this should be fine for our
    # purposes.
    completed_proc = subprocess.run(["git", "diff", "--exit-code", "--quiet"])
    return completed_proc.returncode == 1
