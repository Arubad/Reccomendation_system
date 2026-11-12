"""Git metadata for results files."""

from __future__ import annotations

import subprocess


def git_sha() -> str:
    """Current commit SHA, or "unknown" outside a repo / without git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"
