"""ZeroGPU support. Import this module before torch: `spaces` must load first on ZeroGPU."""

from __future__ import annotations

from collections.abc import Callable

try:  # only present on Hugging Face Spaces
    import spaces as _spaces
except ImportError:
    _spaces = None


def _maybe_gpu(fn: Callable) -> Callable:
    """`spaces.GPU(fn)` when the `spaces` package is importable, else `fn` itself."""
    return fn if _spaces is None else _spaces.GPU(fn)
