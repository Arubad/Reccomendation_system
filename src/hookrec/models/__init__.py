"""Recommender models and the `fit_recommender` registry.

The baselines (and their polars/scipy imports) register lazily, on first access of the registry
names, so importing `hookrec.models.hookseq` / `sasrec` (the demo) stays light.
"""

__all__ = ["REGISTRY", "fit_recommender", "register"]


def __getattr__(name: str):
    if name in __all__:
        from hookrec.models import base, ials, popularity  # noqa: F401  (registers baselines)

        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
