"""Recommender models and the `fit_recommender` registry."""

from hookrec.models import popularity  # noqa: F401  (registers "mostpop" and "decaypop")
from hookrec.models.base import REGISTRY, fit_recommender, register

__all__ = ["REGISTRY", "fit_recommender", "register"]
