"""Recommender models and the `fit_recommender` registry."""

from hookrec.models import ials, popularity  # noqa: F401  (registers the baselines)
from hookrec.models.base import REGISTRY, fit_recommender, register

__all__ = ["REGISTRY", "fit_recommender", "register"]
