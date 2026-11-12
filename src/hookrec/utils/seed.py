"""Global seeding."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed random, numpy and (if importable) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
