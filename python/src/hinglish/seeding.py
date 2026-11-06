"""Deterministic seeding across python, numpy and (optionally) torch.

Kept as a single small function so the seed policy is identical for E3 and E4,
as required by the build spec (matched conditions between the two Transformer
experiments).
"""
from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms where supported; some ops fall back with a
        # warning rather than an error, which is acceptable for this project.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
