"""Pluggable split strategies for train/val/test division.

Use with InsiderThreatDataModule to easily swap different split approaches.
"""

import random
from typing import Protocol, runtime_checkable


@runtime_checkable
class SplitStrategy(Protocol):
    """Abstract protocol for train/val/test splitting strategies.

    Implementations should be deterministic (use seed) for reproducibility.
    """

    def split(self, n: int) -> dict[str, list[int]]:
        """Split indices 0..n-1 into train/val/test.

        Args:
            n: Total number of examples.

        Returns:
            Dict with keys 'train', 'val', 'test', each mapping to a list of indices.
            Indices must be disjoint and cover [0, n).
        """
        ...


class RandomSplit:
    """Random train/val/test split.

    Splits a dataset randomly into three folds with specified sizes.
    Deterministic given a seed.
    """

    def __init__(self, val_size: float = 0.1, test_size: float = 0.2, seed: int = 42):
        """Initialize random split.

        Args:
            val_size: Fraction of data for validation (0.0-1.0).
            test_size: Fraction of data for testing (0.0-1.0).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If val_size + test_size >= 1.0.
        """
        if val_size + test_size >= 1.0:
            raise ValueError(
                f"val_size ({val_size}) + test_size ({test_size}) must be < 1.0"
            )
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    def split(self, n: int) -> dict[str, list[int]]:
        """Split n indices randomly.

        Carves off test set first, then validation, leaving remainder for training.
        """
        if n == 0:
            return {"train": [], "val": [], "test": []}

        rng = random.Random(self.seed)
        indices = list(range(n))
        rng.shuffle(indices)

        n_test = max(1, int(n * self.test_size)) if self.test_size > 0 else 0
        n_val = max(1, int(n * self.val_size)) if self.val_size > 0 else 0

        return {
            "test": indices[:n_test],
            "val": indices[n_test : n_test + n_val],
            "train": indices[n_test + n_val :],
        }
