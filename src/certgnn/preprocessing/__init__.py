"""Preprocessing pipelines for the GNN insider-threat detector.

Two variants live here, sharing the upstream DuckDB feature pipeline in
``common``:

* ``paper_faithful`` — fractional user selection, single chunk stream, the
  per-graph random split is applied at training time.
* ``user_level_split`` — fixed-count scenario-balanced selection, train/val/
  test stratified per scenario at the user level, chunks are split-prefixed
  on disk so there is no leak from sub-session variants.

Pick a variant via ``configs/config.yaml::preprocessing.variant`` and run::

    uv run preprocess
    # or, to bypass the dispatcher and pin a variant:
    uv run preprocess-paper-faithful
    uv run preprocess-user-split
"""

from certgnn.preprocessing import paper_faithful, user_level_split
from certgnn.utils import load_config


def main() -> None:
    """Dispatch to the variant declared in ``preprocessing.variant``."""
    config = load_config()
    variant = config.get("preprocessing", {}).get("variant", "paper_faithful")
    if variant == "paper_faithful":
        paper_faithful.main()
    elif variant == "user_level_split":
        user_level_split.main()
    else:
        raise ValueError(
            f"Unknown preprocessing variant {variant!r}; "
            "expected 'paper_faithful' or 'user_level_split'."
        )


__all__ = ["main", "paper_faithful", "user_level_split"]
