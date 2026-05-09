"""PyTorch Lightning DataModule for insider threat detection.

Supports two chunk layouts via the ``mode`` argument:

* ``"random_split"`` (paper-faithful, mine): one undifferentiated set of
  ``graph_chunk_*.pt`` chunks; train/val/test slices are produced at runtime
  by a ``SplitStrategy`` (e.g. ``RandomSplit``) over the flat dataset and
  served by ``StreamingChunkDataset`` + ``ChunkAwareSampler``.

* ``"presplit"`` (user-level-split, kolegi): chunks already named with
  ``train_/val_/test_`` prefixes; each split streams independently via
  ``SequentialChunkDataset`` (no sampler — chunks are consumed in order).
"""

from pathlib import Path

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from certgnn.data.chunk_store import DvcChunkStore
from certgnn.data.sampler import ChunkAwareSampler
from certgnn.data.streaming_dataset import (
    SequentialChunkDataset,
    StreamingChunkDataset,
)
from certgnn.split import SplitStrategy


def _job_memory_limit_bytes() -> int:
    """Return the effective memory ceiling for this process.

    On SLURM, psutil reports the whole machine's RAM — not the job's cgroup
    limit. Workers get OOM-killed when they collectively exceed that limit
    even though psutil showed plenty of space.  Priority order:

    1. SLURM_MEM_PER_NODE env var (set by SLURM for every job).
    2. cgroup v2  /sys/fs/cgroup/memory.max
    3. cgroup v1  /sys/fs/cgroup/memory/memory.limit_in_bytes
    4. psutil available RAM (local machine fallback).
    """
    import os
    from pathlib import Path as _Path

    slurm_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mb:
        return int(slurm_mb) * 1024**2

    for cgroup_path, unlimited_sentinel in [
        ("/sys/fs/cgroup/memory.max", "max"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None),
    ]:
        try:
            raw = _Path(cgroup_path).read_text().strip()
            if raw == unlimited_sentinel:
                continue
            limit = int(raw)
            if limit < 2**62:  # cgroup v1 uses 2^63-4096 for "unlimited"
                return limit
        except (OSError, ValueError):
            pass

    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return 8 * 1024**3


def _estimate_max_chunks(
    num_chunks: int,
    processed_dir: Path,
    num_workers: int,
    accelerator: str,
) -> int:
    """Pick a chunk-cache size that fits in the job's actual memory budget.

    The LRU cache is per-worker (each DataLoader worker owns its own copy
    after the dataset is pickled to spawn), so total RAM consumed is roughly
    ``num_workers * max_local_chunks * chunk_in_memory_size``. A 380 MB chunk
    file unpickles to ~1 GB of Python heap (Data dict + tensor metadata), so
    we use a 3x file-size multiplier.

    On CUDA, GPU memory is separate from system RAM — reserve only 2 GB for
    the model/scratch/OS and let the cache go bigger (cap 16 chunks). On MPS,
    the Metal pool shares system RAM with the chunks; reserve 5 GB and cap
    6 chunks to avoid swap thrash on a 16 GB MacBook.
    """
    mem_limit = _job_memory_limit_bytes()
    chunk_files = sorted(processed_dir.glob("graph_chunk_*.pt"))
    if chunk_files:
        chunk_size = chunk_files[0].stat().st_size * 3.0
    else:
        chunk_size = 1024 * 1024**2

    if accelerator in ("gpu", "cuda"):
        reserve = 2 * 1024**3
        hard_cap = 16
    else:
        reserve = 5 * 1024**3
        hard_cap = 6

    workers = max(1, num_workers)
    usable = max(0, mem_limit - reserve)
    budget = max(2, int(usable * 0.5 / chunk_size / workers))
    chunks = max(2, min(num_chunks, hard_cap, budget))

    logger.info(
        f"max_local_chunks={chunks} | accelerator={accelerator} | "
        f"mem_limit={mem_limit/1024**3:.1f}GB | reserve={reserve/1024**3:.1f}GB | "
        f"chunk_in_mem~{chunk_size/1024**3:.2f}GB | workers={workers} | "
        f"budget_per_worker={budget} | hard_cap={hard_cap}"
    )
    return chunks


def _detect_accelerator() -> str:
    """Return 'gpu' for CUDA, 'mps' for Apple Metal, otherwise 'cpu'."""
    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _list_split_chunks(processed_dir: Path, split: str) -> list[str]:
    """Return chunk filenames for a given split (train/val/test).

    Looks for the new ``{split}_graph_chunk_*.pt`` naming first, then falls
    back to the legacy ``graph_chunk_*.pt`` naming for backward compatibility.
    """
    store = DvcChunkStore(Path(processed_dir))
    prefix = f"{split}_graph_chunk_"
    split_chunks = [name for name in store.list_chunks() if name.startswith(prefix)]
    if split_chunks:
        return sorted(split_chunks, key=_chunk_sort_key)
    legacy_chunks = [name for name in store.list_chunks() if name.startswith("graph_chunk_")]
    if legacy_chunks:
        return sorted(legacy_chunks, key=_chunk_sort_key)
    raise FileNotFoundError(
        f"No chunk files found for split '{split}' in {processed_dir}. "
        "Run preprocessing first."
    )


def _chunk_sort_key(name: str) -> int:
    import re
    match = re.search(r"\d+", name)
    return int(match.group()) if match else 0


class InsiderThreatDataModule(pl.LightningDataModule):
    """Unified DataModule for both preprocessing variants.

    ``persistent_workers`` defaults are device-aware:
    - CUDA: True (workers stay alive across phases — keeps the LRU cache
      warm and avoids spawn overhead, which dominates throughput when each
      epoch is short relative to chunk-load time).
    - MPS: False (workers are killed per phase so the kernel reclaims their
      heap; Apple Silicon's pymalloc never returns freed pages to the OS,
      which otherwise grows the process to tens of GB after a few epochs).
    """

    def __init__(
        self,
        processed_dir: str | Path,
        mode: str = "random_split",
        split_strategy: SplitStrategy | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        max_local_chunks: int | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
        accelerator: str | None = None,
        sampler_seed: int = 42,
    ):
        super().__init__()
        if mode not in {"random_split", "presplit"}:
            raise ValueError(
                f"mode must be 'random_split' or 'presplit', got {mode!r}"
            )
        if mode == "random_split" and split_strategy is None:
            raise ValueError("split_strategy is required when mode='random_split'")

        self.processed_dir = Path(processed_dir)
        self.mode = mode
        self.split_strategy = split_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler_seed = sampler_seed

        self.accelerator = accelerator if accelerator is not None else _detect_accelerator()
        is_cuda = self.accelerator in ("gpu", "cuda") and torch.cuda.is_available()
        is_mps = self.accelerator == "mps"

        self.pin_memory = is_cuda if pin_memory is None else pin_memory

        if persistent_workers is None:
            self.persistent_workers = (num_workers > 0) and not is_mps
        else:
            self.persistent_workers = persistent_workers and (num_workers > 0)

        if prefetch_factor is None:
            self.prefetch_factor = 4 if is_cuda else 2
        else:
            self.prefetch_factor = prefetch_factor

        if max_local_chunks is None and mode == "random_split":
            store = DvcChunkStore(self.processed_dir)
            num_chunks = len(store.list_chunks())
            self.max_local_chunks = _estimate_max_chunks(
                num_chunks=num_chunks,
                processed_dir=self.processed_dir,
                num_workers=num_workers,
                accelerator=self.accelerator,
            )
        else:
            self.max_local_chunks = max_local_chunks or 2

        self.train_data: Subset | SequentialChunkDataset | None = None
        self.val_data: Subset | SequentialChunkDataset | None = None
        self.test_data: Subset | SequentialChunkDataset | None = None
        self._dataset: StreamingChunkDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.mode == "random_split":
            self._setup_random_split()
        else:
            self._setup_presplit()

    def _setup_random_split(self) -> None:
        self._dataset = StreamingChunkDataset(
            self.processed_dir, max_local_chunks=self.max_local_chunks
        )
        n = len(self._dataset)
        splits = self.split_strategy.split(n)
        self.train_data = Subset(self._dataset, splits["train"])
        self.val_data = Subset(self._dataset, splits["val"])
        self.test_data = Subset(self._dataset, splits["test"])
        logger.info(
            f"DataModule[random_split] | chunks_in_cache={self.max_local_chunks} | "
            f"persistent_workers={self.persistent_workers} | "
            f"total={n:,} | train={len(self.train_data):,} | "
            f"val={len(self.val_data):,} | test={len(self.test_data):,}"
        )

    def _setup_presplit(self) -> None:
        train_chunks = _list_split_chunks(self.processed_dir, "train")
        val_chunks = _list_split_chunks(self.processed_dir, "val")
        test_chunks = _list_split_chunks(self.processed_dir, "test")
        self.train_data = SequentialChunkDataset(self.processed_dir, train_chunks, is_training=True)
        self.val_data = SequentialChunkDataset(self.processed_dir, val_chunks, is_training=False)
        self.test_data = SequentialChunkDataset(self.processed_dir, test_chunks, is_training=False)
        logger.info(
            f"DataModule[presplit] | "
            f"train={len(self.train_data):,} ({len(train_chunks)} chunks) | "
            f"val={len(self.val_data):,} ({len(val_chunks)} chunks) | "
            f"test={len(self.test_data):,} ({len(test_chunks)} chunks)"
        )

    def _make_random_split_loader(self, subset: Subset, shuffle: bool, seed_offset: int) -> DataLoader:
        sampler = ChunkAwareSampler(
            subset,
            active_chunks=self.max_local_chunks,
            shuffle=shuffle,
            seed=self.sampler_seed + seed_offset,
        )
        kwargs: dict = {
            "batch_size": self.batch_size,
            "sampler": sampler,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(subset, **kwargs)

    def _make_presplit_loader(self, dataset: SequentialChunkDataset) -> DataLoader:
        # IterableDataset: no sampler, no shuffle flag — shuffle is internal
        # to the dataset (controlled by is_training).
        kwargs: dict = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        assert self.train_data is not None, "Call setup() first"
        if self.mode == "random_split":
            return self._make_random_split_loader(self.train_data, shuffle=True, seed_offset=0)
        return self._make_presplit_loader(self.train_data)

    def val_dataloader(self) -> DataLoader:
        assert self.val_data is not None, "Call setup() first"
        if self.mode == "random_split":
            return self._make_random_split_loader(self.val_data, shuffle=False, seed_offset=1)
        return self._make_presplit_loader(self.val_data)

    def test_dataloader(self) -> DataLoader:
        assert self.test_data is not None, "Call setup() first"
        if self.mode == "random_split":
            return self._make_random_split_loader(self.test_data, shuffle=False, seed_offset=2)
        return self._make_presplit_loader(self.test_data)
