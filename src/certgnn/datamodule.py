"""PyTorch Lightning DataModule for insider threat detection.

Wraps the streaming chunk-based dataset with a pluggable split strategy,
making it easy to swap between different train/val/test divisions.

Usage:
    from certgnn.split import RandomSplit
    from certgnn.datamodule import InsiderThreatDataModule

    split = RandomSplit(val_size=0.1, test_size=0.2, seed=42)
    dm = InsiderThreatDataModule(
        processed_dir="data/processed/r5.2",
        split_strategy=split,
        batch_size=32,
    )

    # In PyTorch Lightning Trainer:
    trainer = pl.Trainer(...)
    trainer.fit(model, datamodule=dm)
"""

from pathlib import Path

import torch
import pytorch_lightning as pl
from loguru import logger
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from certgnn.sampler import ChunkAwareSampler
from certgnn.split import SplitStrategy
from certgnn.streaming_dataset import StreamingChunkDataset


def _job_memory_limit_bytes() -> int:
    """Return the effective memory ceiling for this process.

    On SLURM, psutil reports the whole machine's RAM — not the job's cgroup
    limit. Workers get OOM-killed when they collectively exceed that limit even
    though psutil showed plenty of space.  Priority order:

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
    after the dataset is pickled to spawn), so total RAM consumed is
    roughly `num_workers * max_local_chunks * chunk_in_memory_size`. A 380 MB
    chunk file unpickles to ~1 GB of Python heap (Data dict + tensor metadata),
    so we use a 3x file-size multiplier.

    On CUDA, GPU memory is separate from system RAM, so we reserve only 2 GB
    for the model/scratch/OS and let the cache go bigger (cap 16 chunks).
    On MPS, the Metal pool shares system RAM with the chunks; reserve 5 GB
    and cap 6 chunks to avoid swap thrash on a 16 GB MacBook.
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


class InsiderThreatDataModule(pl.LightningDataModule):
    """DataModule for streaming insider threat detection graphs.

    Lazily loads chunks from disk/remote via StreamingChunkDataset, applies a
    pluggable split strategy, and provides DataLoaders for training.

    `persistent_workers` defaults are device-aware:
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
        split_strategy: SplitStrategy,
        batch_size: int = 32,
        num_workers: int = 0,
        max_local_chunks: int | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        prefetch_factor: int | None = None,
        accelerator: str | None = None,
        sampler_seed: int = 42,
    ):
        """Initialize the DataModule.

        Args:
            processed_dir: Path to directory with chunks and manifest.
            split_strategy: SplitStrategy instance (e.g., RandomSplit).
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes. With num_workers>0 each
                process gets its own LRU cache copy — safe when chunks are
                local. Recommended: 4-6 on CUDA, 0-1 on macOS, 0 on shared
                low-RAM servers.
            max_local_chunks: How many chunk files to keep cached in memory
                simultaneously per worker. If None (default), auto-detects
                based on available RAM, accelerator, and num_workers.
            pin_memory: Pin CPU tensors for faster GPU transfer. Defaults to
                True when CUDA is available.
            persistent_workers: Keep workers alive between phases. Defaults
                to True on CUDA (with num_workers>0), False on MPS.
            prefetch_factor: Number of batches each worker prefetches.
                Defaults to 4 on CUDA, 2 elsewhere.
            accelerator: 'gpu' | 'mps' | 'cpu'. Auto-detected if None.
        """
        super().__init__()
        self.processed_dir = Path(processed_dir)
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

        if max_local_chunks is None:
            from certgnn.chunk_store import DvcChunkStore

            store = DvcChunkStore(self.processed_dir)
            num_chunks = len(store.list_chunks())
            self.max_local_chunks = _estimate_max_chunks(
                num_chunks=num_chunks,
                processed_dir=self.processed_dir,
                num_workers=num_workers,
                accelerator=self.accelerator,
            )
        else:
            self.max_local_chunks = max_local_chunks

        self.dataset: StreamingChunkDataset | None = None
        self.train_data: Subset | None = None
        self.val_data: Subset | None = None
        self.test_data: Subset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load dataset and apply split strategy.

        Called by PyTorch Lightning. Creates Subset wrappers for each fold.
        """
        self.dataset = StreamingChunkDataset(
            self.processed_dir, max_local_chunks=self.max_local_chunks
        )
        n = len(self.dataset)

        splits = self.split_strategy.split(n)

        self.train_data = Subset(self.dataset, splits["train"])
        self.val_data = Subset(self.dataset, splits["val"])
        self.test_data = Subset(self.dataset, splits["test"])

        logger.info(
            f"Dataset ready | chunks_in_cache={self.max_local_chunks} | "
            f"persistent_workers={self.persistent_workers} | "
            f"prefetch_factor={self.prefetch_factor} | "
            f"total={n:,} | train={len(self.train_data):,} | "
            f"val={len(self.val_data):,} | test={len(self.test_data):,}"
        )

    def _make_loader(self, subset: Subset, shuffle: bool, seed_offset: int) -> DataLoader:
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
        # prefetch_factor is only valid when num_workers > 0
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(subset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader with chunk-aware shuffling."""
        assert self.train_data is not None, "Call setup() first"
        return self._make_loader(self.train_data, shuffle=True, seed_offset=0)

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader (deterministic chunk-sequential order)."""
        assert self.val_data is not None, "Call setup() first"
        return self._make_loader(self.val_data, shuffle=False, seed_offset=1)

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader (deterministic chunk-sequential order)."""
        assert self.test_data is not None, "Call setup() first"
        return self._make_loader(self.test_data, shuffle=False, seed_offset=2)
