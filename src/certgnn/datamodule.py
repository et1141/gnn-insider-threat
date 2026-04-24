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
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from certgnn.sampler import ChunkAwareSampler
from certgnn.split import SplitStrategy
from certgnn.streaming_dataset import StreamingChunkDataset


def _estimate_max_chunks(num_chunks: int, processed_dir: Path) -> int:
    """Return how many chunks fit in ~70% of available RAM."""
    try:
        import psutil

        available_ram = psutil.virtual_memory().available
        chunk_files = sorted(processed_dir.glob("graph_chunk_*.pt"))
        if chunk_files:
            chunk_size = chunk_files[0].stat().st_size * 1.2  # 20% in-memory overhead
        else:
            chunk_size = 300 * 1024**2  # 300 MB fallback when no file on disk yet
        return max(4, min(num_chunks, int(available_ram * 0.7 / chunk_size)))
    except ImportError:
        return min(num_chunks, 16)


class InsiderThreatDataModule(pl.LightningDataModule):
    """DataModule for streaming insider threat detection graphs.

    Lazily loads chunks from disk/remote via StreamingChunkDataset, applies a
    pluggable split strategy, and provides DataLoaders for training.

    With num_workers>0 each worker spawns its own dataset copy (macOS uses
    spawn), so the LRU cache is independent per process — safe for local chunks.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split_strategy: SplitStrategy,
        batch_size: int = 32,
        num_workers: int = 0,
        max_local_chunks: int | None = None,
        pin_memory: bool | None = None,
        sampler_seed: int = 42,
    ):
        """Initialize the DataModule.

        Args:
            processed_dir: Path to directory with chunks and manifest.
            split_strategy: SplitStrategy instance (e.g., RandomSplit).
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes. With num_workers>0 each process gets its own LRU cache copy — safe when chunks are local.
            max_local_chunks: How many chunk files to keep cached in memory simultaneously.
                If None (default), auto-detects based on available RAM.
            pin_memory: Pin CPU tensors for faster GPU transfer. Defaults to True when CUDA is available.
        """
        super().__init__()
        self.processed_dir = Path(processed_dir)
        self.split_strategy = split_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.sampler_seed = sampler_seed

        if max_local_chunks is None:
            from certgnn.chunk_store import DvcChunkStore

            store = DvcChunkStore(self.processed_dir)
            num_chunks = len(store.list_chunks())
            self.max_local_chunks = _estimate_max_chunks(num_chunks, self.processed_dir)
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
        # Load the full streaming dataset
        print(f"[DEBUG] Creating StreamingChunkDataset with max_local_chunks={self.max_local_chunks}")
        self.dataset = StreamingChunkDataset(
            self.processed_dir, max_local_chunks=self.max_local_chunks
        )
        print(f"[DEBUG] StreamingChunkDataset created: max_local_chunks={self.dataset.max_local_chunks}")
        n = len(self.dataset)

        # Apply split strategy
        splits = self.split_strategy.split(n)

        # Create Subset wrappers for each fold
        self.train_data = Subset(self.dataset, splits["train"])
        self.val_data = Subset(self.dataset, splits["val"])
        self.test_data = Subset(self.dataset, splits["test"])

        print(
            f"[DataModule] Using {self.max_local_chunks} chunks cache. Split {n:,} graphs: "
            f"train={len(self.train_data):,}, val={len(self.val_data):,}, test={len(self.test_data):,}"
        )

    def _make_loader(self, subset: Subset, shuffle: bool, seed_offset: int) -> DataLoader:
        sampler = ChunkAwareSampler(
            subset,
            active_chunks=self.max_local_chunks,
            shuffle=shuffle,
            seed=self.sampler_seed + seed_offset,
        )
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
        )

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
