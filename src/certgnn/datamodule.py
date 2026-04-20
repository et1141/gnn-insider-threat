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

import pytorch_lightning as pl
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from certgnn.split import SplitStrategy
from certgnn.streaming_dataset import StreamingChunkDataset


class InsiderThreatDataModule(pl.LightningDataModule):
    """DataModule for streaming insider threat detection graphs.

    Lazily loads chunks from disk/remote via StreamingChunkDataset, applies a
    pluggable split strategy, and provides DataLoaders for training.

    Warning: num_workers must be 0 due to LRU cache in StreamingChunkDataset
    not being multiprocess-safe. Data loading happens in the main process.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        split_strategy: SplitStrategy,
        batch_size: int = 32,
        num_workers: int = 0,
        max_local_chunks: int | None = None,
    ):
        """Initialize the DataModule.

        Args:
            processed_dir: Path to directory with chunks and manifest.
            split_strategy: SplitStrategy instance (e.g., RandomSplit).
            batch_size: Batch size for DataLoaders.
            num_workers: Number of worker processes. MUST be 0 (LRU cache limitation).
            max_local_chunks: How many chunk files to keep cached on disk simultaneously.
                If None (default), auto-detects based on number of chunks to avoid re-downloading.

        Raises:
            ValueError: If num_workers != 0.
        """
        super().__init__()
        if num_workers != 0:
            raise ValueError(
                f"num_workers must be 0 (got {num_workers}). "
                "StreamingChunkDataset's LRU cache is not multiprocess-safe."
            )

        self.processed_dir = Path(processed_dir)
        self.split_strategy = split_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Auto-detect number of chunks if max_local_chunks not specified
        if max_local_chunks is None:
            from certgnn.chunk_store import DvcChunkStore

            store = DvcChunkStore(self.processed_dir)
            num_chunks = len(store.list_chunks())
            # Keep all chunks in cache to avoid re-downloading
            # (better than default 4 which causes thrashing with many chunks)
            self.max_local_chunks = max(4, min(num_chunks, 16))
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

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader with shuffling."""
        assert self.train_data is not None, "Call setup() first"
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader without shuffling."""
        assert self.val_data is not None, "Call setup() first"
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader without shuffling."""
        assert self.test_data is not None, "Call setup() first"
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
