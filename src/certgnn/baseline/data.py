"""Data loading helpers for the CERT baseline."""

from __future__ import annotations

import pickle
from pathlib import Path

from torch_geometric.loader import DataLoader

from certgnn.chunk_store import DvcChunkStore
from certgnn.streaming_dataset import StreamingChunkDataset


def load_processed_metadata(processed_dir: Path) -> dict:
    metadata_path = Path(processed_dir) / "metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing processed metadata at {metadata_path}. "
            "Run preprocessing before training the baseline."
        )

    with open(metadata_path, "rb") as handle:
        return pickle.load(handle)


def list_split_chunks(processed_dir: Path, split: str) -> list[str]:
    prefix = f"{split}_graph_chunk_"
    store = DvcChunkStore(Path(processed_dir))
    split_chunks = [name for name in store.list_chunks() if name.startswith(prefix)]
    if split_chunks:
        return split_chunks

    legacy_chunks = [name for name in store.list_chunks() if name.startswith("graph_chunk_")]
    if legacy_chunks:
        return legacy_chunks

    raise FileNotFoundError(
        f"No chunk files found for split '{split}' in {processed_dir}. "
        "Run preprocessing with split-aware chunk export first."
    )


def build_dataloader(
    processed_dir: Path,
    split: str,
    batch_size: int,
    max_local_chunks: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    delete_after_eviction: bool = True,
) -> DataLoader:
    chunk_names = list_split_chunks(processed_dir, split)
    dataset = StreamingChunkDataset(
        processed_dir=processed_dir,
        max_local_chunks=max_local_chunks,
        chunk_names=chunk_names,
        delete_after_eviction=delete_after_eviction,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=False,
    )