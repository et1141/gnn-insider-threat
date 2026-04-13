"""Streaming PyTorch Dataset for chunked graph data stored on DVC remote.

Keeps at most `max_local_chunks` chunk files on disk at once. When a new chunk
is needed and the cache is full, the oldest chunk is evicted (and deleted from
disk). The next chunk is then pulled from GDrive on demand.

Usage with PyTorch Lightning:
    dataset = StreamingChunkDataset(processed_dir, max_local_chunks=2)
    loader = DataLoader(dataset, batch_size=64, num_workers=0)
    # num_workers=0 required — multi-process access to the LRU cache is unsafe.
"""

import threading
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from certgnn.chunk_store import DvcChunkStore


class StreamingChunkDataset(Dataset):
    """Dataset that streams graph chunks from DVC remote (Google Drive).

    Args:
        processed_dir: Directory containing chunk files and their .dvc pointers.
        max_local_chunks: How many chunk files to keep on disk simultaneously.
        delete_after_eviction: Delete a chunk file from disk when it's evicted
            from the local cache (frees disk space).
    """

    def __init__(
        self,
        processed_dir: Path,
        max_local_chunks: int = 2,
        delete_after_eviction: bool = True,
        chunk_names: list[str] | None = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.store = DvcChunkStore(self.processed_dir)
        self.max_local_chunks = max_local_chunks
        self.delete_after_eviction = delete_after_eviction

        available_chunks = self.store.list_chunks()
        self.chunk_names = chunk_names or available_chunks
        self.chunk_names = [name for name in self.chunk_names if name in available_chunks]
        if not self.chunk_names:
            raise ValueError(
                "Manifest is empty — no chunks found. "
                "Run preprocessing with DvcChunkStore.push_chunk() first."
            )

        # Flat index: position i → (chunk_idx, local_idx_within_chunk)
        self._index: list[tuple[int, int]] = []
        for chunk_idx, name in enumerate(self.chunk_names):
            size = self.store.chunk_size(name)
            self._index.extend((chunk_idx, j) for j in range(size))

        # LRU cache: chunk_name → loaded list[Data]
        self._loaded: OrderedDict[str, list] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        chunk_idx, local_idx = self._index[idx]
        chunk_name = self.chunk_names[chunk_idx]
        chunk = self._get_chunk(chunk_name)
        return chunk[local_idx]

    def _get_chunk(self, chunk_name: str) -> list:
        with self._lock:
            # Cache hit — move to end (most recently used)
            if chunk_name in self._loaded:
                self._loaded.move_to_end(chunk_name)
                return self._loaded[chunk_name]

            # Evict oldest chunk if at capacity
            if len(self._loaded) >= self.max_local_chunks:
                evicted_name, _ = self._loaded.popitem(last=False)
                if self.delete_after_eviction:
                    evicted_path = self.processed_dir / evicted_name
                    if evicted_path.exists():
                        evicted_path.unlink()

            # Pull from remote and load
            chunk_path = self.store.pull_chunk(chunk_name)
            graphs = torch.load(chunk_path, weights_only=False)
            self._loaded[chunk_name] = graphs
            return graphs
