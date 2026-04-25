"""Streaming PyTorch Dataset for chunked graph data stored on DVC remote.

Keeps at most `max_local_chunks` chunk files on disk at once. When a new chunk
is needed and the cache is full, the oldest chunk is evicted (and deleted from
disk). The next chunk is then pulled from GDrive on demand.

Usage with PyTorch Lightning:
    dataset = StreamingChunkDataset(processed_dir, max_local_chunks=2)
    loader = DataLoader(dataset, batch_size=64, num_workers=4)

Multi-worker DataLoaders are supported: __getstate__/__setstate__ drop the
threading.Lock before pickling and recreate it in each worker, so every
worker process ends up with its own independent LRU cache. Note the memory
implication — each worker duplicates up to `max_local_chunks` chunks in RAM.
"""

import ctypes
import ctypes.util
import gc
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from certgnn.chunk_store import DvcChunkStore

_LIBC = ctypes.CDLL(ctypes.util.find_library("c")) if sys.platform == "darwin" else None


def _release_heap_to_os() -> None:
    """Force the allocator to give freed pages back to the OS.

    macOS's malloc keeps freed pages in per-zone caches indefinitely, so after
    repeated torch.load/del cycles the process's physical footprint grows without
    bound even though Python no longer references the data. Calling
    `malloc_zone_pressure_relief(NULL, 0)` walks every zone and unmaps the empty
    regions; on Linux this is a no-op (glibc already does it via M_TRIM).
    """
    gc.collect()
    if _LIBC is not None:
        try:
            _LIBC.malloc_zone_pressure_relief(None, 0)
        except (AttributeError, OSError):
            pass


class StreamingChunkDataset(Dataset):
    """Dataset that streams graph chunks from DVC remote (Google Drive).

    Implements a two-level cache: in-memory LRU (max_local_chunks) + disk cache.
    When a chunk is evicted from memory, it stays on disk; if accessed again,
    it loads from disk without pulling from remote (unless missing entirely).

    Args:
        processed_dir: Directory containing chunk files and their .dvc pointers.
        max_local_chunks: How many chunk files to keep loaded in memory simultaneously.
        delete_after_eviction: If True, delete chunk files from disk when evicted
            from memory (frees disk space but requires re-pulling from remote).
            Default False: keeps files on disk for fast local reload.
    """

    def __init__(
        self,
        processed_dir: Path,
        max_local_chunks: int = 2,
        delete_after_eviction: bool = False,
    ):
        self.processed_dir = Path(processed_dir)
        self.store = DvcChunkStore(self.processed_dir)
        self.max_local_chunks = max_local_chunks
        self.delete_after_eviction = delete_after_eviction

        self.chunk_names = self.store.list_chunks()
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
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
                evicted_name, evicted_data = self._loaded.popitem(last=False)
                del evicted_data
                if self.delete_after_eviction:
                    evicted_path = self.processed_dir / evicted_name
                    if evicted_path.exists():
                        evicted_path.unlink()
                _release_heap_to_os()

            # Pull from remote and load
            chunk_path = self.store.pull_chunk(chunk_name)
            graphs = torch.load(chunk_path, weights_only=False)
            self._loaded[chunk_name] = graphs
            return graphs
