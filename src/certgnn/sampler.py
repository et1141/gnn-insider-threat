"""Chunk-aware sampler for StreamingChunkDataset.

Default shuffle=True yields a globally random permutation of sample indices,
which makes every batch potentially hit several different chunk files. With
large chunks on slow storage (NFS, DVC remote) this thrashes the LRU cache and
starves the GPU on IO.

ChunkAwareSampler keeps at most `active_chunks` chunks "hot" at once. Within
that pool, indices are drawn at random — weighted by remaining samples per
chunk so the marginal distribution over the epoch is uniform. When a chunk is
drained, it is swapped out for the next chunk from a shuffled chunk order.
Each chunk is therefore loaded from storage exactly once per epoch.

Gradient quality: a batch is mixed across up to `active_chunks` chunks, so the
effective shuffle pool is active_chunks × chunk_size samples. For 11 × 10k =
110k that is plenty of diversity.

For shuffle=False (val/test) chunks are visited in order and each chunk is
drained sequentially — still chunk-local, deterministic, and efficient.

Note: active_chunks should be <= the LRU cache size of the underlying
StreamingChunkDataset, otherwise the cache will still evict hot chunks.
"""

import random
from typing import Iterator

from torch.utils.data import Sampler, Subset

from certgnn.streaming_dataset import StreamingChunkDataset


class ChunkAwareSampler(Sampler[int]):
    def __init__(
        self,
        subset: Subset,
        active_chunks: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if not isinstance(subset.dataset, StreamingChunkDataset):
            raise TypeError(
                "ChunkAwareSampler expects a Subset wrapping a StreamingChunkDataset"
            )
        if active_chunks < 1:
            raise ValueError("active_chunks must be >= 1")

        self.subset = subset
        self.dataset: StreamingChunkDataset = subset.dataset
        self.active_chunks = active_chunks
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Map each subset position to the chunk that owns its underlying sample.
        # subset.indices[i] is the flat dataset index; dataset._index[flat] is
        # (chunk_idx, local_idx).
        self._chunk_to_positions: dict[int, list[int]] = {}
        for subset_pos, flat_idx in enumerate(subset.indices):
            chunk_idx, _ = self.dataset._index[flat_idx]
            self._chunk_to_positions.setdefault(chunk_idx, []).append(subset_pos)

        self._chunk_order: list[int] = sorted(self._chunk_to_positions.keys())

    def __iter__(self) -> Iterator[int]:
        self._epoch += 1
        rng = random.Random(self.seed + self._epoch) if self.shuffle else None

        remaining = {
            c: list(positions) for c, positions in self._chunk_to_positions.items()
        }
        order = list(self._chunk_order)

        if self.shuffle:
            rng.shuffle(order)
            for positions in remaining.values():
                rng.shuffle(positions)

        active: list[int] = []
        next_idx = 0
        while len(active) < self.active_chunks and next_idx < len(order):
            active.append(order[next_idx])
            next_idx += 1

        while active:
            if self.shuffle:
                weights = [len(remaining[c]) for c in active]
                chunk = rng.choices(active, weights=weights, k=1)[0]
            else:
                chunk = active[0]

            yield remaining[chunk].pop()

            if not remaining[chunk]:
                active.remove(chunk)
                if next_idx < len(order):
                    active.append(order[next_idx])
                    next_idx += 1

    def __len__(self) -> int:
        return len(self.subset)
