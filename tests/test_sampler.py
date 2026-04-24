"""Tests for ChunkAwareSampler."""

from collections import Counter

import pytest
from torch.utils.data import Subset

from certgnn.sampler import ChunkAwareSampler
from certgnn.streaming_dataset import StreamingChunkDataset


def _fake_dataset(num_chunks: int, per_chunk: int) -> StreamingChunkDataset:
    """Build a StreamingChunkDataset with a synthetic flat index (no disk IO).

    Only the attributes the sampler reads (_index, chunk_names) are populated,
    which is enough — the sampler never calls __getitem__.
    """
    ds = StreamingChunkDataset.__new__(StreamingChunkDataset)
    ds._index = [(c, l) for c in range(num_chunks) for l in range(per_chunk)]
    ds.chunk_names = [f"graph_chunk_{i}.pt" for i in range(num_chunks)]
    return ds


def _chunk_of(ds: StreamingChunkDataset, flat_idx: int) -> int:
    return ds._index[flat_idx][0]


def test_full_permutation_shuffle_true():
    """Every position is yielded exactly once, covering the whole subset."""
    ds = _fake_dataset(num_chunks=4, per_chunk=100)
    subset = Subset(ds, list(range(400)))
    sampler = ChunkAwareSampler(subset, active_chunks=2, shuffle=True, seed=0)

    out = list(sampler)

    assert len(out) == 400
    assert sorted(out) == list(range(400))


def test_full_permutation_shuffle_false():
    """Deterministic mode still yields every position exactly once."""
    ds = _fake_dataset(num_chunks=3, per_chunk=50)
    subset = Subset(ds, list(range(150)))
    sampler = ChunkAwareSampler(subset, active_chunks=2, shuffle=False)

    out = list(sampler)

    assert len(out) == 150
    assert sorted(out) == list(range(150))


def test_shuffle_false_iterates_chunks_sequentially():
    """shuffle=False drains each chunk fully before moving to the next."""
    ds = _fake_dataset(num_chunks=3, per_chunk=50)
    subset = Subset(ds, list(range(150)))
    sampler = ChunkAwareSampler(subset, active_chunks=2, shuffle=False)

    chunks_seen = [_chunk_of(ds, i) for i in sampler]

    assert chunks_seen[:50] == [0] * 50
    assert chunks_seen[50:100] == [1] * 50
    assert chunks_seen[100:] == [2] * 50


def test_shuffle_true_advances_epoch_seed():
    """Calling __iter__ twice yields different permutations."""
    ds = _fake_dataset(num_chunks=4, per_chunk=100)
    subset = Subset(ds, list(range(400)))
    sampler = ChunkAwareSampler(subset, active_chunks=2, shuffle=True, seed=0)

    first = list(sampler)
    second = list(sampler)

    assert first != second
    assert sorted(first) == sorted(second)  # same content, different order


def test_shuffle_true_chunk_locality():
    """Within one epoch only a bounded number of distinct chunks co-occur.

    With active_chunks=K, at any point in the stream the samples come from
    at most K chunks that are currently 'active'. Over the full epoch all
    chunks are visited, but locality is preserved via swap-in/swap-out.
    """
    num_chunks = 8
    per_chunk = 200
    active = 2
    ds = _fake_dataset(num_chunks=num_chunks, per_chunk=per_chunk)
    subset = Subset(ds, list(range(num_chunks * per_chunk)))
    sampler = ChunkAwareSampler(subset, active_chunks=active, shuffle=True, seed=7)

    chunks_seen = [_chunk_of(ds, i) for i in sampler]

    # Every chunk must appear exactly per_chunk times
    counts = Counter(chunks_seen)
    assert set(counts.keys()) == set(range(num_chunks))
    assert all(v == per_chunk for v in counts.values())

    # Over any sliding window of reasonable size, the distinct-chunk count
    # should never explode past active + 1 (the +1 accounts for the transient
    # swap moment when one chunk drains and the next is brought in before
    # the drained chunk is fully flushed from the window).
    window = 50
    max_distinct = max(
        len(set(chunks_seen[i : i + window]))
        for i in range(0, len(chunks_seen) - window, window)
    )
    assert max_distinct <= active + 2  # small slack for swap transients


def test_partial_subset():
    """Sampler works on a Subset that only covers some indices (e.g. train fold)."""
    ds = _fake_dataset(num_chunks=4, per_chunk=100)
    # Pick irregular indices spanning all four chunks
    selected = [5, 50, 105, 150, 205, 250, 305, 399]
    subset = Subset(ds, selected)
    sampler = ChunkAwareSampler(subset, active_chunks=2, shuffle=True, seed=3)

    out = list(sampler)

    assert len(out) == len(selected)
    assert sorted(out) == list(range(len(selected)))  # subset positions 0..7


def test_rejects_non_streaming_dataset():
    class NotAStreamingDataset:
        pass

    subset = Subset(NotAStreamingDataset(), [0, 1])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ChunkAwareSampler(subset, active_chunks=2)


def test_rejects_invalid_active_chunks():
    ds = _fake_dataset(num_chunks=2, per_chunk=10)
    subset = Subset(ds, list(range(20)))
    with pytest.raises(ValueError):
        ChunkAwareSampler(subset, active_chunks=0)


def test_len_matches_subset():
    ds = _fake_dataset(num_chunks=3, per_chunk=100)
    subset = Subset(ds, list(range(250)))
    sampler = ChunkAwareSampler(subset, active_chunks=2)
    assert len(sampler) == 250


def test_single_chunk_subset():
    """Subset covering only one chunk still works (active_chunks > chunks available)."""
    ds = _fake_dataset(num_chunks=5, per_chunk=50)
    subset = Subset(ds, list(range(50)))  # only chunk 0
    sampler = ChunkAwareSampler(subset, active_chunks=4, shuffle=True, seed=0)

    out = list(sampler)
    chunks_seen = {_chunk_of(ds, i) for i in out}

    assert sorted(out) == list(range(50))
    assert chunks_seen == {0}
