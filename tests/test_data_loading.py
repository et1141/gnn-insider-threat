"""Data-loading tests for the chunk-streaming layer.

These exercise the real ``DvcChunkStore`` / dataset / DataModule code path
without touching the DVC remote: chunks are written as local ``.pt`` files in
a tmp dir and the manifest is redirected there by monkeypatching
``get_project_root`` (which is how the store resolves the manifest path).
"""

from __future__ import annotations

import json

import torch
from torch_geometric.data import Batch, Data

import certgnn.data.chunk_store as chunk_store
from certgnn.data import (
    InsiderThreatDataModule,
    SequentialChunkDataset,
    StreamingChunkDataset,
)
from certgnn.split import RandomSplit


def _make_graphs(n: int, num_features: int = 4) -> list[Data]:
    graphs = []
    for i in range(n):
        graphs.append(
            Data(
                x=torch.randn(3, num_features),
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                y_label=torch.tensor(i % 2, dtype=torch.long),
                y_act=torch.tensor(i % 3, dtype=torch.long),
            )
        )
    return graphs


def _setup_chunks(tmp_path, monkeypatch, chunks: dict[str, int]):
    """Write graph chunks + a manifest into a tmp project, redirect the store."""
    monkeypatch.setattr(chunk_store, "get_project_root", lambda: tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (tmp_path / "configs").mkdir()

    for name, count in chunks.items():
        torch.save(_make_graphs(count), processed / name)
    (tmp_path / "configs" / "chunks_manifest.json").write_text(
        json.dumps({"chunks": chunks})
    )
    return processed


def test_streaming_chunk_dataset_local_load(tmp_path, monkeypatch):
    processed = _setup_chunks(
        tmp_path, monkeypatch, {"graph_chunk_0.pt": 4, "graph_chunk_1.pt": 6}
    )

    ds = StreamingChunkDataset(processed, max_local_chunks=1)

    assert len(ds) == 10
    sample = ds[0]
    assert sample.x.shape == (3, 4)
    # Crossing the chunk boundary must evict the first chunk (max_local_chunks=1)
    # and load the second without error.
    assert ds[9].x.shape == (3, 4)


def test_datamodule_random_split_yields_batch(tmp_path, monkeypatch):
    processed = _setup_chunks(tmp_path, monkeypatch, {"graph_chunk_0.pt": 12})

    dm = InsiderThreatDataModule(
        processed_dir=processed,
        mode="random_split",
        split_strategy=RandomSplit(val_size=0.25, test_size=0.25, seed=0),
        batch_size=4,
        num_workers=0,
        max_local_chunks=1,
        accelerator="cpu",
    )
    dm.setup()

    assert len(dm.train_data) + len(dm.val_data) + len(dm.test_data) == 12

    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, Batch)
    assert batch.num_graphs == 4
    assert batch.y_label.shape == (4,)


def test_sequential_chunk_dataset_streams_all_graphs(tmp_path, monkeypatch):
    processed = _setup_chunks(tmp_path, monkeypatch, {"val_graph_chunk_0.pt": 5})

    ds = SequentialChunkDataset(
        processed, ["val_graph_chunk_0.pt"], is_training=False
    )

    assert len(ds) == 5
    collected = list(ds)
    assert len(collected) == 5
    assert all(g.x.shape == (3, 4) for g in collected)
