"""Sequential Streaming PyTorch Dataset for chunked graph data."""

import torch
import random
from pathlib import Path
from torch.utils.data import IterableDataset

from certgnn.chunk_store import DvcChunkStore


class SequentialChunkDataset(IterableDataset):
    """Dataset that streams graph chunks sequentially from DVC.
    
    It pulls one chunk, yields all its graphs, and then deletes it before
    moving to the next one. This prevents disk thrashing and RAM overflow.
    """

    def __init__(self, processed_dir: Path, chunk_names: list[str], is_training: bool = True):
        self.processed_dir = Path(processed_dir)
        self.store = DvcChunkStore(self.processed_dir)
        self.chunk_names = chunk_names
        self.is_training = is_training
        
        self._total_graphs = sum(self.store.chunk_size(name) for name in self.chunk_names)

    def __len__(self) -> int:
        return self._total_graphs

    def __iter__(self):
        chunks_to_process = list(self.chunk_names)

        if self.is_training:
            random.shuffle(chunks_to_process)

        for chunk_name in chunks_to_process:
            chunk_path = self.store.pull_chunk(chunk_name)
            graphs = torch.load(chunk_path, weights_only=False)

            if self.is_training:
                random.shuffle(graphs)

            for graph in graphs:
                yield graph

            del graphs
            try:
                chunk_path.unlink()
            except PermissionError:
                pass
            except Exception:
                pass