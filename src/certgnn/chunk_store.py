"""DVC-based chunk store for streaming processed graph data to/from Google Drive.

Handles per-chunk push during preprocessing and on-demand pull during training,
so the full dataset never needs to live locally at once.

The manifest (configs/chunks_manifest.json) is committed to git and tracks
which chunks exist on remote and how many graphs each contains.
"""

import json
import subprocess
import time
from pathlib import Path

from certgnn.utils import get_project_root

MANIFEST_PATH = get_project_root() / "configs" / "chunks_manifest.json"


class DvcChunkStore:
    """Stream graph chunks to/from DVC remote (Google Drive).

    During preprocessing:
        store = DvcChunkStore(processed_dir)
        store.push_chunk(chunk_path, num_graphs=len(graph_list), delete_local=True)

    During training:
        store = DvcChunkStore(processed_dir)
        chunk_path = store.pull_chunk("graph_chunk_0.pt")
        graphs = torch.load(chunk_path)
        chunk_path.unlink()   # free disk space after use
    """

    def __init__(self, processed_dir: Path):
        self.processed_dir = Path(processed_dir)
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if MANIFEST_PATH.exists():
            return json.loads(MANIFEST_PATH.read_text())
        return {"chunks": {}}  # {chunk_name: num_graphs}

    def _save_manifest(self) -> None:
        MANIFEST_PATH.write_text(json.dumps(self._manifest, indent=2))

    # ------------------------------------------------------------------
    # Push (preprocessing → GDrive)
    # ------------------------------------------------------------------

    def _push_with_retry(
        self, dvc_file: Path, retries: int = 3, delay: float = 10.0
    ) -> None:
        """Run dvc push with exponential backoff on network errors."""
        for attempt in range(1, retries + 1):
            result = subprocess.run(["dvc", "push", str(dvc_file)])
            if result.returncode == 0:
                return
            if attempt < retries:
                wait = delay * attempt
                print(
                    f"  Push failed (attempt {attempt}/{retries}), retrying in {wait:.0f}s..."
                )
                time.sleep(wait)
        raise RuntimeError(f"dvc push failed after {retries} attempts: {dvc_file.name}")

    def _evict_cache_entry(self, dvc_file: Path) -> None:
        """Delete the local DVC cache copy for this chunk.

        After a successful push the blob is safe on remote — the local cache
        copy is just a space-wasting optimisation we don't need in a streaming
        preprocessing flow.  The .dvc pointer file is kept so future
        `dvc pull` calls can still retrieve the chunk from GDrive.
        """
        try:
            import yaml  # PyYAML is a DVC dependency, always present

            meta = yaml.safe_load(dvc_file.read_text())
            md5: str = meta["outs"][0]["md5"]
            # DVC stores blobs at <cache_root>/files/md5/<md5[:2]>/<md5[2:]>
            cache_root = get_project_root() / ".dvc" / "cache" / "files" / "md5"
            cache_entry = cache_root / md5[:2] / md5[2:]
            if cache_entry.exists():
                cache_entry.unlink()
        except Exception as exc:
            # Non-fatal: worst case the cache stays on disk.
            print(f"  Warning: could not evict cache entry ({exc})")

    def push_chunk(
        self,
        chunk_path: Path,
        num_graphs: int,
        delete_local: bool = True,
    ) -> None:
        """Register chunk in DVC cache, push to GDrive, optionally delete local file.

        Creates a `<chunk>.dvc` pointer file next to the chunk so DVC can
        later pull the file back from remote.  The local cache copy is removed
        immediately after push to keep disk usage flat during long preprocessing
        runs.
        """
        chunk_path = Path(chunk_path)

        # Add to DVC cache — creates chunk_path.dvc pointer file
        subprocess.run(["dvc", "add", str(chunk_path)], check=True)

        # Push to GDrive with retry
        dvc_file = Path(str(chunk_path) + ".dvc")
        self._push_with_retry(dvc_file)

        # Remove local cache copy — data is safe on remote, no need to keep it.
        self._evict_cache_entry(dvc_file)

        # Record in manifest
        self._manifest["chunks"][chunk_path.name] = num_graphs
        self._save_manifest()

        if delete_local:
            chunk_path.unlink()
            print(f"  Pushed and deleted local: {chunk_path.name}")

    # ------------------------------------------------------------------
    # Pull (training → local disk)
    # ------------------------------------------------------------------

    def pull_chunk(self, chunk_name: str) -> Path:
        """Pull a chunk from GDrive if not already local. Returns local path."""
        chunk_path = self.processed_dir / chunk_name

        if chunk_path.exists():
            return chunk_path

        dvc_file = self.processed_dir / f"{chunk_name}.dvc"
        if not dvc_file.exists():
            raise FileNotFoundError(
                f"No .dvc pointer for '{chunk_name}'. "
                "Run preprocessing with DvcChunkStore first."
            )

        print(f"  Pulling {chunk_name} from remote...")
        subprocess.run(["dvc", "pull", str(dvc_file)], check=True)
        return chunk_path

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_chunks(self) -> list[str]:
        """Sorted list of all chunk names available on remote."""
        return sorted(self._manifest["chunks"].keys())

    def chunk_size(self, chunk_name: str) -> int:
        """Number of graphs in the given chunk."""
        return self._manifest["chunks"].get(chunk_name, 0)

    def total_graphs(self) -> int:
        return sum(self._manifest["chunks"].values())
