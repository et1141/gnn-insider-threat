"""DVC-based chunk store for streaming processed graph data to/from Google Drive.

Handles per-chunk push during preprocessing and on-demand pull during training,
so the full dataset never needs to live locally at once.
"""

import json
import subprocess
from pathlib import Path

from certgnn.utils import get_project_root

class DvcChunkStore:
    """Stream graph chunks to/from DVC remote (Google Drive)."""

    def __init__(self, processed_dir: Path, manifest_path: Path | None = None):
        self.processed_dir = Path(processed_dir)
        self.manifest_path = manifest_path or (get_project_root() / "configs" / "chunks_manifest.json")
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"chunks": {}}  # {chunk_name: num_graphs}

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    # ------------------------------------------------------------------
    # Push (preprocessing → GDrive)
    # ------------------------------------------------------------------

    def push_chunk(
        self,
        chunk_path: Path,
        num_graphs: int,
        delete_local: bool = True,
    ) -> None:
        chunk_path = Path(chunk_path)

        # Add to DVC cache
        subprocess.run(["dvc", "add", str(chunk_path)], check=True)

        # Push to GDrive
        dvc_file = Path(str(chunk_path) + ".dvc")
        subprocess.run(["dvc", "push", str(dvc_file)], check=True)

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
        return sorted(self._manifest["chunks"].keys())

    def chunk_size(self, chunk_name: str) -> int:
        return self._manifest["chunks"].get(chunk_name, 0)

    def total_graphs(self) -> int:
        return sum(self._manifest["chunks"].values())