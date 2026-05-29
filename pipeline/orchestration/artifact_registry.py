"""Per-run artifact registry for cross-stage dependency resolution.

Each stage registers its output artifacts by a (stage, artifact) key.
Downstream stages look up dependencies from the registry rather than
constructing paths from assumptions.
"""

import json
from pathlib import Path
from typing import Any


class ArtifactRegistry:
    """Persists artifact path pointers inside the run directory."""

    _FILENAME = "artifact_registry.json"

    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / self._FILENAME
        self._data: dict[str, Any] = {}
        if self._path.exists():
            with open(self._path) as fh:
                self._data = json.load(fh)

    def register(self, stage: str, artifact: str, path: Path) -> None:
        """Record an artifact path produced by a stage."""
        self._data.setdefault(stage, {})[artifact] = str(path)
        self._flush()

    def get(self, stage: str, artifact: str) -> Path:
        """Retrieve an artifact path by stage and artifact name."""
        try:
            return Path(self._data[stage][artifact])
        except KeyError:
            raise KeyError(f"Artifact not registered: stage={stage!r} artifact={artifact!r}")

    def list_stage(self, stage: str) -> dict[str, Path]:
        """Return all registered artifacts for a stage."""
        return {k: Path(v) for k, v in self._data.get(stage, {}).items()}

    # ------------------------------------------------------------------ #

    def _flush(self) -> None:
        with open(self._path, "w") as fh:
            json.dump(self._data, fh, indent=2)
