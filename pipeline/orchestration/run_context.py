"""Run context: unique run ID and per-run directory management."""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunContext:
    """Represents a single pipeline execution with a unique, sealed run directory."""

    run_id: str
    run_dir: Path

    @classmethod
    def create(cls, base_dir: Path = Path("runs")) -> "RunContext":
        """Generate a new run ID and create the corresponding directory."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short_hash = hashlib.sha1(timestamp.encode()).hexdigest()[:7]
        run_id = f"run_{timestamp}_{short_hash}"
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(run_id=run_id, run_dir=run_dir)

    @classmethod
    def resume(cls, run_id: str, base_dir: Path = Path("runs")) -> "RunContext":
        """Resume an existing run by ID (directory must already exist)."""
        run_dir = base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return cls(run_id=run_id, run_dir=run_dir)

    def stage_dir(self, stage: str) -> Path:
        """Return (and create) the sub-directory for the given pipeline stage."""
        d = self.run_dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def config_dir(self) -> Path:
        """Return the config snapshot directory for this run."""
        return self.stage_dir("config")
