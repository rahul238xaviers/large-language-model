"""Stage 5 — memory-mapped sequence loader for MLX training.

Reads ``sequences.npy`` produced by Stage 4 (shape ``(N, block_size)``,
dtype ``uint32``) and streams ``mx.array`` mini-batches of shape
``(batch_size, block_size)`` with optional shuffling.

Design notes
------------
* Uses ``np.load(path, mmap_mode='r')`` so only the pages actually accessed
  are loaded from disk.  On Apple Silicon the OS satisfies most reads from
  the Unified Memory cache after the first epoch — effectively zero-copy.
* Token IDs are cast from uint32 → int32 before wrapping in ``mx.array``
  because MLX embedding lookup expects a signed integer type.
* ``reset()`` re-shuffles the index permutation at the start of each epoch
  using a deterministic ``numpy.default_rng`` seeded at construction time.

Public interface:
    SequenceLoader(sequences_path, batch_size, shuffle, seed)
        len(loader)     → int   number of complete batches per epoch
        iter(loader)    → Iterator[mx.array]   shape (batch_size, block_size)
        loader.reset()  → re-shuffle for a new epoch
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx

logger = logging.getLogger(__name__)


class SequenceLoader:
    """Memory-mapped mini-batch iterator over tokenised sequences.

    Args:
        sequences_path: Path to ``sequences.npy`` (shape ``(N, T)``, uint32).
        batch_size:     Number of sequences per mini-batch.
        shuffle:        Shuffle sequence order at the start of each iteration.
        seed:           RNG seed for reproducible shuffles.

    Example::

        loader = SequenceLoader("runs/x/data/tokenised/sequences.npy", batch_size=4)
        for batch in loader:          # mx.array (4, 2048)
            ...
    """

    def __init__(
        self,
        sequences_path: Path | str,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self._path = Path(sequences_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Sequences file not found: {self._path}")

        # Memory-mapped read — only pages we touch are loaded into RAM
        self._arr = np.load(str(self._path), mmap_mode="r")  # (N, T) uint32
        if self._arr.ndim != 2:
            raise ValueError(
                f"Expected 2-D (N, block_size) array, got shape {self._arr.shape}"
            )

        self._n_sequences = self._arr.shape[0]
        self._block_size  = self._arr.shape[1]
        self._batch_size  = batch_size
        self._shuffle     = shuffle
        self._rng         = np.random.default_rng(seed)
        self._indices     = np.arange(self._n_sequences)

        logger.info(
            "SequenceLoader: n=%d  T=%d  batch=%d  batches/epoch=%d",
            self._n_sequences, self._block_size, batch_size, len(self),
        )

    # ── Read-only properties ─────────────────────────────────────────── #

    @property
    def n_sequences(self) -> int:
        """Total number of sequences in the dataset."""
        return self._n_sequences

    @property
    def block_size(self) -> int:
        """Sequence length (tokens per sequence)."""
        return self._block_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        """Number of complete mini-batches per epoch."""
        return self._n_sequences // self._batch_size

    # ── Iteration ────────────────────────────────────────────────────── #

    def reset(self) -> None:
        """Re-shuffle the index permutation for a new epoch.

        Calling this manually is optional — ``__iter__`` calls it automatically.
        """
        if self._shuffle:
            self._rng.shuffle(self._indices)

    def __iter__(self) -> Iterator["mx.array"]:
        """Yield batches as MLX int32 arrays of shape ``(batch_size, block_size)``."""
        import mlx.core as mx

        self.reset()
        n_batches = len(self)
        for i in range(n_batches):
            start = i * self._batch_size
            idx   = self._indices[start : start + self._batch_size]
            # Copy the memmap slice to a contiguous array and cast to int32
            batch_np = np.array(self._arr[idx], dtype=np.int32)
            yield mx.array(batch_np)
