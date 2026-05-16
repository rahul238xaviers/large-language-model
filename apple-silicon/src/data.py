import os
import multiprocessing as mp
import logging
import tiktoken
import pyarrow.parquet as pq
import glob
from typing import Tuple
from utils import timed_log

logger = logging.getLogger("train")

def _token_generator_worker(worker_id, num_workers, token_chunk_size, stop_event, token_queue):
    """
    Multiprocessing worker that tokenizes Parquet text files and pushes
    fixed-size token chunks onto a shared queue.

    Parallelism strategy:
        `num_workers` processes are launched; worker `i` owns file indices
        {i, i + num_workers, i + 2*num_workers, ...} (interleaved assignment
        so large files are balanced across workers rather than one worker
        getting all the big files).

    Tokenisation:
        Uses tiktoken `cl100k_base` (GPT-4 / GPT-3.5 vocabulary, 100 277 tokens).
        Each document is tokenised with `encode_ordinary` (no special tokens)
        and an EOT marker is appended so the model learns document boundaries.

    Chunking:
        Tokens are buffered until `token_chunk_size` accumulate, then emitted
        as a list onto `token_queue`.  Partial trailing chunks are also emitted
        when a worker's file list is exhausted.

    Args:
        worker_id:        Index of this worker in [0, num_workers).
        num_workers:      Total number of parallel worker processes.
        token_chunk_size: Number of tokens per queue item.
        stop_event:       `multiprocessing.Event` — set to signal shutdown.
        token_queue:      `multiprocessing.Queue` — bounded, blocks when full.

    Example (conceptual, not called directly):
        # With 4 workers and 8 Parquet files:
        # worker 0 processes files [0, 4]
        # worker 1 processes files [1, 5]  … etc.
    """
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        local_data_path = os.path.join(os.getcwd(), "data/rust")
        local_files = sorted(glob.glob(os.path.join(local_data_path, "*.parquet")))
        if not local_files: return
        my_files = [f for i, f in enumerate(local_files) if i % num_workers == worker_id]
        token_buffer = []
        
        for file_path in my_files:
            if stop_event.is_set(): break
            parquet_file = pq.ParquetFile(file_path)
            for i in range(parquet_file.num_row_groups):
                if stop_event.is_set(): break
                table = parquet_file.read_row_group(i, columns=["content"])
                for batch in table.to_batches(max_chunksize=100):
                    if stop_event.is_set(): break
                    contents = batch.column("content").to_pylist()
                    for content in contents:
                        if not content: continue
                        tokens = encoder.encode_ordinary(content)
                        tokens.append(encoder.eot_token)
                        if stop_event.is_set(): break
                        token_buffer.extend(tokens)
                        while len(token_buffer) >= token_chunk_size:
                            token_queue.put(token_buffer[:token_chunk_size])
                            token_buffer = token_buffer[token_chunk_size:]

        if token_buffer and not stop_event.is_set():
            token_queue.put(token_buffer)
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}")

class ParallelTokenStream:
    def __init__(self, config):
        """
        Spin up `config.num_worker_threads` tokenizer worker processes and
        create the shared inter-process token queue.

        Design:
            Each process runs `_token_generator_worker` independently and
            pushes `token_chunk_size`-token lists onto a bounded
            `multiprocessing.Queue`.  The main process (or the prefetch thread)
            calls `get_batch()` to consume from that queue.

            Queue bound = `token_queue_max_chunks` items.  Each item is
            `token_chunk_size` integers (int64 = 8 bytes).  Total queue memory:
                max_memory ≈ token_queue_max_chunks × token_chunk_size × 8 bytes
                           = 128 × 4096 × 8 ≈ 4 MB  (negligible)

        Args:
            config: TrainingConfig with num_worker_threads, token_chunk_size,
                    token_queue_max_chunks, and block_size fields.
        """
        self.config = config
        with timed_log("ParallelTokenStream.__init__", getattr(config, "profile_methods", False)):
            self.token_queue = mp.Queue(maxsize=config.token_queue_max_chunks)
            self.stop_event = mp.Event()
            self.processes = []
            self.leftover_tokens = []
            self._batch_count = 0
        
        for i in range(config.num_worker_threads):
            p = mp.Process(
                target=_token_generator_worker,
                args=(
                    i,
                    config.num_worker_threads,
                    config.token_chunk_size,
                    self.stop_event,
                    self.token_queue,
                ),
            )
            p.daemon = True
            p.start()
            self.processes.append(p)

    def get_batch(self) -> Tuple[list, list]:
        """
        Consume exactly `block_size + 1` tokens from the queue and split into
        an input sequence x and a shifted target sequence y.

        Math (causal language modelling objective):
            tokens = [t_0, t_1, ..., t_{T}]   length T+1, T = block_size
            x      = tokens[0:T]              = [t_0, t_1, ..., t_{T-1}]
            y      = tokens[1:T+1]            = [t_1, t_2, ..., t_T]

        At each position i the model predicts t_{i+1} given t_0...t_i.
        Cross-entropy loss is computed over all T positions simultaneously.

        The method maintains `leftover_tokens` between calls so that token
        boundaries align with queue chunk boundaries: no token is ever
        discarded or double-counted.

        Returns:
            x: list of T integers (input token IDs).
            y: list of T integers (target token IDs, shifted right by 1).

        Raises:
            RuntimeError: If the queue is empty for more than 30 seconds
                          (likely indicates all workers have crashed).

        Example:
            x, y = stream.get_batch()   # len(x) == len(y) == block_size == 2048
            # x[0] == tokens[0], y[0] == tokens[1]  (teacher-forcing targets)
        """
        self._batch_count += 1
        log_this = self.config.profile_methods and (self._batch_count <= 5 or self._batch_count % 64 == 0)
        with timed_log(f"ParallelTokenStream.get_batch[{self._batch_count}]", log_this):
            target_len = self.config.block_size + 1
            while len(self.leftover_tokens) < target_len:
                try:
                    chunk = self.token_queue.get(timeout=30)
                    self.leftover_tokens.extend(chunk)
                except: raise RuntimeError("Data stream timeout.")
            out = self.leftover_tokens[:target_len]
            self.leftover_tokens = self.leftover_tokens[target_len:]
            return out[:-1], out[1:]

    def stop(self):
        """
        Gracefully shut down all tokenizer worker processes.

        Shutdown sequence:
            1. Set `stop_event` so workers stop reading after the current file.
            2. Drain `token_queue` to unblock any workers blocked on `put()`.
            3. `terminate()` each process (SIGTERM) and `join()` with a short
               timeout to avoid hanging the main process.

        It is safe to call `stop()` multiple times; subsequent calls are no-ops
        because `stop_event` is already set and the queue is already empty.
        """
        with timed_log("ParallelTokenStream.stop", getattr(self.config, "profile_methods", False)):
            self.stop_event.set()
            while not self.token_queue.empty():
                try: self.token_queue.get_nowait()
                except: break
            for p in self.processes:
                p.terminate()
                p.join(timeout=0.1)
            logger.info("ParallelTokenStream (Multiprocessing) stopped.")

class AsyncBatchPrefetcher:
    """Asynchronously batches tokens and supports full-iteration prefetching."""
    def __init__(self, config, token_stream):
        """
        Start a background daemon thread that assembles complete training
        iterations (all micro-batches) and buffers them in a thread queue.

        Design:
            The prefetch thread calls `token_stream.get_batch()` in a tight
            loop, grouping results into micro-batch MLX arrays:

                For each of `gradient_accumulation_steps` micro-batches:
                    For each of `micro_batch_size` sequences:
                        x, y = token_stream.get_batch()   # (block_size,) each
                    micro_x = mx.array(micro_x)            # (mbs, block_size)
                    micro_y = mx.array(micro_y)            # (mbs, block_size)

            One iteration = `gradient_accumulation_steps` such arrays.
            Completed iterations are pushed onto `batch_queue` (bounded by
            `num_prefetch_batches`) so the GPU never waits for data.

        Memory per buffered iteration:
            tokens  = mbs × acc × block_size × 2 (x+y) × 4 bytes (int32)
                    = 16 × 8 × 2048 × 2 × 4 ≈ 4 MB  (negligible)

        Args:
            config:       TrainingConfig.
            token_stream: An initialised `ParallelTokenStream` instance.
        """
        import queue
        import threading
        self.config = config
        self.token_stream = token_stream
        with timed_log("AsyncBatchPrefetcher.__init__", getattr(config, "profile_methods", False)):
            self.batch_queue = queue.Queue(maxsize=config.num_prefetch_batches)
            self.running = True
            self._prefetch_count = 0
            self.worker = threading.Thread(target=self._prefetch_loop, daemon=True)
            self.worker.start()

    def _prefetch_loop(self):
        """
        Background thread body: continuously assemble and buffer full training
        iterations.

        Each iteration consists of `gradient_accumulation_steps` micro-batches,
        each of shape `(micro_batch_size, block_size)`.  The assembled iteration
        is a pair of lists:
            batch_x[i]: MLX array shape (mbs, block_size) — input tokens
            batch_y[i]: MLX array shape (mbs, block_size) — target tokens

        The loop blocks on `batch_queue.put()` when the queue is full, which
        provides natural backpressure — tokenization workers are not allowed
        to run ahead by more than `num_prefetch_batches` full iterations.
        """
        import mlx.core as mx
        while self.running:
            try:
                self._prefetch_count += 1
                with timed_log(f"AsyncBatchPrefetcher._prefetch_loop[{self._prefetch_count}]", getattr(self.config, "profile_methods", False)):
                    batch_x, batch_y = [], []
                    # Prefetch a full effective iteration (all micro-batches)
                    for _ in range(self.config.gradient_accumulation_steps):
                        micro_x, micro_y = [], []
                        for _ in range(self.config.micro_batch_size):
                            x, y = self.token_stream.get_batch()
                            micro_x.append(x)
                            micro_y.append(y)
                        batch_x.append(mx.array(micro_x))
                        batch_y.append(mx.array(micro_y))
                
                while self.running:
                    try:
                        self.batch_queue.put((batch_x, batch_y), timeout=1.0)
                        break
                    except: continue
            except: break

    def get_full_iteration(self):
        """
        Retrieve the next pre-assembled full training iteration from the buffer.

        Blocks until one iteration is available in the queue (the prefetch
        thread runs ahead, so in practice this is nearly always instant after
        the first iteration).

        Returns:
            Tuple (batch_x, batch_y) where each is a list of
            `gradient_accumulation_steps` MLX arrays of shape
            `(micro_batch_size, block_size)`.

        Example:
            batch_x, batch_y = prefetcher.get_full_iteration()
            # len(batch_x) == gradient_accumulation_steps == 8
            # batch_x[0].shape == (16, 2048)  [mbs=16, block_size=2048]
        """
        with timed_log("AsyncBatchPrefetcher.get_full_iteration", getattr(self.config, "profile_methods", False)):
            return self.batch_queue.get()

    def stop(self):
        """
        Signal the prefetch thread to exit and wait for it to finish.

        Sets `self.running = False` so the thread loop exits at its next
        iteration check.  A 1-second join timeout is used to prevent the main
        process from hanging indefinitely if the thread is blocked on a
        queue operation.
        """
        with timed_log("AsyncBatchPrefetcher.stop", getattr(self.config, "profile_methods", False)):
            self.running = False
            self.worker.join(timeout=1.0)
