import os
import multiprocessing as mp
import logging
import tiktoken
import pyarrow.parquet as pq
import glob
from typing import Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _token_generator_worker(worker_id, num_workers, token_chunk_size, stop_event, token_queue):
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
        self.config = config
        self.token_queue = mp.Queue(maxsize=config.token_queue_max_chunks)
        self.stop_event = mp.Event()
        self.processes = []
        self.leftover_tokens = []
        
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
        import queue
        import threading
        self.config = config
        self.token_stream = token_stream
        self.batch_queue = queue.Queue(maxsize=config.num_prefetch_batches)
        self.running = True
        self.worker = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.worker.start()

    def _prefetch_loop(self):
        import mlx.core as mx
        while self.running:
            try:
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
        """Returns lists of arrays (one for each micro-batch)."""
        return self.batch_queue.get()

    def stop(self):
        self.running = False
        self.worker.join(timeout=1.0)
