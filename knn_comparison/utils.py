import time
import psutil
import os

class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_mem = 0

    def start(self):
        # Force garbage collection for cleaner measurements
        import gc
        gc.collect()
        self.start_mem = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_usage(self):
        current = self.process.memory_info().rss / 1024 / 1024
        return current - self.start_mem
