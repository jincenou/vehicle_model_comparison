"""Bounded, mutation-aware content checks for incremental evaluation intake."""
import hashlib
from pathlib import Path
import time


class Deferred(Exception):
    pass


def signature(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


class HashCache:
    def __init__(self, entries=None, byte_budget=1024**3):
        self.entries = entries if entries is not None else {}
        self.byte_budget = byte_budget
        self.bytes_read = 0
        self.hits = 0

    def check(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        before = signature(path)
        key = str(path)
        cached = self.entries.get(key)
        # Some filesystems coalesce timestamps for rapid same-size writes. Do
        # not reuse a signature for a recently written file, even within a pass.
        settled = time.time_ns() - max(before[3:]) > 2_000_000_000
        if cached and cached['signature'] == before and settled:
            actual = cached['sha256']
            self.hits += 1
        else:
            # Permit one oversized file per pass; otherwise it would starve forever.
            if self.bytes_read and self.bytes_read + before[2] > self.byte_budget:
                raise Deferred('hash_budget_exhausted')
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b''):
                    h.update(block)
                    self.bytes_read += len(block)
            if signature(path) != before:
                self.entries.pop(key, None)
                raise ValueError('file changed during hashing: ' + key)
            actual = h.hexdigest()
            self.entries[key] = dict(signature=before, sha256=actual)
        if expected is not None and actual != expected:
            raise ValueError('SHA256 mismatch: ' + key)
        return actual
