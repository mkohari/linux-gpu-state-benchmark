# SPDX-FileCopyrightText: 2026 Moiz Kohari
# SPDX-License-Identifier: MIT
"""Owned regular files, synchronous completion and measured Linux page-cache purge."""
import ctypes
import json
import mmap
import os
from pathlib import Path
import stat
import sys
import time


def residency(fd):
    length = os.fstat(fd).st_size
    if length == 0:
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.mincore
    call.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    call.restype = ctypes.c_int
    with mmap.mmap(fd, length, access=mmap.ACCESS_COPY) as region:
        anchor = ctypes.c_char.from_buffer(region)
        flags = (ctypes.c_ubyte * ((length + mmap.PAGESIZE - 1) // mmap.PAGESIZE))()
        try:
            if call(ctypes.addressof(anchor), length, flags) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return sum(flag & 1 for flag in flags)
        finally:
            del anchor


def evict(fd):
    if sys.platform != "linux" or not hasattr(os, "posix_fadvise"):
        raise RuntimeError("Purged filesystem measurement requires Linux posix_fadvise/mincore")
    started = time.perf_counter()
    for attempt in range(8):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        if residency(fd) == 0:
            return time.perf_counter() - started
        time.sleep(0.003 * (attempt + 1))
    raise RuntimeError("Owned payload file remains in Linux page cache")


def transfer(path, *, payload=None, purged=True):
    """Write a new file or read a complete existing file. No mmap payload loading."""
    path = Path(path)
    writing = payload is not None
    if writing and not payload:
        raise ValueError("Empty state payload")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL if writing else os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    started = time.perf_counter()
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("State payload must be a regular file")
        purge_seconds = 0.0
        if purged:
            purge_seconds += evict(fd)
        before = residency(fd) if purged else None
        io_start = time.perf_counter()
        if writing:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise IOError("State write did not complete")
                view = view[written:]
            os.fsync(fd)
            data = None
            size = len(payload)
        else:
            size = os.fstat(fd).st_size
            if size <= 0:
                raise IOError("State file is empty")
            chunks, remaining = [], size
            while remaining:
                chunk = os.read(fd, min(16 * 1024**2, remaining))
                if not chunk:
                    raise IOError("State file ended before its declared length")
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if os.read(fd, 1):
                raise IOError("State payload changed size during read")
        io_seconds = time.perf_counter() - io_start
        during = residency(fd) if purged else None
        if purged:
            purge_seconds += evict(fd)
        after = residency(fd) if purged else None
        receipt = {"operation": "write" if writing else "read", "bytes": size,
            "mode": "purged" if purged else "buffered-diagnostic",
            "resident_pages_before": before, "resident_pages_after_io": during,
            "resident_pages_after": after, "io_seconds": io_seconds,
            "purge_seconds": purge_seconds, "file_boundary_seconds": time.perf_counter() - started}
        return data, receipt
    finally:
        os.close(fd)


def append_audit(path, key, receipt):
    with Path(path).open("a", encoding="utf8") as f:
        f.write(json.dumps({"key": key, **receipt}, allow_nan=False) + "\n")
