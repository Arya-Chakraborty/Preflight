"""Exclusive data-dir lock so two Gateways cannot share a WAL set.

Caveat: on POSIX this uses ``fcntl.flock``, which is *advisory* (only cooperating
processes that also take the lock are excluded) and is unreliable on network
filesystems such as NFS/SMB. Keep ``data_dir`` on a local disk. If you must run
on a shared filesystem, set ``allow_multi_writer`` only when you have arranged
exclusivity by other means.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class DataDirLocked(RuntimeError):
    """Another Preflight process already holds this data_dir."""


class DataDirLock:
    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "preflight.lock"
        self._fh = None
        self.held = False

    def acquire(self) -> None:
        if self.held:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        fh.seek(0, 2)
        if fh.tell() == 0:
            fh.write("0")
            fh.flush()
        try:
            _lock_exclusive(fh)
        except OSError as exc:
            fh.close()
            holder = _read_holder(self.path)
            extra = f" (held by pid {holder})" if holder else ""
            raise DataDirLocked(
                f"data_dir {self.path.parent} is already in use{extra}: {exc}"
            ) from exc
        # Write the PID then truncate trailing bytes (rather than truncate-then-write)
        # so a concurrent reader never observes an empty holder file.
        pid = str(os.getpid())
        fh.seek(0)
        fh.write(pid)
        fh.truncate(len(pid))
        fh.flush()
        self._fh = fh
        self.held = True

    def __enter__(self) -> DataDirLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        fh = self._fh
        self._fh = None
        self.held = False
        if fh is None:
            return
        try:
            _unlock(fh)
        finally:
            fh.close()


def _lock_exclusive(fh) -> None:
    if sys.platform == "win32":
        import msvcrt

        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fh) -> None:
    if sys.platform == "win32":
        import msvcrt

        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _read_holder(path: Path) -> str | None:
    try:
        text = path.read_text().strip()
        return text or None
    except OSError:
        return None
