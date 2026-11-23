"""Atomic checkpoints with retry for transient locks on the workspace filesystem."""
import json
import os
from pathlib import Path
import tempfile
import time


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(9):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == 8:
                    raise
                time.sleep(min(.025 * 2**attempt, .3))
    finally:
        temp.unlink(missing_ok=True)
