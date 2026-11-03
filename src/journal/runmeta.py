"""Run metadata written next to every journal-extension result."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(root: Path) -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip())
        return sha + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"


def write_meta(path: Path, experiment_id: str, root: Path, **fields) -> dict:
    meta = dict(experiment_id=experiment_id, timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                git_commit=git_sha(root), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, platform=platform.platform(), **fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str))
    return meta
