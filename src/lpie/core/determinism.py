"""Global seeding and provenance hashes.

Every reported number must be regenerable. `seed_everything` pins every RNG the
pipeline touches; `hashes` records what was actually used so a submission
manifest can prove which code, config and data produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int, *, single_threaded: bool = False) -> None:
    """Pin every RNG. `single_threaded` also removes thread-scheduling nondeterminism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if single_threaded:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"
    try:  # torch is optional at runtime
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
        if single_threaded:
            torch.set_num_threads(1)
    except Exception:  # pragma: no cover - torch not installed
        pass


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    """Stable hash of any JSON-serialisable object (sorted keys, no whitespace drift)."""
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(payload.encode("utf-8"))


def git_sha(root: Path | None = None) -> str | None:
    """Short git SHA, or None when the tree is not a git repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = out.stdout.strip()
        return sha or None
    except Exception:
        return None


def git_dirty(root: Path | None = None) -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except Exception:
        return None


def dataset_hashes(dataset_dir: Path, files: dict[str, str]) -> dict[str, str]:
    """SHA256 of each input file that exists. Missing files are reported as absent."""
    out: dict[str, str] = {}
    for key, name in files.items():
        p = dataset_dir / name
        out[key] = sha256_file(p) if p.exists() else "ABSENT"
    return out


def combined_data_sha256(hashes: dict[str, str]) -> str:
    return sha256_obj(hashes)
