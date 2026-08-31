"""
Ported from sdpype/core/pipeline.py's _calculate_training_data_hash, made
public. The rich-console warning-on-missing-file behavior is dropped in
favor of plain exceptions/return sentinels — this module has no CLI/display
responsibilities.
"""

import hashlib
from pathlib import Path


def calculate_file_hash(file_path: str, max_bytes: int = 1024 * 1024) -> str:
    """
    Calculate SHA256 hash of a data file for unique experiment identification.

    Only the first `max_bytes` of the file are hashed (default 1MB), matching
    the original's tradeoff of speed over hashing full multi-GB files.

    Args:
        file_path: Path to the data file
        max_bytes: Maximum bytes to read for hash calculation

    Returns:
        8-character truncated SHA256 hex digest, or a sentinel string
        ("unknown" if the file doesn't exist, "hashfail" on any other error)
    """
    try:
        if not Path(file_path).exists():
            return "unknown"

        hash_sha256 = hashlib.sha256()
        bytes_read = 0
        with open(file_path, "rb") as f:
            while bytes_read < max_bytes:
                chunk_size = min(8192, max_bytes - bytes_read)
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_sha256.update(chunk)
                bytes_read += len(chunk)

        return hash_sha256.hexdigest()[:8]

    except Exception:
        return "hashfail"
