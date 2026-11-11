#!/usr/bin/env python3
"""Validate the frozen-CLAP audio NPZ format expected at the method boundary.

This script reads user-supplied embeddings; it does not extract or publish them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath

import numpy as np


REQUIRED = {"embeddings", "labels", "sample_ids", "paths", "model_id", "audio_padding"}


def scalar_text(value: np.ndarray) -> str:
    value = np.asarray(value)
    if value.ndim != 0 or value.dtype.kind not in "US":
        raise ValueError("Expected a scalar metadata string")
    item = value.item()
    text = item.decode("utf-8") if isinstance(item, bytes) else item
    if not text or text != text.strip():
        raise ValueError("Expected a nonempty scalar metadata string without surrounding whitespace")
    return text


def text_vector(value: np.ndarray, name: str, count: int) -> list[str]:
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "US" or len(value) != count:
        raise ValueError(f"{name}: expected a one-dimensional string array with {count} rows")
    values = [item.decode("utf-8") if isinstance(item, bytes) else item for item in value.tolist()]
    if any(not item or item != item.strip() for item in values):
        raise ValueError(f"{name}: empty strings or surrounding whitespace are not allowed")
    return values


def canonical_path(value: str) -> str:
    """Return a conservative, case-insensitive portable path identity.

    No filesystem lookup is performed. Case folding deliberately treats paths
    differing only in case as a collision, even on case-sensitive filesystems.
    """
    windows_path = PureWindowsPath(value)
    if windows_path.drive or windows_path.root or Path(value).is_absolute():
        raise ValueError("paths must be relative to the user's audio root, without a drive or root")
    parts = value.replace("\\", "/").split("/")
    if ".." in parts:
        raise ValueError("paths must not contain '..' components")
    parts = [part for part in parts if part not in ("", ".")]
    if not parts:
        raise ValueError("paths must identify a file, not an empty path or '.'")
    if any(part != part.strip() or part.endswith(".") or ":" in part or
           any(ord(character) < 32 for character in part) for part in parts):
        raise ValueError("paths contain ambiguous or nonportable components")
    return "/".join(parts).casefold()


def validate(path: Path, protocol: dict[str, object]) -> tuple[dict[str, object], set[str], set[str]]:
    report, sample_ids, paths, _ = _validate_with_sessions(path, protocol)
    return report, sample_ids, paths


def _validate_with_sessions(
    path: Path, protocol: dict[str, object]
) -> tuple[dict[str, object], set[str], set[str], set[str] | None]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED - set(archive.files)
        if missing:
            raise ValueError(f"{path.name}: missing arrays {sorted(missing)}")
        embeddings = np.asarray(archive["embeddings"])
        dimension = int(protocol["expected_audio_embedding_dimension"])
        if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
            raise ValueError(f"{path.name}: expected a two-dimensional {dimension}-column embedding matrix")
        if embeddings.dtype.kind != "f":
            raise ValueError(f"{path.name}: embeddings must contain real floating-point values")
        count = embeddings.shape[0]
        if count == 0:
            raise ValueError(f"{path.name}: empty embedding matrix")
        labels = text_vector(archive["labels"], "labels", count)
        sample_ids = text_vector(archive["sample_ids"], "sample_ids", count)
        paths = text_vector(archive["paths"], "paths", count)
        session_ids = (set(text_vector(archive["session_ids"], "session_ids", count))
                       if "session_ids" in archive else None)
        model_id = scalar_text(archive["model_id"])
        padding = scalar_text(archive["audio_padding"])
        window = None
        if "analysis_window_seconds" in archive:
            window_array = np.asarray(archive["analysis_window_seconds"])
            if window_array.ndim != 0 or window_array.dtype.kind not in "fiu":
                raise ValueError(f"{path.name}: analysis_window_seconds must be a real numeric scalar")
            window = float(window_array.item())
        # Some historical training NPZ files record an unspecified window as NaN.
        if window is not None and np.isnan(window):
            window = None
        if window is not None and not np.isfinite(window):
            raise ValueError(f"{path.name}: analysis window must be finite or NaN for unspecified")

    if not np.isfinite(embeddings).all():
        raise ValueError(f"{path.name}: non-finite embedding values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
        raise ValueError(f"{path.name}: embeddings are not L2-normalized")
    if not set(labels).issubset(set(protocol["all_labels"])):
        raise ValueError(f"{path.name}: labels outside the five-class task")
    if len(set(sample_ids)) != count:
        raise ValueError(f"{path.name}: duplicate sample IDs")
    normalized_paths = {canonical_path(value) for value in paths}
    if len(normalized_paths) != count:
        raise ValueError(f"{path.name}: duplicate paths after portable case-insensitive normalization")
    if model_id != protocol["expected_audio_model_id"] or padding != protocol["expected_audio_padding"]:
        raise ValueError(f"{path.name}: model ID or padding differs from the evaluated protocol")
    if window is not None and abs(window - float(protocol["audio_window_seconds"])) > 1e-9:
        raise ValueError(f"{path.name}: analysis window differs from the evaluated protocol")
    report = {"file": path.name, "rows": count, "dimensions": embeddings.shape[1],
              "labels_present": sorted(set(labels)), "model_id": model_id, "audio_padding": padding,
              "analysis_window_seconds": window, "path_identity_policy": "portable_casefold",
              "session_ids_present": session_ids is not None,
              "session_count": len(session_ids) if session_ids is not None else None}
    return report, set(sample_ids), normalized_paths, session_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings", nargs="+", type=Path, help="Audio embedding NPZ files")
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "configs/protocol.json")
    parser.add_argument("--disjoint", action="store_true",
                        help="Require no shared sample IDs or normalized paths; check supplied session IDs too")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_sessions: set[str] = set()
    session_files = 0
    reports = []
    for path in args.embeddings:
        report, sample_ids, paths, session_ids = _validate_with_sessions(path, protocol)
        if args.disjoint and (seen_ids & sample_ids or seen_paths & paths):
            raise ValueError(f"Overlap with an earlier split: {path.name}")
        if args.disjoint and session_ids is not None and seen_sessions & session_ids:
            raise ValueError(f"Session overlap with an earlier split: {path.name}")
        seen_ids.update(sample_ids)
        seen_paths.update(paths)
        if session_ids is not None:
            seen_sessions.update(session_ids)
            session_files += 1
        reports.append(report)
    print(json.dumps({"validated": reports, "disjoint_checked": args.disjoint,
                      "disjoint_check_level": "sample_ids_and_normalized_paths" if args.disjoint else "not_requested",
                      "session_disjoint_checked": bool(args.disjoint and len(reports) >= 2 and session_files == len(reports)),
                      "session_metadata_complete": session_files == len(reports),
                      "audio_content_checked": False}, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
