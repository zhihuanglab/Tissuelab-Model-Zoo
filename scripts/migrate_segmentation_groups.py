#!/usr/bin/env python3
"""
Migrate existing .zarr stores to the new Cell-Segmentation layout.

Old layout                              →  New layout
─────────────────────────────────────────────────────────────────
SegmentationNode/                       →  Cell-Segmentation/
  centroids                                 centroids
  contours                                  contours
  embedding         (singular)         →    embeddings        (plural)
  probability       (singular)         →    probabilities     (plural)
  output            (JSON bytes blob)  →    metadata          (group + attrs)

Usage:

    python scripts/migrate_segmentation_groups.py /path/to/folder
    python scripts/migrate_segmentation_groups.py /path/to/folder --dry-run
    python scripts/migrate_segmentation_groups.py /path/to/folder --workers 4

The walker follows symlinks so linked sample slides are migrated too.

Idempotent: re-running on an already-migrated store is a no-op (no group
named SegmentationNode left). Per-store failures are isolated and reported
at the end; one bad zarr does not abort the whole run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import zarr

OLD_GROUP = "SegmentationNode"
NEW_GROUP = "Cell-Segmentation"
DATASET_RENAMES = {
    "embedding": "embeddings",
    "probability": "probabilities",
}
OUTPUT_DATASET = "output"  # legacy JSON bytes blob — promoted to metadata attrs


def find_zarr_stores(root: Path) -> List[Path]:
    """Return every directory ending in .zarr under root (follows symlinks)."""
    stores: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        for d in list(dirnames):
            if d.endswith(".zarr"):
                stores.append(Path(dirpath) / d)
                # Don't descend into a zarr's internals — they contain chunk dirs.
                dirnames.remove(d)
    return stores


def _rename_group_fs(zarr_path: Path) -> bool:
    """
    Rename SegmentationNode → Cell-Segmentation at the filesystem level
    (works only for DirectoryStore-backed zarrs). Returns True if renamed.
    """
    old = zarr_path / OLD_GROUP
    new = zarr_path / NEW_GROUP
    if not old.is_dir():
        return False
    if new.exists():
        # Both exist — pick the most-recently-modified, drop the other.
        # In practice this means a previous migration ran but failed mid-way,
        # so the new dir is incomplete. Caller decides; here we refuse to merge.
        raise RuntimeError(
            f"both '{OLD_GROUP}' and '{NEW_GROUP}' exist in {zarr_path}; manual review required"
        )
    os.rename(old, new)
    return True


def _zarr_copy(src, dst_parent, name):
    """zarr v3 replacement for the removed ``zarr.copy``: deep-copy an array or
    group ``src`` into ``dst_parent`` under ``name``, preserving dtype/attrs."""
    if isinstance(src, zarr.Group):
        dst = dst_parent.require_group(name)
        dst.attrs.update(dict(src.attrs))
        for key in src.keys():
            _zarr_copy(src[key], dst, key)
        return dst
    dst = dst_parent.create_array(name, data=src[...], chunks=src.chunks, overwrite=True)
    dst.attrs.update(dict(src.attrs))
    return dst


def _rename_datasets(seg_grp) -> List[str]:
    """In-place rename embedding→embeddings, probability→probabilities."""
    renamed = []
    for old_key, new_key in DATASET_RENAMES.items():
        if old_key in seg_grp and new_key not in seg_grp:
            _zarr_copy(seg_grp[old_key], seg_grp, new_key)
            del seg_grp[old_key]
            renamed.append(f"{old_key} → {new_key}")
    return renamed


def _promote_output_to_metadata(seg_grp) -> bool:
    """
    Replace legacy /output JSON-bytes blob with /metadata group + attrs.
    Returns True if conversion happened.
    """
    if OUTPUT_DATASET not in seg_grp:
        return False
    raw = bytes(seg_grp[OUTPUT_DATASET][()])
    payload = {}
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except Exception:
        payload = {}
    if "metadata" in seg_grp:
        del seg_grp["metadata"]
    meta = seg_grp.create_group("metadata")
    meta.attrs.update({
        "status": payload.get("status", "unknown"),
        "message": payload.get("message", ""),
        "nuclei_count": int(payload.get("nuclei_count", 0)),
        "has_embeddings": "embeddings" in seg_grp,
        "has_probabilities": "probabilities" in seg_grp,
        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    del seg_grp[OUTPUT_DATASET]
    return True


def migrate_one(zarr_path: Path, dry_run: bool) -> Tuple[Path, str, List[str]]:
    """
    Returns (path, status, actions).
    status ∈ {"migrated", "skipped", "error"}; actions is a list of human-readable
    descriptions of what was changed (or would be changed in dry-run).
    """
    actions: List[str] = []
    try:
        old = zarr_path / OLD_GROUP
        new = zarr_path / NEW_GROUP
        if not old.is_dir() and not new.is_dir():
            return zarr_path, "skipped", ["no segmentation group present"]

        # Step 1 — rename group via filesystem (DirectoryStore only).
        if old.is_dir():
            if new.exists():
                raise RuntimeError(
                    f"both '{OLD_GROUP}' and '{NEW_GROUP}' present — manual review required"
                )
            if dry_run:
                actions.append(f"would rename {OLD_GROUP} → {NEW_GROUP}")
            else:
                _rename_group_fs(zarr_path)
                actions.append(f"renamed {OLD_GROUP} → {NEW_GROUP}")

        if dry_run:
            # Without actually renaming, we open the OLD group to inspect.
            zf = zarr.open_group(str(zarr_path), mode="r")
            seg_grp = zf[OLD_GROUP] if OLD_GROUP in zf else zf.get(NEW_GROUP)
            if seg_grp is None:
                return zarr_path, "skipped", actions
            for old_key, new_key in DATASET_RENAMES.items():
                if old_key in seg_grp:
                    actions.append(f"would rename dataset {old_key} → {new_key}")
            if OUTPUT_DATASET in seg_grp:
                actions.append(f"would promote {OUTPUT_DATASET} → metadata group + attrs")
            return zarr_path, "migrated" if actions and actions[0] != "no segmentation group present" else "skipped", actions

        # Step 2 — open the (renamed) group writable, do dataset renames + output promotion.
        zf = zarr.open_group(str(zarr_path), mode="a")
        if NEW_GROUP not in zf:
            return zarr_path, "skipped", actions or ["nothing to migrate"]
        seg_grp = zf[NEW_GROUP]

        renamed = _rename_datasets(seg_grp)
        actions.extend(renamed)

        if _promote_output_to_metadata(seg_grp):
            actions.append(f"promoted {OUTPUT_DATASET} → metadata (group + attrs)")

        return zarr_path, "migrated" if actions else "skipped", actions
    except Exception as e:
        return zarr_path, "error", [f"{type(e).__name__}: {e}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="folder to scan for *.zarr stores (recursive, follows symlinks)")
    ap.add_argument("--dry-run", action="store_true", help="print actions without modifying files")
    ap.add_argument("--workers", type=int, default=4, help="parallel workers (default: 4)")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        sys.exit(2)

    stores = find_zarr_stores(root)
    print(f"Found {len(stores)} .zarr store(s) under {root}")
    if args.dry_run:
        print("DRY RUN — no files will be modified")

    migrated, skipped, errored = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(migrate_one, p, args.dry_run): p for p in stores}
        for fut in as_completed(futures):
            path, status, actions = fut.result()
            tag = {"migrated": "OK ", "skipped": "-- ", "error": "ERR"}[status]
            print(f"  {tag} {path}")
            for a in actions:
                print(f"        · {a}")
            if status == "migrated":
                migrated.append(path)
            elif status == "skipped":
                skipped.append(path)
            else:
                errored.append((path, actions))

    print()
    print(f"summary: {len(migrated)} migrated, {len(skipped)} skipped, {len(errored)} errored")
    if errored:
        print("errors:")
        for path, msgs in errored:
            for m in msgs:
                print(f"  {path}: {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
