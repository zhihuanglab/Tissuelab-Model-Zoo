#!/usr/bin/env python3
"""
Migrate existing .zarr stores to the new Cell-Classification layout.

Old layout                                →  New layout
─────────────────────────────────────────────────────────────────────
ClassificationNode/                       →  Cell-Classification/
  nuclei_class_id                         →    class_indices
  nuclei_class_probabilities              →    probabilities
  nuclei_class_name        (per-class)    →    classes/name
  nuclei_class_HEX_color   (per-class)    →    classes/color
  (new)                                   →    classes/index   [0, 1, ..., M-1]
  metadata  (JSON bytes blob)             →    metadata        (group + attrs)
  userData/                               →    userData/       (preserved)

Usage:

    python scripts/migrate_classification_groups.py /path/to/folder
    python scripts/migrate_classification_groups.py /path/to/folder --dry-run
    python scripts/migrate_classification_groups.py /path/to/folder --workers 4

Idempotent: re-running on an already-migrated store is a no-op.
Walker follows symlinks. Per-store failures are isolated and reported at the end.
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

import numpy as np
import zarr

OLD_GROUP = "ClassificationNode"
NEW_GROUP = "Cell-Classification"

# Renames that stay flat under the group root.
FLAT_DATASET_RENAMES = {
    "nuclei_class_id": "class_indices",
    "nuclei_class_probabilities": "probabilities",
}

# Per-class datasets that move under the new `classes/` subgroup.
CLASSES_SUBGROUP = "classes"
CLASS_DATASET_MOVES = {
    "nuclei_class_name": "name",
    "nuclei_class_HEX_color": "color",
}

OUTPUT_DATASET = "metadata"  # legacy was a flat JSON-bytes blob with the same name
LEGACY_OUTPUT_DATASET = "output"  # also a flat JSON-bytes blob written by older NuClass


def find_zarr_stores(root: Path) -> List[Path]:
    """Return every directory ending in .zarr under root (follows symlinks)."""
    stores: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        for d in list(dirnames):
            if d.endswith(".zarr"):
                stores.append(Path(dirpath) / d)
                dirnames.remove(d)
    return stores


def _rename_group_fs(zarr_path: Path) -> bool:
    """Rename ClassificationNode → Cell-Classification on disk (DirectoryStore)."""
    old = zarr_path / OLD_GROUP
    new = zarr_path / NEW_GROUP
    if not old.is_dir():
        return False
    if new.exists():
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


def _rename_flat_datasets(grp) -> List[str]:
    """In-place rename for flat per-cell datasets."""
    renamed: List[str] = []
    for old_key, new_key in FLAT_DATASET_RENAMES.items():
        if old_key in grp and new_key not in grp:
            _zarr_copy(grp[old_key], grp, new_key)
            del grp[old_key]
            renamed.append(f"{old_key} → {new_key}")
    return renamed


def _move_class_datasets_to_subgroup(grp) -> List[str]:
    """Move per-class datasets into classes/ subgroup; (re)materialise classes/index."""
    moved: List[str] = []

    # If neither per-class dataset is present, nothing to do (no class taxonomy).
    has_legacy = any(old_key in grp for old_key in CLASS_DATASET_MOVES)
    has_new = CLASSES_SUBGROUP in grp and all(
        new_key in grp[CLASSES_SUBGROUP] for new_key in CLASS_DATASET_MOVES.values()
    )
    if not has_legacy and not has_new:
        return moved

    classes_grp = grp.require_group(CLASSES_SUBGROUP)

    for old_key, new_key in CLASS_DATASET_MOVES.items():
        if old_key not in grp:
            continue
        if new_key in classes_grp:
            del classes_grp[new_key]
        _zarr_copy(grp[old_key], classes_grp, new_key)
        del grp[old_key]
        moved.append(f"{old_key} → {CLASSES_SUBGROUP}/{new_key}")

    # Ensure classes/index exists [0, 1, ..., M-1]. M comes from whichever
    # per-class dataset is in place.
    m = None
    for k in CLASS_DATASET_MOVES.values():
        if k in classes_grp:
            try:
                m = int(classes_grp[k].shape[0])
                break
            except Exception:
                pass
    if m is not None and "index" not in classes_grp:
        classes_grp.create_array("index", data=np.arange(m, dtype=np.int32))
        moved.append(f"classes/index materialised [0..{m - 1}]")

    return moved


def _promote_output_to_metadata(grp) -> bool:
    """Convert legacy /metadata JSON-bytes dataset into a /metadata group + attrs."""
    if OUTPUT_DATASET not in grp:
        return False
    node = grp[OUTPUT_DATASET]
    # If it's already a group, nothing to do.
    if isinstance(node, zarr.Group):
        return False
    raw = bytes(node[()])
    payload = {}
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except Exception:
        payload = {}
    del grp[OUTPUT_DATASET]
    meta = grp.create_group(OUTPUT_DATASET)
    # Heuristic field promotion — keep anything the old blob had, plus a marker.
    promoted = {
        "classification_method": payload.get("classification_method", "unknown"),
        "organ": payload.get("organ", ""),
        "num_classes": int(len(payload.get("nuclei_classes", []) or [])),
        "training_time_sec": float(payload.get("training_time", 0.0)) if payload.get("training_time") else None,
        "testing_time_sec": float(payload.get("testing_time", 0.0)) if payload.get("testing_time") else None,
        "created": payload.get("created", ""),
        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Drop None entries so zarr attrs stays clean.
    meta.attrs.update({k: v for k, v in promoted.items() if v is not None})
    return True


def migrate_one(zarr_path: Path, dry_run: bool) -> Tuple[Path, str, List[str]]:
    actions: List[str] = []
    try:
        old = zarr_path / OLD_GROUP
        new = zarr_path / NEW_GROUP
        if not old.is_dir() and not new.is_dir():
            return zarr_path, "skipped", ["no classification group present"]

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
            zf = zarr.open_group(str(zarr_path), mode="r")
            grp = zf[OLD_GROUP] if OLD_GROUP in zf else zf.get(NEW_GROUP)
            if grp is None:
                return zarr_path, "skipped", actions
            for old_key, new_key in FLAT_DATASET_RENAMES.items():
                if old_key in grp:
                    actions.append(f"would rename {old_key} → {new_key}")
            for old_key, new_key in CLASS_DATASET_MOVES.items():
                if old_key in grp:
                    actions.append(f"would move {old_key} → {CLASSES_SUBGROUP}/{new_key}")
            if OUTPUT_DATASET in grp and not isinstance(grp[OUTPUT_DATASET], zarr.Group):
                actions.append(f"would promote {OUTPUT_DATASET} (bytes) → metadata group + attrs")
            return zarr_path, "migrated" if actions else "skipped", actions

        zf = zarr.open_group(str(zarr_path), mode="a")
        if NEW_GROUP not in zf:
            return zarr_path, "skipped", actions or ["nothing to migrate"]
        grp = zf[NEW_GROUP]

        actions.extend(_rename_flat_datasets(grp))
        actions.extend(_move_class_datasets_to_subgroup(grp))
        if _promote_output_to_metadata(grp):
            actions.append(f"promoted {OUTPUT_DATASET} → metadata group + attrs")
        # Drop the older sibling "output" JSON-bytes blob if present — newer
        # NuClass writes everything into metadata attrs, so it is dead data.
        if LEGACY_OUTPUT_DATASET in grp and not isinstance(grp[LEGACY_OUTPUT_DATASET], zarr.Group):
            del grp[LEGACY_OUTPUT_DATASET]
            actions.append(f"dropped legacy {LEGACY_OUTPUT_DATASET} bytes blob")

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
