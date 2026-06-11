#!/usr/bin/env python3
"""
One-shot migration: rename the cell annotation-count dataset
``User-Annotations/class_counts`` → ``User-Annotations/cell_class_counts``
in existing ``.zarr`` stores.

Why: the backend now writes the cell per-class counts under
``cell_class_counts`` so the name is symmetric with the patch side
(``patch_class_counts``). Older stores still have the dataset under the
legacy name ``class_counts``; the new code does NOT read the old name
(no backward-compat shim), so run this once over your store folder to
migrate them.

What it does NOT touch:
  - ``patch_class_counts`` (already correctly named)
  - the API JSON response key ``class_counts`` (that's a wire-format key,
    not a stored dataset — unaffected by this script)
  - the actual annotation arrays ``cell`` / ``patch``

Properties: idempotent (skips stores already migrated), follows symlinks,
per-store failures are isolated, ``--dry-run`` previews without writing.

Usage:

    python scripts/rename_class_counts_to_cell.py /path/to/folder
    python scripts/rename_class_counts_to_cell.py /path/to/folder --dry-run
    python scripts/rename_class_counts_to_cell.py /path/to/one_store.svs.zarr
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zarr

OLD_NAME = "class_counts"
NEW_NAME = "cell_class_counts"
GROUP = "User-Annotations"


def find_zarr_stores(root: Path) -> List[Path]:
    """Return every ``*.zarr`` store dir under ``root`` (or ``root`` itself)."""
    if root.name.endswith(".zarr") and root.is_dir():
        return [root]
    stores: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        keep = []
        for d in dirnames:
            if d.endswith(".zarr"):
                stores.append(Path(dirpath) / d)
                # don't descend into a store we already matched
            else:
                keep.append(d)
        dirnames[:] = keep
    return stores


def migrate_store(store: Path, dry_run: bool) -> Tuple[str, str]:
    """Migrate a single store. Returns (status, detail).

    status ∈ {"migrated", "skipped", "no-op", "error"}.
    """
    try:
        root = zarr.open_group(str(store), mode="r" if dry_run else "r+")
    except Exception as e:
        return ("error", f"cannot open as zarr group: {e}")

    if GROUP not in root:
        return ("no-op", f"no {GROUP} group")
    ua = root[GROUP]

    has_old = OLD_NAME in ua
    has_new = NEW_NAME in ua

    if has_new and not has_old:
        return ("skipped", "already migrated")
    if not has_old:
        return ("no-op", f"no {OLD_NAME} dataset")
    if has_new and has_old:
        # Both exist — new wins; just drop the stale old one.
        if dry_run:
            return ("migrated", f"would delete stale {OLD_NAME} ({NEW_NAME} already present)")
        del ua[OLD_NAME]
        return ("migrated", f"deleted stale {OLD_NAME} ({NEW_NAME} already present)")

    # The normal case: copy old → new (preserving dtype/shape), delete old.
    old_ds = ua[OLD_NAME]
    dtype = old_ds.dtype
    shape = old_ds.shape
    value = old_ds[()]
    try:
        preview = (bytes(value).rstrip(b"\x00").decode("utf-8", "replace")
                   if isinstance(value, (bytes, bytearray, np.bytes_)) else str(value))
    except Exception:
        preview = "<unprintable>"

    if dry_run:
        return ("migrated", f"would rename {OLD_NAME} -> {NEW_NAME} (dtype={dtype}, value={preview})")

    ua.create_dataset(NEW_NAME, data=value, shape=shape, dtype=dtype, overwrite=True)
    del ua[OLD_NAME]
    return ("migrated", f"renamed {OLD_NAME} -> {NEW_NAME} (value={preview})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="folder to scan recursively, or a single .zarr store")
    ap.add_argument("--dry-run", action="store_true", help="preview without writing")
    args = ap.parse_args()

    root = Path(args.path).expanduser()
    if not root.exists():
        print(f"[ERROR] path does not exist: {root}", file=sys.stderr)
        return 2

    stores = find_zarr_stores(root)
    if not stores:
        print(f"[WARN] no .zarr stores found under {root}")
        return 0

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Scanning {len(stores)} store(s) under {root}\n")
    tally = {"migrated": 0, "skipped": 0, "no-op": 0, "error": 0}
    for store in sorted(stores):
        try:
            status, detail = migrate_store(store, args.dry_run)
        except Exception as e:
            status, detail = "error", f"{e}\n{traceback.format_exc()}"
        tally[status] += 1
        if status in ("migrated", "error"):
            print(f"  [{status:8}] {store.name}: {detail}")

    print(
        f"\nDone. migrated={tally['migrated']} skipped={tally['skipped']} "
        f"no-op={tally['no-op']} error={tally['error']}"
        + ("  (dry-run, nothing written)" if args.dry_run else "")
    )
    return 1 if tally["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
