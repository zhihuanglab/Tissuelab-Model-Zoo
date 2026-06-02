#!/usr/bin/env python3
"""
Migrate existing .zarr stores from the legacy ``user_annotation`` layout into
the new unified ``User-Annotations`` group with aligned cell/patch subgroups.

Old layout                                  →  New layout
─────────────────────────────────────────────────────────────────────
user_annotation/                            →  User-Annotations/
  nuclei_annotations  (structured array)    →    cell  (structured array, same row count)
    fields: cell_class, cell_color, ...     →      fields: class, color, ...
    (region_x1..y2 unchanged)
  tissue_annotations  (JSON bytes blob)     →    patch (dense structured array, length=N_patches)
                                                    same dtype as cell, region_*=-1
                                                    (unannotated rows: class=-1)
  class_counts        (JSON bytes blob)     →    cell.attrs.class_counts (dict)
  reclassification_metadata  (bytes)        →    cell.attrs.reclassification_metadata
attrs.class_names                           →    cell.attrs.class_names
attrs.class_colors                          →    cell.attrs.class_colors
attrs.tissue_class_names                    →    patch.attrs.class_names
attrs.tissue_class_colors                   →    patch.attrs.class_colors

Idempotent. Walker follows symlinks. Per-store failures isolated.

Usage:

    python scripts/migrate_user_annotations.py /path/to/folder
    python scripts/migrate_user_annotations.py /path/to/folder --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zarr

OLD_GROUP = "user_annotation"
NEW_GROUP = "User-Annotations"
OLD_CELL_DS = "nuclei_annotations"
NEW_CELL_DS = "cell"
OLD_PATCH_DS = "tissue_annotations"
NEW_PATCH_DS = "patch"
PATCH_SEG_GROUP = "Patch-Segmentation"


# Same dtype as TissueLab-Dev/app/service/app/services/tasks.py:_get_annotation_dtype
ANNOTATION_DTYPE = np.dtype([
    ('class', 'i4'),
    ('color', 'i4'),
    ('datetime', 'i8'),
    ('region_x1', 'i8'),
    ('region_y1', 'i8'),
    ('region_x2', 'i8'),
    ('region_y2', 'i8'),
    ('method', 'U32'),
    ('annotator', 'U64'),
])

# Field renames inside the existing structured array (legacy cell rows).
CELL_FIELD_RENAMES = {
    'cell_class': 'class',
    'cell_color': 'color',
}


def find_zarr_stores(root: Path) -> List[Path]:
    stores: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        for d in list(dirnames):
            if d.endswith(".zarr"):
                stores.append(Path(dirpath) / d)
                dirnames.remove(d)
    return stores


def _rename_fields_in_array(arr: np.ndarray, renames: dict) -> np.ndarray:
    """Copy structured array into a new dtype where some field names are renamed."""
    new_dtype = np.dtype([
        (renames.get(name, name), arr.dtype.fields[name][0])
        for name in arr.dtype.names
    ])
    out = np.empty(arr.shape, dtype=new_dtype)
    for old_name in arr.dtype.names:
        out[renames.get(old_name, old_name)] = arr[old_name]
    return out


def _hex_to_int(s: str) -> int:
    if not isinstance(s, str):
        return -1
    s = s.strip().lstrip('#')
    if len(s) == 6:
        try:
            return int(s, 16)
        except ValueError:
            return -1
    return -1


def _patch_dict_to_array(ann_dict: dict, n_patches: int, class_names: List[str]) -> np.ndarray:
    """Materialise a sparse patch annotation dict (legacy JSON) into a dense
    structured array of length n_patches. Unannotated rows have class=-1.

    ``ann_dict`` keys are patch_id strings; values are old JSON entries with
    ``tissue_class`` (string name) and ``tissue_color`` (hex string).
    """
    arr = np.zeros(n_patches, dtype=ANNOTATION_DTYPE)
    arr['class'][:] = -1
    arr['color'][:] = -1
    for region in ('region_x1', 'region_y1', 'region_x2', 'region_y2'):
        arr[region][:] = -1

    name_to_idx = {n: i for i, n in enumerate(class_names)}

    for k, v in ann_dict.items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if i < 0 or i >= n_patches:
            continue
        if not isinstance(v, dict):
            continue
        # Old JSON used 'tissue_class' (string) and 'tissue_color' (hex).
        # The bulk sed in the codebase may also have produced 'class' / 'color'
        # in newer dumps. Honour both for safety.
        cls_name = v.get('tissue_class') if v.get('tissue_class') is not None else v.get('class')
        col_hex = v.get('tissue_color') if v.get('tissue_color') is not None else v.get('color')

        if cls_name is None:
            continue
        cls_idx = name_to_idx.get(str(cls_name), -1)
        if cls_idx < 0:
            # Class not in our taxonomy (e.g. class deleted) — leave as -1.
            continue
        arr['class'][i] = cls_idx
        if isinstance(col_hex, str):
            arr['color'][i] = _hex_to_int(col_hex)
        elif isinstance(col_hex, (int, np.integer)):
            arr['color'][i] = int(col_hex)
        try:
            arr['datetime'][i] = int(v.get('datetime', 0)) if str(v.get('datetime', 0)).isdigit() else 0
        except (TypeError, ValueError):
            arr['datetime'][i] = 0
        arr['method'][i] = str(v.get('method', ''))[:32]
        arr['annotator'][i] = str(v.get('annotator', ''))[:64]

    return arr


def migrate_one(zarr_path: Path, dry_run: bool) -> Tuple[Path, str, List[str]]:
    actions: List[str] = []
    try:
        try:
            zf = zarr.open_group(str(zarr_path), mode="r" if dry_run else "a")
        except (zarr.errors.GroupNotFoundError, zarr.errors.ContainsArrayError, KeyError):
            return zarr_path, "skipped", ["not a group-style zarr"]
        if OLD_GROUP not in zf and NEW_GROUP not in zf:
            return zarr_path, "skipped", ["no user_annotation group present"]

        # Already migrated? (NEW_GROUP exists, OLD_GROUP gone)
        if NEW_GROUP in zf and OLD_GROUP not in zf:
            return zarr_path, "skipped", ["already migrated"]

        # ── Phase 1: rename top-level group ─────────────────────────────────
        if OLD_GROUP in zf and NEW_GROUP not in zf:
            if dry_run:
                actions.append(f"would rename {OLD_GROUP} → {NEW_GROUP}")
            else:
                old = zarr_path / OLD_GROUP
                new = zarr_path / NEW_GROUP
                if new.exists():
                    raise RuntimeError(f"both '{OLD_GROUP}' and '{NEW_GROUP}' exist")
                os.rename(old, new)
                actions.append(f"renamed {OLD_GROUP} → {NEW_GROUP}")
                # Re-open with new layout.
                zf = zarr.open_group(str(zarr_path), mode="a")

        if NEW_GROUP not in zf and not dry_run:
            return zarr_path, "error", ["rename produced no group"]

        if dry_run:
            ua = zf[OLD_GROUP] if OLD_GROUP in zf else zf[NEW_GROUP]
        else:
            ua = zf[NEW_GROUP]

        # Pull legacy attrs first; we'll re-pin them to the right subgroup later.
        legacy_attrs = dict(ua.attrs) if hasattr(ua, 'attrs') else {}
        cell_class_names = legacy_attrs.get('class_names') or []
        cell_class_colors = legacy_attrs.get('class_colors') or []
        patch_class_names = legacy_attrs.get('tissue_class_names') or []
        patch_class_colors = legacy_attrs.get('tissue_class_colors') or []

        # ── Phase 2: cell side — rename nuclei_annotations → cell ───────────
        if OLD_CELL_DS in ua and NEW_CELL_DS not in ua:
            if dry_run:
                actions.append(f"would rename {OLD_CELL_DS} → {NEW_CELL_DS} (and field renames)")
            else:
                old_arr = ua[OLD_CELL_DS][:]
                needs_field_rename = any(k in (old_arr.dtype.names or ()) for k in CELL_FIELD_RENAMES)
                if needs_field_rename:
                    new_arr = _rename_fields_in_array(old_arr, CELL_FIELD_RENAMES)
                else:
                    new_arr = old_arr
                ua.create_dataset(NEW_CELL_DS, data=new_arr, dtype=new_arr.dtype)
                del ua[OLD_CELL_DS]
                actions.append(f"renamed {OLD_CELL_DS} → {NEW_CELL_DS}")

        # Move attrs.class_names/class_colors → cell.attrs
        if not dry_run and NEW_CELL_DS in ua:
            cell_ds = ua[NEW_CELL_DS]
            cell_attrs_payload = {}
            if cell_class_names:
                cell_attrs_payload['class_names'] = list(cell_class_names)
            if cell_class_colors:
                cell_attrs_payload['class_colors'] = list(cell_class_colors)
            # class_counts dataset → cell.attrs.class_counts dict
            if 'class_counts' in ua:
                try:
                    raw = bytes(ua['class_counts'][()])
                    counts = json.loads(raw.decode('utf-8')) if raw else {}
                    cell_attrs_payload['class_counts'] = counts
                except Exception:
                    pass
                del ua['class_counts']
                actions.append("class_counts dataset → cell.attrs.class_counts")
            # reclassification_metadata if present → cell.attrs
            if 'reclassification_metadata' in ua:
                try:
                    raw = bytes(ua['reclassification_metadata'][()])
                    meta = json.loads(raw.decode('utf-8')) if raw else {}
                    cell_attrs_payload['reclassification_metadata'] = meta
                except Exception:
                    pass
                del ua['reclassification_metadata']
                actions.append("reclassification_metadata → cell.attrs.reclassification_metadata")
            if cell_attrs_payload:
                cell_ds.attrs.update(cell_attrs_payload)
                actions.append(f"cell.attrs ← class_names/class_colors/...")

        # ── Phase 3: patch side — tissue_annotations (JSON) → patch (dense) ─
        if OLD_PATCH_DS in ua and NEW_PATCH_DS not in ua:
            if dry_run:
                actions.append(f"would convert {OLD_PATCH_DS} JSON → {NEW_PATCH_DS} dense array")
            else:
                # 1) Decode the JSON dict.
                try:
                    raw = bytes(ua[OLD_PATCH_DS][()])
                    ann_dict = json.loads(raw.decode('utf-8')) if raw else {}
                except Exception:
                    ann_dict = {}
                # 2) Figure out N_patches from Patch-Segmentation/coordinates.
                n_patches = 0
                try:
                    if PATCH_SEG_GROUP in zf and 'coordinates' in zf[PATCH_SEG_GROUP]:
                        n_patches = int(zf[PATCH_SEG_GROUP]['coordinates'].shape[0])
                except Exception:
                    n_patches = 0
                if n_patches == 0:
                    # Fall back to "max patch_id + 1" so we still preserve user data.
                    keys_as_int = []
                    for k in ann_dict.keys():
                        try:
                            keys_as_int.append(int(k))
                        except (TypeError, ValueError):
                            pass
                    n_patches = (max(keys_as_int) + 1) if keys_as_int else 0
                # 3) Materialise dense array.
                patch_arr = _patch_dict_to_array(ann_dict, n_patches, list(patch_class_names))
                ua.create_dataset(NEW_PATCH_DS, data=patch_arr, dtype=patch_arr.dtype)
                del ua[OLD_PATCH_DS]
                actions.append(f"{OLD_PATCH_DS} JSON({len(ann_dict)}) → {NEW_PATCH_DS} dense({n_patches})")

        # Move patch attrs (renaming tissue_* → bare class_*).
        if not dry_run and NEW_PATCH_DS in ua:
            patch_ds = ua[NEW_PATCH_DS]
            patch_attrs_payload = {}
            if patch_class_names:
                patch_attrs_payload['class_names'] = list(patch_class_names)
            if patch_class_colors:
                patch_attrs_payload['class_colors'] = list(patch_class_colors)
            # patch_class_counts dataset under user_annotation? optional
            if 'patch_class_counts' in ua:
                try:
                    raw = bytes(ua['patch_class_counts'][()])
                    counts = json.loads(raw.decode('utf-8')) if raw else {}
                    patch_attrs_payload['class_counts'] = counts
                except Exception:
                    pass
                del ua['patch_class_counts']
                actions.append("patch_class_counts → patch.attrs.class_counts")
            if patch_attrs_payload:
                patch_ds.attrs.update(patch_attrs_payload)
                actions.append("patch.attrs ← class_names/class_colors/...")

        # ── Phase 4: drop the legacy attrs from the parent group ────────────
        if not dry_run and hasattr(ua, 'attrs'):
            for k in ('class_names', 'class_colors', 'tissue_class_names', 'tissue_class_colors',
                      'annotation_format'):
                if k in ua.attrs:
                    del ua.attrs[k]

        return zarr_path, "migrated" if actions else "skipped", actions
    except Exception as e:
        return zarr_path, "error", [f"{type(e).__name__}: {e}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
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
        sys.exit(1)


if __name__ == "__main__":
    main()
