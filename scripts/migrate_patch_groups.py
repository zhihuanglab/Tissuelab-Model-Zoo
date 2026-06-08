#!/usr/bin/env python3
"""
Migrate existing .zarr stores from the legacy `MuskNode` group into the new
split layout:

    MuskNode/                              →  Patch-Segmentation/
      embeddings                                embeddings
      coordinates                               coordinates
      probability    (placeholder ones)         (dropped — was useless)
      output         (bytes blob)               (dropped)
      metadata       (bytes blob)               metadata (group + attrs)

    MuskNode/                              →  Patch-Classification/
      tissue_class_id                           class_indices
      tissue_class_probabilities                probabilities
      tissue_class_name        (per-class)      classes/name
      tissue_class_HEX_color   (per-class)      classes/color
      (new)                                     classes/index   [0, 1, ..., M-1]
      (metadata bytes blob)                     metadata (group + attrs)

Patch userData (if present) is preserved on the embedding side
(`Patch-Segmentation/userData`) because that's the tasknode that "owns" the
shared user-config historically.

Usage:

    python scripts/migrate_patch_groups.py /path/to/folder
    python scripts/migrate_patch_groups.py /path/to/folder --dry-run
    python scripts/migrate_patch_groups.py /path/to/folder --workers 4

Idempotent. Walker follows symlinks. Per-store failures isolated.
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

OLD_GROUP = "MuskNode"
PATCH_SEG = "Patch-Segmentation"
PATCH_CLS = "Patch-Classification"

# Datasets that belong to embedding side. Accept both legacy singular
# ('embedding') and post-cell-rename plural ('embeddings').
SEG_FLAT_RENAMES = {
    "embeddings": "embeddings",
    "embedding": "embeddings",
    "coordinates": "coordinates",
}
SEG_DROPPED_DATASETS = {"probability", "output", "classification_output"}

# Datasets that belong to classification side. Per-cell flat moves first.
CLS_FLAT_RENAMES = {
    "tissue_class_id": "class_indices",
    "tissue_class_probabilities": "probabilities",
}

# Per-class datasets that move under classes/ subgroup in the new layout.
CLS_CLASSES_SUBGROUP = "classes"
CLS_CLASS_DATASET_MOVES = {
    "tissue_class_name": "name",
    "tissue_class_HEX_color": "color",
}

# Legacy bytes blob with run-summary JSON (old NuClass / MUSK pattern).
LEGACY_METADATA_DATASET = "metadata"

USER_DATA_DATASET = "userData"


def find_zarr_stores(root: Path) -> List[Path]:
    stores: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root, followlinks=True):
        for d in list(dirnames):
            if d.endswith(".zarr"):
                stores.append(Path(dirpath) / d)
                dirnames.remove(d)
    return stores


def _is_array(node) -> bool:
    """True iff node is a zarr Array (not a Group)."""
    return not isinstance(node, zarr.hierarchy.Group)


def _read_bytes_blob(node) -> bytes:
    """Read a legacy metadata blob that may be stored as bytes, str, or a numpy
    scalar (some stores wrote an object-dtype string instead of uint8 bytes)."""
    val = node[()]
    if isinstance(val, bytes):  # also covers np.bytes_
        return val
    if isinstance(val, str):  # also covers np.str_
        return val.encode("utf-8")
    try:
        return bytes(val)
    except TypeError:
        return str(val).encode("utf-8")


def _split_old_node(zf, dry_run: bool) -> List[str]:
    """
    Inspect MuskNode and (in non-dry-run mode) migrate its contents into
    Patch-Segmentation + Patch-Classification.
    """
    actions: List[str] = []
    old = zf[OLD_GROUP]
    has_embedding_data = (
        "embeddings" in old or "embedding" in old or "coordinates" in old
    )
    has_classification_data = any(k in old for k in CLS_FLAT_RENAMES) or any(
        k in old for k in CLS_CLASS_DATASET_MOVES
    )

    # ── Embedding side ────────────────────────────────────────────────────
    if has_embedding_data:
        if dry_run:
            actions.append(f"would create {PATCH_SEG}/ with embeddings + coordinates")
        else:
            seg = zf.require_group(PATCH_SEG)
            for old_key, new_key in SEG_FLAT_RENAMES.items():
                if old_key in old:
                    if new_key in seg:
                        del seg[new_key]
                    zarr.copy(old[old_key], seg, name=new_key)
                    actions.append(f"{OLD_GROUP}/{old_key} → {PATCH_SEG}/{new_key}")
            # Promote legacy metadata bytes blob if it carries patch-embedding params.
            if LEGACY_METADATA_DATASET in old and _is_array(old[LEGACY_METADATA_DATASET]):
                raw = _read_bytes_blob(old[LEGACY_METADATA_DATASET])
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except Exception:
                    payload = {}
                if "patch_size" in payload or "level" in payload:
                    if LEGACY_METADATA_DATASET in seg:
                        del seg[LEGACY_METADATA_DATASET]
                    meta_grp = seg.create_group(LEGACY_METADATA_DATASET)
                    meta_attrs = {
                        "model": "MUSK",
                        "patch_size": int(payload.get("patch_size", 0)) if payload.get("patch_size") else None,
                        "level": int(payload.get("level", 0)) if payload.get("level") is not None else None,
                        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    meta_grp.attrs.update({k: v for k, v in meta_attrs.items() if v is not None})
                    actions.append(f"{OLD_GROUP}/metadata (bytes) → {PATCH_SEG}/metadata (group + attrs)")
            # Carry userData (legacy shared between embedding & classification).
            if USER_DATA_DATASET in old:
                if USER_DATA_DATASET in seg:
                    del seg[USER_DATA_DATASET]
                zarr.copy(old[USER_DATA_DATASET], seg, name=USER_DATA_DATASET)
                actions.append(f"{OLD_GROUP}/{USER_DATA_DATASET} → {PATCH_SEG}/{USER_DATA_DATASET}")

    elif dry_run and (LEGACY_METADATA_DATASET in old and _is_array(old[LEGACY_METADATA_DATASET])):
        actions.append(f"would inspect {OLD_GROUP}/metadata bytes blob")

    # ── Classification side ───────────────────────────────────────────────
    if has_classification_data:
        if dry_run:
            actions.append(f"would create {PATCH_CLS}/ with class_indices/probabilities/classes/metadata")
        else:
            cls = zf.require_group(PATCH_CLS)
            for old_key, new_key in CLS_FLAT_RENAMES.items():
                if old_key in old:
                    if new_key in cls:
                        del cls[new_key]
                    zarr.copy(old[old_key], cls, name=new_key)
                    actions.append(f"{OLD_GROUP}/{old_key} → {PATCH_CLS}/{new_key}")

            # Per-class taxonomy into classes/ subgroup.
            has_legacy_per_class = any(k in old for k in CLS_CLASS_DATASET_MOVES)
            if has_legacy_per_class:
                classes_grp = cls.require_group(CLS_CLASSES_SUBGROUP)
                for old_key, new_key in CLS_CLASS_DATASET_MOVES.items():
                    if old_key in old:
                        if new_key in classes_grp:
                            del classes_grp[new_key]
                        zarr.copy(old[old_key], classes_grp, name=new_key)
                        actions.append(
                            f"{OLD_GROUP}/{old_key} → {PATCH_CLS}/{CLS_CLASSES_SUBGROUP}/{new_key}"
                        )
                # Materialise classes/index if missing.
                m = None
                for k in CLS_CLASS_DATASET_MOVES.values():
                    if k in classes_grp:
                        try:
                            m = int(classes_grp[k].shape[0])
                            break
                        except Exception:
                            pass
                if m is not None and "index" not in classes_grp:
                    classes_grp.create_dataset("index", data=np.arange(m, dtype=np.int32))
                    actions.append(f"{PATCH_CLS}/{CLS_CLASSES_SUBGROUP}/index materialised [0..{m - 1}]")

            # Promote classification metadata bytes blob if not already used by embedding side.
            if (
                LEGACY_METADATA_DATASET in old
                and _is_array(old[LEGACY_METADATA_DATASET])
                and not has_embedding_data
            ):
                raw = _read_bytes_blob(old[LEGACY_METADATA_DATASET])
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except Exception:
                    payload = {}
                if LEGACY_METADATA_DATASET in cls:
                    del cls[LEGACY_METADATA_DATASET]
                meta_grp = cls.create_group(LEGACY_METADATA_DATASET)
                meta_attrs = {
                    "model": "MUSK-Classification",
                    "classification_method": payload.get("classification_method", "unknown"),
                    "num_classes": int(len(payload.get("tissue_classes", []) or [])),
                    "training_time_sec": float(payload.get("training_time", 0.0)) if payload.get("training_time") else None,
                    "testing_time_sec": float(payload.get("testing_time", 0.0)) if payload.get("testing_time") else None,
                    "created": payload.get("created", ""),
                    "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                meta_grp.attrs.update({k: v for k, v in meta_attrs.items() if v is not None})
                actions.append(f"{OLD_GROUP}/metadata (bytes) → {PATCH_CLS}/metadata (group + attrs)")

    # ── Finally: delete MuskNode if we migrated anything ───────────────────
    if not dry_run and (has_embedding_data or has_classification_data):
        del zf[OLD_GROUP]
        actions.append(f"dropped {OLD_GROUP}/ after split")

    return actions


# Dead datasets that earlier MUSK code dropped into the new groups before we
# stripped them out of the writer. They are placeholder/legacy blobs only;
# safe to delete unconditionally.
DEAD_DATASETS_IN_PATCH_SEG = ("output", "probability", "classification_output")
DEAD_DATASETS_IN_PATCH_CLS = ("output", "classification_output")


def _drop_legacy_artifacts(zf, dry_run: bool) -> List[str]:
    """Drop placeholder/legacy datasets from the new patch groups (idempotent)."""
    actions: List[str] = []
    for grp_name, dead in (
        (PATCH_SEG, DEAD_DATASETS_IN_PATCH_SEG),
        (PATCH_CLS, DEAD_DATASETS_IN_PATCH_CLS),
    ):
        if grp_name not in zf:
            continue
        grp = zf[grp_name]
        for key in dead:
            if key in grp and _is_array(grp[key]):
                if dry_run:
                    actions.append(f"would drop dead {grp_name}/{key}")
                else:
                    del grp[key]
                    actions.append(f"dropped dead {grp_name}/{key}")
    return actions


def migrate_one(zarr_path: Path, dry_run: bool) -> Tuple[Path, str, List[str]]:
    actions: List[str] = []
    try:
        try:
            zf = zarr.open_group(str(zarr_path), mode="r" if dry_run else "a")
        except (zarr.errors.GroupNotFoundError, zarr.errors.ContainsArrayError, KeyError):
            # Not a group-style zarr (likely an image pyramid root array). Nothing to migrate.
            return zarr_path, "skipped", ["not a group-style zarr (skipping)"]

        # 1) Split MuskNode if it's still there.
        if OLD_GROUP in zf:
            actions.extend(_split_old_node(zf, dry_run))

        # 2) Drop dead placeholder datasets from the new groups (always — covers
        #    stores that were written by an old MUSK Embedding before the
        #    placeholder writes were removed).
        actions.extend(_drop_legacy_artifacts(zf, dry_run))

        if not actions:
            return zarr_path, "skipped", ["nothing to migrate"]
        return zarr_path, "migrated", actions
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
