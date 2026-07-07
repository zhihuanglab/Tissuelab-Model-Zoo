"""
Standalone script: same logic as musk_embedding_taskNode.run_patch_classification.
Process WSI(s) and save Patch-Segmentation embeddings/coordinates to zarr
(one .zarr dir per WSI).
"""
from musk_for_embedding import MUSK
import time
import os
import sys
import argparse
import glob
import json
import zarr
import numpy as np
from PIL import Image
import tiffslide

# Same defaults as musk_embedding_taskNode.parse_args
PATCH_SIZE = 224
LEVEL = 0
BATCH_SIZE = 4
TISSUE_THRESHOLD = 0.1
WSI_EXTENSIONS = (".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs", ".bif")


def _resolve_model_path():
    """Same model resolution as musk_embedding_taskNode.load_model_at_init."""
    base_path = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
    model_path = os.path.join(base_path, "checkpoints", "model.safetensors")
    if not os.path.exists(model_path):
        for alt in [
            "checkpoints/model.safetensors",
            os.path.join(base_path, "model", "model.safetensors"),
            "model/model.safetensors",
        ]:
            if os.path.exists(alt):
                model_path = alt
                break
    return model_path


def get_wsi_paths(wsi_input):
    """Return list of WSI file paths. wsi_input can be a single file or a directory."""
    if os.path.isfile(wsi_input):
        return [wsi_input]
    if os.path.isdir(wsi_input):
        paths = []
        for ext in WSI_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(wsi_input, "*" + ext)))
        return sorted(paths)
    return []


def _stored_patch_size_from_coordinates(coordinates):
    """Infer patch_size from coordinate spacing (same as taskNode)."""
    if coordinates is None or len(coordinates) < 2:
        return None
    coords_array = np.array(coordinates)
    unique_x = np.unique(coords_array[:, 0])
    unique_y = np.unique(coords_array[:, 1])
    if len(unique_x) >= 2:
        return int(np.min(np.diff(np.sort(unique_x))))
    if len(unique_y) >= 2:
        return int(np.min(np.diff(np.sort(unique_y))))
    return None


def run_patch_classification_one(musk, wsi_path, output_dir, patch_size=224, level=0,
                                 batch_size=4, tissue_threshold=0.1, zarr_group="Patch-Segmentation"):
    """
    One-WSI processing: same logic as musk_embedding_taskNode.run_patch_classification.
    If zarr already exists and patch_size matches => skip. Otherwise process and save.
    """
    os.makedirs(output_dir, exist_ok=True)
    wsi_basename = os.path.basename(wsi_path)
    zarr_path = os.path.join(output_dir, f"{wsi_basename}.zarr")

    # Step 1: Check if zarr exists and patch_size matches => skip (same as taskNode)
    if os.path.exists(zarr_path):
        try:
            zf = zarr.open_group(zarr_path, mode="r")
            if zarr_group in zf:
                coordinates = zf[f"{zarr_group}/coordinates"][()]
                embeddings = zf[f"{zarr_group}/embeddings"][()]
                stored_patch_size = _stored_patch_size_from_coordinates(coordinates)
                if stored_patch_size is not None and stored_patch_size == patch_size:
                    print(f"Using existing embeddings {zarr_path} (patch_size={stored_patch_size}) => skip")
                    return True
                if stored_patch_size is not None:
                    print(f"Patch size mismatch: stored={stored_patch_size}, current={patch_size} => re-process")
        except Exception as e:
            print(f"Existing zarr invalid or missing data: {e} => re-process")

    start_time = time.time()

    def progress_callback(stage, payload):
        if stage == "encode":
            try:
                percent = int(payload)
                if percent % 10 == 0 or percent == 100:
                    print(f"  encode progress: {percent}%")
            except Exception:
                pass

    # Same call as taskNode (save_patches=False, output_mask_path=None, progress_callback)
    patch_embeddings, patch_coordinates = musk.process_whole_wsi(
        wsi_path=wsi_path,
        patch_size=patch_size,
        level=level,
        batch_size=batch_size,
        tissue_threshold=tissue_threshold,
        save_patches=False,
        output_mask_path=None,
        progress_callback=progress_callback,
    )

    if patch_embeddings is None or len(patch_coordinates) == 0:
        print(f"No patches for {wsi_path}, skip.")
        return False

    embeddings = patch_embeddings.cpu().numpy()
    coordinates = np.array(patch_coordinates)

    # Same canonical layout as musk_embedding_taskNode writes:
    #   <zarr_group>/embeddings, /coordinates, /metadata (group + attrs).
    zf = zarr.open_group(zarr_path, mode="w")
    if zarr_group in zf:
        del zf[zarr_group]
    node_grp = zf.create_group(zarr_group)
    node_grp.create_array("embeddings", data=embeddings)
    node_grp.create_array("coordinates", data=coordinates)
    meta_grp = node_grp.require_group("metadata")
    meta_grp.attrs.update({"patch_size": int(patch_size), "level": int(level)})

    elapsed = time.time() - start_time
    print(f"Saved {embeddings.shape} -> {zarr_path} ({elapsed:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="MUSK embedding (same logic as musk_embedding_taskNode)")
    parser.add_argument("--wsi_dir", type=str, default=r"E:\experiments\UPENN-GBM\training_WSI",
                        help="Single WSI file or directory of WSI files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for .zarr (default: <wsi_dir>/result_zarr or script dir result_zarr)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE)
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--tissue_threshold", type=float, default=TISSUE_THRESHOLD)
    args = parser.parse_args()

    script_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = args.model_path or _resolve_model_path()
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        return
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    musk = MUSK(model_path=model_path)

    wsi_list = get_wsi_paths(args.wsi_dir)
    if not wsi_list:
        print(f"No WSI files under: {args.wsi_dir}")
        return

    if args.output_dir is None:
        args.output_dir = (
            os.path.join(os.path.abspath(args.wsi_dir), "result_zarr")
            if os.path.isdir(args.wsi_dir)
            else os.path.join(script_dir, "result_zarr")
        )

    print(f"WSI count: {len(wsi_list)}, output_dir: {args.output_dir}")
    print(f"patch_size={args.patch_size}, level={args.level}, batch_size={args.batch_size}, tissue_threshold={args.tissue_threshold}")
    for wsi_path in wsi_list:
        run_patch_classification_one(
            musk, wsi_path, args.output_dir,
            patch_size=args.patch_size, level=args.level,
            batch_size=args.batch_size, tissue_threshold=args.tissue_threshold,
        )


if __name__ == "__main__":
    main()
