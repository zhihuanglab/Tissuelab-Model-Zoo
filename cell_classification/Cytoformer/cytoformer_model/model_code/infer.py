"""Inference with a trained Xenium H&E cell-type classifier.

Loads a model directory (best.pth + organ_celltype_map.json) and predicts a cell type for each
input cell patch image. The organ is known at input and routes the per-organ head; predictions
are restricted to that organ's cell types. Each patch should be ~the cell's 56 um field of view
(any pixel size — it is resized to 224 px).

Usage:
  python infer.py --model_dir xenium_noncommercial --backbone uni2 \
      --patches patch_dir/ --organ skin --out preds.parquet
  # or a list of image files instead of a folder:
  python infer.py --model_dir xenium_commercial --backbone hoptimus \
      --patches a.png b.png c.png --organ breast --out preds.parquet

Output parquet columns: cell_id, file, organ, pred_celltype, prob.
"""
import os, sys, argparse, glob
import numpy as np, pandas as pd, torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

IN_PX = 224
_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def load_model(model_dir, backbone, device):
    os.environ["NUCLASS_ORGAN_MAP"] = os.path.join(model_dir, "organ_celltype_map.json")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common
    from model import CellClassifier
    net = CellClassifier(backbone=backbone)                       # foundation weights optional (ckpt has all)
    ck = torch.load(os.path.join(model_dir, "best.pth"), map_location="cpu")
    sd = ck.get("model_state_dict", ck)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}   # tolerate a torch.compile prefix
    net.load_state_dict(sd, strict=True)
    net.eval().to(device)
    return net, common


@torch.inference_mode()
def predict(net, common, files, organ, backbone, device, batch=256):
    assert organ in common.ORGAN_IDX, f"organ '{organ}' not in {list(common.ORGAN_IDX)}"
    mean, std = common.BACKBONE_NORM[backbone]
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    oid = torch.full((batch,), common.ORGAN_IDX[organ], dtype=torch.long, device=device)
    GLOBAL = np.array(common.GLOBAL_CLASSES)
    preds, probs = [], []
    for i in range(0, len(files), batch):
        chunk = files[i:i + batch]
        arr = np.stack([np.array(Image.open(f).convert("RGB").resize((IN_PX, IN_PX))) for f in chunk])
        x = torch.from_numpy(arr).to(device).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit = net(x, oid[:len(chunk)]).float()
        p = torch.softmax(logit, 1); conf, idx = p.max(1)
        preds.extend(GLOBAL[idx.cpu().numpy()]); probs.extend(conf.cpu().numpy())
    return preds, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="dir with best.pth + organ_celltype_map.json")
    ap.add_argument("--backbone", default="uni2", choices=["uni2", "hoptimus"])
    ap.add_argument("--organ", required=True, help="one of the 16 organs (see organ_celltype_map.json)")
    ap.add_argument("--patches", required=True, nargs="+", help="a folder of cell-patch images, or image files")
    ap.add_argument("--out", default="preds.parquet")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(args.patches) == 1 and os.path.isdir(args.patches[0]):
        files = sorted(f for f in glob.glob(os.path.join(args.patches[0], "*")) if f.lower().endswith(_EXT))
    else:
        files = [f for f in args.patches if f.lower().endswith(_EXT)]
    assert files, f"no images found in {args.patches}"

    net, common = load_model(args.model_dir, args.backbone, device)
    preds, probs = predict(net, common, files, args.organ, args.backbone, device, args.batch)
    out = pd.DataFrame({"cell_id": range(len(files)), "file": [os.path.basename(f) for f in files],
                        "organ": args.organ, "pred_celltype": preds, "prob": np.round(probs, 4)})
    out.to_parquet(args.out, index=False)
    print(f"predicted {len(out):,} cells (organ={args.organ}) -> {args.out}")
    print(out["pred_celltype"].value_counts().to_string())


if __name__ == "__main__":
    main()
