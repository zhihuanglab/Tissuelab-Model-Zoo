#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:42:07 2022

@author: zhihuang

"""
import argparse
import os
import sys
import time
import json
import h5py
import uvicorn
import requests
import platform

import numpy as np
import multiprocess as mp
import cv2

from fastapi import FastAPI
from typing import Dict, Any
from pathlib import Path

# ============ from original code ============

from nuc_seg import SlideSegmentation
from nuc_stat import SlideProperty

app = FastAPI()

# --------- Global variables for Node ---------
ARGS = None                # record /read stage parameters
IS_MODEL_INITED = False    # whether /init initilized
H5_PATH = None             # h5 file path
NODE_NAME = None
DEPENDENCIES = []

# ============ 1) parse_args (for local debug) ============
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8005, help='port')  # default
    parser.add_argument('--name', type=str, default='NucSegNode', help='node name')  # default
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['openslide','tiffslide','PIL','numpy'])
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str,
                        choices=['2D_versatile_fluo','2D_paper_dsb2018','2D_versatile_he'])
    parser.add_argument('--isIHC', default=False, type=bool)
    parser.add_argument('--calculate_features', default=False, type=bool)
    return parser.parse_args()


# ============ 2) features ============
def calculate_features(args, centroids, contours):
    """
    from original code:
    uses SlideProperty to get mask, get_nucstat_parallel, returns features
    """
    print('Number of CPU: %d' % mp.cpu_count())
    print('Working on %s ...' % args.slidepath)
    dt = SlideProperty(args, centroids, contours)
    dt.get_mask()
    dt.get_nucstat_parallel()
    nuclei_stat = dt.nuc_stat_processed
    features = nuclei_stat.values.astype(np.float16)
    feature_names = [f"{v[0]}-{v[1]}" for v in nuclei_stat.columns]
    class_vector = np.zeros(len(centroids))
    class_names = {0: 'negative_control'}
    return features, feature_names, class_vector, class_names

# ============ 3) main(args) run_segmentation ============

def run_segmentation(args):
    """
    original main function, run segmentation and calculate features
    return result dict: { "status": "...", "message": "...", "nuclei_count": 0 }
    """

    import traceback

    result = {
        "status": "success",
        "message": "",
        "nuclei_count": 0
    }
    try:
        start_time = time.time()

        slidepath = args.slidepath
        h5_path = slidepath + ".h5"

        ALREADY_HAVE_NUCLEI_SEGMENTATION = False
        APPEND_FEATURES = False
        APPEND_EMBEDDINGS = False
        centroids = None
        contours = None

        # --- if already have h5 file(the h5 file in the Example folder) ---
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as hf:
                if 'nuclei_segmentation' in hf:
                    seg_grp = hf['nuclei_segmentation']
                    try:
                        centroids = seg_grp['centroids'][()].copy()
                        contours = seg_grp['contours'][()].copy()
                        ALREADY_HAVE_NUCLEI_SEGMENTATION = True
                    except:
                        print("Error: nuclei_segmentation group is corrupted.")

                    has_features = ('features' in seg_grp)
                    has_embeddings = ('cell_embeddings' in seg_grp)

                    if ALREADY_HAVE_NUCLEI_SEGMENTATION and has_features and has_embeddings:
                        result["nuclei_count"] = len(centroids)
                        result["message"] = "Using existing nuclei segmentation, embeddings, and features."
                    elif ALREADY_HAVE_NUCLEI_SEGMENTATION and len(centroids) > 0:
                        if not has_embeddings:
                            print("Calculating embeddings for existing nuclei segmentation...")
                            APPEND_EMBEDDINGS = True
                        if (not has_features) and args.calculate_features:
                            print("Calculating features for existing nuclei segmentation...")
                            APPEND_FEATURES = True

        # if need to append embeddings
        if APPEND_EMBEDDINGS and centroids is not None:
            print("Generating embeddings for existing segmentation...")
            from nuc_embedding import NucleiEmbedding
            ne = NucleiEmbedding(args, centroids)
            embeddings = ne.generate_embeddings()

            with h5py.File(h5_path, 'a') as hf_write:
                seg_grp = hf_write['nuclei_segmentation']
                if 'cell_embeddings' in seg_grp:
                    del seg_grp['cell_embeddings']
                seg_grp.create_dataset('cell_embeddings', data=embeddings)

        # if just need to append features
        if APPEND_FEATURES and centroids is not None and contours is not None:
            features, feature_names, class_vector, class_names = calculate_features(args, centroids, contours)
            with h5py.File(h5_path, 'a') as hf_write:
                seg_grp = hf_write['nuclei_segmentation']
                seg_grp.create_dataset('features', data=features)
                seg_grp.create_dataset('feature_names', data=feature_names)
                seg_grp.create_dataset('class_vector', data=class_vector)
                class_names_json = json.dumps(class_names)
                seg_grp.create_dataset('class_names', data=class_names_json, dtype=h5py.string_dtype())

            result["nuclei_count"] = len(centroids)
            result["message"] = "Using existing nuclei segmentation, newly calculated features."

        # if no segmentation in h5 file, do segmentation
        if not ALREADY_HAVE_NUCLEI_SEGMENTATION:
            print(f'Working on {slidepath} with stardist_pretrain={args.stardist_pretrain}, isIHC={args.isIHC}')

            from nuc_seg import SlideSegmentation
            ss = SlideSegmentation(
                args,
                tile_size=4096,
                overlap=256,
                prob_thresh=0.3,
                nms_thresh=0.3,
                n_tiles=(2,2,1),
                stardist_pretrain=args.stardist_pretrain,
                isIHC=args.isIHC
            )
            ss.run_WSI_segmentation()

            contours = ss.final_coord.astype(np.int32)
            centroids = ss.final_points.astype(np.int32)
            probability = ss.prob_all

            mode = 'a' if os.path.exists(h5_path) else 'w'
            with h5py.File(h5_path, mode) as hf:
                seg_grp = hf.create_group('nuclei_segmentation')
                seg_grp.create_dataset('contours', data=contours)
                seg_grp.create_dataset('centroids', data=centroids)
                seg_grp.create_dataset('probability', data=probability)

            # embeddings

            print("Generating nuclei embeddings for new segmentation ...")
            from nuc_embedding import NucleiEmbedding
            ne = NucleiEmbedding(args, centroids)
            embeddings = ne.generate_embeddings()
            with h5py.File(h5_path, 'a') as hf:
                seg_grp = hf['nuclei_segmentation']
                seg_grp.create_dataset('cell_embeddings', data=embeddings)

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        result["message"] = "Segmentation completed successfully"

        result["nuclei_count"] = len(centroids)

        # write workflow h5 file output
        if H5_PATH and os.path.exists(H5_PATH):
            # 1. copy nuclei_segmentation ot workflow h5
            with h5py.File(h5_path, 'r') as src, h5py.File(H5_PATH, 'a') as dst:
                src_grp = src['nuclei_segmentation']

                # if already have nuclei_segmentation in workflow h5, delete it
                if 'nuclei_segmentation' in dst:
                    del dst['nuclei_segmentation']

                # create new group in workflow h5
                dst_grp = dst.create_group('nuclei_segmentation')

                # just copy the datasets
                for name in ['centroids', 'contours', 'probability', 'cell_embeddings']:
                    if name in src_grp:
                        dst_grp.create_dataset(name, data=src_grp[name][()])

                if args.calculate_features:
                    for name in ['features', 'feature_names', 'class_vector', 'class_names']:
                        if name in src_grp:
                            dst_grp.create_dataset(name, data=src_grp[name][()])

            # 2. write into output
            with h5py.File(H5_PATH, 'a') as hf:
                node_out_path = f"{NODE_NAME}/output"
                if node_out_path in hf:
                    del hf[node_out_path]
                out_str = json.dumps({
                    "status": "success",
                    "message": result["message"],
                    "nuclei_count": result["nuclei_count"]
                })
                hf.create_dataset(node_out_path, data=out_str.encode("utf-8"))

        return result

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Traceback:")
        import traceback
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "nuclei_count": 0
        }



# =========== define /status, /init, /read, /execute four routers ===========

@app.get("/status")
def get_status():

    return {"status": "nuc_seg_node running"}

@app.post("/init")
def init_node():
    """
    load model
    """
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[NucSegNode] /init => inited model/resources")
        return {"status": "ok", "message":"NucSegNode init done"}
    else:
        print("[NucSegNode] /init => already done.")
        return {"status": "ok", "message":"Already init."}

@app.post("/read")
def read_node(data: Dict[str, Any]):
    """
            Read upstream node outputs + frontend user inputs (stored in h5 file),
          Save needed content as current Node's member variables for execute use.

          data: dict. Usually contains key "h5_path" for h5 file location.
       """
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS

    NODE_NAME = data.get("node_name", "NucSegNode")
    DEPENDENCIES = data.get("dependencies", [])
    H5_PATH = data.get("h5_path", None)

    print(f"[NucSegNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")

    if (not H5_PATH) or (not os.path.exists(H5_PATH)):
        print("[NucSegNode] no h5 file => skip read.")
        return {"status": "ok", "message": "no H5 file found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            path="",
            read_image_method="tiffslide",
            stardist_pretrain="2D_versatile_he",
            isIHC=False,
            calculate_features=False
        )

    # read userData from h5 file
    with h5py.File(H5_PATH, "r") as hf:
        user_data_path = f"{NODE_NAME}/userData"
        if user_data_path in hf:
            for k in hf[user_data_path].keys():
                raw_bytes = hf[user_data_path][k][()]
                raw_str = raw_bytes.decode("utf-8")
                try:
                    val_json = json.loads(raw_str)
                except:
                    val_json = raw_str
                print(f"[NucSegNode] user param {k} => {val_json}")

                # val_json
                if k == "path":
                    ARGS.slidepath = val_json
                elif k == "read_image_method":
                    ARGS.read_image_method = val_json
                elif k == "stardist_pretrain":
                    ARGS.stardist_pretrain = val_json
                elif k == "isIHC":
                    ARGS.isIHC = (val_json in [True, "true", "True"])
                elif k == "calculate_features":
                    ARGS.calculate_features = (val_json is True or val_json == "true")


        # read dependency outputs
        for dep in DEPENDENCIES:
            out_path = f"{dep}/output"
            if out_path in hf:
                out_bytes = hf[out_path][()]
                out_str = out_bytes.decode("utf-8")
                try:
                    out_json = json.loads(out_str)
                except:
                    out_json = out_str
                print(f"[NucSegNode] sees {dep}'s output => {out_json}")

    return {"status": "ok", "message": "NucSegNode read done"}

@app.post("/execute")
def execute_node():
    """
    Execute actual model inference / data processing / analysis.
   Write results back to h5 file (e.g. /myNode/output etc).
   Finally return a dict as node's external execution result (for logs/viewing only).
    """
    global IS_MODEL_INITED, ARGS, H5_PATH, NODE_NAME

    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}

    if not ARGS or not getattr(ARGS, "slidepath", None):
        print("[NucSegNode NEW] no path => skip.")
        out_val = {
            "status": "ok",
            "message": "no path, skipping.",
            "nuclei_count": 0
        }
    else:
        print(
            f"[NucSegNode NEW] /execute => run_segmentation with slidepath={ARGS.slidepath}, stardist_pretrain={ARGS.stardist_pretrain}, isIHC={ARGS.isIHC}, calc_features={ARGS.calculate_features}")
        out_val = run_segmentation(ARGS)

    #  out_val write to H5 => /NucSegNode/output

    if H5_PATH and os.path.exists(H5_PATH):
        with h5py.File(H5_PATH, "a") as hf:
            node_out_path = f"{NODE_NAME}/output"
            if node_out_path in hf:
                del hf[node_out_path]
            out_str = json.dumps(out_val, ensure_ascii=False)
            hf.create_dataset(node_out_path, data=out_str.encode("utf-8"))

    return {"status": "ok", "output": out_val}

# =========== main: start Node and register to manager =============

def main():
    try:
        args = parse_args()
        print(f"Starting NucSegNode with port={args.port}")

        def run_uvicorn():
            uvicorn.run(app, host="0.0.0.0", port=args.port)

        # create and start a new thread to run uvicorn
        import threading
        t = threading.Thread(target=run_uvicorn, daemon=True)
        t.start()

        time.sleep(3)  # wait for uvicorn to start

        # register to manager
        this_file_path = str(Path(__file__).resolve())
        create_payload = {
            "service_name": args.name,
            "file_path": this_file_path,
            "port": args.port
        }
        url_create = f"{args.manager_host}/api/tasks/v1/create_node"

        try:
            resp = requests.post(url_create, json=create_payload, timeout=10)
            resp.raise_for_status()
            print(f"[{args.name}] create_node success => {resp.json()}")
        except Exception as e:
            print(f"[{args.name}] create_node request failed: {e}")
            print("keep running...")

        print(f"[{args.name}] Serving at port={args.port}, Press Ctrl+C to exit.")
        t.join()

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting service: {e}")

if __name__ == "__main__":
    main()