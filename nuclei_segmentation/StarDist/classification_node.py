#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Node for nuclei feature extraction and embedding generation
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
import pandas as pd

from fastapi import FastAPI
from typing import Dict, Any
from pathlib import Path

from nuc_stat import SlideProperty
from nuc_embedding import NucleiEmbedding

app = FastAPI()

# Global variables
ARGS = None
IS_MODEL_INITED = False
H5_PATH = None
NODE_NAME = None
DEPENDENCIES = ["SegmentationNode"]  # Must depend on SegmentationNode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8006, help='port')
    parser.add_argument('--name', type=str, default='ClassificationNode', help='node name')
    parser.add_argument('--manager_host', type=str, default='http://localhost:5001', help='manager service URL')

    parser.add_argument('--slidepath', default='', type=str)
    parser.add_argument('--read_image_method', default='tiffslide', type=str,
                        choices=['openslide', 'tiffslide', 'PIL', 'numpy'])
    parser.add_argument('--calculate_features', default=False, type=bool)
    return parser.parse_args()



def calculate_features(args, centroids, contours):
    """Calculate features using SlideProperty"""
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


def run_classification(args):
    """
    只生成 nuclei embeddings，读取 SegmentationNode 产生的分割结果
    并将 embedding 保存到 H5 文件中，返回生成情况。
    """
    result = {
        "status": "success",
        "message": "",
        "embedding_count": 0
    }
    try:
        start_time = time.time()

        # 读取来自 SegmentationNode 的分割结果
        if not H5_PATH or not os.path.exists(H5_PATH):
            raise ValueError("H5 file not found")

        with h5py.File(H5_PATH, 'r') as hf:
            seg_path = "SegmentationNode"
            if seg_path not in hf:
                raise ValueError("SegmentationNode results not found in H5 file")
            try:
                centroids = np.array(hf[f"{seg_path}/centroids"][()], dtype=np.int32)
                contours = np.array(hf[f"{seg_path}/contours"][()], dtype=np.int32)
                print(f"Read centroids shape: {centroids.shape}, dtype: {centroids.dtype}")
                print(f"Read contours shape: {contours.shape}, dtype: {contours.dtype}")
            except Exception as e:
                raise ValueError(f"Failed to read segmentation data: {str(e)}")

        slide_basename = os.path.basename(args.slidepath)
        temp_h5_path = f"/path/to/temp_dir/temp_classification_{slide_basename}.h5"

        # 3) 检查 temp_h5_path 是否有 embedding
        have_cached_embedding = False
        embedding_data = None
        if os.path.exists(temp_h5_path):
            with h5py.File(temp_h5_path, "r") as tf:
                if "embedding" in tf:
                    e = tf["embedding"][()]
                    if len(e) == len(centroids):
                        print("Found existing embedding in temp => skip embed calc")
                        have_cached_embedding = True
                        embedding_data = e
        # 4) 如果没有 => 计算 embedding
        if not have_cached_embedding:
            print("No cached embedding => generate new")
            # 生成embedding
            ne = NucleiEmbedding(args, centroids)
            embedding_data = ne.generate_embeddings()

            # 存到temp
            with h5py.File(temp_h5_path, "w") as tf:
                tf.create_dataset("embedding", data=embedding_data)

        # 5) 把embedding_data写到当前 workflow H5 => ClassificationNode/cell_embeddings
        with h5py.File(H5_PATH, "a") as hf:
            if "ClassificationNode" not in hf:
                grp = hf.create_group("ClassificationNode")
            else:
                grp = hf["ClassificationNode"]
            if "cell_embeddings" in grp:
                del grp["cell_embeddings"]
            grp.create_dataset("cell_embeddings", data=embedding_data)

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        result["message"] = "Embedding generation completed successfully"
        result["embedding_count"] = len(embeddings)
        return result

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": str(e),
            "embedding_count": 0
        }



@app.get("/status")
def get_status():
    return {"status": "classification_node running"}


@app.post("/init")
def init_node():
    global IS_MODEL_INITED
    if not IS_MODEL_INITED:
        IS_MODEL_INITED = True
        print("[ClassificationNode] /init => inited model/resources")
        return {"status": "ok", "message": "ClassificationNode init done"}
    else:
        print("[ClassificationNode] /init => already done.")
        return {"status": "ok", "message": "Already init."}


@app.post("/read")
def read_node(data: Dict[str, Any]):
    global NODE_NAME, DEPENDENCIES, H5_PATH, ARGS

    NODE_NAME = data.get("node_name", "ClassificationNode")
    DEPENDENCIES = data.get("dependencies", ["SegmentationNode"])
    H5_PATH = data.get("h5_path", None)

    print(f"[ClassificationNode] /read => node_name={NODE_NAME}, deps={DEPENDENCIES}, h5_path={H5_PATH}")

    if (not H5_PATH) or (not os.path.exists(H5_PATH)):
        print("[ClassificationNode] no h5 file => skip read.")
        return {"status": "ok", "message": "no H5 file found."}

    if ARGS is None:
        ARGS = argparse.Namespace(
            slidepath="",
            read_image_method="tiffslide",
            calculate_features=False
        )

        # 读取 userData（如果有的话）
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
                    print(f"[ClassificationNode] user param {k} => {val_json}")
                    if k == "path":
                        ARGS.slidepath = val_json
                    elif k == "read_image_method":
                        ARGS.read_image_method = val_json
                    elif k == "calculate_features":
                        ARGS.calculate_features = (val_json in [True, "true", "True"])
        return {"status": "ok", "message": "ClassificationNode read done"}


@app.post("/execute")
def execute_node():
    global IS_MODEL_INITED, ARGS, H5_PATH, NODE_NAME
    if not IS_MODEL_INITED:
        return {"status": "error", "message": "Please /init first."}
    if not ARGS or not getattr(ARGS, "slidepath", None):
        print("[ClassificationNode] no path => skip.")
        out_val = {"status": "ok", "message": "no path, skipping.", "embedding_count": 0}
    else:
        print(f"[ClassificationNode] /execute => run_classification with slidepath={ARGS.slidepath}")
        out_val = run_classification(ARGS)
    if H5_PATH and os.path.exists(H5_PATH):
        with h5py.File(H5_PATH, "a") as hf:
            node_out_path = f"{NODE_NAME}/output"
            if node_out_path in hf:
                del hf[node_out_path]
            out_str = json.dumps(out_val, ensure_ascii=False)
            hf.create_dataset(node_out_path, data=out_str.encode("utf-8"))
    return {"status": "ok", "output": out_val}


def main():
    try:
        args = parse_args()
        print(f"Starting ClassificationNode with port={args.port}")

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