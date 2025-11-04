# -*- mode: python ; coding: utf-8 -*-

import sys
import imagecodecs
from PyInstaller.utils.hooks import collect_all
import argparse
import multiprocessing as mp
import os

block_cipher = None

# Collect all necessary data for packages
stardist_datas, stardist_binaries, stardist_hiddenimports = collect_all('stardist')
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')
xgboost_datas, xgboost_binaries, xgboost_hiddenimports = collect_all('xgboost')

hiddenimport_data = (
    ["imagecodecs." + x for x in imagecodecs._extensions()]
    + [
        'stardist',
        'stardist.models',
        'transformers',
        'xgboost',
        'scikit-learn',
        'scipy',
        'matplotlib',
        'cv2',
        'tiffslide',
        'multiprocess',
        'fastdist',
        'pandas',
        'numpy',
        'tqdm',
        'natsort',
        'multiprocess',
        'transformers',
        # Missing dependencies from requirements.txt
        'requests',
        'sse_starlette',
        'fastapi',
        'uvicorn',
        # Zarr for data storage (replaced h5py)
        'zarr',
        'zarr.hierarchy',
        'zarr.core',
        'tissuelab_sdk',
        'torch',
        'torch.nn',
        'torch.optim',
        'torch.utils',
        *stardist_hiddenimports,
        *transformers_hiddenimports,
        *xgboost_hiddenimports,
    ])

a = Analysis(
    ['classification_taskNode.py'],  # Main entry script
    pathex=[],
    binaries=xgboost_binaries,
    datas=[
        ('checkpoints/contrastive_checkpoint_epoch_0.pt', 'checkpoints'),
        ('checkpoints/checkpoint_step_10000.pt', 'checkpoints'),
        ('checkpoints', 'checkpoints'),
        ('negative_control_example_vectors.npy', '.'),
        ('negative_control_examples', 'negative_control_examples'),
        ('classifier.xgb', '.'),
        *stardist_datas,
        *transformers_datas,
        *xgboost_datas,
    ],
    hiddenimports=hiddenimport_data,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TissueLab_Classification',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    runtime_tmpdir=None,
    icon='TissueLab_logo.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TissueLab_Classification',
)

parser = argparse.ArgumentParser()
# add custom arguments
# parser.add_argument('--your-arg', type=str, help='Your argument description')
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')