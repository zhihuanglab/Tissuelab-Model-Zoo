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

hiddenimport_data = (
    ["imagecodecs." + x for x in imagecodecs._extensions()]
    + [
        'tensorflow',
        'tensorflow.python',
        'tensorflow.python.framework',
        'tensorflow.python.ops',
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
        'einops',
        'multiprocess',
        'transformers',
        *stardist_hiddenimports,
        *transformers_hiddenimports,
    ])

a = Analysis(
    ['classification_taskNode.py'],  # Main entry script
    pathex=[],
    binaries=[],
    datas=[
        ('checkpoints/contrastive_checkpoint_epoch_0.pt', 'checkpoints'),
        ('checkpoints', 'checkpoints'),
        ("Resources\\imagecodecs\\_zlib.cp39-win_amd64.pyd", "imagecodecs"),
        ("Resources\\imagecodecs\\_jpeg8.cp39-win_amd64.pyd", "imagecodecs"),
        ("Resources\\imagecodecs\\_jpeg2k.cp39-win_amd64.pyd", "imagecodecs"),
        ("Resources\\imagecodecs\\_imcd.cp39-win_amd64.pyd", "imagecodecs"),
        ("Resources\\imagecodecs\\_shared.cp39-win_amd64.pyd", "imagecodecs"),
        *stardist_datas,
        *transformers_datas,
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
    name='ClassificationNode',
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ClassificationNode',
)

parser = argparse.ArgumentParser()
# add custom arguments
# parser.add_argument('--your-arg', type=str, help='Your argument description')
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')