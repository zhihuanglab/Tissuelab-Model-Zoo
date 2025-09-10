# -*- mode: python ; coding: utf-8 -*-

import sys
import imagecodecs
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files
import argparse
import multiprocessing as mp
import os

block_cipher = None

# Collect all necessary data for packages
stardist_datas, stardist_binaries, stardist_hiddenimports = collect_all('stardist')
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')
# Bundle XGBoost native library (.dylib) explicitly
xgboost_binaries = collect_dynamic_libs('xgboost')
xgboost_datas = collect_data_files('xgboost', includes=['VERSION'])

# macOS-specific hidden imports (omit tensorflow unless explicitly needed)
hiddenimport_data = (
    ["imagecodecs." + x for x in imagecodecs._extensions()]
    + [
        # stardist removed on macOS unless needed and installed
        'transformers',
        'xgboost',
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
        *stardist_hiddenimports,
        *transformers_hiddenimports,
    ])

a = Analysis(
    ['classification_taskNode.py'],  # Main entry script
    pathex=[],
    binaries=xgboost_binaries,
    datas=[
        # Include entire checkpoints directory so runtime lookup finds the right file
        ('checkpoints', 'checkpoints'),
        ('negative_control_example_vectors.npy', '.'),
        ('negative_control_examples', 'negative_control_examples'),
        # Windows-only imagecodecs .pyd files intentionally omitted on macOS
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
    name='TissueLab_Classification_Mac',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Prefer disabled on macOS
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
    upx=False,
    upx_exclude=[],
    name='TissueLab_Classification_Mac',
)

parser = argparse.ArgumentParser()
# add custom arguments
# parser.add_argument('--your-arg', type=str, help='Your argument description')
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')


