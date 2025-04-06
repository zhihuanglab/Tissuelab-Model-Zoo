# -*- mode: python ; coding: utf-8 -*-

import sys
import imagecodecs
from PyInstaller.utils.hooks import collect_all
import argparse
import multiprocessing as mp

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
        'czifile',
        'win32con',
        'win32api',
        'win32security',
        'pywin32',
        'pydicom',
        'pylibCZIrw',
        *stardist_hiddenimports,
        *transformers_hiddenimports,
        'multiprocess.spawn',
        'multiprocess.context',
        'multiprocess.popen_spawn_win32',
        'multiprocess.reduction',
        'multiprocess.resource_tracker',
        'openslide',
        'openslide.lowlevel',
    ])

a = Analysis(
    ['segmentation_taskNode.py'],  # Main entry script
    pathex=[],
    binaries=[],
    datas=[
        ('nuc_seg.py', '.'),
        ('nuc_stat.py', '.'),
        ('nuc_embedding.py', '.'),
        ('checkpoints', 'checkpoints'),
        ('checkpoints/contrastive_checkpoint_epoch_0.pt', 'checkpoints'),
        ('histomicstk_scripts', 'histomicstk_scripts'),
        ('models', 'models'),
        ("Resources\\imagecodecs\\_zlib.cp39-win_amd64.pyd", "imagecodecs"),
        ("Resources\\imagecodecs\\_jpeg8.cp39-win_amd64.pyd", "imagecodecs"),
        ("Resources\\imagecodecs\\_jpeg2k.cp39-win_amd64.pyd", "imagecodecs"),
        ("Resources\\imagecodecs\\_imcd.cp39-win_amd64.pyd", "imagecodecs"),
        ("Resources\\imagecodecs\\_shared.cp39-win_amd64.pyd", "imagecodecs"),
        *stardist_datas,
        *transformers_datas
    ],
    hiddenimports=hiddenimport_data,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['mp_hook.py'],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TissueLab_Segmentation',
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
    name='TissueLab_Segmentation',
)