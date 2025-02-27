# -*- mode: python ; coding: utf-8 -*-

import sys
import imagecodecs
from PyInstaller.utils.hooks import collect_all

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
        *stardist_hiddenimports,
        *transformers_hiddenimports,
    ])

a = Analysis(
    ['nuc_seg_task_node.py'],  # Main entry script
    pathex=[],
    binaries=[],
    datas=[
        ('nuc_embedding.py', '.'),
        ('nuc_seg.py', '.'),
        ('nuc_stat.py', '.'),
        ('checkpoints', 'checkpoints'),
        ('histomicstk_scripts', 'histomicstk_scripts'),
        ('models', 'models'),
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
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NucSegNode',
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NucSegNode',
)