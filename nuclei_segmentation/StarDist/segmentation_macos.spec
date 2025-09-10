# -*- mode: python ; coding: utf-8 -*-

import sys
try:
    import imagecodecs
    _imagecodecs_hidden = ["imagecodecs." + x for x in imagecodecs._extensions()]
except Exception:
    _imagecodecs_hidden = []
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_data_files
import argparse
import multiprocessing as mp

block_cipher = None

# Collect package data
stardist_datas, stardist_binaries, stardist_hiddenimports = collect_all('stardist')
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')

hiddenimport_data = (
    _imagecodecs_hidden
    + [
        # Omit tensorflow on mac unless needed
        'stardist',
        'stardist.models',
        'transformers',
        'scipy',
        'matplotlib',
        'cv2',
        'tiffslide',
        'multiprocess',
        'pandas',
        'numpy',
        'tqdm',
        'natsort',
        'einops',
        'pydicom',
        'czifile',
        'pylibCZIrw',
        'openslide',
        'openslide.lowlevel',
        *stardist_hiddenimports,
        *transformers_hiddenimports,
    ])

a = Analysis(
    ['segmentation_taskNode.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('nuc_seg.py', '.'),
        ('nuc_stat.py', '.'),
        ('nuc_embedding.py', '.'),
        ('checkpoints', 'checkpoints'),
        ('histomicstk_scripts', 'histomicstk_scripts'),
        ('models', 'models'),
        # Windows-only imagecodecs .pyd omitted for macOS
        *stardist_datas,
        *transformers_datas,
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
    name='TissueLab_Segmentation_Mac',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    name='TissueLab_Segmentation_Mac',
)

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')


