# -*- mode: python ; coding: utf-8 -*-

import sys
try:
    import imagecodecs
    _imagecodecs_hidden = ["imagecodecs." + x for x in imagecodecs._extensions()]
except Exception:
    _imagecodecs_hidden = []
from PyInstaller.utils.hooks import collect_all
import argparse
import multiprocessing as mp
import os

block_cipher = None

# Cellpose loads its networks and transforms by name; the cpsam_v2 weights are
# downloaded to ~/.cellpose/models on first use, not bundled.
cellpose_datas, cellpose_binaries, cellpose_hiddenimports = collect_all('cellpose')

hiddenimport_data = (
    _imagecodecs_hidden
    + [
        'cellpose',
        'cellpose.models',
        'scipy',
        'skimage',
        'cv2',
        'tiffslide',
        'tifffile',
        'pydicom',
        'pandas',
        'numpy',
        'tqdm',
        'requests',
        'sse_starlette',
        'fastapi',
        'uvicorn',
        'zarr',
        'tissuelab_sdk',
        'torch',
        'torchvision',
        *cellpose_hiddenimports,
    ])

# Resolve shared runtime hook path robustly even when __file__ is undefined
try:
    _spec_dir = os.path.abspath(os.path.dirname(sys.argv[0])) if (hasattr(sys, 'argv') and sys.argv and sys.argv[0].endswith('.spec')) else os.getcwd()
except Exception:
    _spec_dir = os.getcwd()
RUNTIME_HOOKS_DIR = os.path.abspath(os.path.join(_spec_dir, "..", "..", "runtime_hooks"))
rth_spawn_guard = os.path.join(RUNTIME_HOOKS_DIR, "rth_spawn_guard.py")

a = Analysis(
    ['cellpose_sam_taskNode.py'],
    pathex=[],
    binaries=cellpose_binaries,
    datas=[
        ('cellpose_nuc_seg.py', '.'),
        *cellpose_datas,
    ],
    hiddenimports=hiddenimport_data,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[rth_spawn_guard],
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
    name='TissueLab_CellposeSAM_Mac',
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
    name='TissueLab_CellposeSAM_Mac',
)

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')
