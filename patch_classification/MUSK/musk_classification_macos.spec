# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, collect_data_files
import argparse
import multiprocessing as mp

block_cipher = None

# Collect package data
fastapi_datas, fastapi_binaries, fastapi_hiddenimports = collect_all('fastapi')
uvicorn_datas, uvicorn_binaries, uvicorn_hiddenimports = collect_all('uvicorn')

hiddenimport_data = (
    [
        'tiffslide',
        'torch',
        'torchvision',
        'torchvision.io.image',
        'xgboost',
        'numpy',
        'pandas',
        'fastapi',
        'uvicorn',
        'sse_starlette.sse',
        'colorsys',
    ]
    + fastapi_hiddenimports
    + uvicorn_hiddenimports
)

# Resolve shared runtime hook path robustly even when __file__ is undefined
try:
    _spec_dir = os.path.abspath(os.path.dirname(sys.argv[0])) if (hasattr(sys, 'argv') and sys.argv and sys.argv[0].endswith('.spec')) else os.getcwd()
except Exception:
    _spec_dir = os.getcwd()
RUNTIME_HOOKS_DIR = os.path.abspath(os.path.join(_spec_dir, "..", "..", "runtime_hooks"))
rth_spawn_guard = os.path.join(RUNTIME_HOOKS_DIR, "rth_spawn_guard.py")

# Bundle XGBoost native library and VERSION file (like NuClass)
xgboost_binaries = collect_dynamic_libs('xgboost')
xgboost_datas = collect_data_files('xgboost', includes=['VERSION'])

a = Analysis(
    ['musk_classification_tasknode.py'],
    pathex=[],
    binaries=xgboost_binaries,
    datas=[
        # Bundle model/checkpoints and optional negative control vectors
        ('checkpoints', 'checkpoints'),
        ('model', 'model'),
        ('negative_control_vectors_1024d.npy', '.'),
        *xgboost_datas,
        *fastapi_datas,
        *uvicorn_datas,
    ],
    hiddenimports=hiddenimport_data,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[rth_spawn_guard],
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
    name='TissueLab_MUSK_Classification_Mac',
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
    name='TissueLab_MUSK_Classification_Mac',
)

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')


