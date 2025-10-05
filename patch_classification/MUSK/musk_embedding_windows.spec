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

# Collect package data as needed (timm/torch/torchvision have their own hooks)
fastapi_datas, fastapi_binaries, fastapi_hiddenimports = collect_all('fastapi')
uvicorn_datas, uvicorn_binaries, uvicorn_hiddenimports = collect_all('uvicorn')

hiddenimport_data = (
    _imagecodecs_hidden
    + [
        'tiffslide',
        'torch',
        'torchvision',
        'torchvision.io.image',
        'timm',
        'PIL',
        'sse_starlette.sse',
        'fastapi',
        'uvicorn',
        'numpy',
        'opencv-python',
        # Additional dependencies from requirements.txt
        'xgboost',
        'pandas',
        'scipy',
        'transformers',
        'tissuelab_sdk',
        'einops',
        'requests',
        'skimage',  # scikit-image
        'h5py',
        'safetensors',
        'huggingface_hub',
        'safe_h5_utils',
    ]
    + fastapi_hiddenimports
    + uvicorn_hiddenimports
)

# Resolve shared runtime hook path
try:
    _spec_dir = os.path.abspath(os.path.dirname(sys.argv[0])) if (hasattr(sys, 'argv') and sys.argv and sys.argv[0].endswith('.spec')) else os.getcwd()
except Exception:
    _spec_dir = os.getcwd()
RUNTIME_HOOKS_DIR = os.path.abspath(os.path.join(_spec_dir, "..", "..", "runtime_hooks"))
rth_spawn_guard = os.path.join(RUNTIME_HOOKS_DIR, "rth_spawn_guard.py")

a = Analysis(
    ['musk_embedding_taskNode.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Bundle model assets so the binary can find them under _MEIPASS
        ('model', 'model'),
        ('checkpoints', 'checkpoints'),
        ('safe_h5_utils.py', '.'),
        ('TissueLab_logo.ico', '.'),
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
    name='TissueLab_MUSK_Embedding_Win',
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
    name='TissueLab_MUSK_Embedding_Win',
)

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')


