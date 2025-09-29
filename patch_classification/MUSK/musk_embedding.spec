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
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')
transformers_datas, transformers_binaries, transformers_hiddenimports = collect_all('transformers')
imagecodecs_datas, imagecodecs_binaries, imagecodecs_hiddenimports = collect_all('imagecodecs')

# Specify CUDA version for PyTorch
CUDA_VERSION = "12.1"  # Change this to your desired CUDA version

hiddenimport_data = (
    [
        'tiffslide',
        'torch',
        'torch.nn',
        'torch.optim',
        'torch.utils',
        'torchvision',
        'torchvision.io.image',
        'torchaudio',
        'timm',
        'PIL',
        'sse_starlette.sse',
        'fastapi',
        'uvicorn',
        'numpy',
        'opencv-python',
        'cv2',
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
        'blobfile',
        'mypy',
        'pytest',
        'tensorboardX',
        'ftfy',
        'sentencepiece',
        'pyarrow',
        'pytorch_lightning',
        'nltk',
        'rouge',
        'accelerate',
        'fairscale',
        'ruamel.yaml',
        'wandb',
        'future',
        'scikit_survival',
        'torchmetrics',
        'open_clip_torch',
        'pycocoevalcap',
        'webdataset',
        'huggingface_hub',
        'safetensors',
        'starlette',
        'datasets',
        # Windows-specific
        'win32con',
        'win32api',
        'win32security',
        'pywin32',
        # Image codecs
        'imagecodecs',
        'imagecodecs.jpeg8',
        'imagecodecs.jpeg2k',
        'imagecodecs.zlib',
        'imagecodecs.imcd',
        'imagecodecs.shared',
    ]
    + fastapi_hiddenimports
    + uvicorn_hiddenimports
    + torch_hiddenimports
    + transformers_hiddenimports
    + imagecodecs_hiddenimports
)

# Resolve shared runtime hook path robustly even when __file__ is undefined
try:
    _spec_dir = os.path.abspath(os.path.dirname(sys.argv[0])) if (hasattr(sys, 'argv') and sys.argv and sys.argv[0].endswith('.spec')) else os.getcwd()
except Exception:
    _spec_dir = os.getcwd()
RUNTIME_HOOKS_DIR = os.path.abspath(os.path.join(_spec_dir, "..", "..", "runtime_hooks"))
rth_spawn_guard = os.path.join(RUNTIME_HOOKS_DIR, "rth_spawn_guard.py")

a = Analysis(
    ['musk_embedding_taskNode.py'],
    pathex=[],
    binaries=torch_binaries + imagecodecs_binaries,
    datas=[
        # Bundle model assets so the binary can find them under _MEIPASS
        ('model', 'model'),
        ('checkpoints', 'checkpoints'),
        *fastapi_datas,
        *uvicorn_datas,
        *torch_datas,
        *transformers_datas,
        *imagecodecs_datas,
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
    name='TissueLab_MUSK_Embedding',
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
    name='TissueLab_MUSK_Embedding',
)

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')
