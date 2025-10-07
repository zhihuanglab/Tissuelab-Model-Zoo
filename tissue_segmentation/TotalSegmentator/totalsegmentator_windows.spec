# -*- mode: python ; coding: utf-8 -*-

import sys
import os
try:
    import imagecodecs
    _imagecodecs_hidden = ["imagecodecs." + x for x in imagecodecs._extensions()]
except Exception:
    _imagecodecs_hidden = []
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules
import argparse
import multiprocessing as mp

block_cipher = None

# Collect package data
print("[Build] Collecting TotalSegmentator data...")
totalseg_datas, totalseg_binaries, totalseg_hiddenimports = collect_all('totalsegmentator')

print("[Build] Collecting nibabel data...")
nibabel_datas, nibabel_binaries, nibabel_hiddenimports = collect_all('nibabel')

print("[Build] Collecting SimpleITK data...")
sitk_datas, sitk_binaries, sitk_hiddenimports = collect_all('SimpleITK')

print("[Build] Collecting torch data...")
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

print("[Build] Collecting FastAPI/Uvicorn data...")
fastapi_datas, fastapi_binaries, fastapi_hiddenimports = collect_all('fastapi')
uvicorn_datas, uvicorn_binaries, uvicorn_hiddenimports = collect_all('uvicorn')

print("[Build] Collecting nnUNet data...")
nnunet_datas, nnunet_binaries, nnunet_hiddenimports = collect_all('nnunetv2')

# Collect all scipy submodules (important for TotalSegmentator)
scipy_submodules = collect_submodules('scipy')

hiddenimport_data = (
    _imagecodecs_hidden
    + [
        # Core dependencies
        'totalsegmentator',
        'totalsegmentator.python_api',
        'totalsegmentator.libs',
        'totalsegmentator.map_to_binary',
        'nibabel',
        'nibabel.freesurfer',
        'nibabel.gifti',
        'nibabel.nifti1',
        'nibabel.nifti2',
        'SimpleITK',
        'dicom2nifti',
        'pydicom',
        'pydicom.encoders',
        'pydicom.encoders.gdcm',
        'pydicom.encoders.pylibjpeg',
        
        # Deep learning
        'torch',
        'torchvision',
        'nnunetv2',
        'nnunetv2.inference',
        'nnunetv2.imageio',
        'acvl_utils',
        
        # Scientific computing
        'numpy',
        'scipy',
        'scipy.ndimage',
        'scipy.spatial',
        'skimage',
        'skimage.transform',
        'skimage.measure',
        'cv2',
        
        # Image processing
        'PIL',
        'PIL.Image',
        'tifffile',
        
        # Web service
        'fastapi',
        'uvicorn',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'sse_starlette',
        'sse_starlette.sse',
        
        # Utilities
        'h5py',
        'pandas',
        'tqdm',
        'requests',
        'safe_h5_utils',
        
        # Collected imports
        *totalseg_hiddenimports,
        *nibabel_hiddenimports,
        *sitk_hiddenimports,
        *torch_hiddenimports,
        *fastapi_hiddenimports,
        *uvicorn_hiddenimports,
        *nnunet_hiddenimports,
        *scipy_submodules,
    ]
)

# Resolve shared runtime hook path
try:
    _spec_dir = os.path.abspath(os.path.dirname(sys.argv[0])) if (hasattr(sys, 'argv') and sys.argv and sys.argv[0].endswith('.spec')) else os.getcwd()
except Exception:
    _spec_dir = os.getcwd()
RUNTIME_HOOKS_DIR = os.path.abspath(os.path.join(_spec_dir, "..", "..", "runtime_hooks"))
rth_spawn_guard = os.path.join(RUNTIME_HOOKS_DIR, "rth_spawn_guard.py")

a = Analysis(
    ['totalsegmentator_tasknode.py'],
    pathex=[],
    binaries=[
        *totalseg_binaries,
        *nibabel_binaries,
        *sitk_binaries,
        *torch_binaries,
        *fastapi_binaries,
        *uvicorn_binaries,
        *nnunet_binaries,
    ],
    datas=[
        ('safe_h5_utils.py', '.'),
        # Include local TotalSegmentator source if exists
        ('TotalSegmentator-src', 'TotalSegmentator-src') if os.path.exists('TotalSegmentator-src') else ('safe_h5_utils.py', '.'),  # Dummy if not exists
        # Include local model weights if exists
        ('models', 'models') if os.path.exists('models') else ('safe_h5_utils.py', '.'),  # Dummy if not exists
        *totalseg_datas,
        *nibabel_datas,
        *sitk_datas,
        *torch_datas,
        *fastapi_datas,
        *uvicorn_datas,
        *nnunet_datas,
    ],
    hiddenimports=hiddenimport_data,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[rth_spawn_guard] if os.path.exists(rth_spawn_guard) else [],
    excludes=[
        'matplotlib',  # Exclude heavy packages not needed
        'IPython',
        'jupyter',
        'notebook',
    ],
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
    name='TissueLab_TotalSegmentator_Win',
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='TissueLab_TotalSegmentator_Win',
)

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')
