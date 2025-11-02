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
import os

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
        'tissuelab_sdk',
        *stardist_hiddenimports,
        *transformers_hiddenimports,
    ])

# Resolve shared runtime hook path robustly even when __file__ is undefined
try:
    _spec_dir = os.path.abspath(os.path.dirname(sys.argv[0])) if (hasattr(sys, 'argv') and sys.argv and sys.argv[0].endswith('.spec')) else os.getcwd()
except Exception:
    _spec_dir = os.getcwd()
RUNTIME_HOOKS_DIR = os.path.abspath(os.path.join(_spec_dir, "..", "..", "runtime_hooks"))
rth_spawn_guard = os.path.join(RUNTIME_HOOKS_DIR, "rth_spawn_guard.py")

a = Analysis(
    ['segmentation_taskNode.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('nuc_seg_mac.py', '.'),
        ('nuc_stat.py', '.'),
        ('nuc_embedding_mac.py', '.'),
        ('safe_h5_utils.py', '.'),
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
    runtime_hooks=['mp_hook.py', rth_spawn_guard],
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
    name='TissueLab_Segmentation_Win',
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
    name='TissueLab_Segmentation_Win',
)

parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

if __name__ == '__main__':
    mp.set_start_method('spawn')


