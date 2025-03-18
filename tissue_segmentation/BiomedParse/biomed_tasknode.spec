# -*- mode: python ; coding: utf-8 -*-

import os
import imagecodecs

block_cipher = None

mpi4py_path = './mpi4py'

added_files = [
    ('configs/*.yaml', 'configs'),  # config
    ('biomedparse_v1.pt', '.'),    # weights
    ('modeling/', 'modeling/'),     # modeling folder
    ('utilities/', 'utilities/'),   # utilities folder
    ('inference_utils/', 'inference_utils/'),  # inference_utils folder
    ('Resources\\mpi4py\\', 'mpi4py\\'),  # local mpi4py resources
    # Imagecodecs pyds you manually placed in "Resources"
    ("Resources\\imagecodecs\\_zlib.cp39-win_amd64.pyd", "imagecodecs"),
    ("Resources\\imagecodecs\\_jpeg8.cp39-win_amd64.pyd", "imagecodecs"),
    ("Resources\\imagecodecs\\_jpeg2k.cp39-win_amd64.pyd", "imagecodecs"),
    ("Resources\\imagecodecs\\_imcd.cp39-win_amd64.pyd", "imagecodecs"),
    ("Resources\\imagecodecs\\_shared.cp39-win_amd64.pyd", "imagecodecs"),
]

hiddenimport_data = (
    ["imagecodecs." + x for x in imagecodecs._extensions()]
    + [
        "imagecodecs._shared",
        "imagecodecs._imcd",
        'mpi4py',
        'mpi4py.MPI',
        'mpi4py.libmpi',
        'mpi4py._mpiabi',
        'mpi4py.dl',
        'mpi4py.run',
        # FastAPI related
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        # main dependencies
        'PIL',
        'cv2',
        'numpy',
        'torch',
        'torchvision',
        'tiffslide',
        'h5py',
        'skimage',
        'skimage.transform',
        'yaml',
        'json_tricks',
        'yacs',
        'sklearn',
        'pandas',
        'timm',
        'einops',
        'fvcore',
        'transformers',
        'sentencepiece',
        'ftfy',
        'regex',
        'nltk',
        'vision_datasets',
        'pycocotools',
        'scikit-image',
        'mup',
        'accelerate',
        'kornia',
        'open_clip_torch',
    ])

a = Analysis(
    ['biomed_tasknode.py'],
    pathex=['.'],  # current folder
    binaries=[],
    datas=added_files,
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
    name='biomed_tasknode',
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
    name='biomed_tasknode',
)