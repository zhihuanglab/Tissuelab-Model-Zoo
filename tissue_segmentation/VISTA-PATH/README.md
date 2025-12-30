## 1. conda envirnment

```
conda create -n PathSeg python=3.12
conda activate PathSeg

conda install -c conda-forge scikit-image opencv pandas pillow numpy

conda install -c conda-forge openslide openslide-python

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia

pip install transformers==4.46.1

pip install pycocotools matplotlib scikit-learn

pip install accelerate==0.26.0

conda install -c conda-forge opencv

conda install -c conda-forge albumentations

pip install zarr
```


## 2. checkpoint 

download checkpoint from

`https://github.com/pxliang/PASeg/tree/main/checkpoints` to `Tissuelab-Model-Zoo/tissue_segmentation/VISTA-PATH/checkpoints`

## 3. prepare zarr file

### get the zarr in the format:

```
HE_img.png.zarr/
└── SegNode/
    ├── images        ← (2448, 512, 512, 3) uint8  
    ├── coordinates   ← (2448, 4) int64            
    ├── patch_id                                   
    └── userData/
        ├── path
        ├── tiling_params
        └── SegNode_config
```

```
python3 python prepare_segnode_zarr.py \\
      --wsi_path /path/to/slide.svs \\
      --zarr_path /path/to/output.zarr \\
      --zarr_group SegNode \\
      --patch_size 512 \\
      --stride 512 \\
      --level 0
```

