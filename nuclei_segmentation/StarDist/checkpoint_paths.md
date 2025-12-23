# Checkpoint Paths 说明

在 `segmentation_taskNode.py` 中，有两个模型需要加载 checkpoint：

## 1. StarDist 模型（用于细胞核分割）

### Checkpoint Path
```
{StarDist目录}/models/{stardist_pretrain}/
```

### 具体位置
- **代码位置**: `nuc_seg.py` 第 98-104 行
- **默认值**: `stardist_pretrain = '2D_versatile_he'`
- **完整路径示例**: 
  ```
  /home/peixian/Tissuelab-Model-Zoo/nuclei_segmentation/StarDist/models/2D_versatile_he/
  ```

### 代码逻辑
```python
# nuc_seg.py 第 98-104 行
local_model_path = os.path.join(os.path.dirname(__file__), 'models', stardist_pretrain)
if os.path.exists(local_model_path):
    print(f"Loading StarDist model from local path: {local_model_path}")
    self.model = StarDist2D(None, name=stardist_pretrain, 
                           basedir=os.path.join(os.path.dirname(__file__), 'models'))
else:
    print(f"Local model not found at {local_model_path}, attempting to download...")
    self.model = StarDist2D.from_pretrained(stardist_pretrain)
```

### 可选的 pretrain 模型
- `'2D_versatile_he'` (默认，用于 H&E 染色)
- `'2D_versatile_fluo'` (用于荧光图像)
- `'2D_paper_dsb2018'` (用于特定数据集)

### 如何设置
通过 `--stardist_pretrain` 参数或 zarr 中的 `userData/stardist_pretrain` 设置。

---

## 2. NucleiEmbedding 模型（用于生成 embedding）

### Checkpoint Path
```
{StarDist目录}/checkpoints/checkpoint_step_10000.pt
```

### 具体位置
- **代码位置**: `nuc_embedding.py` 第 558 行
- **完整路径示例**:
  ```
  /home/peixian/Tissuelab-Model-Zoo/nuclei_segmentation/StarDist/checkpoints/checkpoint_step_10000.pt
  ```

### 代码逻辑
```python
# nuc_embedding.py 第 557-572 行
checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'checkpoint_step_10000.pt')
if os.path.exists(checkpoint_path):
    print(f"Loading trained checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    
    # Load model state
    self.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load image projection layer
    self.image_projection.load_state_dict(checkpoint['image_projection_state_dict'])
    print("Successfully loaded checkpoint")
else:
    raise FileNotFoundError(f"Required checkpoint not found at {checkpoint_path}. Cannot proceed without trained model.")
```

### 重要说明
- **必需文件**: 如果 checkpoint 不存在，会抛出 `FileNotFoundError`
- **包含内容**: 
  - `model_state_dict`: PLIP 模型的状态
  - `image_projection_state_dict`: 图像投影层的状态

---

## 目录结构

```
StarDist/
├── segmentation_taskNode.py
├── nuc_seg.py              # StarDist 模型加载
├── nuc_embedding.py        # Embedding 模型加载
├── models/                 # StarDist 预训练模型目录
│   ├── 2D_versatile_he/    # H&E 染色模型
│   ├── 2D_versatile_fluo/  # 荧光模型
│   └── 2D_paper_dsb2018/  # DSB2018 模型
└── checkpoints/            # Embedding 模型 checkpoint
    └── checkpoint_step_10000.pt
```

---

## 总结

| 模型 | Checkpoint Path | 参数控制 | 必需性 |
|------|----------------|---------|--------|
| StarDist | `models/{stardist_pretrain}/` | `--stardist_pretrain` | 可选（不存在会下载） |
| NucleiEmbedding | `checkpoints/checkpoint_step_10000.pt` | 无（硬编码） | **必需** |





