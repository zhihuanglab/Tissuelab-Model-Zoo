# TotalSegmentator 路径配置说明

## ✅ 确认：路径配置完全正确！

根据验证结果，TotalSegmentator **能够正确找到**下载到 `models/nnunet/results/` 的所有权重。

---

## 📂 路径流程详解

### 1. TaskNode 设置环境变量

```python
# 在 totalsegmentator_tasknode.py (lines 30-46)

SCRIPT_DIR = Path(__file__).parent.absolute()
# → E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator

LOCAL_MODELS = SCRIPT_DIR / "models"
# → E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator\models

os.environ['TOTALSEG_HOME_DIR'] = str(LOCAL_MODELS)
# 设置环境变量指向本地 models 文件夹
```

### 2. TotalSegmentator 查找权重

```python
# 在 TotalSegmentator-src/totalsegmentator/config.py

def get_totalseg_dir():
    if "TOTALSEG_HOME_DIR" in os.environ:
        return Path(os.environ["TOTALSEG_HOME_DIR"])
    else:
        return Path.home() / ".totalsegmentator"

# 因为我们设置了 TOTALSEG_HOME_DIR，所以返回：
# → E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator\models


def get_weights_dir():
    if "TOTALSEG_WEIGHTS_PATH" in os.environ:
        return Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
    else:
        totalseg_dir = get_totalseg_dir()
        return totalseg_dir / "nnunet/results"

# 我们没有设置 TOTALSEG_WEIGHTS_PATH，所以返回：
# → E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator\models\nnunet\results
```

### 3. nnUNet 环境变量

```python
# 在 setup_nnunet() 函数中

weights_dir = config_dir / "nnunet/results"

os.environ["nnUNet_raw"] = str(weights_dir)
os.environ["nnUNet_preprocessed"] = str(weights_dir)
os.environ["nnUNet_results"] = str(weights_dir)

# 所有 nnUNet 相关环境变量都指向同一个目录：
# → E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator\models\nnunet\results
```

---

## 🔍 路径验证结果

### 环境变量
- `TOTALSEG_HOME_DIR` = `E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator\models`
- `nnUNet_results` = `E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator\models\nnunet\results`

### 找到的模型权重
```
models/nnunet/results/
├── Dataset258_lung_vessels_248subj/            (236 MB)
├── Dataset291_TotalSegmentator_part1_organs_1559subj/  (241 MB)
├── Dataset292_TotalSegmentator_part2_vertebrae_1532subj/  (241 MB)
├── Dataset293_TotalSegmentator_part3_cardiac_1559subj/  (241 MB)
├── Dataset294_TotalSegmentator_part4_muscles_1559subj/  (241 MB)
├── Dataset295_TotalSegmentator_part5_ribs_1559subj/  (241 MB)
├── Dataset297_TotalSegmentator_total_3mm_1559subj/  (158 MB) ⭐ 主要模型
├── Dataset298_TotalSegmentator_total_6mm_1559subj/  (158 MB)
└── Dataset299_body_1559subj/                   (240 MB)
```

**总计：9 个模型，约 2 GB**

---

## 📋 路径流程图

```
TaskNode 启动
    ↓
设置 TOTALSEG_HOME_DIR = models/
    ↓
导入 totalsegmentator
    ↓
config.get_totalseg_dir()
    → 检查 TOTALSEG_HOME_DIR ✓
    → 返回: models/
    ↓
config.get_weights_dir()
    → 检查 TOTALSEG_WEIGHTS_PATH ✗ (未设置)
    → 返回: get_totalseg_dir() / "nnunet/results"
    → 返回: models/nnunet/results/
    ↓
config.setup_nnunet()
    → 设置 nnUNet_results = models/nnunet/results/
    ↓
TotalSegmentator 加载模型
    → 从 models/nnunet/results/Dataset297_*/
    → 加载 checkpoint_final.pth ✓
```

---

## ✅ 关键点总结

1. **`TOTALSEG_HOME_DIR`** 是关键环境变量
   - TaskNode 设置为：`<TotalSegmentator文件夹>/models/`
   
2. **权重查找路径** 固定为：
   - `$TOTALSEG_HOME_DIR/nnunet/results/`
   
3. **实际路径** 完整展开为：
   - `E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator\models\nnunet\results\`

4. **模型文件结构**：
   ```
   Dataset<ID>_<name>/
   └── nnUNetTrainer*__nnUNetPlans__3d_fullres/
       ├── dataset.json
       ├── plans.json
       └── fold_0/
           └── checkpoint_final.pth  ← 实际权重文件
   ```

5. **验证方法**：
   ```bash
   python verify_paths.py
   ```

---

## 🎯 实际使用

### TaskNode 运行时会自动找到权重

```bash
# 1. 进入 TotalSegmentator 目录
cd E:\Tissuelab-Model-Zoo\tissue_segmentation\TotalSegmentator

# 2. 激活环境
conda activate totalsegmentator_tissuelab

# 3. 启动 TaskNode（会自动使用本地权重）
python totalsegmentator_tasknode.py --port 8010 --name TotalSegmentator
```

**输出会显示：**
```
[TotalSegmentator] Using local source: ...TotalSegmentator-src
[TotalSegmentator] Using local model weights: ...models
```

这表示本地权重配置成功！

---

## 🔧 如果需要更改路径

### 方法 1: 修改 `TOTALSEG_HOME_DIR`（推荐）

```python
# 在 totalsegmentator_tasknode.py 修改 LOCAL_MODELS
LOCAL_MODELS = SCRIPT_DIR / "models"  # 当前
LOCAL_MODELS = Path("D:/MyModels")     # 改为其他位置

# 权重会在：D:/MyModels/nnunet/results/
```

### 方法 2: 使用 `TOTALSEG_WEIGHTS_PATH`

```python
# 直接指定权重目录（跳过 nnunet/results 子目录）
os.environ['TOTALSEG_WEIGHTS_PATH'] = "E:/CustomPath/weights"

# 权重直接在：E:/CustomPath/weights/Dataset297_*/
```

---

## 📦 打包时的注意事项

使用 PyInstaller 打包时，`models/` 文件夹会被包含：

```python
# 在 totalsegmentator_windows.spec

datas=[
    (str(TOTALSEG_SRC), 'TotalSegmentator-src'),
    (str(LOCAL_MODELS), 'models'),  # ← 包含所有权重
]
```

打包后的可执行文件会自动使用内嵌的权重，无需额外配置。

---

## ⚙️ 环境变量优先级

```
TOTALSEG_WEIGHTS_PATH（最高优先级）
    ↓ (如果未设置)
TOTALSEG_HOME_DIR / nnunet / results
    ↓ (如果未设置)
~/.totalsegmentator / nnunet / results（默认）
```

**我们的配置**：使用第二级（`TOTALSEG_HOME_DIR`），既保持灵活性，又符合 TotalSegmentator 的标准结构。

---

## 🧪 测试权重是否可用

```bash
# 快速测试
python -c "
from pathlib import Path
import os
os.environ['TOTALSEG_HOME_DIR'] = str(Path('models').absolute())
from totalsegmentator.config import get_weights_dir
print(f'Weights dir: {get_weights_dir()}')
print(f'Exists: {get_weights_dir().exists()}')
"
```

**预期输出：**
```
Weights dir: E:\...\TotalSegmentator\models\nnunet\results
Exists: True
```

---

## 📚 相关文件

- **路径设置**：`totalsegmentator_tasknode.py` (lines 30-46)
- **路径查找**：`TotalSegmentator-src/totalsegmentator/config.py` (lines 16-51)
- **权重下载**：`download_to_results.py`
- **路径验证**：`verify_paths.py`

---

**创建时间：** 2025-10-07  
**验证状态：** ✅ 通过  
**模型数量：** 9 个  
**总大小：** ~2 GB
