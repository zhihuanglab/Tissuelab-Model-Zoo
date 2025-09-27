---
license: cc-by-nc-nd-4.0
language:
- en
tags:
- pathology
- vision
- vision language
- precision oncology
- pytorch
extra_gated_prompt: >-
  This model and associated code are released under the CC-BY-NC-ND 4.0 license
  and may only be used for non-commercial, academic research purposes with
  proper attribution. Any commercial use, sale, or other monetization of the
  MUSK model and its derivatives, which include models trained on outputs from
  the MUSK model or datasets created from the MUSK model, is prohibited and
  requires prior approval. By downloading this model, you
  agree not to distribute, publish or reproduce a copy of the model.  If you are a commercial entity, please contact the
  corresponding author. 
  
extra_gated_fields:
  Full name: text
  Affiliation: text
  Type of affiliation:
    type: select
    options:
    - Academia
    - Industry
    - label: Other
      value: other
  Official email (must match primary email in your Hugging Face account; work email instead of personal): text
  Please explain your intended research use: text
  I agree to all terms outlined above: checkbox
  I agree to use this model for non-commercial, academic purposes only: checkbox
  I agree not to distribute the model, if another user within your organization wishes to use the MUSK model, they must register as an individual user: checkbox
pipeline_tag: image-to-text
---


## MUSK: A Vision-Language Foundation Model for Precision Oncology
(Nature 2025)

Jinxi Xiang‡, Xiyue Wang‡, Xiaoming Zhang, Yinghua Xi, Feyisope Eweje, Yijiang Chen, Yuchen
Li, Colin Bergstrom, Matthew Gopaulchan, Ted Kim, Kun-Hsing Yu, Sierra Willens, Francesca Maria
Olguin, Jeffrey J. Nirschl, Joel Neal, Maximilian Diehn, Sen Yang<sup>+</sup>, Ruijiang Li<sup>+</sup> (‡Equal Contribution)

_Lead Contact_: [Ruijiang Li](https://med.stanford.edu/lilab.html), Ph.D.

Stanford University, Harvard University


## Application

Please ensure that the official email is used as the primary email for Hugging Face accounts, rather than personal domains  (gmail/qq/outlook.com).  
Thank you for your understanding!  

## Installation

First clone the repo and cd into the directory:
```shell
git clone https://github.com/lilab-stanford/MUSK
cd MUSK
```

Create a new enviroment with anaconda.
```shell
conda create -n musk python=3.10 -y --no-default-packages
conda activate musk
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Model Code

The MUSK code can be accessed from [GitHub](https://github.com/lilab-stanford/MUSK).

You need to agree to the terms to access the models and login with your HuggingFace write token:
```python
from huggingface_hub import login
login(<huggingface write token>)
```


## Basic Usage: MUSK as a Vision-Language Encoder

Please refer to `demo.ipynb` for a demonstration. 


1. Load the MUSK model  

```python
from musk import utils, modeling
from timm.models import create_model
model = create_model("musk_large_patch16_384")
utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
model.to(device="cuda", dtype=torch.float16)
model.eval()
```

2. Encode images with MUSK  (refer to `demo.ipynb` for complete implementation)
```python
import torchvision
from PIL import Image
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(384, interpolation=3, antialias=True),
    torchvision.transforms.CenterCrop((384, 384)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
])

img = Image.open('assets/lungaca1014.jpeg').convert("RGB")  # input image
img_tensor = transform(img).unsqueeze(0)
with torch.inference_mode():
    image_embeddings = model(
        image=img_tensor.to("cuda", dtype=torch.float16),
        with_head=False,
        out_norm=False,
        ms_aug=True,
        return_global=True  
        )[0]  # return (vision_cls, text_cls)

```
- `with_head=True`: Enable head for image-text retrieval.  
- `out_norm=True`: Apply normalization.  
- `ms_aug=True`: Use multiscale augmentation (for tasks, e.g., linear probe classification, MIL).  
- `return_global=True`: Return only [CLS] token, exclude patch tokens.  


3. Encode texts with MUSK (refer to `demo.ipynb` for complete implementation)
```python
tokenizer = XLMRobertaTokenizer("./musk/models/tokenizer.spm")
text = ['histopathology image of lung adenocarcinoma']
txt_ids, pad = utils.xlm_tokenizer(txt, tokenizer, max_len=100)

with torch.inference_mode():
   text_embeddings = model(
      text_description=txt_ids,
      padding_mask=pad,
      with_head=False, 
      out_norm=True,
      ms_aug=False,
      return_global=True 
   )[1]  # return (vision_cls, text_cls)
```


## Evaluation on Patch-level Benchmarks

Please refer to `./benchmarks/demo.ipynb` for a demonstration. 

Patch-level benchmarks include image-text retrieval, zero-shot/few-shot/linear probe image classification, image-image retrieval, and more. The evaluation code is all-in-one which adapted from the [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark).

First, download the necessary datasets. For demonstrations, we provide example datasets [here](https://drive.google.com/file/d/1FCGRn6mtdrw8l3WAR_U76V0eRBnsQxD1/view?usp=sharing). Download and unzip it to a local path, for example `/root/to/downstreams_demo`, then, change the directory path `dataset_root=/root/to/downstreams_demo`. The code will automatically extract features and perform evaluations.


The main file is `clip_benchmark.cli` and includes the following options:
- `--pretrained_model`: Specifies the model name and the path to its weights.
- `--dataset`: Indicates the evaluation dataset(s); multiple datasets can be specified.
- `--dataset_root`: The root of datasets.
- `--task`: Defines the evaluation task.
- `--batch_size`: Sets the batch size for feature extraction.
- `--output`: Specifies where to save the output results.

Set the `models.txt` file with entries in the format: `(model_name, model_path)`. If you want to run both MUSK and [CONCH](https://github.com/mahmoodlab/CONCH) for comparison, your `models.txt` might look like this:
```shell
musk_large_patch16_384,hf_hub:xiangjx/musk
conch,/path/to/conch.pt
```
Alternatively, you can remove the CONCH entry and run MUSK alone.

Some example commands:

```shell
# >>>>>>>>>>> zero-shot image-text retrieval >>>>>>>>>>> #
 python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
        --dataset   "pathmmu_retrieval"  \
        --task "zeroshot_retrieval" \
        --batch_size 256 \
        --num_workers 8 \
        --seed 42 \
        --recall_k 1 10 50 \
        --dataset_root "/root/to/downstreams_demo" \
        --output "./results/benchmark_mm_retrieval.json"
```


```shell
# >>>>>>>>>>> few-shot image classification >>>>>>>>>>> #
for k_shot in "${shot_list[@]}"
do
  for seed in "${seed_list[@]}"
  do
      python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
          --dataset  "skin" "pannuke" "unitopatho" \
          --task "linear_probe" \
          --batch_size 256 \
          --num_workers 8 \
          --fewshot_k $k_shot \
          --seed $seed \
          --dataset_root "/root/to/downstreams_demo" \
          --output "./results/benchmark_fs_${k_shot}shot_seed${seed}.json"
  done
done
```

```shell
# >>>>>>>>>>> zero-shot image2image retrieval >>>>>>>>>>> #
python3 -m clip_benchmark.cli eval --pretrained_model models.txt \
        --dataset   "unitopatho_retrieval" \
        --task "image_retrieval" \
        --batch_size 256 \
        --num_workers 8 \
        --seed 41 \
        --dataset_root "/root/to/downstreams_demo" \
        --output "./results/benchmark_image_retrieval.json"
```

and more tasks in `./benchmarks/demo.ipynb`.


## Acknowledgements

The project was built on many amazing open-source repositories: [Quilt1M](https://github.com/wisdomikezogwo/quilt1m), [PathAsst](https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology), [torchscale](https://github.com/microsoft/torchscale), [accelerate](https://github.com/huggingface/accelerate) (model pretraining), [deepspeed](https://github.com/microsoft/DeepSpeed) (model pretraining), [pytorch-lightning](https://github.com/Lightning-AI/pytorch-lightning) (downstream finetuning), and [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) (model evaluation). We thank the authors and developers for their contributions.

## Issues
- Please open new threads or address questions to xiangjx@stanford.edu or xiyue.wang.scu@gmail.com

## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the MUSK model and its derivatives, which include models trained on outputs from the MUSK model or datasets created from the MUSK model, is prohibited and requires prior approval.
