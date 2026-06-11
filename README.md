# CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities

**[Paper (arXiv)](https://arxiv.org/abs/2603.26425)** | **CVPR Findings 2026**

Official repository for CPUBone, a family of vision backbones optimized for CPU-based inference.
Authors: Moritz Nottebaum, Matteo Dunnhofer, Christian Micheloni.

(model checkpoints will follow in the next 1-2 weeks.)

---

## Overview

CPUs, unlike GPUs and other high-parallelization platforms, require models that balance the number of operations (MACs) and hardware-efficient execution measured by MACs per second (MACpS). CPUBone investigates two modifications to standard convolutions — **grouped convolutions** (groups=2) and **reduced kernel sizes** (2×2 instead of 3×3) — that reduce MACs while preserving MACpS on CPUs, achieving state-of-the-art speed–accuracy trade-offs across a wide range of devices.

CPUBone also transfers its efficiency to downstream tasks (object detection, semantic segmentation).

Latency benchmarks are run on exported models: desktop and embedded CPU numbers (RPi 5, Xeon) use ONNX Runtime, while mobile numbers (Pixel 7 Pro) use TFLite.

The full model definition is self-contained in [`cpubone_model.py`](cpubone_model.py) and requires only PyTorch.

---

## Model Zoo

### ImageNet-1K Classification

| Model | Params (M) | MACs (M) | Top-1 (%) | RPi 5 (ms) | Pixel 7 Pro (ms) | Xeon W-2125 (ms) |
|-------|-----------|----------|-----------|-----------|-----------------|-----------------|
| CPUBone-B0 | 10.4 | 519 | 77.6 | 24.2 | 13.8 | 6.9 |
| CPUBone-B1 | 12.4 | 746 | 78.7 | 33.5 | 18.6 | 9.4 |
| CPUBone-B2 | 23.9 | 1354 | 80.3 | 60.3 | 32.4 | 16.1 |
| CPUBone-B3 | 40.7 | 4054 | 83.1 | 199.8 | 83.1 | 34.1 |

Models and ablations can be found [here](https://www.dropbox.com/scl/fo/ugvs2mpeexp99o7vhhb7e/AHo9-C9GjeCFwD_KDZPfRi4?rlkey=44d7f611sc8iyuhp5trjm0ng0&st=wo6doxpl&dl=0) (just put the folder in the root directory). 
Latency measured at batch size 1. RPi 5 and Xeon numbers use ONNX Runtime; Pixel 7 Pro numbers use TFLite.

### Tiny Backbones

| Model | Params (M) | MACs (M) | Top-1 (%) | RPi 5 (ms) |
|-------|-----------|----------|-----------|-----------|
| CPUBone-Nano | 6.52 | 190 | 72.67 | — |
| CPUBone-T0 | 7.54 | 269 | 74.85 | 12.47 |
| CPUBone-S0 | 8.73 | 359 | 75.89 | 15.53 |


### Downstream Tasks

**Object Detection on COCO 2017** (RetinaNet framework):

| Model | AP (%) |
|-------|--------|
| CPUBone-B0 | 37.5 |
| CPUBone-B1 | 39.0 |
| CPUBone-B2 | 40.4 |
| CPUBone-B3 | 42.9 |

**Semantic Segmentation on ADE20K** (Semantic FPN framework):

| Model | mIoU (%) |
|-------|----------|
| CPUBone-B0 | 37.9 |
| CPUBone-B1 | 39.2 |
| CPUBone-B2 | 42.1 |
| CPUBone-B3 | 44.1 |

---

## Quick Start: Using a Pretrained Model

`cpubone_model.py` is a standalone file — the only dependency is PyTorch.

Model loading requires the YAML config files from the `configs/` directory of this repository. Each config defines the architecture hyperparameters (width, depth, etc.) for a given model variant.

```python
from cpubone_model import get_cpubone_b1
import torch

# Requires configs/cls/imagenet/cpubone_b1.yaml to be present
model = get_cpubone_b1(pretrained=True)
model.eval()

x = torch.randn(1, 3, 224, 224)
out = model(x)  # (1, 1000)
print(out.shape)
```

Available convenience functions: `get_cpubone_b0`, `get_cpubone_b1`, `get_cpubone_b15`, `get_cpubone_b2`, `get_cpubone_b3`.

Pretrained weights are expected at `.exp/cls/imagenet/cpubone_<name>/checkpoint/evalmodel.pt`. You can simply download  [checkpoint folder](https://www.dropbox.com/scl/fo/ugvs2mpeexp99o7vhhb7e/AHo9-C9GjeCFwD_KDZPfRi4?rlkey=44d7f611sc8iyuhp5trjm0ng0&st=wo6doxpl&dl=0) and put it into this project directory. It contains all the models and ablations.
Both the config path and checkpoint path can be passed explicitly:

```python
from cpubone_model import get_cpubone

model = get_cpubone(
    config_path="configs/cls/imagenet/cpubone_b1.yaml",
    checkpoint_path="path/to/evalmodel.pt",
    pretrained=True,
)
```

---

## Installation

```bash
conda create -n cpubone python=3.11
conda activate cpubone
pip install -r requirements.txt
```

For ImageNet-1K training and evaluation, set the dataset path in the config file (`configs\cls\imagenet\default.yml`).
Checkpoints can be downloaded from [here](https://www.dropbox.com/scl/fo/ugvs2mpeexp99o7vhhb7e/AHo9-C9GjeCFwD_KDZPfRi4?rlkey=44d7f611sc8iyuhp5trjm0ng0&st=wo6doxpl&dl=0).

---

## Evaluation

```bash
python eval_cls_model.py --model cpubone_b1
```

Additional flags:
- `--testrun` — skip weight loading, useful for architecture checks
- `--latency` — measure CPU/GPU latency with TorchScript
- `--bench` — full throughput benchmark

---

## Training

**Single GPU:**
```bash
torchrun --nproc_per_node=1 train_cls_model.py \
    configs/cls/imagenet/cpubone_b1.yaml \
    --path .exp/cls/imagenet/cpubone_b1
```

**Multi-GPU (8 GPUs):**
```bash
torchrun --nproc_per_node=8 train_cls_model.py \
    configs/cls/imagenet/cpubone_b1.yaml \
    --path .exp/cls/imagenet/cpubone_b1
```

To resume training, add `--resume`.

Gradient accumulation is controlled via the `bsizemult` parameter in the run config. Learning rate is scaled automatically with world size when `total_lr` and `total_batch_size` are set in the config.

---

## Custom Models

The easiest way to define a custom CPUBone variant is via the `custom` model name in a config file:

```yaml
net_config:
  name: custom
  width_list: [16, 32, 64, 128, 256]
  depth_list: [0, 1, 1, 4, 4]
  fastit: true
```

Or directly in Python by passing kwargs to `cpubone_backbone_b1`:

```python
from cpubone_model import cpubone_backbone_b1

backbone, width_list = cpubone_backbone_b1(
    name="custom",
    width_list=[16, 32, 64, 128, 256],
    depth_list=[0, 1, 1, 4, 4],
    fastit=True,
)
```

---

## Acknowledgements

The training infrastructure in the `cpubone/` package is built on top of [EfficientViT](https://github.com/mit-han-lab/efficientvit). The code in this repository also builds on our prior work [LowFormer](https://github.com/altair199797/LowFormer).

---

## Citation

```bibtex
@inproceedings{nottebaum2026cpubone,
  title     = {CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities},
  author    = {Nottebaum, Moritz and Dunnhofer, Matteo and Micheloni, Christian},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```
