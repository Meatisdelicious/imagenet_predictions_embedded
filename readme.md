# ImageNet Predictions Embedded

This repository contains a ResNet50 ImageNet project under:

`pt_resnet50_imagenet_224_224_8.2G_3.0/`

## Python Version

Use **Python 3.12** with `uv` for local development.

Note: some Vitis AI Docker images may pin older Python/tooling. In that case, follow the container's version constraints.

## Setup with `uv` (recommended)

From the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv sync
```

This uses:

- `pyproject.toml` for dependency declarations
- `uv.lock` for fully pinned, reproducible installs

### Update dependencies

```bash
# add or update a runtime dependency
uv add <package>

# add a dev dependency
uv add --dev <package>

# refresh lockfile after dependency changes
uv lock

# sync .venv to the lockfile
uv sync
```

## Legacy requirements-based setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Dataset Layout

`inference_vitis.py` expects an ImageNet-style folder structure:

```text
data/imagenet-val/
  val/
    n01440764/
      ILSVRC2012_val_00002138.JPEG
    n01443537/
    ...
```

## Download Dataset

ImageNet-1K validation set (Kaggle):

`https://www.kaggle.com/datasets/titericz/imagenet1k-val`

After download/extract, place it so `--data_dir data/imagenet-val` works.

## Run `inference_vitis.py`

From `pt_resnet50_imagenet_224_224_8.2G_3.0/`:

```bash
python code/test/inference_vitis.py \
  --model_path float/resnet50_pretrained.pth \
  --data_dir data/imagenet-val \
  --split val \
  --image_path data/imagenet-val/n01440764/ILSVRC2012_val_00002138.JPEG \
  --device gpu
```

### Run with Vitis AI Inspector (CPU mode)

```bash
python code/test/inference_vitis.py \
  --model_path float/resnet50_pretrained.pth \
  --data_dir data/imagenet-val \
  --split val \
  --image_path data/imagenet-val/n01440764/ILSVRC2012_val_00002138.JPEG \
  --device cpu \
  --inspect \
  --dpu_arch DPUCVDX8G_ISA3_C32B6 \
  --inspect_out vitis_ai_inspector_results
```

Note: `--inspect` requires `pytorch_nndct` from a Vitis AI environment.

## Evaluate with `eval_imagenet.py`

From `pt_resnet50_imagenet_224_224_8.2G_3.0/`:

```bash
python code/test/eval_imagenet.py \
  --model_name resnet50 \
  --model_path float/resnet50_pretrained.pth \
  --data_dir data/imagenet-val \
  --split val \
  --batch_size 16 \
  --device gpu
```

## Expected Baseline Metrics

|Model|input_size|FLOPs|Params|Float top-1/top-5 acc(%)|Quantized top-1/top-5 acc(%)|
|---|---|---|---|---|---|
|Baseline|224x224|8.2G|25.56M|76.1/92.9|76.1/92.9|

## Quantized Model Path (`.xmodel`)

If you are looking for `resnet_int.xmodel`, the quantized xmodel in this repository is:

`pt_resnet50_imagenet_224_224_8.2G_3.0/code/quantization_output/compiled/ResNet_int.xmodel`

## Model Specs

- Model: `ResNet-50` (ImageNet, 1000 classes)
- Input size: `224 x 224`
- Input tensor shape: `[N, 3, 224, 224]` (NCHW)
- Output tensor shape: `[N, 1000]` (class logits)
- FLOPs: `8.2G`
- Params: `25.56M`

### Preprocessing (used by `inference.py`, `inference_vitis.py`, `eval_imagenet.py`)

1. Load image in `RGB`
2. Resize shorter side to `256`
3. Center-crop to `224 x 224`
4. Convert to tensor in `[0, 1]`
5. Normalize with ImageNet statistics:
   - mean = `[0.485, 0.456, 0.406]`
   - std = `[0.229, 0.224, 0.225]`

### Post-processing

1. Run forward pass to get logits `[N, 1000]`
2. For single-image inference, predicted class = `argmax(logits)`
3. For evaluation, report `top-1` and `top-5` accuracy over the validation set
