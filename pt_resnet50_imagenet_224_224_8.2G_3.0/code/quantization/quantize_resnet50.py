import argparse
import glob
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from resnet50 import resnet50

# python code/quantization/quantize_resnet50.py \
#   --model_path float/resnet50_pretrained.pth \
#   --data_dir data/imagenet-val \
#   --split val \
#   --quant_mode calib \
#   --target DPUCVDX8G_ISA3_C32B6 \
#   --quant_dir ./code/quantization_output \
#   --batch_size 1 \
#   --max_calib_samples 50 \
#   --device gpu

# # 2) Test + deploy + optional single-image inference
# python code/quantization/quantize_resnet50.py \
#   --model_path float/resnet50_pretrained.pth \
#   --data_dir data/imagenet-val \
#   --split val \
#   --quant_mode test \
#   --target DPUCVDX8G_ISA3_C32B6 \
#   --quant_dir ./code/quantization_output \
#   --export_dir ./code/quantization_output \
#   --max_test_samples 50 \
#   --image_path data/imagenet-val/n01440764/ILSVRC2012_val_00002138.JPEG \
#   --deploy \
#   --arch_json ./code/quantization/arch.json \
#   --compile_dir ./code/quantization_output/compiled \
#   --net_name resnet50_int \
#   --device gpu

try:
    from pytorch_nndct.apis import torch_quantizer  # type: ignore

    _HAS_QUANT = True
except Exception:
    _HAS_QUANT = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="PTQ flow for ResNet50: calibration, test, export xmodel, optional compile."
    )
    parser.add_argument("--model_path", required=True, help="Path to float .pth checkpoint.")
    parser.add_argument("--data_dir", default="data/imagenet-val", help="ImageNet-style dataset directory.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split to use.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Run device.")
    parser.add_argument("--model_name", default="resnet50", choices=["resnet50"])
    parser.add_argument("--num_classes", type=int, default=None, help="Optional classifier output classes.")
    parser.add_argument("--quant_mode", required=True, choices=["calib", "test"])
    parser.add_argument("--target", default=None, help="Vitis target, e.g. DPUCVDX8G_ISA3_C32B6.")
    parser.add_argument("--quant_dir", default="quantized", help="Quantizer output dir (quant_info.json, etc).")
    parser.add_argument("--export_dir", default="quantized", help="Where export_xmodel writes .xmodel.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max_calib_samples", type=int, default=100)
    parser.add_argument("--max_test_samples", type=int, default=0, help="0 means full split.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--deploy", action="store_true", help="In test mode: export .xmodel.")
    parser.add_argument("--arch_json", default=None, help="Optional arch.json path for vai_c_xir compile.")
    parser.add_argument("--compile_dir", default=None, help="Output directory for compiled artifacts.")
    parser.add_argument("--net_name", default="resnet50_int", help="Network name for vai_c_xir.")
    parser.add_argument("--image_path", default=None, help="Optional single image inference in test mode.")
    return parser.parse_args()


def pick_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(model_name):
    size = 224
    resize = 256
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_imagefolder_dataset(data_dir, split, transform):
    split_dir = os.path.join(data_dir, split)
    if os.path.isdir(split_dir):
        root = split_dir
    elif os.path.isdir(data_dir):
        root = data_dir
    else:
        raise FileNotFoundError(
            f"Could not find dataset directory. Checked: '{split_dir}' and '{data_dir}'"
        )
    return torchvision.datasets.ImageFolder(root=root, transform=transform)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def infer_num_classes(state_dict, default=1000):
    fc_weight = state_dict.get("fc.weight")
    if fc_weight is None:
        return default
    return int(fc_weight.shape[0])


def load_model(model_path, device, num_classes=None):
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    if num_classes is None:
        num_classes = infer_num_classes(state_dict)
    model = resnet50(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def run_calibration(quant_model, loader, device):
    quant_model.eval()
    seen = 0
    with torch.no_grad():
        for images, _labels in tqdm(loader, desc="Calibrating"):
            images = images.to(device)
            _ = quant_model(images)
            seen += images.size(0)
    print(f"[QANT] Calibration complete. Samples seen: {seen}")


def evaluate_classifier(model, loader, device):
    top1_sum = 0.0
    top5_sum = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
            batch = images.size(0)
            top1_sum += float(acc1[0].item()) * batch
            top5_sum += float(acc5[0].item()) * batch
            total += batch
    if total == 0:
        return 0.0, 0.0
    return top1_sum / total, top5_sum / total


def run_single_image_inference(model, image_path, transform, dataset, device):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    expected_class = os.path.basename(os.path.dirname(image_path))
    expected_idx = dataset.class_to_idx.get(expected_class, -1)
    with torch.no_grad():
        logits = model(image_tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())
    predicted_class = dataset.classes[pred_idx]
    print(f"[TEST] Image path: {image_path}")
    if expected_idx >= 0:
        print(f"[TEST] Expected class: {expected_class} (idx={expected_idx})")
    else:
        print(f"[TEST] Expected class: {expected_class} (idx=unknown)")
    print(f"[TEST] Predicted class: {predicted_class} (idx={pred_idx})")


def find_exported_xmodel(export_dir, prefer_pattern="*_int.xmodel"):
    candidates = sorted(glob.glob(os.path.join(export_dir, prefer_pattern)))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(export_dir, "*.xmodel")))
    if not candidates:
        raise FileNotFoundError(f"No .xmodel found in export_dir={export_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def compile_with_vai_c_xir(xmodel_path, arch_json, out_dir, net_name):
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["vai_c_xir", "-x", xmodel_path, "-a", arch_json, "-o", out_dir, "-n", net_name]
    print("[COMPILER] Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[COMPILER] Compiled artifacts written to: {out_dir}")


def make_subset(dataset, max_samples, seed):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_samples])


def main():
    args = parse_args()
    if not _HAS_QUANT:
        raise RuntimeError("pytorch_nndct is not available. Run in a Vitis AI environment.")

    os.makedirs(args.quant_dir, exist_ok=True)
    os.makedirs(args.export_dir, exist_ok=True)

    device = pick_device(args.device)
    print(f"Using device: {device}")
    transform = build_transform(args.model_name)
    dataset = load_imagefolder_dataset(args.data_dir, args.split, transform)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    model = load_model(args.model_path, device=device, num_classes=args.num_classes)

    dummy_input = torch.zeros(1, 3, 224, 224, dtype=torch.float32).to(device)
    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,
        module=model,
        input_args=(dummy_input,),
        device=device,
        output_dir=args.quant_dir,
        bitwidth=8,
        target=args.target,
    )
    quant_model = quantizer.quant_model.to(device).eval()

    if args.quant_mode == "calib":
        calib_dataset = make_subset(dataset, args.max_calib_samples, args.seed)
        calib_loader = DataLoader(
            calib_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        print(f"[QANT] Calibration samples: {len(calib_dataset)}")
        run_calibration(quant_model, calib_loader, device)
        quantizer.export_quant_config()
        print(f"[QANT] Quant config exported to: {args.quant_dir}")
        return

    test_dataset = make_subset(dataset, args.max_test_samples, args.seed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    print(f"[QANT] Test samples: {len(test_dataset)}")
    top1, top5 = evaluate_classifier(quant_model, test_loader, device)
    print(f"[QANT] Quantized top-1 / top-5: {top1:.1f} / {top5:.1f}")

    if args.image_path:
        run_single_image_inference(quant_model, args.image_path, transform, dataset, device)

    if args.deploy:
        print("[QANT] Exporting .xmodel ...")
        quantizer.export_xmodel(output_dir=args.export_dir, deploy_check=True)
        xmodel_path = find_exported_xmodel(args.export_dir)
        print(f"[QANT] Exported xmodel: {xmodel_path}")
        if args.arch_json:
            compile_out = args.compile_dir or os.path.join(args.export_dir, "compiled")
            compile_with_vai_c_xir(
                xmodel_path=xmodel_path,
                arch_json=args.arch_json,
                out_dir=compile_out,
                net_name=args.net_name,
            )
        else:
            print("[QANT] --arch_json not set, skipping vai_c_xir compilation.")


if __name__ == "__main__":
    main()
