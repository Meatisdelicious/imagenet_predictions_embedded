import argparse
import os
import random
import sys
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from resnet50 import resnet50

# Simple inference testing without vitis AI, on the original model
# Test inference on a class fish
# python code/test/inference.py \
#   --model_path float/resnet50_pretrained.pth \
#   --data_dir data/imagenet-val \
#   --split val \
#   --image_path data/imagenet-val/n01440764/ILSVRC2012_val_00002138.JPEG \
#   --device gpu

# Test inference on a class fish with another fish picture
# python code/test/inference.py \
#   --model_path float/resnet50_pretrained.pth \
#   --data_dir data/imagenet-val \
#   --split val \
#   --image_path data/imagenet-val/n01440764/ILSVRC2012_val_00007197.JPEG \
#   --device gpu




def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image inference on ImageNet-style dataset.")
    parser.add_argument("--data_dir", default="data/imagenet-val", help="Dataset directory.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split.")
    parser.add_argument("--model_path", required=True, help="Path to .pth model file.")
    parser.add_argument("--model_name", default="resnet50", choices=["resnet50"])
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument(
        "--image_path",
        default=None,
        help="Path to a specific image to run inference on.",
    )
    parser.add_argument(
        "--sample_index",
        default=0,
        type=int,
        help="Index in dataset to run inference on (ignored if --random_sample is set).",
    )
    parser.add_argument(
        "--random_sample",
        action="store_true",
        help="Pick a random image from the dataset.",
    )
    parser.add_argument(
        "--num_classes",
        default=None,
        type=int,
        help="Override classifier output classes. If omitted, inferred from checkpoint.",
    )
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
    if model_name == "inceptionv3":
        size = 299
        resize = 299
    else:
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


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    transform = build_transform(args.model_name)
    dataset = load_imagefolder_dataset(args.data_dir, args.split, transform)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    if args.image_path is not None:
        image_path = args.image_path
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_tensor = transform(Image.open(image_path).convert("RGB"))
        expected_class = os.path.basename(os.path.dirname(image_path))
        target_idx = dataset.class_to_idx.get(expected_class, -1)
    else:
        if args.random_sample:
            sample_index = random.randint(0, len(dataset) - 1)
        else:
            if args.sample_index < 0 or args.sample_index >= len(dataset):
                raise IndexError(
                    f"sample_index {args.sample_index} is out of bounds for dataset of size {len(dataset)}"
                )
            sample_index = args.sample_index

        image_tensor, target_idx = dataset[sample_index]
        image_path, _ = dataset.samples[sample_index]
        expected_class = dataset.classes[target_idx]

    model = load_model(args.model_path, device, args.num_classes)
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        pred_idx = int(torch.argmax(logits, dim=1).item())
    predicted_class = dataset.classes[pred_idx]

    print(f"Image path: {image_path}")
    if target_idx >= 0:
        print(f"Expected class: {expected_class} (idx={target_idx})")
    else:
        print(f"Expected class: {expected_class} (idx=unknown, not found in dataset classes)")
    print(f"Predicted class: {predicted_class} (idx={pred_idx})")


if __name__ == "__main__":
    main()
