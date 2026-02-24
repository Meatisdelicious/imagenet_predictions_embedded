# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Command to test the accuracy of the model, on the dataset imgnet of 1000 classes. 
# it's on the validation dataset
# run in the pt_resnet50... folder
# python code/test/eval_imagenet.py \
#   --model_name resnet50 \
#   --data_dir data/imagenet-val \
#   --split val \
#   --model_path float/resnet50_pretrained.pth \
#   --batch_size 32 \
#   --device cpu \
#   --workers 1


import os
import sys
import argparse
import random
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / 'models'))
from resnet50 import resnet50


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default='data/tiny-imagenet-200',
    help='Dataset directory containing train/val/test folders')
parser.add_argument(
    '--model_name',
    default='resnet50',
    choices=['inceptionv3', 'resnet50'])
parser.add_argument(
    '--model_path',
    required=True)
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole split if it is not set')
parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument(
    '--split',
    default='val',
    choices=['train', 'val'],
    help='Dataset split to evaluate')
parser.add_argument(
    '--data_type',
    default='float32',
    choices=['float32', 'float16'])
parser.add_argument(
    '--workers',
    default=2,
    type=int,
    help='Number of DataLoader workers')
parser.add_argument(
    '--device',
    default='gpu',
    choices=['gpu', 'cpu'])
parser.add_argument(
    '--num_classes',
    default=None,
    type=int,
    help='Override classifier output classes. If omitted, inferred from checkpoint for .pth models.')
args, _ = parser.parse_known_args()

if args.device == 'cpu':
    device = torch.device('cpu')
    print(f'Using device: {device}')
else:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Using device: {device}')
    else:
        device = torch.device('cpu')



class TinyImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform
        val_images_dir = os.path.join(data_dir, 'val', 'images')
        val_annotations = os.path.join(data_dir, 'val', 'val_annotations.txt')
        wnids_path = os.path.join(data_dir, 'wnids.txt')

        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f if line.strip()]
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}

        self.samples = []
        with open(val_annotations, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                if wnid not in self.class_to_idx:
                    continue
                img_path = os.path.join(val_images_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def load_data(data_dir='dataset/imagenet',
              batch_size=128,
              subset_len=None,
              sample_method='random',
              split='val',
              workers=4,
              model_name='resnet18',
              **kwargs):
    split_dir = os.path.join(data_dir, split)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if model_name == 'inceptionv3':
        size = 299
        resize = 299
    else:
        size = 224
        resize = 256
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])

    if split == 'val' and os.path.isfile(os.path.join(data_dir, 'val', 'val_annotations.txt')):
        dataset = TinyImageNetValDataset(data_dir, transform)
    else:
        # Support both:
        # 1) <data_dir>/<split>/<class>/*.JPEG
        # 2) <data_dir>/<class>/*.JPEG  (e.g. data/imagenet-val)
        if os.path.isdir(split_dir):
            imagefolder_root = split_dir
        elif os.path.isdir(data_dir):
            imagefolder_root = data_dir
        else:
            raise FileNotFoundError(
                f"Could not find dataset directory. Checked: '{split_dir}' and '{data_dir}'")
        dataset = torchvision.datasets.ImageFolder(imagefolder_root, transform)

    if subset_len:
        assert subset_len <= len(dataset)
        if sample_method == 'random':
            dataset = torch.utils.data.Subset(
                dataset, random.sample(range(0, len(dataset)), subset_len))
        else:
            dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, **kwargs)
    return data_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


def evaluate(model, val_loader, loss_fn, model_type, data_type='float32'):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for iteration, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if model_type == 'pth':
                images = images.to(device)
                outputs = model(images)
            else:
                if data_type == 'float16':
                    images = images.half()
                outputs = model.run({'input': images.cpu().numpy()})[0]
                outputs = torch.tensor(np.array(outputs))

            outputs = outputs.to(device)
            labels = labels.to(device)
            total += images.size(0)

            batch_loss = loss_fn(outputs.to(torch.float32), labels)
            total_loss += batch_loss.item() * images.size(0)

            acc1, acc5 = accuracy(outputs.to(torch.float32), labels, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if iteration != 0 and iteration % 1000 == 0:
                print('image_size=%d,\t top1=%.1f,\t top5=%.1f' % (images.size(2), top1.avg, top5.avg))

    return top1.avg, top5.avg, total_loss / total


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']

    cleaned = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        cleaned[k] = v
    return cleaned


def _infer_num_classes(state_dict, default=1000):
    fc_weight = state_dict.get('fc.weight')
    if fc_weight is None:
        return default
    return int(fc_weight.shape[0])


def _load_resnet50_checkpoint(model_path, num_classes):
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = _extract_state_dict(checkpoint)

    if num_classes is None:
        num_classes = _infer_num_classes(state_dict)

    model = resnet50(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    return model


def main(model_name=''):
    data_dir = args.data_dir
    batch_size = args.batch_size
    subset_len = args.subset_len
    split = args.split

    print('=== Load pretrained model ===')

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    val_loader = load_data(
        subset_len=subset_len,
        batch_size=batch_size,
        sample_method='',
        data_dir=data_dir,
        split=split,
        workers=args.workers,
        model_name=model_name)

    print('Model name:', model_name)
    model_type = args.model_path.split('.')[-1]
    print('Model type:', model_type)
    if model_type not in ['onnx', 'pth']:
        print(f'[Error] Evaluating model type "{model_type}" is not supported yet')
        return

    print('Loading model:', args.model_path)
    if model_type == 'pth':
        model = _load_resnet50_checkpoint(args.model_path, args.num_classes)
        model = model.to(device)
        model.eval()
    else:
        import migraphx
        print('Evaluating onnx model using AMD MIGRAPHX')
        model = migraphx.parse_onnx(args.model_path)
        model.compile(migraphx.get_target('gpu'))

    acc1_gen, acc5_gen, loss_gen = evaluate(model, val_loader, loss_fn, model_type, args.data_type)

    print('top-1 / top-5 accuracy: %.1f / %.1f' % (acc1_gen, acc5_gen))
    print('loss: %.4f' % loss_gen)


if __name__ == '__main__':
    print('-------- Start {} test '.format(args.model_name))
    main(model_name=args.model_name)
    print('-------- End of {} test '.format(args.model_name))
