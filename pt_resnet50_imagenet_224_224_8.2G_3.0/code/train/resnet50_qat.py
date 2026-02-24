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

'''
This file support quantization-aware-traing and export quantized model
mode:
    train -> qat
    test -> test float qat model
    deploy -> get quantiezd qat model
    qat_test -> test quantized qat model
'''
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import re
import sys
import pdb
import random
import torchvision
from ipdb import set_trace

if os.environ["W_QUANT"]=='1':
    from pytorch_nndct import nn as nndct_nn
    from pytorch_nndct.nn.modules import functional
    from pytorch_nndct import QatProcessor
    
sys.path.append('./code/')
#sys.path.append('./code/models/')
from models.resnet50_qat_model import resnet50

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    default='/scratch/workspace/dataset/imagenet/pytorch',
    help='Data set directory.')
parser.add_argument(
    '--pretrained',
    default=None,
    help='Pre-trained model file path.')
parser.add_argument(
    '--workers',
    default=4,
    type=int,
    help='Number of data loading workers to be used.')
parser.add_argument('--epochs', default=5, type=int, help='Training epochs.')
parser.add_argument(
    '--quantizer_lr',
    default=1e-2,
    type=float,
    help='Initial learning rate of quantizer.')
parser.add_argument(
    '--quantizer_lr_decay',
    default=0.5,
    type=int,
    help='Learning rate decay ratio of quantizer.')
parser.add_argument(
    '--weight_lr',
    default=1e-5,
    type=float,
    help='Initial learning rate of network weights.')
parser.add_argument(
    '--weight_lr_decay',
    default=0.94,
    type=int,
    help='Learning rate decay ratio of network weights.')
parser.add_argument(
    '--train_batch_size', default=24, type=int, help='Batch size for training.')
parser.add_argument(
    '--val_batch_size',
    default=100,
    type=int,
    help='Batch size for validation.')
parser.add_argument(
    '--weight_decay', default=1e-4, type=float, help='Weight decay.')
parser.add_argument(
    '--display_freq',
    default=100,
    type=int,
    help='Display training metrics every n steps.')
parser.add_argument(
    '--val_freq', default=1000, type=int, help='Validate model every n steps.')
parser.add_argument(
    '--quantizer_norm',
    default=True,
    type=bool,
    help='Use normlization for quantizer.')
parser.add_argument(
    '--mode',
    default='train',
    choices=['train', 'test', 'deploy', 'qat_test'],
    help='Running mode.')
parser.add_argument(
    '--deployable',
    default='./deployable.pth',
    help='Deployable model file path.')
parser.add_argument(
    '--qat_model_path',
    help='float qat model file path.')
parser.add_argument(
    '--save_dir',
    default='./qat_models',
    help='Directory to save trained models.')
parser.add_argument(
    '--output_dir', default='qat_result', help='Directory to save qat result.')
parser.add_argument( '--distillation', action='store_true')
args, _ = parser.parse_known_args()


def train_one_step(model, inputs, criterion, optimizer, step, device):
  # switch to train mode
  model.train()

  images, target = inputs
  
  model = model.to(device)
  images = images.cuda(non_blocking=True)
  target = target.cuda(non_blocking=True)
  
  # compute output
  output = model(images)
  loss = criterion(output, target)

  l2_decay = 1e-4
  l2_norm = 0.0
  for param in model.quantizer_parameters():
    l2_norm += torch.pow(param, 2.0)[0]
  if args.quantizer_norm:
    loss += l2_decay * torch.sqrt(l2_norm)

  # measure accuracy and record loss
  acc1, acc5 = accuracy(output, target, topk=(1, 5))

  # compute gradient and do SGD step
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss, acc1, acc5

def validate(val_loader, model, criterion, device):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(val_loader), [batch_time, losses, top1, top5], prefix='Test: ')

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
      model = model.to(device)
      images = images.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

      # compute output
      output = model(images)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

    # TODO: this should also be done with the ProgressMeter
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

  return top1.avg

def mkdir_if_not_exist(x):
  if not x or os.path.isdir(x):
    return
  os.mkdir(x)
  if not os.path.isdir(x):
    raise RuntimeError("Failed to create dir %r" % x)

def save_checkpoint(state, is_best, directory):
  mkdir_if_not_exist(directory)

  filepath = os.path.join(directory, 'model.pth')
  torch.save(state, filepath)
  if is_best:
    best_acc1 = state['best_acc1'].item()
    best_filepath = os.path.join(directory, 'model_best_%5.3f.pth' % best_acc1)
    shutil.copyfile(filepath, best_filepath)
    print('Saving best ckpt to {}, acc1: {}'.format(best_filepath, best_acc1))

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

class ProgressMeter(object):

  def __init__(self, num_batches, meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, step):
  """Sets the learning rate to the initial LR decayed by decay ratios"""

  weight_lr_decay_steps = 3000 * (24 / args.train_batch_size)
  quantizer_lr_decay_steps = 1000 * (24 / args.train_batch_size)

  for param_group in optimizer.param_groups:
    group_name = param_group['name']
    if group_name == 'weight' and step % weight_lr_decay_steps == 0:
      lr = args.weight_lr * (
          args.weight_lr_decay**(step / weight_lr_decay_steps))
      param_group['lr'] = lr
      print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
          epoch, step, group_name, lr))
    if group_name == 'quantizer' and step % quantizer_lr_decay_steps == 0:
      lr = args.quantizer_lr * (
          args.quantizer_lr_decay**(step / quantizer_lr_decay_steps))
      param_group['lr'] = lr
      print('Adjust lr at epoch {}, step {}: group_name={}, lr={}'.format(
          epoch, step, group_name, lr))

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

def train(model, train_loader, val_loader, criterion, device):
  best_acc1 = 0

  num_train_batches_per_epoch = int(len(train_loader) / args.train_batch_size)

  batch_time = AverageMeter('Time', ':6.3f')
  data_time = AverageMeter('Data', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')

  param_groups = [{
      'params': model.quantizer_parameters(),
      'lr': args.quantizer_lr,
      'name': 'quantizer'
  }, {
      'params': model.non_quantizer_parameters(),
      'lr': args.weight_lr,
      'name': 'weight'
  }]
  optimizer = torch.optim.Adam(
      param_groups, args.weight_lr, weight_decay=args.weight_decay)

  for epoch in range(args.epochs):
    progress = ProgressMeter(
        len(train_loader) * args.epochs,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch[{}], Step: ".format(epoch))

    for i, (images, target) in enumerate(train_loader):
      end = time.time()
      # measure data loading time
      data_time.update(time.time() - end)

      step = len(train_loader) * epoch + i

      adjust_learning_rate(optimizer, epoch, step)
      loss, acc1, acc5 = train_one_step(model, (images, target), criterion,
                                        optimizer, step, device=torch.device("cuda"))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      if step % args.display_freq == 0:
        progress.display(step)

      if step>0 and step % args.val_freq == 0:
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1
            }, is_best, args.save_dir)

def main():
  print('Used arguments:', args)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if args.mode == 'deploy':
      device = torch.device('cpu')
  
  traindir = os.path.join(args.data_dir, 'train')
  valdir = os.path.join(args.data_dir, 'val')
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_transform = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
  ])
  val_transforms = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
  ])

  if args.mode == 'train':
      train_dataset = datasets.ImageFolder(
          traindir,
          transform= train_transform)
    
      train_loader = torch.utils.data.DataLoader(
          train_dataset,
          batch_size=args.train_batch_size,
          shuffle=True,
          num_workers=args.workers,
          pin_memory=True)

  val_dataset = datasets.ImageFolder(
      valdir, val_transforms)

  val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.val_batch_size,
      shuffle=False,
      num_workers=args.workers,
      pin_memory=True)

  model = resnet50()
#  model.load_state_dict(torch.load(args.pretrained))
  if args.pretrained:
      print('Load pretrain weights:', args.pretrained)
      ckpt = torch.load(args.pretrained)
      if 'state_dict' in ckpt:
          ckpt = ckpt['state_dict']
      params = {}
      for i in ckpt.keys():
          if i.split('.')[0]=='module':
              params[i[7:]]=ckpt[i]
          else:
              params[i]=ckpt[i]
      model.load_state_dict(params)
  else:
      print('No pretrain weights is loaded')

  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss()

  gpu = 0
  model = model.to(device)
  inputs = torch.randn([1, 3, 224, 224],
                       dtype=torch.float32).to(device)
  qat_processor = QatProcessor(
      model, inputs, bitwidth=8, device=device)

  if args.mode == 'train':
    # Step 1: Get quantized model and train it.
    quantized_model = qat_processor.trainable_model()

    criterion = criterion.cuda()
    train(quantized_model, train_loader, val_loader, criterion, device=torch.device("cuda"))

    # Step 2: Get deployable model and test it.
    # There may be some slight differences in accuracy with the quantized model.
    deployable_model = qat_processor.to_deployable(quantized_model,
                                                   args.output_dir)
    validate(val_loader, deployable_model, criterion, device=torch.device("cuda"))
  elif args.mode == 'test':
    quantized_model = qat_processor.trainable_model()
    deployable_model = qat_processor.deployable_model(args.output_dir, used_for_xmodel=False)
    validate(val_loader, deployable_model, criterion, device=torch.device("cuda"))

  elif args.mode == 'deploy':
    quantized_model = qat_processor.trainable_model()
    quantized_model.load_state_dict(torch.load(args.qat_model_path)['state_dict']) # load float qat model
    deployable_model = qat_processor.to_deployable(quantized_model, output_dir=args.output_dir)
    from pytorch_nndct.apis import torch_quantizer
    quantizer = torch_quantizer('test', model, inputs, device=device, output_dir=args.output_dir)
    quantizer.quant_model.load_state_dict(deployable_model.state_dict())
    quantizer.quant_model.eval()
    quantizer.quant_model(inputs)
    quantizer.export_xmodel(args.output_dir, deploy_check=True)
    quantizer.export_torch_script(args.output_dir)
    quantizer.export_onnx_model(args.output_dir)
    
  else:
    raise ValueError('mode must be one of [train, test, deploy, qat_test]')

if __name__ == '__main__':
  main()
