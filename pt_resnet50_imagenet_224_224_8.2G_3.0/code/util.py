import torch
from torch import Tensor
import torchvision.transforms as transforms
from collections.abc import Sequence
from typing import List, Optional, Tuple

import math
import numpy as np
import cv2
from PIL import Image

def f_divergence(q_logits, p_logits, alpha, iw_clip=1e3):
    assert isinstance(alpha, float)
    q_prob = torch.nn.functional.softmax(q_logits, dim=1).detach()
    p_prob = torch.nn.functional.softmax(p_logits, dim=1).detach()
    q_log_prob = torch.nn.functional.log_softmax(q_logits, dim=1) #gradient is only backpropagated here
  
    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio = importance_ratio.clamp(0, iw_clip)
        f = -importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio.log() - 1.0
    elif abs(alpha - 1.0) < 1e-3:
        f = importance_ratio * importance_ratio.log()
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha)
        iw_alpha = iw_alpha.clamp(0, iw_clip)
        f = iw_alpha / alpha / (alpha - 1.0)
        f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha / alpha + f_base
  
    loss = torch.sum(q_prob * (f - f_base), dim=1)
    grad_loss = -torch.sum(q_prob * rho_f * q_log_prob, dim=1)
    return loss, grad_loss

def AdaptiveKDLossSoft(pred, target, alpha_min=-1.0, alpha_max=1.0, iw_clip=5.0):
    loss_left, grad_loss_left = f_divergence(pred, target, alpha_min, iw_clip=iw_clip)
    loss_right, grad_loss_right = f_divergence(pred, target, alpha_max, iw_clip=iw_clip)
  
    ind = torch.gt(loss_left, loss_right).float()
    # loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right
    loss = (ind * grad_loss_left + (1.0 - ind) * grad_loss_right) + 1.5
  
    return loss.mean()
  
class RandomCrop(torch.nn.Module):
    def __init__(
        self,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
    ):
        super().__init__()
        #_log_api_usage_once(self)

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        #_, height, width = F.get_dimensions(img)
        width, height= img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped image
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return transforms.functional.crop(img, i, j, h, w)

class Cv2Resize(torch.nn.Module):
    def __init__(self, resize_shape):
        super().__init__()
        self.resize_shape = resize_shape

    #def __call__(self, img):
    def forward(self, img):
        img = np.asarray(img)
        img = cv2.resize(img, self.resize_shape)
        return Image.fromarray(img)
